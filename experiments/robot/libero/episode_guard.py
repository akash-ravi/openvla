"""Invalid-episode handling for `run_libero_eval.py --save_all_layers` (shared-cache kit, must-fix 4 / choice E2).

SAFE's script re-raises ANY exception from the step loop, so one bad episode kills a whole collection job. With
`--save_all_layers` the patched script instead calls `EpisodeGuard.record_invalid()`:

  * ordinary per-episode error (e.g. a simulator error for one initial state):
      - every output of that episode that already exists (`task{t}--ep{e}--succ*`: .mp4/.pkl/.layers.pt/.csv and
        their .tmp files, e.g. left over from an earlier, killed attempt) is removed FIRST,
      - then the marker `task{t}--ep{e}--invalid.json` is written atomically (per-process tmp file, fsync, rename),
      - the episode writes NO .mp4/.pkl/.layers.pt/.csv (the replay video is dropped too: a `--succ0.mp4` would
        claim an outcome the episode never had; the marker's traceback and step index are the debugging record, and
        the episode can be replayed from its fixed initial state),
      - the env is reset (rebuilt if the reset itself fails) and the job continues with the next episode.
  * systemic error (see `is_systemic`): re-raised unchanged, no marker. These would hit every later episode too.
  * more than `max_invalid_episodes` markers in the job's task range (markers already on disk from earlier runs
    count, so restarts cannot reset the count): the marker is written, then `TooManyInvalidEpisodes` aborts the job
    (non-zero exit). If the cause turns out to be systemic, delete the listed markers before resuming.

Re-attempting an episode (a run without --resume restarts at episode 0) deletes its old marker first.

Resume: SAFE's `--resume` starts each task after the highest episode index that has a `.csv`. With `--save_all_layers`
the markers are added to that set (`invalid_markers`), so a deterministic error is not retried forever. Delete a
marker to retry that episode.

Errors before the first episode (model load, processor, env construction, provenance) happen outside the step loop
and are never caught here. Exceptions outside the step loop's try block (env.reset / set_init_state at the start of
an episode, writing the outputs) are not caught either and still abort the job, as in SAFE.

Marker format (JSON object, `format` = 1): format, kind="invalid_episode", task_suite_name, task_id, episode_idx,
init_state_index, seed, step_index (env step counter t incl. the num_steps_wait settling steps), num_steps_wait,
policy_steps_completed, exception_type (module-qualified), exception_message, traceback, timestamp_utc (ISO 8601),
job {slurm_job_id, slurm_array_task_id, modal_task_id, pid}, invalid_count_in_job_range, max_invalid_episodes.
"""
from __future__ import annotations

import datetime as dt
import errno
import json
import os
import re
import sys
import traceback as tb
from pathlib import Path
from typing import Callable, Iterable, Optional

MARKER_FORMAT = 1
MARKER_RE = re.compile(r"^task(\d+)--ep(\d+)--invalid\.json$")

# Substrings (case-insensitive) that mark an error as systemic regardless of its Python type: CUDA / driver / cuBLAS /
# cuDNN failures, device-side asserts, OOM, flash-attn refusing the GPU, and a broken offscreen-render (EGL/OpenGL) context.
SYSTEMIC_PATTERNS = (
    "cuda error", "cuda out of memory", "out of memory", "device-side assert", "illegal memory access",
    "unspecified launch failure", "cublas_status", "cudnn_status", "cudnn error", "nccl error", "cuda driver",
    "no cuda gpus", "ecc error", "flashattention", "flash_attn", "egl_", "glerror", "opengl",
)
# Python types that are never per-episode: interpreter/OOM signals and code/installation problems.
SYSTEMIC_TYPES = (MemoryError, ImportError, NameError, SyntaxError)
# Disk full / quota / read-only filesystem: nothing further can be written, not even a marker.
SYSTEMIC_ERRNOS = {errno.ENOSPC, errno.EROFS} | ({errno.EDQUOT} if hasattr(errno, "EDQUOT") else set())


class TooManyInvalidEpisodes(RuntimeError):
    """Raised (chained to the episode's exception) when the invalid-episode cap is exceeded."""


def marker_path(save_folder: Path, task_id: int, episode_idx: int) -> Path:
    return Path(save_folder) / f"task{task_id}--ep{episode_idx}--invalid.json"


def invalid_markers(save_folder: Path, task_id: int) -> list[Path]:
    """Markers of one task, in the same shape as SAFE's `glob(f"task{t}--ep*--succ*.csv")` result for --resume."""
    return sorted(p for p in Path(save_folder).glob(f"task{task_id}--ep*--invalid.json") if MARKER_RE.match(p.name))


def _cuda_oom_type():
    torch = sys.modules.get("torch")
    return getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None) if torch is not None else None


def is_systemic(exc: BaseException) -> bool:
    """True if `exc` (or anything in its __cause__/__context__ chain) must abort the whole job."""
    seen, cur = set(), exc
    oom = _cuda_oom_type()
    while cur is not None and id(cur) not in seen and len(seen) < 20:
        seen.add(id(cur))
        if not isinstance(cur, Exception):  # KeyboardInterrupt, SystemExit, GeneratorExit
            return True
        if isinstance(cur, SYSTEMIC_TYPES) or (oom is not None and isinstance(cur, oom)):
            return True
        if isinstance(cur, OSError) and cur.errno in SYSTEMIC_ERRNOS:
            return True
        text = f"{type(cur).__name__}: {cur}".lower()
        if any(p in text for p in SYSTEMIC_PATTERNS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def atomic_write_json(obj: dict, path: Path) -> None:
    """Per-process tmp file in the same directory, fsync, os.replace (atomic on POSIX); tmp removed on any failure."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def remove_episode_outputs(save_folder: Path, task_id: int, episode_idx: int) -> list[Path]:
    """Delete every `task{t}--ep{e}--succ*` file (mp4/pkl/layers.pt/csv and their .tmp). `--ep1--` never matches ep10."""
    removed = []
    for p in sorted(Path(save_folder).glob(f"task{task_id}--ep{episode_idx}--succ*")):
        if p.is_file():
            p.unlink()
            removed.append(p)
    return removed


def _job_identity() -> dict:
    return {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"), "pid": os.getpid()}


class EpisodeGuard:
    """One per job. `task_ids` = the job's task range; the cap counts markers of those tasks on disk."""

    def __init__(self, save_folder: Path, max_invalid_episodes: int, task_ids: Iterable[int],
                 task_suite_name: str = "", seed: Optional[int] = None):
        if int(max_invalid_episodes) < 0:
            raise ValueError(f"max_invalid_episodes must be >= 0, got {max_invalid_episodes}")
        self.save_folder = Path(save_folder)
        self.max_invalid = int(max_invalid_episodes)
        self.task_ids = list(task_ids)
        self.task_suite_name = task_suite_name
        self.seed = seed
        self.written: list[Path] = []  # markers written by this process

    def invalid_count(self) -> int:
        return sum(len(invalid_markers(self.save_folder, t)) for t in self.task_ids)

    def record_invalid(self, exc: BaseException, *, task_id: int, episode_idx: int, step_index: int,
                       num_steps_wait: int = 0, policy_steps_completed: int = 0) -> bool:
        """Returns False for a systemic error (caller re-raises it unchanged). Otherwise removes the episode's partial
        outputs, writes its marker and returns True (caller drops the episode and continues), unless the cap is now
        exceeded, in which case TooManyInvalidEpisodes is raised from `exc`."""
        if is_systemic(exc):
            return False
        remove_episode_outputs(self.save_folder, task_id, episode_idx)  # before the marker: never both on disk
        path = marker_path(self.save_folder, task_id, episode_idx)
        count = self.invalid_count() + (0 if path.exists() else 1)
        record = {
            "format": MARKER_FORMAT,
            "kind": "invalid_episode",
            "task_suite_name": self.task_suite_name,
            "task_id": int(task_id),
            "episode_idx": int(episode_idx),
            "init_state_index": int(episode_idx),
            "seed": self.seed,
            "step_index": int(step_index),
            "num_steps_wait": int(num_steps_wait),
            "policy_steps_completed": int(policy_steps_completed),
            "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "exception_message": str(exc),
            "traceback": "".join(tb.format_exception(type(exc), exc, exc.__traceback__)),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "job": _job_identity(),
            "invalid_count_in_job_range": count,
            "max_invalid_episodes": self.max_invalid,
        }
        atomic_write_json(record, path)
        self.written.append(path)
        print(f"[invalid episode] task {task_id} ep {episode_idx} at step {step_index}: "
              f"{record['exception_type']}: {exc}\n  marker: {path} ({count}/{self.max_invalid} allowed)", flush=True)
        if count > self.max_invalid:
            raise TooManyInvalidEpisodes(
                f"{count} invalid episodes in tasks {self.task_ids} exceed max_invalid_episodes={self.max_invalid}; "
                f"aborting the job. Markers written by this run: {[str(p) for p in self.written]}. If the cause is "
                f"systemic, delete those markers before --resume (resume treats a marker as done).") from exc
        return True

    def discard_marker(self, task_id: int, episode_idx: int) -> None:
        """Called when an episode is (re-)attempted, e.g. by a run without --resume: its old marker no longer applies
        and must not end up next to the new outputs."""
        marker_path(self.save_folder, task_id, episode_idx).unlink(missing_ok=True)

    @staticmethod
    def recover_env(env, make_env: Callable[[], object]):
        """Reset the env after an invalid episode; if the reset fails (non-systemically), close it and build a new one.
        A failure of the rebuild propagates (the job cannot continue without an env)."""
        try:
            env.reset()
            return env
        except Exception as e:
            if is_systemic(e):
                raise
            print(f"[invalid episode] env.reset() failed ({type(e).__name__}: {e}); rebuilding the env", flush=True)
        try:
            env.close()
        except Exception:
            pass
        return make_env()
