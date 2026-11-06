"""
collect_io.py -- CPU-importable helpers shared by run_libero_collect.py, audit_collection.py and the tests.

Atomic writes, per-job temp files, O_EXCL episode locks, quarantine, per-episode seeds and episode-file
validation. No libero / robosuite / model imports, so it works on a login node or laptop.
"""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ACTION_DIM = 7
REQUIRED_META = ("task_suite", "task_id", "init_state_idx", "seed", "invalid", "length", "success", "error")


# --------------------------------------------------------------------------------------------------
# Atomic IO + validation
# --------------------------------------------------------------------------------------------------
def episode_path(root: Path, task_id: int, init_state_idx: int, seed: int) -> Path:
    return root / f"task{task_id:02d}" / f"ep{init_state_idx:03d}_seed{seed}.pt"


def job_identity() -> dict:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "restart": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "time": time.time(),
    }


def _tmp_id() -> str:
    return os.environ.get("SLURM_JOB_ID") or str(os.getpid())


def derive_episode_seed(base_seed: int, task_id: int, init_state_idx: int) -> int:
    """Deterministic per-episode seed (independent of which episodes were skipped on resume)."""
    h = hashlib.sha256(f"{base_seed}:{task_id}:{init_state_idx}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31 - 1)


def _slurm_job_running(job_id: str) -> Optional[bool]:
    """True/False if squeue can tell us, None if unknown (no squeue / error)."""
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if out.returncode != 0:
        # squeue exits non-zero with "Invalid job id" for finished jobs on many configs, but also on real errors
        return False if "Invalid job id" in out.stderr else None
    return out.stdout.strip() != ""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def holder_alive(holder: dict, max_age_h: float) -> bool:
    me = job_identity()
    same_job = holder.get("job_id") is not None and holder.get("job_id") == me["job_id"]
    if same_job and holder.get("restart") != me["restart"]:
        return False  # same Slurm job id, earlier incarnation (requeue): dead
    if holder.get("host") == me["host"]:
        return _pid_alive(int(holder.get("pid", -1)))  # same node: check the process directly
    if holder.get("job_id"):
        running = _slurm_job_running(holder["job_id"])
        if running is not None:
            return running
    return (time.time() - float(holder.get("time", 0))) < max_age_h * 3600  # unknown: trust lock until it is old


def lock_path(path: Path) -> Path:
    return path.with_suffix(".lock")


def try_claim(path: Path, max_age_h: float) -> bool:
    """O_EXCL claim of an episode. False if another live job holds it."""
    lp = lock_path(path)
    lp.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = json.loads(lp.read_text())
            except Exception:
                holder = {}  # unreadable lock (being written / corrupt): treat by age below
                if time.time() - lp.stat().st_mtime < 60:
                    return False
            if holder and holder_alive(holder, max_age_h):
                return False
            # stale: move it aside (atomic) so we never delete a fresh lock another job just created
            aside = lp.with_name(lp.name + f".stale.{_tmp_id()}.{int(time.time())}")
            try:
                os.rename(lp, aside)
                print(f"[lock] reclaimed stale lock {lp.name} (holder={holder})")
                aside.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(job_identity()))
            f.flush()
            os.fsync(f.fileno())
        return True
    return False


def release_claim(path: Path) -> None:
    lp = lock_path(path)
    try:
        holder = json.loads(lp.read_text())
        if holder.get("pid") == os.getpid() and holder.get("host") == socket.gethostname():
            lp.unlink()
    except Exception:
        pass


def atomic_save(obj, path: Path) -> int:
    """Write to a per-job temp file in the same directory, fsync, then rename. Returns bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{_tmp_id()}.tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        size = os.path.getsize(tmp)
        os.replace(tmp, path)  # atomic on POSIX within one filesystem
        return size
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


SIDECAR_FORMAT = 1


def sidecar_path(pt_path: Path) -> Path:
    """ep{NNN}_seed{S}.pt -> ep{NNN}_seed{S}.meta.json"""
    assert pt_path.name.endswith(".pt"), pt_path
    return pt_path.with_name(pt_path.name[: -len(".pt")] + ".meta.json")


def _tensor_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def build_sidecar(episode: dict, pt_bytes: int) -> dict:
    """Small JSON-able summary of an episode: its meta plus sizes, so audits can run without loading the .pt."""
    return {
        "format": SIDECAR_FORMAT,
        "meta": json.loads(json.dumps(episode["meta"], default=str)),  # JSON round-trip, exactly what is compared later
        "pt_bytes": int(pt_bytes),
        "feature_bytes": _tensor_bytes(episode.get("steps", {}).get("features", {})),
        "input_bytes": _tensor_bytes(episode.get("inputs", {})),
        "token_level_bytes": _tensor_bytes(episode.get("token_level_sample", {})),
    }


def atomic_write_json(obj, path: Path) -> int:
    """Same rules as atomic_save: per-job temp name, fsync, os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{_tmp_id()}.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        size = os.path.getsize(tmp)
        os.replace(tmp, path)
        return size
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def save_episode(episode: dict, path: Path) -> int:
    """Atomic .pt write, THEN the atomic .meta.json sidecar. A crash between the two leaves a valid .pt without
    sidecar; the collector regenerates it on resume (ensure_sidecar). Returns .pt bytes."""
    nbytes = atomic_save(episode, path)
    atomic_write_json(build_sidecar(episode, nbytes), sidecar_path(path))
    return nbytes


def read_sidecar(pt_path: Path) -> Optional[dict]:
    sp = sidecar_path(pt_path)
    try:
        return json.loads(sp.read_text())
    except Exception:
        return None


def sidecar_mismatch(sidecar: dict, episode: dict, pt_bytes: int) -> Optional[str]:
    """None if the sidecar agrees with the loaded .pt (meta and size), else a reason."""
    if sidecar.get("format") != SIDECAR_FORMAT:
        return f"sidecar format {sidecar.get('format')}"
    want = json.loads(json.dumps(episode["meta"], default=str))
    if sidecar.get("meta") != want:
        diff = sorted(k for k in set(want) | set(sidecar.get("meta", {})) if want.get(k) != sidecar.get("meta", {}).get(k))
        return f"meta differs in {diff}"
    if sidecar.get("pt_bytes") != pt_bytes:
        return f"pt_bytes {sidecar.get('pt_bytes')} != {pt_bytes}"
    return None


def ensure_sidecar(pt_path: Path, episode: dict) -> bool:
    """Write the sidecar if missing or inconsistent with the (valid) .pt. Returns True if (re)written."""
    sc = read_sidecar(pt_path)
    size = pt_path.stat().st_size
    if sc is not None and sidecar_mismatch(sc, episode, size) is None:
        return False
    atomic_write_json(build_sidecar(episode, size), sidecar_path(pt_path))
    return True


def remove_stale_tmp(root: Path, max_age_h: float) -> int:
    """Remove only temp files older than max_age_h whose owning job (id in the file name) is not running."""
    n = 0
    for p in root.glob("task*/*.tmp"):
        if time.time() - p.stat().st_mtime < max_age_h * 3600:
            continue
        owner = p.name[: -len(".tmp")].rsplit(".", 1)[-1]
        if owner.isdigit() and _slurm_job_running(owner):
            continue
        p.unlink()
        n += 1
    return n


def quarantine(path: Path, root: Path, reason: str) -> Path:
    """Move a bad episode file aside for debugging. Never deletes."""
    qdir = root / "_quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    dest = qdir / f"{path.parent.name}__{path.name}.{time.strftime('%Y%m%dT%H%M%S')}.{_tmp_id()}"
    shutil.move(str(path), str(dest))
    sp = sidecar_path(path)
    if sp.exists():
        shutil.move(str(sp), str(qdir / (dest.name + ".meta.json")))
    print(f"[quarantine] {path} -> {dest} ({reason})")
    with open(qdir / "quarantine_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{path}\t{dest}\t{reason}\n")
    return dest


def load_and_validate(path: Path, task_id: int, init_state_idx: int, seed: int):
    """Read-only. Returns (episode or None, reason or None). Loads the file once so callers can reuse the data."""
    if not path.exists():
        return None, "missing"
    try:
        ep = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # truncated / corrupt
        return None, f"unreadable: {e!r}"
    reason = _validate_loaded(ep, task_id, init_state_idx, seed)
    return (ep if reason is None else None), reason


def validate_episode_file(path: Path, task_id: int, init_state_idx: int, seed: int) -> Optional[str]:
    """Return None if the file is a complete, self-consistent episode; else a reason string."""
    return load_and_validate(path, task_id, init_state_idx, seed)[1]


def _validate_loaded(ep, task_id: int, init_state_idx: int, seed: int) -> Optional[str]:
    if not isinstance(ep, dict):
        return "not a dict"
    meta, steps = ep.get("meta"), ep.get("steps")
    if meta is None or steps is None:
        return "missing meta/steps"
    miss = [k for k in REQUIRED_META if k not in meta]
    if miss:
        return f"meta missing {miss}"
    if (meta["task_id"], meta["init_state_idx"], meta["seed"]) != (task_id, init_state_idx, seed):
        return "meta does not match filename"
    if meta["invalid"]:
        return None if meta["error"] else "invalid episode without error message"
    T = meta["length"]
    if meta["success"] is None or not isinstance(meta["success"], bool):
        return "valid episode without boolean success"
    if meta.get("label_failure") != int(not meta["success"]):
        return "label_failure != 1 - success"
    if T <= 0 or len(steps["s"]) != T or len(steps["env_t"]) != T:
        return "length mismatch (s/env_t)"
    if tuple(np.asarray(steps["action_token_ids"]).shape) != (T, ACTION_DIM):
        return "action_token_ids shape"
    if tuple(np.asarray(steps["executed_action"]).shape) != (T, ACTION_DIM):
        return "executed_action shape"
    for k in ("n_out_of_range_token_steps", "out_of_range_token_rate", "action_bin_token_range",
              "n_tokens_at_bin_min_edge", "n_tokens_at_pad_edge"):
        if meta.get(k) is None:
            return f"meta.{k} missing"
    if len(steps.get("token_out_of_range", [])) != T:
        return "token_out_of_range length"
    if int(np.asarray(steps["token_out_of_range"]).sum()) != meta["n_out_of_range_token_steps"]:
        return "n_out_of_range_token_steps != sum(token_out_of_range)"
    if int(np.asarray(steps["action_token_ids"]).reshape(-1).tolist().count(meta["action_bin_token_range"][1] + 1)) != meta["n_tokens_at_pad_edge"]:
        return "n_tokens_at_pad_edge inconsistent with action_token_ids"
    if int((np.asarray(steps["action_token_ids"]) == meta["action_bin_token_range"][0]).sum()) != meta["n_tokens_at_bin_min_edge"]:
        return "n_tokens_at_bin_min_edge inconsistent with action_token_ids"
    for k in ("token_probs", "token_entropy"):
        if tuple(np.asarray(steps.get(k, [])).shape) != (T, ACTION_DIM):
            return f"{k} shape"
    feats = steps.get("features", {})
    for k, v in feats.items():
        if v.shape[0] != T:
            return f"feature {k} length {v.shape[0]} != {T}"
    if meta.get("features_captured"):
        keys, pooling = meta.get("layer_keys"), meta.get("pooling")
        if not keys or not pooling:
            return "features_captured but meta.layer_keys/pooling missing"
        if sorted(feats) != sorted(keys):
            return f"feature keys {sorted(feats)} != meta.layer_keys {sorted(keys)}"
        for k, v in feats.items():
            if v.dtype != torch.bfloat16 or v.ndim != 3 or v.shape[1] != len(pooling):
                return f"feature {k} dtype/shape {v.dtype}/{tuple(v.shape)} (expected bf16 [T,{len(pooling)},D])"
        for need in ("layer_mean_def", "attn_implementation"):
            if need not in meta:
                return f"meta.{need} missing"
    tl = ep.get("token_level_sample")
    if tl is not None:
        tk = meta.get("token_level_keys")
        if not tk or tl.dtype != torch.bfloat16 or tl.ndim != 4 or tl.shape[0] != T or tl.shape[2] != len(tk):
            return "token_level_sample dtype/shape inconsistent with meta.token_level_keys"
    return None


