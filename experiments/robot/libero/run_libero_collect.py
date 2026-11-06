"""
run_libero_collect.py  (A4 draft -- NOT RUN, NOT TESTED)

Resumable, exception-tolerant LIBERO rollout collector. New script; run_libero_eval.py is untouched.

Policy behaviour mirrors run_libero_eval.py (greedy, center_crop=True, num_steps_wait=10, max_steps=520 for
libero_10, same unnorm-key logic, same gripper post-processing). Only the bookkeeping differs:

  * One file per episode: {out_root}/task{task_id:02d}/ep{init_state_idx:03d}_seed{seed}.pt
  * Written atomically (temp file `ep{NNN}_seed{S}.pt.{SLURM_JOB_ID|pid}.tmp` in the same dir -> fsync ->
    os.replace). A killed job never leaves a partial file under the final name. At startup only temp files
    older than --stale_tmp_hours whose job is no longer running are removed.
  * Each episode is claimed with an O_EXCL lock file (`ep{NNN}_seed{S}.lock`: job id, host, pid). An episode
    whose lock is held by a live job is skipped; locks of dead jobs are reclaimed. Safe for concurrent and
    requeued Slurm jobs on the same task directory.
  * On restart an episode is skipped iff its final file exists AND validates (see validate_episode_file).
    A file that fails validation is MOVED to {out_root}/_quarantine/ (timestamp suffix, logged), never deleted,
    and the episode is re-collected.
  * Deterministic per-episode seed derived from (base_seed, task_id, init_state_idx), recorded in meta, so a
    resumed episode is identical to the same episode in an uninterrupted run. INTENTIONAL DEVIATION from SAFE,
    which seeds once at startup (see Safe-Research docs/compute_notes.md).
  * token_probs / token_entropy [T,7] are stored, computed with the fork's compute_token_uncertainty_metrics
    (identical to what run_libero_eval.py logs).
  * Exceptions are recorded per episode (meta.invalid=True, meta.error=<traceback>) and collection continues.
    An invalid episode is NEVER counted as a failure: success/label_failure are None. By default invalid files
    count as "done" on restart (so a deterministic crash is not retried forever); pass --retry_invalid to redo them.
  * After an exception the env is rebuilt, since its state is unknown.

TODO (depends on spec freeze P0-1, intentionally stubbed): feature capture (FeatureCapture), the final dtype /
layout / key set, inputs (A2, Dhyan), metadata fields (A3, Dhyan).
"""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union

import draccus
import numpy as np
import torch
from tqdm import tqdm

from libero.libero import benchmark

sys.path.append("../..")
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.openvla_utils import get_processor
from experiments.robot.unc_utils import compute_token_uncertainty_metrics
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

ACTION_DIM = 7
MAX_STEPS_LIBERO_10 = 520  # same as run_libero_eval.py (longest training demo has 505 steps)
REQUIRED_META = ("task_suite", "task_id", "init_state_idx", "seed", "invalid", "length", "success", "error")


@dataclass
class CollectConfig:
    # fmt: off
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True                         # asserted True, as in SAFE
    n_samples: int = 1
    attn_implementation: str = "flash_attention_2"   # same as SAFE default; recorded in meta, keep fixed for ALL collection
    output_logits: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = True                # needed once FeatureCapture is implemented

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    task_ids: Optional[List[int]] = None             # default: all tasks; use one job per task
    episode_start: int = 0                           # init_state_idx range [start, end)
    episode_end: int = 50
    seed: int = 7

    out_root: str = "./collect"
    retry_invalid: bool = False
    stale_tmp_hours: float = 24.0                    # only temp files older than this (and with no live job) are removed
    stale_lock_hours: float = 24.0                   # fallback lock expiry when job liveness cannot be determined
    # fmt: on


# --------------------------------------------------------------------------------------------------
# Feature capture: STUB (spec-dependent, P0-1)
# --------------------------------------------------------------------------------------------------
class FeatureCapture:
    """Placeholder. Real version: forward hooks on layers[3,7,...,31] + pre-hook on final norm, capture per
    generation state j (7 per action), pool first/last/mean AFTER per-token capture, move to CPU every step,
    clear buffers every action. See tests/gpu/test_layer_mapping.py (Safe-Research) for verified module paths."""

    def attach(self, model) -> None:
        pass

    def begin_action(self) -> None:
        pass

    def end_action(self, generated_outputs) -> dict:
        """Return {feature_key: tensor[3, 4096] bf16} for this env step. Stub returns nothing."""
        return {}

    def detach(self) -> None:
        pass


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
    print(f"[quarantine] {path} -> {dest} ({reason})")
    with open(qdir / "quarantine_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{path}\t{dest}\t{reason}\n")
    return dest


def validate_episode_file(path: Path, task_id: int, init_state_idx: int, seed: int) -> Optional[str]:
    """Return None if the file is a complete, self-consistent episode; else a reason string."""
    if not path.exists():
        return "missing"
    try:
        ep = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # truncated / corrupt
        return f"unreadable: {e!r}"
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
    if len(steps.get("token_out_of_range", [])) != T or meta.get("n_out_of_range_token_steps") is None:
        return "token_out_of_range / meta count missing"
    for k in ("token_probs", "token_entropy"):
        if tuple(np.asarray(steps.get(k, [])).shape) != (T, ACTION_DIM):
            return f"{k} shape"
    for k, v in steps.get("features", {}).items():
        if v.shape[0] != T:
            return f"feature {k} length {v.shape[0]} != {T}"
    return None


# --------------------------------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------------------------------
def run_episode(cfg, model, processor, env, task_description, init_state, max_steps, resize_size, capture):
    assert cfg.output_logits, "token_probs require output_logits=True"
    """Same control flow as run_libero_eval.py. Returns (steps_dict, success, n_steps)."""
    env.reset()
    obs = env.set_init_state(init_state)
    t = 0
    s_list, env_t_list, tok_list, act_list, prob_list, ent_list, oor_list = [], [], [], [], [], [], []
    # Action-bin token range = exactly the logits slice used for token_probs/entropy (unc_utils.py) and the range
    # predict_action decodes: ids in [vocab_size - n_action_bins + 1, vocab_size].
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    bin_lo, bin_hi = vocab_size - model.config.n_action_bins + 1, vocab_size
    feats = {}
    success = False
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue
        img = get_libero_image(obs, resize_size)  # lossless copy of this is what A2 will store (pre center-crop)
        observation = {
            "full_image": img,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }
        capture.begin_action()
        actions = get_action(cfg, model, observation, task_description, processor=processor, n_samples=cfg.n_samples)
        actions, generated_outputs = actions
        step_feats = capture.end_action(generated_outputs)
        for k, v in step_feats.items():
            feats.setdefault(k, []).append(v)

        actions = normalize_gripper_action(actions, binarize=True)
        actions = invert_gripper_action(actions)
        assert actions.ndim == 1, "collector supports n_samples == 1 only"

        s_list.append(t - cfg.num_steps_wait)
        env_t_list.append(t)
        tok = generated_outputs["sequences"][0, -ACTION_DIM:].cpu().numpy()
        tok_list.append(tok)
        # Under greedy decoding the chosen token can fall outside the action bins; the fork then clips the action
        # and token_probs is NOT the chosen token's probability. We keep SAFE's prob definition and just count these.
        oor_list.append(bool(((tok < bin_lo) | (tok > bin_hi)).any()))
        act_list.append(np.asarray(actions, dtype=np.float32))
        # Same computation as run_libero_eval.py logs: softmax over the action-bin slice of the logits;
        # "prob" = max bin prob (== chosen token's prob under greedy when the argmax is a bin token).
        m = compute_token_uncertainty_metrics(generated_outputs, model)
        prob_list.append([m[f"token_{i}_prob"] for i in range(ACTION_DIM)])
        ent_list.append([m[f"token_{i}_entropy"] for i in range(ACTION_DIM)])

        obs, reward, done, info = env.step(actions.tolist())
        if done:
            success = True
            break
        t += 1

    steps = {
        "s": np.asarray(s_list, dtype=np.int64),
        "env_t": np.asarray(env_t_list, dtype=np.int64),
        "action_token_ids": np.stack(tok_list) if tok_list else np.zeros((0, ACTION_DIM), dtype=np.int64),
        "executed_action": np.stack(act_list) if act_list else np.zeros((0, ACTION_DIM), dtype=np.float32),
        "features": {k: torch.stack(v) for k, v in feats.items()},  # [T, 3, 4096] once capture is real
        "token_out_of_range": np.asarray(oor_list, dtype=bool),  # [T]: any of the 7 chosen ids outside the bins
        "token_probs": np.asarray(prob_list, dtype=np.float32).reshape(-1, ACTION_DIM),
        "token_entropy": np.asarray(ent_list, dtype=np.float32).reshape(-1, ACTION_DIM),
    }
    steps["action_bin_range"] = (int(bin_lo), int(bin_hi))
    return steps, success, len(s_list)


def _quat2axisangle(q):
    from experiments.robot.libero.libero_utils import quat2axisangle
    return quat2axisangle(q)


def base_meta(cfg, task_id, task_description, init_state_idx, unnorm_key):
    return {
        "task_suite": cfg.task_suite_name, "task_id": task_id, "task_description": task_description,
        "init_state_idx": init_state_idx, "seed": cfg.seed,
        "checkpoint": str(cfg.pretrained_checkpoint),  # TODO(A3): append @<revision>
        "dtype": "bfloat16", "center_crop": cfg.center_crop, "unnorm_key": unnorm_key,
        "num_steps_wait": cfg.num_steps_wait, "max_steps": MAX_STEPS_LIBERO_10,
        "attn_implementation": cfg.attn_implementation,  # must be identical across the whole collection
        "torch": torch.__version__,
        "features_captured": False,  # TODO(P0-1): set by real FeatureCapture
        # TODO(A3): versions, git commits, layer_keys, pooling, image_boundary
    }


@draccus.wrap()
def collect(cfg: CollectConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must not be None!"
    assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert cfg.task_suite_name == "libero_10" and cfg.n_samples == 1
    set_seed_everywhere(cfg.seed)

    # Un-norm key: identical to run_libero_eval.py:121-132
    unnorm_key = cfg.task_suite_name
    model = get_model(cfg)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key  # read by get_action()
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    root = Path(cfg.out_root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Removed {remove_stale_tmp(root, cfg.stale_tmp_hours)} stale temp files")
    (root / "collect_config.json").write_text(json.dumps(asdict(cfg), default=str, indent=2))

    capture = FeatureCapture()
    capture.attach(model)

    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task_ids = cfg.task_ids if cfg.task_ids is not None else list(range(suite.n_tasks))
    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        stats = dict(done=0, skipped=0, invalid=0, success=0, failure=0)
        for idx in tqdm(range(cfg.episode_start, cfg.episode_end), desc=f"task {task_id}"):
            path = episode_path(root, task_id, idx, cfg.seed)
            if path.exists():
                reason = validate_episode_file(path, task_id, idx, cfg.seed)
                if reason is None:
                    ep_meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
                    if not (ep_meta["invalid"] and cfg.retry_invalid):
                        stats["skipped"] += 1
                        continue
                    quarantine(path, root, "invalid episode, --retry_invalid")
                else:
                    quarantine(path, root, f"failed validation: {reason}")
            if not try_claim(path, cfg.stale_lock_hours):
                print(f"[lock] task {task_id} ep {idx} held by another live job; skipping")
                stats["skipped"] += 1
                continue
            try:
                # Re-check after claiming: another job may have finished it between our check and our claim.
                if path.exists() and validate_episode_file(path, task_id, idx, cfg.seed) is None:
                    stats["skipped"] += 1
                    continue
                ep_seed = derive_episode_seed(cfg.seed, task_id, idx)
                set_seed_everywhere(ep_seed)  # intentional deviation from SAFE (seeds once); makes resume reproducible
                meta = base_meta(cfg, task_id, task_description, idx, unnorm_key)
                meta["episode_seed"] = ep_seed
                meta["episode_seed_derivation"] = "sha256(f'{base_seed}:{task_id}:{init_state_idx}')[:4] % (2**31-1)"
                t0 = time.time()
                try:
                    steps, success, n = run_episode(
                        cfg, model, processor, env, task_description, initial_states[idx],
                        MAX_STEPS_LIBERO_10, resize_size, capture,
                    )
                    oor = steps.pop("action_bin_range")
                    meta.update(success=bool(success), label_failure=int(not success), length=n, invalid=False, error=None,
                                n_out_of_range_token_steps=int(steps["token_out_of_range"].sum()),
                                out_of_range_token_rate=float(steps["token_out_of_range"].mean()) if n else 0.0,
                                action_bin_token_range=list(oor))
                    episode = {"meta": meta, "steps": steps}  # TODO(A2): "inputs"
                    stats["success" if success else "failure"] += 1
                except Exception:
                    # Record and continue. NOT a failure: success/label_failure are None.
                    meta.update(success=None, label_failure=None, length=0, invalid=True, error=traceback.format_exc())
                    episode = {"meta": meta, "steps": {}}
                    stats["invalid"] += 1
                    print(f"[invalid] task {task_id} ep {idx}:\n{meta['error']}")
                    try:
                        env.close()
                    except Exception:
                        pass
                    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)  # fresh env
                    capture.begin_action()  # clear any half-filled hook buffers
                meta["wall_time_s"] = time.time() - t0
                nbytes = atomic_save(episode, path)
                stats["done"] += 1
                print(f"task {task_id} ep {idx}: len={meta['length']} invalid={meta['invalid']} "
                      f"steps/s={meta['length'] / max(meta['wall_time_s'], 1e-9):.2f} bytes={nbytes}")
            finally:
                release_claim(path)
        env.close()
        print(f"task {task_id} summary: {stats}")
    capture.detach()


if __name__ == "__main__":
    collect()
