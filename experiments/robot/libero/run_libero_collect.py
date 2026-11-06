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

Feature capture is feature_capture.FeatureCapture (layer list, pooling, layer-mean definition are config values; defaults
= the draft contract, frozen by P0-1). Each episode also gets an ep{NNN}_seed{S}.meta.json sidecar (written atomically
after the .pt; regenerated on resume if missing/inconsistent) so audits can run with --fast.

TODO: inputs (A2, Dhyan), extra metadata fields (A3, Dhyan).
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

MAX_STEPS_LIBERO_10 = 520  # same as run_libero_eval.py (longest training demo has 505 steps)


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
    output_hidden_states: bool = True                # required by FeatureCapture(verify=True)

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

    # Feature capture (defaults = draft data contract; frozen by P0-1)
    blocks: str = "4,8,12,16,20,24,28,32"            # raw block_k keys (1-based)
    final_norm: bool = True
    layer_mean_all: bool = True
    mean_blocks: Optional[str] = None                # None => mean of raw blocks 1..L (L=32)
    pooling: str = "first,last,mean"
    verify_features: bool = True                     # assert hooks == hidden_states every action (cheap)
    token_level_episodes_per_task: int = 0           # episodes with init_state_idx < N also store token_level_sample [T,7,n_keys,D]
    # fmt: on


# Atomic IO / locks / validation live in collect_io.py (CPU-importable, shared with audit_collection.py)
from experiments.robot.libero.feature_capture import FeatureCapture, FeatureCaptureConfig, parse_int_list  # noqa: E402
from experiments.robot.libero.collect_io import (  # noqa: E402
    ACTION_DIM, REQUIRED_META, derive_episode_seed, episode_path, holder_alive, job_identity,
    ensure_sidecar, lock_path, quarantine, release_claim, remove_stale_tmp, save_episode, try_claim,
    validate_episode_file,
)

# --------------------------------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------------------------------
def run_episode(cfg, model, processor, env, task_description, init_state, max_steps, resize_size, capture, want_token_level=False):
    assert cfg.output_logits, "token_probs require output_logits=True"
    """Same control flow as run_libero_eval.py. Returns (steps_dict, success, n_steps)."""
    env.reset()
    obs = env.set_init_state(init_state)
    t = 0
    s_list, env_t_list, tok_list, act_list, prob_list, ent_list, oor_list = [], [], [], [], [], [], []
    n_min_edge = n_pad_edge = 0
    tl_list = []
    # TRUE tokenizer action range (prismatic/vla/action_tokenizer.py: token = vocab_size - digitize(a), digitize in
    # 1..n_bins): [vocab_size - n_bins, vocab_size - 1]  (31744..31999 for Llama-2 / 256 bins). Derived from config.
    # NOTE: SAFE's uncertainty slice (unc_utils.py:29) is [vocab_size - n_bins + 1, vocab_size], off by one; we keep
    # it for token_probs/token_entropy (baseline parity) but count the two edge cases below.
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    bin_lo, bin_hi = vocab_size - model.config.n_action_bins, vocab_size - 1
    min_edge, pad_edge = vocab_size - model.config.n_action_bins, vocab_size
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
        if want_token_level:
            tl_list.append(capture.last_token_level)  # [7, n_keys, D] bf16 CPU

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
        # token counts (not step counts): valid action at the -1.0.. edge whose SAFE prob is the max over OTHER bins,
        # and the padding token that SAFE's slice wrongly includes
        n_min_edge += int((tok == min_edge).sum())
        n_pad_edge += int((tok == pad_edge).sum())
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
        "features": {k: torch.stack(v) for k, v in feats.items()},  # bf16 [T, len(pooling), 4096]
        "token_out_of_range": np.asarray(oor_list, dtype=bool),  # [T]: any of the 7 chosen ids outside the bins
        "token_probs": np.asarray(prob_list, dtype=np.float32).reshape(-1, ACTION_DIM),
        "token_entropy": np.asarray(ent_list, dtype=np.float32).reshape(-1, ACTION_DIM),
    }
    steps["action_bin_range"] = (int(bin_lo), int(bin_hi))
    steps["edge_counts"] = (n_min_edge, n_pad_edge)
    token_level = torch.stack(tl_list) if (want_token_level and tl_list) else None  # [T,7,n_keys,D]
    return steps, success, len(s_list), token_level


def _quat2axisangle(q):
    from experiments.robot.libero.libero_utils import quat2axisangle
    return quat2axisangle(q)


def base_meta(cfg, task_id, task_description, init_state_idx, unnorm_key, capture):
    return {
        "task_suite": cfg.task_suite_name, "task_id": task_id, "task_description": task_description,
        "init_state_idx": init_state_idx, "seed": cfg.seed,
        "checkpoint": str(cfg.pretrained_checkpoint),  # TODO(A3): append @<revision>
        "dtype": "bfloat16", "center_crop": cfg.center_crop, "unnorm_key": unnorm_key,
        "num_steps_wait": cfg.num_steps_wait, "max_steps": MAX_STEPS_LIBERO_10,
        "attn_implementation": cfg.attn_implementation,  # must be identical across the whole collection
        "torch": torch.__version__,
        # TODO(A3): versions, git commits, image_boundary
        **capture.describe(),  # features_captured, layer_keys, pooling, layer_mean_def, num_blocks, ...
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

    capture = FeatureCapture(FeatureCaptureConfig(
        blocks=parse_int_list(cfg.blocks), final_norm=cfg.final_norm, layer_mean_all=cfg.layer_mean_all,
        mean_blocks=parse_int_list(cfg.mean_blocks) if cfg.mean_blocks else None,
        pooling=tuple(x for x in cfg.pooling.replace(" ", "").split(",") if x), verify=cfg.verify_features))
    assert cfg.output_hidden_states or not cfg.verify_features, "verify_features needs output_hidden_states=True"
    capture.attach(model)
    print(f"[capture] keys={capture.layer_keys} pooling={capture.cfg.pooling} L={capture.L}")

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
                        if try_claim(path, cfg.stale_lock_hours):  # sidecar repair writes: take the episode lock
                            try:
                                if ensure_sidecar(path, torch.load(path, map_location="cpu", weights_only=False)):
                                    print(f"[resume] (re)wrote sidecar for {path.name}")
                            finally:
                                release_claim(path)
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
                meta = base_meta(cfg, task_id, task_description, idx, unnorm_key, capture)
                meta["episode_seed"] = ep_seed
                meta["episode_seed_derivation"] = "sha256(f'{base_seed}:{task_id}:{init_state_idx}')[:4] % (2**31-1)"
                t0 = time.time()
                try:
                    want_tl = idx < cfg.token_level_episodes_per_task
                    steps, success, n, token_level = run_episode(
                        cfg, model, processor, env, task_description, initial_states[idx],
                        MAX_STEPS_LIBERO_10, resize_size, capture, want_token_level=want_tl,
                    )
                    oor = steps.pop("action_bin_range")
                    n_min_edge, n_pad_edge = steps.pop("edge_counts")
                    meta.update(success=bool(success), label_failure=int(not success), length=n, invalid=False, error=None,
                                n_out_of_range_token_steps=int(steps["token_out_of_range"].sum()),
                                out_of_range_token_rate=float(steps["token_out_of_range"].mean()) if n else 0.0,
                                action_bin_token_range=list(oor),
                                n_tokens_at_bin_min_edge=int(n_min_edge), n_tokens_at_pad_edge=int(n_pad_edge))
                    episode = {"meta": meta, "steps": steps}  # TODO(A2): "inputs"
                    if token_level is not None:
                        episode["token_level_sample"] = token_level
                        meta["token_level_keys"] = list(capture.layer_keys)
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
                nbytes = save_episode(episode, path)  # .pt then .meta.json sidecar, both atomic
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
