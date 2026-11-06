"""
parity_check.py  (A6 check #9 -- WRITTEN, NOT RUN)

Proves the trimmed collector loop (run_libero_collect.py) behaves identically to SAFE's run_libero_eval.py.
Runs BOTH on the same task / init states / seed on the same GPU, same checkpoint, same attention implementation,
then asserts per step: action token IDs, executed actions, token probs/entropy, timesteps; per episode: length
and success. Greedy decoding => expected EXACT equality (floats compared after casting the eval CSV to float32,
which is how the collector stores them).

run_libero_eval.py is NOT modified: it does not save token IDs, so this script wraps its `get_action` at runtime
(monkeypatch of the imported name) to record `generated_outputs["sequences"][0, -7:]`.

Seeds: eval seeds once at startup with --seed. For episode 0 we pass the collector's derived per-episode seed
(derive_episode_seed(base_seed, task_id, 0)) as eval's --seed, so both runs use the same seed for episode 0.
Episodes >0 additionally show that actions don't depend on RNG state (greedy), which is what makes
per-episode re-seeding safe.

Usage (GPU node, collection env, MUJOCO_GL=egl):
  python experiments/robot/libero/parity_check.py --checkpoint $VLA_CKPT --task_id 0 --n_episodes 2 \
      --workdir $SCRATCH/parity
Exit code 0 = all identical; 1 = mismatch (details in {workdir}/parity_report.json).
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append("../..")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
import experiments.robot.libero.run_libero_eval as ev
import experiments.robot.libero.run_libero_collect as col

ACTION_DIM = 7
ACTION_COLS = ["action/dx", "action/dy", "action/dz", "action/droll", "action/dpitch", "action/dyaw", "action/dgripper"]


def run_eval(args, ep_seed, out_dir):
    recorded = []  # per action step: token ids [7]
    orig = ev.get_action

    def recording_get_action(*a, **k):
        out = orig(*a, **k)
        _, gen = out
        recorded.append(gen["sequences"][0, -ACTION_DIM:].cpu().numpy().copy())
        return out

    ev.get_action = recording_get_action
    try:
        cfg = ev.GenerateConfig(
            pretrained_checkpoint=args.checkpoint, task_suite_name="libero_10",
            task_start_index=args.task_id, task_end_index=args.task_id, num_trials_per_task=args.n_episodes,
            seed=ep_seed, use_wandb=False, output_hidden_states=True, output_logits=True,
            attn_implementation=args.attn, run_id_note="parity_eval", save_root=str(out_dir), save_logs=True,
        )
        ev.eval_libero(cfg)  # draccus.wrap accepts an explicit config object
    finally:
        ev.get_action = orig
    return recorded


def run_collect(args, out_dir):
    cfg = col.CollectConfig(
        pretrained_checkpoint=args.checkpoint, attn_implementation=args.attn, task_ids=[args.task_id],
        episode_start=0, episode_end=args.n_episodes, seed=args.base_seed, out_root=str(out_dir),
    )
    col.collect(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--n_episodes", type=int, default=2)
    ap.add_argument("--base_seed", type=int, default=7)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    ep_seed0 = col.derive_episode_seed(args.base_seed, args.task_id, 0)
    recorded = run_eval(args, ep_seed0, work / "eval")
    gc.collect()
    torch.cuda.empty_cache()
    run_collect(args, work / "collect")

    eval_dir = work / "eval" / "parity_eval" / "libero_10"
    report, ok, cursor = [], True, 0
    for i in range(args.n_episodes):
        csvs = list(eval_dir.glob(f"task{args.task_id}--ep{i}--succ*.csv"))
        assert len(csvs) == 1, f"expected one eval CSV for ep {i}, found {csvs}"
        df = pd.read_csv(csvs[0])
        eval_success = bool(int(csvs[0].stem.split("--succ")[1]))
        T = len(df)
        eval_tokens = np.stack(recorded[cursor:cursor + T])
        cursor += T
        ep = torch.load(col.episode_path(work / "collect", args.task_id, i, args.base_seed), map_location="cpu", weights_only=False)
        meta, st = ep["meta"], ep["steps"]
        res = {"episode": i, "eval_len": T, "collect_len": meta["length"], "eval_success": eval_success,
               "collect_success": meta["success"], "collect_invalid": meta["invalid"]}
        if meta["invalid"]:
            res["error"] = meta["error"]; res["ok"] = False
        else:
            res["length_equal"] = T == meta["length"]
            res["success_equal"] = eval_success == meta["success"]
            if res["length_equal"]:
                res["tokens_equal"] = bool(np.array_equal(eval_tokens, np.asarray(st["action_token_ids"])))
                res["actions_equal"] = bool(np.array_equal(df[ACTION_COLS].to_numpy().astype(np.float32), np.asarray(st["executed_action"])))
                res["env_t_equal"] = bool(np.array_equal(df["action/timestep"].to_numpy(), np.asarray(st["env_t"])))
                pcols = [f"action/token_{j}_prob" for j in range(ACTION_DIM)]
                ecols = [f"action/token_{j}_entropy" for j in range(ACTION_DIM)]
                res["token_probs_equal"] = bool(np.array_equal(df[pcols].to_numpy().astype(np.float32), np.asarray(st["token_probs"])))
                res["token_entropy_equal"] = bool(np.array_equal(df[ecols].to_numpy().astype(np.float32), np.asarray(st["token_entropy"])))
                if not res["actions_equal"]:
                    d = np.abs(df[ACTION_COLS].to_numpy().astype(np.float32) - np.asarray(st["executed_action"]))
                    res["max_abs_action_diff"] = float(d.max())
                    res["first_diverging_step"] = int(np.argmax((d > 0).any(1)))
            res["ok"] = all(v for k, v in res.items() if k.endswith("_equal"))
        ok &= res["ok"]
        report.append(res)
        print(res)
    (work / "parity_report.json").write_text(json.dumps({"args": vars(args), "episodes": report, "all_ok": bool(ok)}, indent=2))
    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
