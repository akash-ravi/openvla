"""
project_collection.py -- project GPU-hours, SUs, storage and a Slurm job layout for the full collection from a
pilot audit JSON (produced by audit_collection.py --json). Pure arithmetic, CPU only.

Every assumption is a CLI flag. Every number in the output is tagged:
  [measured]  taken from the audit JSON (pilot)
  [override]  given explicitly on the command line
  [assumed]   a default/guess used because it was not measured -- replace with a measurement when possible

  python experiments/robot/libero/project_collection.py audit.json --episodes-per-task 50 --partition debug
"""
import argparse
import json
import math
import sys
from pathlib import Path

M, O, A = "measured", "override", "assumed"


def _pick(override, measured, assumed=None):
    if override is not None:
        return override, O
    if measured:
        return measured, M
    return assumed, A


def project(audit: dict, episodes_per_task=50, num_tasks=10, su_per_gpu_minute=8.0, partition="debug",
            walltime_hours=None, margin=0.2, job_startup_min=5.0, length_basis="mean", max_steps=520,
            steps_per_sec=None, feature_bytes_per_step=None, input_bytes_per_step=None,
            assumed_steps_per_sec=None, assumed_feature_bytes_per_step=221184, assumed_input_bytes_per_step=150528):
    if walltime_hours is None:
        walltime_hours = {"debug": 1.0, "gpu": 2.0}[partition]
    o = audit["overall"]
    tasks = audit["tasks"]
    sps_overall = o.get("steps_per_sec_pooled")
    steps_overall = o["total_steps"]
    mean_len_overall = o["lengths"]["mean"] if o["lengths"] else None
    fb_overall = (o["bytes_feature_total"] / steps_overall) if steps_overall else 0
    ib_overall = (o["bytes_input_total"] / steps_overall) if steps_overall else 0

    rows, tot = [], dict(steps=0.0, gpu_s=0.0, jobs=0, feat_b=0.0, inp_b=0.0, alloc_s=0.0)
    usable_s = walltime_hours * 3600 * (1 - margin) - job_startup_min * 60
    if usable_s <= 0:
        raise ValueError("walltime too short for the margin + startup")
    for t in range(num_tasks):
        T = tasks.get(str(t))
        have = bool(T and T["valid"] > 0)
        L = T["lengths"] if have else {}
        if length_basis == "max":
            mean_len, ls = float(max_steps), A
        else:
            mean_len, ls = (L["mean"], M) if have else (mean_len_overall, M if mean_len_overall else A)
            if mean_len is None:
                mean_len, ls = float(max_steps), A
        sps_task = T["steps_per_sec"]["median"] if have and T["steps_per_sec"] else None
        sps, ss = _pick(steps_per_sec, sps_task or sps_overall, assumed_steps_per_sec)
        if sps is None:
            raise ValueError("no steps/sec in the audit and no --steps-per-sec / --assumed-steps-per-sec given")
        fb_task = (T["bytes_feature_total"] / T["total_steps"]) if have and T["total_steps"] else 0
        ib_task = (T["bytes_input_total"] / T["total_steps"]) if have and T["total_steps"] else 0
        fb, fs = _pick(feature_bytes_per_step, fb_task or fb_overall, assumed_feature_bytes_per_step)
        ib, is_ = _pick(input_bytes_per_step, ib_task or ib_overall, assumed_input_bytes_per_step)
        # planning length for the job layout always uses the chosen basis; totals use measured mean length
        plan_len = mean_len
        ep_s = plan_len / sps
        eps_per_job = int(usable_s // ep_s)
        if eps_per_job < 1:
            raise ValueError(f"task {t}: one episode ({ep_s:.0f}s) does not fit in usable walltime ({usable_s:.0f}s)")
        eps_per_job = min(eps_per_job, episodes_per_task)
        jobs = math.ceil(episodes_per_task / eps_per_job)
        steps = episodes_per_task * mean_len
        gpu_s = steps / sps + jobs * job_startup_min * 60
        rows.append({"task": t, "mean_len": mean_len, "mean_len_src": ls, "steps_per_sec": sps, "sps_src": ss,
                     "feature_bytes_per_step": fb, "feature_src": fs, "input_bytes_per_step": ib, "input_src": is_,
                     "episodes_per_job": eps_per_job, "jobs": jobs, "steps": steps, "gpu_hours": gpu_s / 3600,
                     "feature_gb": steps * fb / 1e9, "input_gb": steps * ib / 1e9,
                     "job_walltime_needed_h": (eps_per_job * ep_s + job_startup_min * 60) / (1 - margin) / 3600})
        tot["steps"] += steps; tot["gpu_s"] += gpu_s; tot["jobs"] += jobs
        tot["feat_b"] += steps * fb; tot["inp_b"] += steps * ib; tot["alloc_s"] += jobs * walltime_hours * 3600
    gpu_h = tot["gpu_s"] / 3600
    return {
        "assumptions": {"episodes_per_task": episodes_per_task, "num_tasks": num_tasks, "su_per_gpu_minute": su_per_gpu_minute,
                        "partition": partition, "walltime_hours": walltime_hours, "margin": margin,
                        "job_startup_min": job_startup_min, "length_basis": length_basis, "max_steps": max_steps,
                        "pilot_episodes_used": o["valid"]},
        "tasks": rows,
        "totals": {"steps": tot["steps"], "gpu_hours": gpu_h, "sus": gpu_h * 60 * su_per_gpu_minute,
                   "sus_if_every_job_uses_full_walltime": tot["alloc_s"] / 60 * su_per_gpu_minute,
                   "jobs": tot["jobs"], "feature_gb": tot["feat_b"] / 1e9, "input_gb": tot["inp_b"] / 1e9,
                   "storage_gb": (tot["feat_b"] + tot["inp_b"]) / 1e9},
    }


def to_markdown(p: dict) -> str:
    a, t = p["assumptions"], p["totals"]
    out = ["# Collection projection\n",
           f"Based on {a['pilot_episodes_used']} valid pilot episodes. Tags: [measured] from audit, [override] CLI, [assumed] default/guess.\n",
           "## Assumption parameters (all [override]/[assumed] CLI flags)",
           f"- episodes/task {a['episodes_per_task']} × {a['num_tasks']} tasks; SU rate {a['su_per_gpu_minute']} SU/GPU-min [assumed unless set]",
           f"- partition `{a['partition']}`, walltime {a['walltime_hours']} h, margin {a['margin']:.0%}, job startup {a['job_startup_min']} min [assumed]",
           f"- planning length basis: {a['length_basis']} (max_steps {a['max_steps']} [assumed from SAFE])\n",
           "## Totals",
           f"- steps: {t['steps']:.0f}",
           f"- GPU-hours: **{t['gpu_hours']:.1f}** → SUs: **{t['sus']:.0f}** (upper bound if every job used its full walltime: {t['sus_if_every_job_uses_full_walltime']:.0f})",
           f"- storage: features {t['feature_gb']:.1f} GB + inputs {t['input_gb']:.1f} GB = **{t['storage_gb']:.1f} GB**",
           f"- Slurm jobs: {t['jobs']}\n",
           "## Per task and job layout",
           "| task | mean len (src) | steps/s (src) | feat B/step (src) | input B/step (src) | eps/job | jobs | GPU-h | features GB | inputs GB |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in p["tasks"]:
        out.append(f"| {r['task']} | {r['mean_len']:.0f} ({r['mean_len_src']}) | {r['steps_per_sec']:.2f} ({r['sps_src']}) | "
                   f"{r['feature_bytes_per_step']:.0f} ({r['feature_src']}) | {r['input_bytes_per_step']:.0f} ({r['input_src']}) | "
                   f"{r['episodes_per_job']} | {r['jobs']} | {r['gpu_hours']:.2f} | {r['feature_gb']:.1f} | {r['input_gb']:.1f} |")
    out.append("\nAssumed storage numbers mean the pilot had no real features/inputs (e.g. capture still stubbed); re-run after the real pilot.")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit_json", type=Path)
    ap.add_argument("--episodes-per-task", type=int, default=50)
    ap.add_argument("--num-tasks", type=int, default=10)
    ap.add_argument("--su-per-gpu-minute", type=float, default=8.0, help="[assumed] CARC SU rate for A40/A100; confirm in the portal")
    ap.add_argument("--partition", choices=["debug", "gpu"], default="debug", help="debug => 1 h walltime, gpu => 2 h")
    ap.add_argument("--walltime-hours", type=float, default=None, help="overrides --partition")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--job-startup-min", type=float, default=5.0, help="[assumed] model load + env setup per job")
    ap.add_argument("--length-basis", choices=["mean", "max"], default="mean", help="episode length used for job sizing; max = max_steps (all failures)")
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--steps-per-sec", type=float, default=None, help="override measured")
    ap.add_argument("--assumed-steps-per-sec", type=float, default=None, help="used only if the audit has no timing")
    ap.add_argument("--feature-bytes-per-step", type=float, default=None, help="override measured")
    ap.add_argument("--input-bytes-per-step", type=float, default=None, help="override measured")
    ap.add_argument("--assumed-feature-bytes-per-step", type=float, default=221184, help="9 reps x 3 poolings x 4096 x 2 B")
    ap.add_argument("--assumed-input-bytes-per-step", type=float, default=150528, help="224*224*3 uint8")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    p = project(json.loads(a.audit_json.read_text()), a.episodes_per_task, a.num_tasks, a.su_per_gpu_minute, a.partition,
                a.walltime_hours, a.margin, a.job_startup_min, a.length_basis, a.max_steps, a.steps_per_sec,
                a.feature_bytes_per_step, a.input_bytes_per_step, a.assumed_steps_per_sec,
                a.assumed_feature_bytes_per_step, a.assumed_input_bytes_per_step)
    print(to_markdown(p))
    if a.json:
        a.json.write_text(json.dumps(p, indent=2))


if __name__ == "__main__":
    sys.exit(main())
