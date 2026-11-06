"""
audit_collection.py -- STRICTLY READ-ONLY summary of a collection folder.

Never deletes, moves, creates or modifies anything (episode files, locks, temp files, quarantine). Reuses the
collector's own validation and lock-liveness helpers from collect_io.py. Dataset statistics only: no detector
metrics, so it is fine to run on every task.

  python experiments/robot/libero/audit_collection.py /scratch1/akashrav/data/<run> --json audit.json

--fast reads ONLY the .meta.json sidecars (plus os.stat of the .pt files): no .pt is opened, so it is quick on large
collections but cannot detect a corrupt/truncated .pt. Full mode loads every .pt, validates it, and verifies that each
sidecar matches its .pt meta (SIDECAR_MISMATCH).

Exit code: 0 = clean, 1 = something flagged (use --warn-only to always exit 0).
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # fork repo root
from experiments.robot.libero import collect_io as cio  # noqa: E402

EP_RE = re.compile(r"^ep(\d{3})_seed(\d+)\.pt$")
SC_RE = re.compile(r"^ep(\d{3})_seed(\d+)\.meta\.json$")
TASK_RE = re.compile(r"^task(\d{2})$")
CONSISTENCY_FIELDS = ("task_suite", "checkpoint", "processor_revision", "attn_implementation", "center_crop", "dtype",
                      "unnorm_key", "num_steps_wait", "max_steps", "git_commit", "versions")


def error_type(err: str) -> str:
    """Exception class from the last line of a formatted traceback."""
    lines = [l for l in (err or "").strip().splitlines() if l.strip()]
    if not lines:
        return "UnknownError"
    m = re.match(r"^([\w\.]+(?:Error|Exception|Exit|Interrupt|Warning)?)\s*:", lines[-1])
    return m.group(1) if m else lines[-1][:60]


def tensor_bytes(obj) -> int:
    """In-memory byte size of tensors/arrays in a (nested) container."""
    import numpy as np
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def _freeze(v):
    return json.dumps(v, sort_keys=True, default=str)


def audit(root: Path, num_tasks=10, n_episodes=50, expect_seed=None, stale_tmp_hours=24.0, stale_lock_hours=24.0,
          oor_rate_warn=1e-3, edge_rate_warn=1e-3, fast=False) -> dict:
    now = time.time()
    flags = []

    def flag(code, task, detail):
        flags.append({"code": code, "task": task, "detail": detail})

    task_dirs = {}
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        m = TASK_RE.match(d.name)
        if d.is_dir() and m:
            task_dirs[int(m.group(1))] = d
    expected_tasks = list(range(num_tasks))
    for t in expected_tasks:
        if t not in task_dirs:
            flag("MISSING_TASK", t, "no task directory")
    unexpected_dirs = sorted(t for t in task_dirs if t not in expected_tasks)
    for t in unexpected_dirs:
        flag("UNEXPECTED_TASK", t, "task directory outside expected range")

    meta_values = defaultdict(Counter)  # field -> Counter(frozen value)
    per_task = {}
    all_lengths_valid = []
    for t in sorted(set(expected_tasks) | set(task_dirs)):
        d = task_dirs.get(t)
        T = {
            "task_id": t, "n_files": 0, "valid": 0, "invalid": 0, "failed_validation": 0, "quarantined": 0,
            "invalid_error_types": {}, "failed_validation_reasons": {}, "success": 0, "failure": 0,
            "present_idx": [], "missing_idx": [], "duplicate_idx": [], "unrecognized_files": [],
            "lengths": {}, "at_max_steps": 0, "success_at_max": 0, "failure_not_at_max": 0,
            "total_steps": 0, "wall_time_s_total": 0.0, "steps_per_sec": {}, "bytes_file_total": 0,
            "bytes_feature_total": 0, "bytes_input_total": 0, "bytes_per_episode": {},
            "token_checks": {}, "locks": [], "tmp_files": [],
        }
        by_idx = defaultdict(list)
        sidecar_keys = set()
        lens_valid, sps_list, file_bytes = [], [], []
        oor_steps = min_edge = pad_edge = 0
        err_types = Counter()
        fv_reasons = Counter()
        if d is not None:
            for p in sorted(d.iterdir()):
                if not p.is_file():
                    continue
                name = p.name
                if name.endswith(".tmp"):
                    st = p.stat()
                    owner = name[: -len(".tmp")].rsplit(".", 1)[-1]
                    running = cio._slurm_job_running(owner) if owner.isdigit() else None
                    age_h = (now - st.st_mtime) / 3600
                    stale = age_h >= stale_tmp_hours and not running
                    T["tmp_files"].append({"file": name, "age_h": round(age_h, 2), "bytes": st.st_size,
                                           "owner": owner, "owner_running": running, "stale": stale})
                    if stale:
                        flag("STALE_TMP", t, f"{name} age {age_h:.1f} h, owner {owner} not running")
                    continue
                if name.endswith(".lock"):
                    try:
                        holder = json.loads(p.read_text())
                    except Exception:
                        holder = {}
                    age_h = (now - p.stat().st_mtime) / 3600
                    alive = bool(holder) and cio.holder_alive(holder, stale_lock_hours)
                    jid = holder.get("job_id")
                    T["locks"].append({"file": name, "holder": holder, "age_h": round(age_h, 2), "live": alive,
                                       "job_id": jid, "job_in_squeue": cio._slurm_job_running(jid) if jid else None})
                    if not alive:
                        flag("STALE_LOCK", t, f"{name} holder={holder or 'unreadable'} age {age_h:.1f} h")
                    continue
                ms = SC_RE.match(name)
                if ms:
                    sidecar_keys.add((int(ms.group(1)), int(ms.group(2))))
                    continue
                m = EP_RE.match(name)
                if not m:
                    if ".stale." in name:
                        continue  # leftover of a lock reclaim; harmless
                    T["unrecognized_files"].append(name)
                    flag("UNRECOGNIZED_FILE", t, name)
                    continue
                idx, seed = int(m.group(1)), int(m.group(2))
                T["n_files"] += 1
                size = p.stat().st_size
                T["bytes_file_total"] += size
                by_idx[idx].append((seed, p, size))
        qdir = root / "_quarantine"
        if qdir.is_dir():
            T["quarantined"] = sum(1 for q in qdir.iterdir()
                                   if q.name.startswith(f"task{t:02d}__") and not q.name.endswith(".meta.json"))
            if T["quarantined"]:
                flag("QUARANTINED", t, f"{T['quarantined']} file(s) in _quarantine/")

        for (i_, s_) in sorted(sidecar_keys):
            if not any(sd == s_ for sd, _p, _sz in by_idx.get(i_, [])):
                flag("SIDECAR_ORPHAN", t, f"ep{i_:03d}_seed{s_}.meta.json has no .pt")
        seeds_seen = set()
        for idx in sorted(by_idx):
            entries = by_idx[idx]
            if len(entries) > 1:
                T["duplicate_idx"].append(idx)
                flag("DUPLICATE_EPISODES", t, f"init_state_idx {idx}: {[p.name for _, p, _ in entries]}")
            for seed, p, size in entries:
                seeds_seen.add(seed)
                if expect_seed is not None and seed != expect_seed:
                    flag("UNEXPECTED_SEED", t, f"{p.name} (expected seed {expect_seed})")
                sc = cio.read_sidecar(p)
                if fast:
                    # sidecar only: never opens the .pt
                    if sc is None:
                        T["sidecar_missing"] = T.get("sidecar_missing", 0) + 1
                        flag("SIDECAR_MISSING", t, f"{p.name}: no readable sidecar (--fast cannot assess this episode)")
                        continue
                    meta = sc["meta"]
                    if (meta.get("task_id"), meta.get("init_state_idx"), meta.get("seed")) != (t, idx, seed):
                        T["failed_validation"] += 1
                        flag("FAILED_VALIDATION", t, f"{p.name}: sidecar meta does not match filename")
                        continue
                    if sc.get("pt_bytes") != size:
                        flag("SIDECAR_MISMATCH", t, f"{p.name}: sidecar pt_bytes {sc.get('pt_bytes')} != size on disk {size}")
                    feat_b, inp_b = sc.get("feature_bytes", 0), sc.get("input_bytes", 0)
                else:
                    ep, reason = cio.load_and_validate(p, t, idx, seed)
                    if reason is not None:
                        T["failed_validation"] += 1
                        fv_reasons[reason.split(":")[0]] += 1
                        flag("FAILED_VALIDATION", t, f"{p.name}: {reason}")
                        continue
                    meta, steps = ep["meta"], ep["steps"]
                    feat_b = tensor_bytes(steps.get("features", {}))
                    inp_b = tensor_bytes(ep.get("inputs", {}))
                    if sc is None:
                        T["sidecar_missing"] = T.get("sidecar_missing", 0) + 1
                        flag("SIDECAR_MISSING", t, f"{p.name}: no readable sidecar")
                    else:
                        why = cio.sidecar_mismatch(sc, ep, size)
                        if why:
                            flag("SIDECAR_MISMATCH", t, f"{p.name}: {why}")
                    del ep, steps
                T["present_idx"].append(idx)
                for f in CONSISTENCY_FIELDS:
                    meta_values[f][_freeze(meta.get(f))] += 1
                if meta["invalid"]:
                    T["invalid"] += 1
                    et = error_type(meta["error"])
                    err_types[et] += 1
                    flag("INVALID_EPISODE", t, f"{p.name}: {et}")
                    continue
                T["valid"] += 1
                n = meta["length"]
                max_steps = meta.get("max_steps")
                lens_valid.append(n)
                T["total_steps"] += n
                wall = meta.get("wall_time_s")
                if wall:
                    T["wall_time_s_total"] += wall
                    sps_list.append(n / wall)
                file_bytes.append(size)
                T["bytes_feature_total"] += feat_b
                T["bytes_input_total"] += inp_b
                if meta["success"]:
                    T["success"] += 1
                    if max_steps and n >= max_steps:
                        T["success_at_max"] += 1
                        flag("SUCCESS_AT_MAX", t, f"{p.name}: success with length {n} == max_steps")
                else:
                    T["failure"] += 1
                    if max_steps and n < max_steps:
                        T["failure_not_at_max"] += 1
                        flag("FAILURE_NOT_AT_MAX", t, f"{p.name}: failure with length {n} < max_steps {max_steps}")
                if max_steps and n >= max_steps:
                    T["at_max_steps"] += 1
                oor_steps += meta["n_out_of_range_token_steps"]
                min_edge += meta["n_tokens_at_bin_min_edge"]
                pad_edge += meta["n_tokens_at_pad_edge"]
        exp = set(range(n_episodes))
        T["present_idx"] = sorted(set(T["present_idx"]))
        T["missing_idx"] = sorted(exp - set(by_idx))  # no file at all
        T["not_collected_idx"] = sorted(exp - set(T["present_idx"]))  # no *valid-or-invalid* episode
        if d is not None and T["missing_idx"]:
            flag("MISSING_EPISODES", t, f"{len(T['missing_idx'])} missing init states: {T['missing_idx']}")
        extra = sorted(set(by_idx) - exp)
        if extra:
            flag("UNEXPECTED_EPISODES", t, f"init_state_idx outside 0..{n_episodes - 1}: {extra}")
        if T["valid"] > 0:
            if T["success"] == 0 or T["failure"] == 0:
                missing_cls = "success" if T["success"] == 0 else "failure"
                flag("MISSING_CLASS", t, f"no {missing_cls} episodes ({T['success']} succ / {T['failure']} fail)")
            T["lengths"] = {"min": min(lens_valid), "median": statistics.median(lens_valid), "max": max(lens_valid),
                            "mean": statistics.fmean(lens_valid)}
            all_lengths_valid += lens_valid
            if T["at_max_steps"] != T["failure"]:
                flag("MAX_STEPS_MISMATCH", t, f"{T['at_max_steps']} episodes at max_steps vs {T['failure']} failures")
            tot = T["total_steps"]
            T["token_checks"] = {
                "out_of_range_steps": oor_steps, "out_of_range_step_rate": oor_steps / tot,
                "n_tokens_at_bin_min_edge": min_edge, "min_edge_token_rate": min_edge / (7 * tot),
                "n_tokens_at_pad_edge": pad_edge, "pad_edge_token_rate": pad_edge / (7 * tot),
            }
            if T["token_checks"]["out_of_range_step_rate"] > oor_rate_warn:
                flag("TOKEN_OUT_OF_RANGE", t, f"rate {T['token_checks']['out_of_range_step_rate']:.4g} > {oor_rate_warn}")
            for k, nm in (("min_edge_token_rate", "TOKEN_MIN_EDGE"), ("pad_edge_token_rate", "TOKEN_PAD_EDGE")):
                if T["token_checks"][k] > edge_rate_warn:
                    flag(nm, t, f"rate {T['token_checks'][k]:.4g} > {edge_rate_warn}")
            if sps_list:
                T["steps_per_sec"] = {"median": statistics.median(sps_list), "min": min(sps_list), "max": max(sps_list)}
            T["bytes_per_episode"] = {"mean": statistics.fmean(file_bytes), "max": max(file_bytes)}
        T["invalid_error_types"] = dict(err_types)
        T["failed_validation_reasons"] = dict(fv_reasons)
        per_task[t] = T

    # consistency across ALL readable episodes (valid and invalid)
    consistency = {}
    for f, cnt in meta_values.items():
        consistency[f] = {k: v for k, v in cnt.items()}
        if len(cnt) > 1:
            flag("INCONSISTENT_META", None, f"{f}: {dict(cnt)}")

    # overall
    tasks = list(per_task.values())
    sps_all = [tt["steps_per_sec"]["median"] for tt in tasks if tt["steps_per_sec"]]
    overall = {
        "n_files": sum(x["n_files"] for x in tasks), "valid": sum(x["valid"] for x in tasks),
        "invalid": sum(x["invalid"] for x in tasks), "failed_validation": sum(x["failed_validation"] for x in tasks),
        "quarantined": sum(x["quarantined"] for x in tasks), "success": sum(x["success"] for x in tasks),
        "failure": sum(x["failure"] for x in tasks), "total_steps": sum(x["total_steps"] for x in tasks),
        "wall_time_s_total": sum(x["wall_time_s_total"] for x in tasks),
        "bytes_feature_total": sum(x["bytes_feature_total"] for x in tasks),
        "bytes_input_total": sum(x["bytes_input_total"] for x in tasks),
        "bytes_episode_files": sum(x["bytes_file_total"] for x in tasks),
        "expected_episodes": num_tasks * n_episodes,
        "lengths": ({"min": min(all_lengths_valid), "median": statistics.median(all_lengths_valid),
                     "max": max(all_lengths_valid), "mean": statistics.fmean(all_lengths_valid)} if all_lengths_valid else {}),
        "steps_per_sec_median_of_tasks": statistics.median(sps_all) if sps_all else None,
    }
    if overall["wall_time_s_total"]:
        overall["steps_per_sec_pooled"] = overall["total_steps"] / overall["wall_time_s_total"]
    # bytes on disk (everything under root)
    disk = {"total": 0, "episodes": 0, "tmp": 0, "locks": 0, "quarantine": 0, "other": 0}
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            try:
                sz = os.stat(os.path.join(dp, fn)).st_size
            except OSError:
                continue
            disk["total"] += sz
            if "_quarantine" in Path(dp).parts:
                disk["quarantine"] += sz
            elif fn.endswith(".tmp"):
                disk["tmp"] += sz
            elif fn.endswith(".lock"):
                disk["locks"] += sz
            elif EP_RE.match(fn):
                disk["episodes"] += sz
            else:
                disk["other"] += sz
    overall["bytes_on_disk"] = disk
    return {"root": str(root), "mode": "fast" if fast else "full", "generated_unix": now, "params": {
                "num_tasks": num_tasks, "n_episodes": n_episodes, "expect_seed": expect_seed,
                "stale_tmp_hours": stale_tmp_hours, "stale_lock_hours": stale_lock_hours},
            "overall": overall, "tasks": {str(k): v for k, v in per_task.items()},
            "consistency": consistency, "flags": flags}


def _mb(b):
    return f"{b / 1e6:.1f} MB" if b < 1e9 else f"{b / 1e9:.2f} GB"


def to_markdown(r: dict) -> str:
    o, out = r["overall"], []
    out.append(f"# Collection audit: `{r['root']}`\n")
    if r["mode"] == "fast":
        out.append("> **FAST mode**: only `.meta.json` sidecars were read; `.pt` files were not opened, so corrupt or truncated episodes are NOT detected.\n")
    out.append(f"**{len(r['flags'])} flag(s)** — {'CLEAN' if not r['flags'] else 'ATTENTION NEEDED'}\n")
    out.append("## Episodes and outcomes (per task)\n")
    out.append("| task | files | valid | invalid | failed-valid. | quarantined | missing | dup | succ | fail | len min/med/max | at max | sps (median) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, t in r["tasks"].items():
        L = t["lengths"]
        ls = f"{L['min']}/{L['median']}/{L['max']}" if L else "-"
        sps = f"{t['steps_per_sec']['median']:.2f}" if t["steps_per_sec"] else "-"
        out.append(f"| {k} | {t['n_files']} | {t['valid']} | {t['invalid']} | {t['failed_validation']} | {t['quarantined']} | "
                   f"{len(t['missing_idx'])} | {len(t['duplicate_idx'])} | {t['success']} | {t['failure']} | {ls} | {t['at_max_steps']} | {sps} |")
    L = o["lengths"]
    out.append(f"| **all** | {o['n_files']} | {o['valid']} | {o['invalid']} | {o['failed_validation']} | {o['quarantined']} | | | "
               f"{o['success']} | {o['failure']} | {('%s/%s/%s' % (L['min'], L['median'], L['max'])) if L else '-'} | | "
               f"{o['steps_per_sec_median_of_tasks'] and round(o['steps_per_sec_median_of_tasks'], 2)} |\n")
    out.append(f"Expected episodes: {o['expected_episodes']} (num_tasks × n_episodes).\n")
    bad = [(k, t) for k, t in r["tasks"].items() if t["invalid_error_types"]]
    if bad:
        out.append("### Invalid episodes by error type")
        for k, t in bad:
            out.append(f"- task {k}: {t['invalid_error_types']}")
        out.append("")
    bad = [(k, t) for k, t in r["tasks"].items() if t["failed_validation_reasons"]]
    if bad:
        out.append("### Files that failed validation (by reason)")
        for k, t in bad:
            out.append(f"- task {k}: {t['failed_validation_reasons']}")
        out.append("")
    out.append("## Coverage\n")
    for k, t in r["tasks"].items():
        if t["missing_idx"] or t["duplicate_idx"] or t.get("not_collected_idx"):
            out.append(f"- task {k}: no file for {t['missing_idx']}; duplicates {t['duplicate_idx']}; "
                       f"not usable (no validating file) {t.get('not_collected_idx')}")
    if not any(t["missing_idx"] or t["duplicate_idx"] for t in r["tasks"].values()):
        out.append("- all expected init states present, no duplicates")
    out.append("\n## Token checks (per task)\n")
    out.append("| task | out-of-range step rate | tokens at min edge (31744) | rate | tokens at pad edge (32000) | rate |")
    out.append("|---|---|---|---|---|---|")
    for k, t in r["tasks"].items():
        c = t["token_checks"]
        if c:
            out.append(f"| {k} | {c['out_of_range_step_rate']:.4g} ({c['out_of_range_steps']} steps) | {c['n_tokens_at_bin_min_edge']} | "
                       f"{c['min_edge_token_rate']:.4g} | {c['n_tokens_at_pad_edge']} | {c['pad_edge_token_rate']:.4g} |")
    out.append("\n## Throughput and storage\n")
    out.append(f"- pooled steps/s: {o.get('steps_per_sec_pooled', 'n/a')}; total steps {o['total_steps']}")
    for k, t in r["tasks"].items():
        if t["bytes_per_episode"]:
            out.append(f"- task {k}: {_mb(t['bytes_per_episode']['mean'])}/episode on disk (mean); features {_mb(t['bytes_feature_total'])}, inputs {_mb(t['bytes_input_total'])} (in-memory tensor bytes)")
    d = o["bytes_on_disk"]
    out.append(f"- bytes on disk: total {_mb(d['total'])} (episodes {_mb(d['episodes'])}, quarantine {_mb(d['quarantine'])}, tmp {_mb(d['tmp'])}, locks/other {_mb(d['locks'] + d['other'])})")
    out.append("\n## Housekeeping\n")
    n_lock = 0
    for k, t in r["tasks"].items():
        for lk in t["locks"]:
            n_lock += 1
            out.append(f"- task {k} lock `{lk['file']}`: {'LIVE' if lk['live'] else 'STALE'}, job {lk['job_id']} "
                       f"(in squeue: {lk['job_in_squeue']}), host {lk['holder'].get('host')}, age {lk['age_h']} h")
        for tf in t["tmp_files"]:
            n_lock += 1
            out.append(f"- task {k} temp `{tf['file']}`: age {tf['age_h']} h, owner {tf['owner']} running={tf['owner_running']}, {'STALE' if tf['stale'] else 'recent'}")
    if not n_lock:
        out.append("- no locks or temp files")
    out.append("\n## Metadata consistency (all readable episodes)\n")
    for f, cnt in r["consistency"].items():
        out.append(f"- `{f}`: {'OK' if len(cnt) == 1 else 'MIXED'} — {cnt if len(cnt) > 1 else list(cnt)[0]}")
    out.append("\n## Flags\n")
    if r["flags"]:
        for fl in r["flags"]:
            out.append(f"- **{fl['code']}** (task {fl['task']}): {fl['detail']}")
    else:
        out.append("- none")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write machine-readable audit here (outside the collection folder)")
    ap.add_argument("--num-tasks", type=int, default=10)
    ap.add_argument("--n-episodes", type=int, default=50, help="expected init_state_idx values are 0..n-1")
    ap.add_argument("--expect-seed", type=int, default=None)
    ap.add_argument("--stale-tmp-hours", type=float, default=24.0)
    ap.add_argument("--stale-lock-hours", type=float, default=24.0)
    ap.add_argument("--oor-rate-warn", type=float, default=1e-3)
    ap.add_argument("--edge-rate-warn", type=float, default=1e-3)
    ap.add_argument("--fast", action="store_true", help="read only the .meta.json sidecars; do not open any .pt")
    ap.add_argument("--warn-only", action="store_true", help="always exit 0")
    a = ap.parse_args(argv)
    if a.json is not None and a.root.resolve() in a.json.resolve().parents:
        ap.error("--json must be written outside the collection folder (the audit never writes there)")
    r = audit(a.root, a.num_tasks, a.n_episodes, a.expect_seed, a.stale_tmp_hours, a.stale_lock_hours,
              a.oor_rate_warn, a.edge_rate_warn, fast=a.fast)
    print(to_markdown(r))
    if a.json:
        a.json.write_text(json.dumps(r, indent=2, default=str))
    return 0 if (a.warn_only or not r["flags"]) else 1


if __name__ == "__main__":
    sys.exit(main())
