"""CPU tests for audit_collection.py / project_collection.py. Fake episodes are written with the collector's own
atomic writer (collect_io.atomic_save); features are stubbed with tiny tensors. No GPU, no model, no libero."""
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.robot.libero import audit_collection as audit  # noqa: E402
from experiments.robot.libero import collect_io as cio  # noqa: E402
from experiments.robot.libero import project_collection as proj  # noqa: E402

MAX_STEPS = 20
BASE_META = dict(task_suite="libero_10", checkpoint="openvla/openvla-7b-finetuned-libero-10@abc", processor_revision="p1",
                 attn_implementation="flash_attention_2", center_crop=True, dtype="bfloat16", unnorm_key="libero_10_no_noops",
                 num_steps_wait=10, max_steps=MAX_STEPS, git_commit={"safe_research": "a", "openvla_fork": "b"}, versions={"torch": "2.2.0"})


def make_episode(task, idx, seed=7, success=True, T=8, wall=4.0, ids=None, oor=None, min_edge=0, pad=0, attn=None, inputs=False):
    ids = np.full((T, 7), 31900, dtype=np.int64) if ids is None else ids
    oor_arr = np.zeros(T, dtype=bool) if oor is None else oor
    meta = dict(BASE_META, task_id=task, init_state_idx=idx, seed=seed, invalid=False, error=None, success=success,
                label_failure=int(not success), length=T, wall_time_s=wall,
                n_out_of_range_token_steps=int(oor_arr.sum()), out_of_range_token_rate=float(oor_arr.mean()),
                action_bin_token_range=[31744, 31999], n_tokens_at_bin_min_edge=min_edge, n_tokens_at_pad_edge=pad)
    if attn:
        meta["attn_implementation"] = attn
    ep = {"meta": meta, "steps": {
        "s": np.arange(T), "env_t": np.arange(T) + 10, "action_token_ids": ids, "executed_action": np.zeros((T, 7), np.float32),
        "token_probs": np.ones((T, 7), np.float32), "token_entropy": np.zeros((T, 7), np.float32), "token_out_of_range": oor_arr,
        "features": {"block_4": torch.zeros(T, 3, 8, dtype=torch.bfloat16)}}}
    if inputs:
        ep["inputs"] = {"images": np.zeros((T, 4, 4, 3), np.uint8)}
    return ep


def make_invalid(task, idx, seed=7):
    meta = dict(BASE_META, task_id=task, init_state_idx=idx, seed=seed, invalid=True, success=None, label_failure=None,
                length=0, error="Traceback (most recent call last):\n  File x\nValueError: boom", wall_time_s=1.0)
    return {"meta": meta, "steps": {}}


def save(root, ep):
    m = ep["meta"]
    p = cio.episode_path(root, m["task_id"], m["init_state_idx"], m["seed"])
    cio.atomic_save(ep, p)
    return p


def snapshot(root):
    snap = {}
    for dp, dn, fns in os.walk(root):
        for n in dn:
            p = Path(dp) / n
            snap[str(p.relative_to(root)) + "/"] = (p.stat().st_mtime_ns,)
        for n in fns:
            p = Path(dp) / n
            snap[str(p.relative_to(root))] = (p.stat().st_size, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
    snap["."] = (root.stat().st_mtime_ns,)
    return snap


@pytest.fixture
def dirty_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cio, "_slurm_job_running", lambda jid: str(jid) == "999")  # mock squeue: only job 999 runs
    root = tmp_path / "collect"
    # task 0: 6 valid; 4 successes, 2 failures at max_steps; one episode with a different attn implementation; inputs on one
    for i in range(4):
        save(root, make_episode(0, i, success=True, T=8, inputs=(i == 0)))
    save(root, make_episode(0, 4, success=False, T=MAX_STEPS))
    save(root, make_episode(0, 5, success=False, T=MAX_STEPS, attn="sdpa"))
    (root / "task00" / "notes.txt").write_text("stray")
    # task 1: failure, invalid (exception), corrupt, duplicate (two seeds), missing idx 4,5
    save(root, make_episode(1, 0, success=False, T=MAX_STEPS))
    save(root, make_invalid(1, 1))
    cio.episode_path(root, 1, 2, 7).write_bytes(b"not a torch file")
    save(root, make_episode(1, 3, seed=7, success=True))
    save(root, make_episode(1, 3, seed=8, success=True))
    qd = root / "_quarantine"
    qd.mkdir()
    (qd / "task01__ep002_seed7.pt.20261106T000000.1").write_bytes(b"old corrupt")
    # task 2: no failures; stale lock (dead pid, this host); old temp file of a finished job (888)
    for i in range(6):
        save(root, make_episode(2, i, success=True))
    lock = cio.lock_path(cio.episode_path(root, 2, 5, 7))
    lock.write_text(json.dumps(dict(job_id=None, restart="0", host=socket.gethostname(), pid=2**22 + 777, time=time.time())))
    old = cio.episode_path(root, 2, 0, 7).with_name("ep000_seed7.pt.888.tmp")
    old.write_bytes(b"x" * 10)
    t_old = time.time() - 48 * 3600
    os.utime(old, (t_old, t_old))
    # task 3: live lock (job 999 running on another host), recent temp of job 999, token-edge episode, success AT max, failure NOT at max
    ids = np.full((8, 7), 31900, dtype=np.int64)
    ids[0, 0], ids[1, 0] = 32000, 31744
    oor = np.zeros(8, dtype=bool)
    oor[0] = True
    save(root, make_episode(3, 0, success=True, T=8, ids=ids, oor=oor, min_edge=1, pad=1))
    save(root, make_episode(3, 1, success=True, T=MAX_STEPS))
    save(root, make_episode(3, 2, success=False, T=15))
    cio.lock_path(cio.episode_path(root, 3, 3, 7)).write_text(
        json.dumps(dict(job_id="999", restart="0", host="otherhost", pid=1, time=time.time())))
    cio.episode_path(root, 3, 3, 7).with_name("ep003_seed7.pt.999.tmp").write_bytes(b"partial")
    return root


def run_audit(root, tmp_path, *extra):
    out = tmp_path / "audit.json"  # outside the collection folder
    code = audit.main([str(root), "--num-tasks", "4", "--n-episodes", "6", "--json", str(out), *extra])
    return code, json.loads(out.read_text())


def codes(r):
    return {(f["code"], f["task"]) for f in r["flags"]}


def test_audit_flags_everything_and_is_read_only(dirty_root, tmp_path, capsys):
    before = snapshot(dirty_root)
    code, r = run_audit(dirty_root, tmp_path)
    after = snapshot(dirty_root)
    assert before == after, "audit modified the collection folder"
    assert code == 1
    c = codes(r)
    t = r["tasks"]
    # valid / invalid / corrupt / quarantined
    assert t["0"]["valid"] == 6 and t["1"]["valid"] == 3
    assert t["1"]["invalid"] == 1 and t["1"]["invalid_error_types"] == {"ValueError": 1}
    assert ("INVALID_EPISODE", 1) in c
    assert t["1"]["failed_validation"] == 1 and ("FAILED_VALIDATION", 1) in c
    assert t["1"]["failed_validation_reasons"] == {"unreadable": 1}
    assert t["1"]["quarantined"] == 1 and ("QUARANTINED", 1) in c
    # coverage
    assert ("DUPLICATE_EPISODES", 1) in c and t["1"]["duplicate_idx"] == [3]
    assert t["1"]["missing_idx"] == [4, 5] and ("MISSING_EPISODES", 1) in c
    assert t["3"]["missing_idx"] == [3, 4, 5]
    # outcomes
    assert t["0"]["success"] == 4 and t["0"]["failure"] == 2
    assert ("MISSING_CLASS", 2) in c and ("MISSING_CLASS", 0) not in c
    # lengths
    assert t["0"]["lengths"]["max"] == MAX_STEPS and t["0"]["at_max_steps"] == 2 and ("MAX_STEPS_MISMATCH", 0) not in c
    assert ("SUCCESS_AT_MAX", 3) in c and ("FAILURE_NOT_AT_MAX", 3) in c  # counts coincide (1 vs 1) so only per-episode flags fire
    # token checks
    tc = t["3"]["token_checks"]
    assert tc["out_of_range_steps"] == 1 and tc["n_tokens_at_bin_min_edge"] == 1 and tc["n_tokens_at_pad_edge"] == 1
    assert {("TOKEN_OUT_OF_RANGE", 3), ("TOKEN_MIN_EDGE", 3), ("TOKEN_PAD_EDGE", 3)} <= c
    # housekeeping
    assert ("STALE_LOCK", 2) in c and ("STALE_TMP", 2) in c
    assert ("STALE_LOCK", 3) not in c and ("STALE_TMP", 3) not in c
    live = t["3"]["locks"][0]
    assert live["live"] and live["job_id"] == "999" and live["job_in_squeue"] is True
    assert [x["stale"] for x in t["3"]["tmp_files"]] == [False]
    assert ("UNRECOGNIZED_FILE", 0) in c
    # consistency: mixed attn flagged, others consistent
    assert ("INCONSISTENT_META", None) in c
    mixed = [f for f in r["flags"] if f["code"] == "INCONSISTENT_META"]
    assert len(mixed) == 1 and mixed[0]["detail"].startswith("attn_implementation")
    # storage split + throughput
    assert t["0"]["bytes_feature_total"] == 6 * 0 + (4 * 8 + 2 * MAX_STEPS) * 3 * 8 * 2
    assert t["0"]["bytes_input_total"] == 8 * 4 * 4 * 3
    assert t["0"]["steps_per_sec"]["median"] == pytest.approx(2.0)  # 8 steps / 4 s (20-step failures: 5 steps/s)
    assert r["overall"]["bytes_on_disk"]["quarantine"] == len(b"old corrupt")
    md = capsys.readouterr().out
    assert "ATTENTION NEEDED" in md and "STALE" in md and "LIVE" in md


def test_warn_only_exit_zero(dirty_root, tmp_path):
    code, _ = run_audit(dirty_root, tmp_path, "--warn-only")
    assert code == 0


def test_clean_audit_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(cio, "_slurm_job_running", lambda jid: False)
    root = tmp_path / "clean"
    for task in range(2):
        save(root, make_episode(task, 0, success=True))
        save(root, make_episode(task, 1, success=False, T=MAX_STEPS))
    out = tmp_path / "a.json"
    code = audit.main([str(root), "--num-tasks", "2", "--n-episodes", "2", "--json", str(out)])
    r = json.loads(out.read_text())
    assert r["flags"] == [] and code == 0


def test_json_inside_collection_folder_is_refused(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    with pytest.raises(SystemExit):
        audit.main([str(root), "--json", str(root / "a.json")])


# ----------------------------------------------------------------------------- projection
def _audit_fixture(feature_b=1_000_000, input_b=500_000):
    task = {"valid": 10, "lengths": {"mean": 100.0}, "steps_per_sec": {"median": 2.0}, "total_steps": 1000,
            "bytes_feature_total": feature_b, "bytes_input_total": input_b}
    return {"overall": {"valid": 10, "total_steps": 1000, "lengths": {"mean": 100.0}, "steps_per_sec_pooled": 2.0,
                        "bytes_feature_total": feature_b, "bytes_input_total": input_b},
            "tasks": {"0": task}}  # task 1 absent from the pilot -> falls back to overall


def test_projection_fixed_numbers():
    p = proj.project(_audit_fixture(), episodes_per_task=10, num_tasks=2, su_per_gpu_minute=8, partition="debug")
    # usable = 3600*0.8 - 300 = 2580 s ; episode = 100 steps / 2 sps = 50 s -> 51 eps/job, capped at 10 -> 1 job per task
    assert [r["episodes_per_job"] for r in p["tasks"]] == [10, 10] and [r["jobs"] for r in p["tasks"]] == [1, 1]
    t = p["totals"]
    assert t["steps"] == 2000
    assert t["gpu_hours"] == pytest.approx((2 * (10 * 100 / 2 + 300)) / 3600)  # 1600 s
    assert t["sus"] == pytest.approx(1600 / 60 * 8)
    assert t["feature_gb"] == pytest.approx(0.002) and t["input_gb"] == pytest.approx(0.001) and t["storage_gb"] == pytest.approx(0.003)
    assert t["sus_if_every_job_uses_full_walltime"] == pytest.approx(2 * 3600 / 60 * 8)
    assert p["tasks"][0]["mean_len_src"] == "measured" and p["tasks"][0]["sps_src"] == "measured"


def test_projection_job_split_and_overrides():
    p = proj.project(_audit_fixture(), episodes_per_task=100, num_tasks=1, partition="debug")
    assert p["tasks"][0]["episodes_per_job"] == 51 and p["tasks"][0]["jobs"] == 2
    p2 = proj.project(_audit_fixture(), episodes_per_task=100, num_tasks=1, partition="gpu")  # 2 h: usable 5460 s -> 109 eps
    assert p2["tasks"][0]["episodes_per_job"] == 100 and p2["tasks"][0]["jobs"] == 1
    p3 = proj.project(_audit_fixture(), episodes_per_task=100, num_tasks=1, steps_per_sec=4.0, length_basis="max", max_steps=520)
    r = p3["tasks"][0]
    assert r["sps_src"] == "override" and r["mean_len_src"] == "assumed" and r["mean_len"] == 520
    assert r["episodes_per_job"] == int((3600 * 0.8 - 300) // (520 / 4.0))


def test_projection_marks_assumed_storage_when_pilot_has_no_bytes():
    p = proj.project(_audit_fixture(0, 0), episodes_per_task=1, num_tasks=1)
    r = p["tasks"][0]
    assert r["feature_src"] == "assumed" and r["feature_bytes_per_step"] == 221184 and r["input_src"] == "assumed"
    p2 = proj.project(_audit_fixture(0, 0), episodes_per_task=1, num_tasks=1, feature_bytes_per_step=10.0)
    assert p2["tasks"][0]["feature_src"] == "override"


def test_projection_rejects_episode_that_does_not_fit():
    with pytest.raises(ValueError):
        proj.project(_audit_fixture(), episodes_per_task=1, num_tasks=1, steps_per_sec=0.01)
