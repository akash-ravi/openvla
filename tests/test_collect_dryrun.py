"""End-to-end CPU dry run: collector episode loop -> files -> audit (full and --fast), with a mock env, a tiny random Llama
as policy, one injected exception and one real SIGKILL mid-episode followed by a resume. Mocks live in
tests/mock_collect_env.py (test-only); production code has no mock branches. Needs transformers==4.40.1, scipy, sklearn."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.40.1":
    pytest.skip(f"needs transformers==4.40.1, found {transformers.__version__}", allow_module_level=True)
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.robot.libero import audit_collection as audit  # noqa: E402
from experiments.robot.libero import collect_io as cio  # noqa: E402

SPEC = dict(n_tasks=2, n_init=4, max_steps=12, wait=2, success_len=6, blocks="2,4", token_level_episodes_per_task=1)
KEYS = ["block_2", "block_4", "final_norm", "layer_mean_all"]


def run_collect(root, tmp_path, name, **over):
    spec = dict(SPEC, root=str(root), **over)
    sp = tmp_path / f"{name}.json"
    sp.write_text(json.dumps(spec))
    return subprocess.run([sys.executable, str(REPO / "tests" / "mock_collect_env.py"), str(sp)], cwd=REPO,
                          capture_output=True, text=True, timeout=600)


def pt_sc_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("task*/*")) if p.suffix == ".pt" or p.name.endswith(".meta.json")}


def codes(r):
    return {(f["code"], f["task"]) for f in r["flags"]}


@pytest.fixture(scope="module")
def killed_then_resumed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dry")
    root = tmp / "collect"
    # run A: task 1 / init 1 is SIGKILLed 2 actions into the episode
    a = run_collect(root, tmp, "A", fail_at=[[1, 2]], kill_at=[1, 1, 2])
    assert a.returncode == -signal.SIGKILL, a.stdout[-2000:] + a.stderr[-2000:]
    state_after_kill = {"files": sorted(str(p.relative_to(root)) for p in root.glob("task*/*")), "hashes": pt_sc_hashes(root)}
    # simulate extra crash artefacts before resuming: sidecar lost between the two renames; an old temp of a dead job
    victim = cio.episode_path(root, 0, 3, 7)
    cio.sidecar_path(victim).unlink()
    old_tmp = victim.with_name(victim.name + ".99999.tmp")
    old_tmp.write_bytes(b"partial")
    t_old = time.time() - 72 * 3600
    os.utime(old_tmp, (t_old, t_old))
    hashes_before_resume = pt_sc_hashes(root)
    # run B: resume (same command, no kill)
    b = run_collect(root, tmp, "B", fail_at=[[1, 2]])
    assert b.returncode == 0 and "COLLECT_DONE" in b.stdout, b.stdout[-3000:] + b.stderr[-3000:]
    return dict(tmp=tmp, root=root, killed=state_after_kill, before=hashes_before_resume, resume_out=b.stdout, victim=victim, old_tmp=old_tmp)


def test_kill_left_expected_crash_state(killed_then_resumed):
    names = killed_then_resumed["killed"]["files"]  # directory listing right after the SIGKILL, before the resume
    assert {"task00/ep000_seed7.pt", "task00/ep003_seed7.pt", "task01/ep000_seed7.pt"} <= set(names)  # finished before the kill
    assert "task01/ep001_seed7.lock" in names                        # the killed episode's lock was left behind
    assert "task01/ep001_seed7.pt" not in names and "task01/ep001_seed7.meta.json" not in names
    assert "task01/ep002_seed7.pt" not in names and "task01/ep003_seed7.pt" not in names  # never reached
    assert not any(n.endswith(".tmp") for n in names)
    assert (killed_then_resumed["root"] / "task01" / "ep001_seed7.pt").exists()  # produced by the resume


def test_resume_reclaims_lock_repairs_sidecar_cleans_old_tmp_and_does_not_touch_finished_episodes(killed_then_resumed):
    r = killed_then_resumed
    out = r["resume_out"]
    assert "reclaimed stale lock" in out and "ep001_seed7.lock" in out
    assert "(re)wrote sidecar" in out
    assert not r["old_tmp"].exists()
    assert cio.sidecar_path(r["victim"]).exists()
    after = pt_sc_hashes(r["root"])
    for k, h in r["before"].items():  # every episode that was finished before the resume is byte-identical afterwards
        if k.endswith(".meta.json") and k.endswith("task00/ep003_seed7.meta.json"):
            continue  # repaired
        assert after[k] == h, k
    assert not list(r["root"].glob("task*/*.lock")) and not list(r["root"].glob("task*/*.tmp"))


@pytest.mark.parametrize("fast", [False, True], ids=["full", "fast"])
def test_audit_after_kill_and_resume(killed_then_resumed, tmp_path, fast):
    root = killed_then_resumed["root"]
    before = pt_sc_hashes(root)
    out = tmp_path / "a.json"
    code = audit.main([str(root), "--num-tasks", "2", "--n-episodes", "4", "--expect-seed", "7", "--json", str(out)] + (["--fast"] if fast else []))
    r = json.loads(out.read_text())
    assert pt_sc_hashes(root) == before  # audit is read-only
    o, t = r["overall"], r["tasks"]
    assert r["mode"] == ("fast" if fast else "full")
    assert (o["valid"], o["invalid"], o["failed_validation"], o["quarantined"]) == (7, 1, 0, 0)
    assert (o["success"], o["failure"]) == (3, 4)
    assert t["0"]["valid"] == 4 and t["1"]["valid"] == 3 and t["1"]["invalid"] == 1
    assert t["1"]["invalid_error_types"] == {"RuntimeError": 1}
    assert all(not x["missing_idx"] and not x["duplicate_idx"] for x in t.values())
    assert sorted(p.name for p in (root / "task01").glob("*.pt")) == [f"ep00{i}_seed7.pt" for i in range(4)]  # no duplicates
    assert {x["lengths"]["min"] for x in t.values()} == {6} and {x["lengths"]["max"] for x in t.values()} == {12}
    assert all(x["at_max_steps"] == x["failure"] for x in t.values())  # failures run to max_steps, successes stop early
    assert t["0"]["token_checks"]["out_of_range_steps"] == 0 and t["0"]["token_checks"]["n_tokens_at_pad_edge"] == 0
    # the only flag is the injected exception; no SIDECAR_*, no stale locks/tmp, no inconsistent metadata
    assert codes(r) == {("INVALID_EPISODE", 1)}, r["flags"]
    assert code == 1
    assert audit.main([str(root), "--num-tasks", "2", "--n-episodes", "4", "--warn-only", "--json", str(tmp_path / "b.json")] + (["--fast"] if fast else [])) == 0
    assert r["overall"]["bytes_feature_total"] > 0


def test_episode_contents_features_keys_shapes_and_pooling(killed_then_resumed):
    root = killed_then_resumed["root"]
    for task in (0, 1):
        for idx in range(4):
            p = cio.episode_path(root, task, idx, 7)
            assert cio.validate_episode_file(p, task, idx, 7) is None
            ep = torch.load(p, weights_only=False)
            m = ep["meta"]
            if m["invalid"]:
                assert (task, idx) == (1, 2) and m["success"] is None and m["label_failure"] is None and "injected sim crash" in m["error"]
                continue
            T = m["length"]
            assert m["success"] == (idx % 2 == 0) and m["label_failure"] == int(idx % 2 == 1)
            assert (m["layer_keys"], m["pooling"], m["attn_implementation"]) == (KEYS, ["first", "last", "mean"], "mock-eager")
            assert "layer_mean_def" in m and m["episode_seed"] == cio.derive_episode_seed(7, task, idx)
            assert list(ep["steps"]["features"]) == KEYS
            for k, v in ep["steps"]["features"].items():
                assert v.shape == (T, 3, 64) and v.dtype == torch.bfloat16, k
            assert tuple(ep["steps"]["action_token_ids"].shape) == (T, 7) and tuple(ep["steps"]["token_probs"].shape) == (T, 7)
            assert list(ep["steps"]["s"]) == list(range(T)) and list(ep["steps"]["env_t"]) == [s + 2 for s in range(T)]
            assert m["n_out_of_range_token_steps"] == 0 and m["n_tokens_at_pad_edge"] == 0
            if idx == 0:  # token_level_episodes_per_task=1 -> only init state 0 stores token-level features
                tl = ep["token_level_sample"]
                assert tl.shape == (T, 7, 4, 64) and m["token_level_keys"] == KEYS
                for i, k in enumerate(KEYS):  # pooled features are exactly recomputable from the token-level sample
                    x = tl[:, :, i]  # [T,7,D]
                    f = ep["steps"]["features"][k]
                    assert torch.equal(f[:, 0], x[:, 0]) and torch.equal(f[:, 1], x[:, 6])
                    assert torch.equal(f[:, 2], x.float().mean(1).to(torch.bfloat16))
            else:
                assert "token_level_sample" not in ep


def test_resumed_run_matches_uninterrupted_run(killed_then_resumed, tmp_path):
    """Per-episode seeding + deterministic policy: kill+resume gives the same data as one clean run."""
    clean_root = tmp_path / "clean"
    c = run_collect(clean_root, tmp_path, "C", fail_at=[[1, 2]])
    assert c.returncode == 0, c.stdout[-2000:] + c.stderr[-2000:]
    for task in (0, 1):
        for idx in range(4):
            a = torch.load(cio.episode_path(killed_then_resumed["root"], task, idx, 7), weights_only=False)
            b = torch.load(cio.episode_path(clean_root, task, idx, 7), weights_only=False)
            assert a["meta"]["invalid"] == b["meta"]["invalid"]
            if a["meta"]["invalid"]:
                continue
            for k in ("action_token_ids", "executed_action", "token_probs", "token_entropy", "s", "env_t"):
                assert (torch.as_tensor(a["steps"][k]) == torch.as_tensor(b["steps"][k])).all(), (task, idx, k)
            for k in a["steps"]["features"]:
                assert torch.equal(a["steps"]["features"][k], b["steps"]["features"][k]), (task, idx, k)
            assert a["meta"]["episode_seed"] == b["meta"]["episode_seed"]


def test_second_resume_is_a_noop(killed_then_resumed, tmp_path):
    root = killed_then_resumed["root"]
    before = pt_sc_hashes(root)
    r = run_collect(root, tmp_path, "D", fail_at=[[1, 2]])
    assert r.returncode == 0 and "COLLECT_DONE" in r.stdout
    assert pt_sc_hashes(root) == before
    assert "'done': 0" in r.stdout and "len=" not in r.stdout  # nothing was re-collected
