"""Must-fix 4 (choice E2): invalid-episode markers in the patched run_libero_eval.py --save_all_layers.

CPU only. LIBERO, robosuite, tensorflow, wandb and the OpenVLA model are replaced by in-process fakes installed in
sys.modules before run_libero_eval is imported; the real episode loop, layer_cache.py and episode_guard.py run.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.robot.libero import episode_guard as eg  # noqa: E402

N_LAYERS, D, N_TOK, SUCCESS_AT = 33, 8, 7, 3  # every non-failing fake episode succeeds after 3 policy steps

# ----------------------------------------------------------------------------------------------------------------------
# Fakes for the simulator / model / logging. Behaviour is driven by SCEN, reset per test.
# ----------------------------------------------------------------------------------------------------------------------
SCEN: dict = {}


def _new_scenario(**kw):
    SCEN.clear()
    SCEN.update(fail={}, reset_fail=set(), attempts=[], envs_built=0, resets=0)
    SCEN.update(kw)


def _maybe_raise(where, task, ep, policy_step):
    spec = SCEN["fail"].get((task, ep))
    if spec and spec[0] == where and spec[1] == policy_step:
        exc = spec[2]
        raise exc() if isinstance(exc, type) else exc


class FakeEnv:
    def __init__(self, task_id):
        self.task_id, self.ep, self.policy_t = task_id, None, 0
        self.env = SimpleNamespace(parsed_problem={"goal_state": []})
        SCEN["envs_built"] += 1

    def reset(self):
        SCEN["resets"] += 1
        if (self.task_id, SCEN["resets"]) in SCEN["reset_fail"]:
            raise ValueError("fake reset failure")

    def set_init_state(self, state):
        self.ep, self.policy_t = state[1], 0
        SCEN["attempts"].append(state)
        return self._obs()

    def _obs(self):
        return {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2), "task": self.task_id, "ep": self.ep}

    def step(self, action):
        if action != [0.0] * 7:  # not a settling (dummy) step
            _maybe_raise("step", self.task_id, self.ep, self.policy_t)
            self.policy_t += 1
        return self._obs(), 0.0, self.policy_t >= SUCCESS_AT, {}

    def close(self):
        pass


def fake_get_action(cfg, model, observation, task_description, processor=None, n_samples=1):
    t, e = observation["_task_ep"]
    _maybe_raise("action", t, e, SCEN.get("_policy_t", {}).get((t, e), 0))
    SCEN.setdefault("_policy_t", {})
    SCEN["_policy_t"][(t, e)] = SCEN["_policy_t"].get((t, e), 0) + 1
    hs = tuple(tuple(torch.full((1, 1, D), float(k * 100 + l), dtype=torch.bfloat16) for l in range(N_LAYERS))
               for k in range(N_TOK))
    return np.full(7, 0.1), {"hidden_states": hs}


class FakeSuite:
    n_tasks = 2

    def get_task(self, i):
        return SimpleNamespace(name=f"task{i}", language=f"fake task {i}", id=i)

    def get_task_init_states(self, i):
        return [(i, e) for e in range(50)]


def _install_fakes(monkeypatch):
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    mod("wandb", init=lambda **k: None, log=lambda *a, **k: None, finish=lambda **k: None, Image=lambda x: x)
    libero = mod("libero")
    libero.libero = mod("libero.libero")
    libero.libero.benchmark = mod("libero.libero.benchmark", get_benchmark_dict=lambda: {"libero_10": FakeSuite})

    def get_libero_image(obs, resize_size):  # smuggle the episode id to fake_get_action through the observation
        SCEN["_cur"] = (obs["task"], obs["ep"])
        return np.zeros((4, 4, 3), np.uint8)

    def save_video(images, path):
        Path(path).write_bytes(b"mp4")

    mod("experiments.robot.libero.libero_utils", get_libero_dummy_action=lambda fam: [0.0] * 7,
        get_libero_env=lambda task, fam, resolution=256: (FakeEnv(task.id), task.language),
        get_libero_image=get_libero_image, quat2axisangle=lambda q: np.zeros(3),
        save_rollout_video_given_path=save_video)
    mod("experiments.robot.openvla_utils", get_processor=lambda cfg: None, get_text_tokens=None)

    def get_action(cfg, model, observation, task_description, processor=None, n_samples=1):
        observation["_task_ep"] = SCEN["_cur"]
        return fake_get_action(cfg, model, observation, task_description, processor, n_samples)

    mod("experiments.robot.robot_utils", get_action=get_action, get_image_resize_size=lambda cfg: 224,
        get_model=lambda cfg: SimpleNamespace(norm_stats={"libero_10": {}}, config=SimpleNamespace()),
        invert_gripper_action=lambda a: a, normalize_gripper_action=lambda a, binarize=True: a,
        set_seed_everywhere=lambda s: None)

    class PoseCumulator:
        cum_pos = cum_rot = 0.0

        def update(self, *a):
            pass

    mod("experiments.robot.viz_utils", PoseCumulator=PoseCumulator)
    mod("experiments.robot.unc_utils", compute_token_uncertainty_metrics=lambda out, model: {"m": 0.5},
        compute_samples_uncertainty_metrics=None)


@pytest.fixture()
def rle(monkeypatch):
    _install_fakes(monkeypatch)
    spec = importlib.util.spec_from_file_location("rle_under_test", REPO / "experiments/robot/libero/run_libero_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # Fixes 3 and 5 also act on the loaded model; they have their own tests (test_kit_run_guards.py,
    # test_kit_raw_block32.py). Here the model is a stand-in, so stub those two entry points.
    m.require_flash_attention_2 = lambda model, requested: {
        "attn_implementation": "flash_attention_2", "attention_classes": ["FakeFlashAttention2"]}
    m.layer_names_for = lambda model, raw_block32: [f"layer{i:02d}" for i in range(N_LAYERS)]
    _new_scenario()
    return m


def make_cfg(rle, tmp_path, **kw):
    base = dict(pretrained_checkpoint="fake", task_suite_name="libero_10", num_steps_wait=2, num_trials_per_task=4,
                use_wandb=False, save_root=str(tmp_path), run_id_note="t", output_hidden_states=True,
                save_all_layers=True, save_raw_block32=False, task_start_index=0, task_end_index=0,
                max_invalid_episodes=3)
    base.update(kw)
    return rle.GenerateConfig(**base)


def folder(tmp_path):
    return tmp_path / "t" / "libero_10"


def files_of(tmp_path, t, e):
    return sorted(p.name for p in folder(tmp_path).glob(f"task{t}--ep{e}--*"))


# ----------------------------------------------------------------------------------------------------------------------
# End-to-end through the real episode loop
# ----------------------------------------------------------------------------------------------------------------------
def test_invalid_episode_is_recorded_and_job_continues(rle, tmp_path):
    out = folder(tmp_path)
    out.mkdir(parents=True)
    # leftovers of an earlier attempt of ep1 that was killed between SAFE's non-atomic .pkl and .csv writes
    for name in ("task0--ep1--succ0.mp4", "task0--ep1--succ0.pkl", "task0--ep1--succ0.layers.pt",
                 "task0--ep1--succ1.layers.pt.tmp", "task0--ep10--succ1.csv", "task1--ep1--succ1.csv"):
        (out / name).write_text("stale")
    SCEN["fail"][(0, 1)] = ("step", 1, ValueError("MuJoCo: unstable simulation"))
    rle.eval_libero(make_cfg(rle, tmp_path))

    assert files_of(tmp_path, 0, 1) == ["task0--ep1--invalid.json"]  # no csv/pkl/layers.pt/mp4/tmp left
    assert (out / "task0--ep10--succ1.csv").exists() and (out / "task1--ep1--succ1.csv").exists()  # others untouched
    rec = json.loads((out / "task0--ep1--invalid.json").read_text())
    assert rec["format"] == 1 and rec["kind"] == "invalid_episode"
    assert (rec["task_id"], rec["episode_idx"], rec["init_state_index"], rec["seed"]) == (0, 1, 1, 7)
    assert rec["exception_type"] == "builtins.ValueError" and "unstable simulation" in rec["exception_message"]
    assert "Traceback" in rec["traceback"] and "ValueError" in rec["traceback"]
    assert rec["step_index"] == 2 + 1 and rec["num_steps_wait"] == 2 and rec["policy_steps_completed"] == 2
    assert rec["timestamp_utc"].endswith("+00:00") and rec["invalid_count_in_job_range"] == 1
    assert rec["max_invalid_episodes"] == 3 and "pid" in rec["job"]
    for e in (0, 2, 3):  # the job went on with the next episodes
        names = files_of(tmp_path, 0, e)
        assert [n.split(".", 1)[1] for n in names] == ["csv", "layers.pt", "mp4", "pkl"], names
    assert not list(out.glob("*.tmp"))
    assert SCEN["resets"] == 4 + 1  # one reset per episode + the recovery reset after the invalid one


def test_resume_counts_marker_as_done_even_when_it_is_the_last_episode(rle, tmp_path):
    SCEN["fail"][(0, 3)] = ("step", 0, ValueError("deterministic"))
    rle.eval_libero(make_cfg(rle, tmp_path))
    assert [e for _, e in SCEN["attempts"]] == [0, 1, 2, 3]
    _new_scenario(fail={(0, 3): ("step", 0, ValueError("deterministic"))})
    # task 1 still has work: SAFE's own script ends with env.close() and crashes (UnboundLocalError) when --resume
    # finds every task of the range complete, independent of this fix
    rle.eval_libero(make_cfg(rle, tmp_path, resume=True, task_end_index=1))
    assert all(t == 1 for t, _ in SCEN["attempts"])  # task 0 ep3 is not retried: 3 csv + 1 marker = complete


def test_resume_with_every_task_done_exits_cleanly(rle, tmp_path):
    # SAFE's script ends with env.close(); when --resume finds the whole range complete no env exists.
    # With save_all_layers the job must still finish normally (a retry loop would read a crash as failure).
    SCEN["fail"][(0, 3)] = ("step", 0, ValueError("deterministic"))
    rle.eval_libero(make_cfg(rle, tmp_path))
    _new_scenario()
    rle.eval_libero(make_cfg(rle, tmp_path, resume=True))  # must not raise UnboundLocalError
    assert SCEN.get("attempts", []) == []


def test_resume_with_every_task_done_without_flag_is_unchanged(rle, tmp_path):
    rle.eval_libero(make_cfg(rle, tmp_path, save_all_layers=False))
    with pytest.raises(UnboundLocalError):  # SAFE's original behaviour, deliberately untouched
        rle.eval_libero(make_cfg(rle, tmp_path, save_all_layers=False, resume=True))


def test_resume_after_kill_starts_after_the_marker(rle, tmp_path):
    SCEN["fail"][(0, 1)] = ("step", 0, ValueError("bad init state"))
    SCEN["fail"][(0, 2)] = ("step", 1, KeyboardInterrupt)  # the job is killed during ep2
    with pytest.raises(KeyboardInterrupt):
        rle.eval_libero(make_cfg(rle, tmp_path))
    assert files_of(tmp_path, 0, 2) == [] and files_of(tmp_path, 0, 1) == ["task0--ep1--invalid.json"]
    _new_scenario()
    rle.eval_libero(make_cfg(rle, tmp_path, resume=True))
    assert [e for _, e in SCEN["attempts"]] == [2, 3]  # highest done = marker of ep1 -> restart at ep2


@pytest.mark.parametrize("where,exc", [
    ("action", RuntimeError("CUDA error: device-side assert triggered")),
    ("action", torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB")),
    ("action", RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx")),
    ("action", MemoryError()),
    ("step", KeyboardInterrupt),
    ("step", SystemExit(3)),
])
def test_systemic_errors_abort_without_marker(rle, tmp_path, where, exc):
    SCEN["fail"][(0, 1)] = (where, 1, exc)
    with pytest.raises(type(exc) if not isinstance(exc, type) else exc):
        rle.eval_libero(make_cfg(rle, tmp_path))
    assert not list(folder(tmp_path).glob("*invalid.json*"))
    assert [e for _, e in SCEN["attempts"]] == [0, 1]


def test_cap_aborts_job_and_counts_markers_already_on_disk(rle, tmp_path):
    out = folder(tmp_path)
    out.mkdir(parents=True)
    eg.atomic_write_json({"format": 1}, out / "task0--ep9--invalid.json")  # from an earlier run of this task range
    eg.atomic_write_json({"format": 1}, out / "task1--ep0--invalid.json")  # other task: outside the range, not counted
    SCEN["fail"][(0, 0)] = ("step", 0, ValueError("a"))
    SCEN["fail"][(0, 1)] = ("action", 0, ValueError("b"))
    with pytest.raises(eg.TooManyInvalidEpisodes, match="max_invalid_episodes=2") as ei:
        rle.eval_libero(make_cfg(rle, tmp_path, max_invalid_episodes=2))
    assert isinstance(ei.value.__cause__, ValueError)
    assert files_of(tmp_path, 0, 1) == ["task0--ep1--invalid.json"]  # the marker that crossed the cap is kept
    assert [e for _, e in SCEN["attempts"]] == [0, 1]  # ep2 never started


def test_rerun_without_resume_replaces_stale_marker(rle, tmp_path):
    SCEN["fail"][(0, 2)] = ("step", 0, ValueError("transient"))
    rle.eval_libero(make_cfg(rle, tmp_path))
    assert files_of(tmp_path, 0, 2) == ["task0--ep2--invalid.json"]
    _new_scenario()
    rle.eval_libero(make_cfg(rle, tmp_path))  # no --resume: SAFE redoes every episode; ep2 now completes
    assert [n.split(".", 1)[1] for n in files_of(tmp_path, 0, 2)] == ["csv", "layers.pt", "mp4", "pkl"]


def test_cap_zero_aborts_on_first_invalid(rle, tmp_path):
    SCEN["fail"][(0, 0)] = ("step", 0, ValueError("a"))
    with pytest.raises(eg.TooManyInvalidEpisodes):
        rle.eval_libero(make_cfg(rle, tmp_path, max_invalid_episodes=0))


def test_without_save_all_layers_safe_behaviour_is_unchanged(rle, tmp_path):
    SCEN["fail"][(0, 1)] = ("step", 1, ValueError("MuJoCo: unstable simulation"))
    with pytest.raises(ValueError, match="unstable"):
        rle.eval_libero(make_cfg(rle, tmp_path, save_all_layers=False))
    assert not list(folder(tmp_path).glob("*invalid*"))
    assert [e for _, e in SCEN["attempts"]] == [0, 1]


def test_without_save_all_layers_resume_ignores_markers(rle, tmp_path):
    out = folder(tmp_path)
    out.mkdir(parents=True)
    (out / "task0--ep3--invalid.json").write_text("{}")
    rle.eval_libero(make_cfg(rle, tmp_path, save_all_layers=False, resume=True))
    assert [e for _, e in SCEN["attempts"]] == [0, 1, 2, 3]


def test_env_is_rebuilt_when_recovery_reset_fails(rle, tmp_path):
    SCEN["fail"][(0, 0)] = ("step", 0, ValueError("a"))
    SCEN["reset_fail"].add((0, 2))  # 2nd reset = the recovery reset after the invalid ep0
    rle.eval_libero(make_cfg(rle, tmp_path))
    assert SCEN["envs_built"] == 2
    assert [e for _, e in SCEN["attempts"]] == [0, 1, 2, 3]


def test_task_where_every_episode_is_invalid_does_not_crash_the_summary(rle, tmp_path):
    for e in range(4):
        SCEN["fail"][(0, e)] = ("step", 0, ValueError("x"))
    rle.eval_libero(make_cfg(rle, tmp_path, max_invalid_episodes=10))  # no ZeroDivisionError
    assert len(list(folder(tmp_path).glob("task0--ep*--invalid.json"))) == 4
    assert not list(folder(tmp_path).glob("task0--*succ*"))


# ----------------------------------------------------------------------------------------------------------------------
# Unit tests of episode_guard.py
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("exc,systemic", [
    (ValueError("MuJoCo: unstable simulation"), False),
    (AssertionError("x"), False),
    (KeyError("robot0_eef_pos"), False),
    (RuntimeError("shape mismatch"), False),
    (KeyboardInterrupt(), True),
    (SystemExit(1), True),
    (MemoryError(), True),
    (torch.cuda.OutOfMemoryError("CUDA out of memory"), True),
    (RuntimeError("CUDA error: an illegal memory access was encountered"), True),
    (RuntimeError("CUDA error: device-side assert triggered"), True),
    (RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR"), True),
    (RuntimeError("FlashAttention only supports Ampere GPUs or newer."), True),
    (RuntimeError("EGL_NOT_INITIALIZED"), True),
    (ModuleNotFoundError("No module named 'flash_attn'"), True),
    (NameError("x"), True),
    (OSError(28, "No space left on device"), True),
    (OSError(2, "No such file"), False),
])
def test_is_systemic(exc, systemic):
    assert eg.is_systemic(exc) is systemic


def test_is_systemic_follows_exception_chain():
    try:
        try:
            raise RuntimeError("CUDA error: unspecified launch failure")
        except RuntimeError as inner:
            raise ValueError("wrapped") from inner
    except ValueError as outer:
        assert eg.is_systemic(outer)


def test_atomic_write_leaves_no_tmp_on_failure(tmp_path, monkeypatch):
    def boom(a, b):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(eg.os, "replace", boom)
    with pytest.raises(OSError):
        eg.atomic_write_json({"a": 1}, tmp_path / "task0--ep0--invalid.json")
    assert list(tmp_path.iterdir()) == []


def test_invalid_markers_ignore_tmp_and_other_tasks(tmp_path):
    for n in ("task1--ep2--invalid.json", "task1--ep3--invalid.json.123.tmp", "task11--ep0--invalid.json",
              "task1--ep4--succ1.csv"):
        (tmp_path / n).write_text("{}")
    assert [p.name for p in eg.invalid_markers(tmp_path, 1)] == ["task1--ep2--invalid.json"]


def test_record_invalid_returns_false_for_systemic_and_writes_nothing(tmp_path):
    g = eg.EpisodeGuard(tmp_path, 3, [0])
    (tmp_path / "task0--ep0--succ0.pkl").write_text("x")
    assert g.record_invalid(RuntimeError("CUDA error: x"), task_id=0, episode_idx=0, step_index=5) is False
    assert sorted(p.name for p in tmp_path.iterdir()) == ["task0--ep0--succ0.pkl"]


def test_negative_cap_rejected(tmp_path):
    with pytest.raises(ValueError):
        eg.EpisodeGuard(tmp_path, -1, [0])
