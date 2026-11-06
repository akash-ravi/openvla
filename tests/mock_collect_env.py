"""TEST-ONLY harness: runs run_libero_collect.collect() end to end on CPU with a mock LIBERO env and a tiny random
Llama as the policy. Nothing here is imported by production code. All mocks are injected by replacing the names the
collector already imports (get_model, get_action, get_libero_env, benchmark, ...), plus stub modules for libero /
robosuite-dependent imports. Run as a subprocess:  python tests/mock_collect_env.py spec.json

spec.json: {root, n_tasks, n_init, max_steps, wait, success_len, fail_at:[[task,idx],..], kill_at:[task,idx,step]|null,
            blocks, token_level_episodes_per_task}
Outcome rule: init_state_idx even -> success after `success_len` actions; odd -> never succeeds (runs to max_steps).
"""
import json
import os
import signal
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = json.loads(Path(sys.argv[1]).read_text())

# ---- stub modules for imports the collector needs but that require libero / robosuite / tensorflow -------------------
def stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m

stub("draccus", wrap=lambda *a, **k: (lambda f: f))
stub("libero")
stub("libero.libero", benchmark=None)
stub("experiments.robot.libero.libero_utils", get_libero_dummy_action=None, get_libero_env=None, get_libero_image=None,
     quat2axisangle=lambda q: np.zeros(3))
stub("experiments.robot.openvla_utils", get_processor=None)
stub("experiments.robot.robot_utils", get_action=None, get_image_resize_size=None, get_model=None,
     invert_gripper_action=None, normalize_gripper_action=None, set_seed_everywhere=None)

import experiments.robot.libero as _pkg  # noqa: E402,F401  (namespace package)
from transformers import LlamaConfig, LlamaForCausalLM, LogitsProcessor, LogitsProcessorList  # noqa: E402
import experiments.robot.libero.run_libero_collect as col  # noqa: E402  (real unc_utils is imported by it)

V, NBINS, PAD, DIM = 96, 16, 4, 64  # effective vocab = 100 - 4 = 96 ; action bins 80..95 (true range), SAFE slice 81..96


class MockEnv:
    def __init__(self, task_id):
        self.task_id, self.idx, self.calls = task_id, None, 0

    def reset(self):
        self.calls = 0

    def set_init_state(self, state):
        self.idx, self.calls = state["idx"], 0
        return self._obs()

    def _obs(self):
        img = ((np.arange(128 * 128 * 3, dtype=np.int64).reshape(128, 128, 3) + 7 * self.idx + 3 * self.calls + 11 * self.task_id) % 256).astype(np.uint8)
        return {"agentview_image": img, "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2)}

    def step(self, action):
        self.calls += 1
        if spec.get("kill_at") and [self.task_id, self.idx] == spec["kill_at"][:2] and self.calls == spec["wait"] + spec["kill_at"][2]:
            os.kill(os.getpid(), signal.SIGKILL)  # a real kill -9 mid-episode: no cleanup runs, lock stays behind
        if [self.task_id, self.idx] in spec.get("fail_at", []) and self.calls == spec["wait"] + 3:
            raise RuntimeError("injected sim crash")
        done = self.idx % 2 == 0 and self.calls >= spec["wait"] + spec["success_len"]
        return self._obs(), 0.0, done, {}

    def close(self):
        pass


class MockSuite:
    n_tasks = spec["n_tasks"]

    def get_task(self, i):
        return types.SimpleNamespace(task_id=i)

    def get_task_init_states(self, i):
        return [{"idx": k} for k in range(spec["n_init"])]


class MaskToActionBins(LogitsProcessor):  # chosen tokens land strictly inside the action-bin range (no edge tokens)
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, V - NBINS + 1: V] = 0
        return scores + mask


class MockVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cfg = LlamaConfig(vocab_size=100, hidden_size=DIM, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=4, max_position_embeddings=128)
        torch.manual_seed(1234)
        self.language_model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
        self.config = types.SimpleNamespace(text_config=types.SimpleNamespace(vocab_size=100), pad_to_multiple_of=PAD, n_action_bins=NBINS)
        self.norm_stats = {"libero_10_no_noops": {}}


def get_action(cfg, model, obs, task_label, processor=None, n_samples=1):
    seed = int(obs["full_image"].astype(np.int64).sum()) % 100003
    ids = torch.randint(1, 90, (1, 10), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        out = model.language_model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=7, min_new_tokens=7, do_sample=False,
            output_hidden_states=True, output_logits=True, return_dict_in_generate=True, use_cache=True, pad_token_id=0,
            logits_processor=LogitsProcessorList([MaskToActionBins()]))
    tok = out["sequences"][0, -7:].float()
    return ((tok - 88.0) / 8.0).numpy().astype(np.float64), out


def normalize_gripper_action(a, binarize=True):  # same maths as the fork's robot_utils
    a[..., -1] = 2 * a[..., -1] - 1
    if binarize:
        a[..., -1] = np.sign(a[..., -1])
    return a


def invert_gripper_action(a):
    a[..., -1] = a[..., -1] * -1.0
    return a


# ---- inject through the collector's existing names ------------------------------------------------------------------
col.get_model = lambda cfg: MockVLA()
col.get_processor = lambda cfg: None
col.get_action = get_action
col.get_image_resize_size = lambda cfg: 224
col.get_libero_env = lambda task, family, resolution=256: (MockEnv(task.task_id), f"mock task {task.task_id}")
col.get_libero_image = lambda obs, size: np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
col.get_libero_dummy_action = lambda family: [0, 0, 0, 0, 0, 0, -1]
col.normalize_gripper_action = normalize_gripper_action
col.invert_gripper_action = invert_gripper_action
col.set_seed_everywhere = lambda seed: torch.manual_seed(seed)
col.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_10": MockSuite})
col.MAX_STEPS_LIBERO_10 = spec["max_steps"]

cfg = col.CollectConfig(pretrained_checkpoint="mock", attn_implementation="mock-eager", out_root=spec["root"],
                        episode_start=0, episode_end=spec["n_init"], num_steps_wait=spec["wait"], seed=7,
                        blocks=spec.get("blocks", "2,4"),
                        token_level_episodes_per_task=spec.get("token_level_episodes_per_task", 1))
col.collect(cfg)
print("COLLECT_DONE")
