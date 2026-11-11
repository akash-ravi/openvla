"""Fixes 3, 4 and 5 together in the real run_libero_eval.py episode loop (CPU only).

Each fix has its own tests; this file checks that they compose: one --save_all_layers run with raw block 32 on,
a run fingerprint, and one crashing episode. Reuses the fakes of test_kit_episode_guard.py, but gives the model a
real LlamaRMSNorm final norm that the fake get_action calls once per generated token, as OpenVLA's generate does.
"""
import importlib.util
import json

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

import test_kit_episode_guard as base
from test_kit_episode_guard import D, N_LAYERS, N_TOK, SCEN, folder, make_cfg

N_BLOCKS = N_LAYERS - 1  # 33 hidden states = embeddings + 32 blocks


class FakeVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.model = torch.nn.Module()
        self.language_model.model.layers = torch.nn.ModuleList(torch.nn.Identity() for _ in range(N_BLOCKS))
        self.language_model.model.norm = LlamaRMSNorm(D)
        self.norm_stats = {"libero_10": {}}


@pytest.fixture()
def rle(monkeypatch):
    base._install_fakes(monkeypatch)
    spec = importlib.util.spec_from_file_location("rle_integration", base.REPO / "experiments/robot/libero/run_libero_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    base._new_scenario()
    model = FakeVLA()
    monkeypatch.setattr(m, "get_model", lambda cfg: model)
    # flash-attn cannot be installed on a laptop: accept the stand-in model, but build the REAL fingerprint
    monkeypatch.setattr(m, "require_flash_attention_2", lambda mdl, requested: {
        "attn_implementation": "flash_attention_2", "attention_classes": ["FakeFlashAttention2"]})
    inner = m.get_action

    def get_action_through_norm(cfg, mdl, observation, task_description, processor=None, n_samples=1):
        out = inner(cfg, mdl, observation, task_description, processor, n_samples)
        for _ in range(N_TOK):  # one language-model forward per generated token -> pre-hook fires each time
            mdl.language_model.model.norm(torch.ones(1, 1, D))
        return out

    monkeypatch.setattr(m, "get_action", get_action_through_norm)
    return m


def test_fixes_3_4_5_compose_in_one_run(rle, tmp_path, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    SCEN["fail"][(0, 1)] = ("step", 1, ValueError("bad init state"))
    rle.eval_libero(make_cfg(rle, tmp_path, save_raw_block32=True))

    out = folder(tmp_path)
    assert json.loads((out / "task0--ep1--invalid.json").read_text())["exception_type"] == "builtins.ValueError"
    assert not list(out.glob("task0--ep1--succ*"))
    layer_files = sorted(out.glob("task0--ep*--succ*.layers.pt"))
    assert [p.name.split("--")[1] for p in layer_files] == ["ep0", "ep2", "ep3"]
    for p in layer_files:
        payload = torch.load(p)
        feats = payload["features"]
        assert feats.shape[2] == N_LAYERS + 1                       # fix 5: index 33 added
        assert payload["layer_names"][-2:] == ["block32_postnorm", "block32_prenorm"]
        assert len(payload["layer_names"]) == feats.shape[2]
        fp = payload["run_fingerprint"]                              # fix 3
        assert fp["attn_implementation"] == "flash_attention_2"
        assert fp["render_backend"] == {"MUJOCO_GL": "egl"}
        # raw block 32 = the norm's input; the fake feeds ones, so every pooled value is exactly 1
        assert torch.equal(feats[:, :, -1].float(), torch.ones_like(feats[:, :, -1].float()))
    prov = next((out / "provenance").glob("*.json"))
    assert json.loads(prov.read_text())["run_fingerprint"]["attn_implementation"] == "flash_attention_2"
