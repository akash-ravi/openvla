"""CPU tests for experiments/robot/libero/run_guards.py (must-fix 3, run-time half, + E3 run fingerprint).

LIBERO, robosuite, tensorflow and flash-attn are not installable on the dev Mac, so these tests exercise run_guards
directly with mocked models and tiny real transformers-4.40.1 Llama models, not the rollout loop.
"""
import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.robot.libero import run_guards as rg  # noqa: E402

FA2 = "flash_attention_2"


class LlamaFlashAttention2(nn.Module):
    pass


class LlamaSdpaAttention(nn.Module):
    pass


class LlamaAttention(nn.Module):
    pass


def fake_vla(top_attn=FA2, lm_attn=FA2, attn_cls=LlamaFlashAttention2, n_layers=2):
    """Mimics OpenVLAForActionPrediction: .config and .language_model (with its own .config)."""
    vla = nn.Module()
    vla.config = SimpleNamespace(_attn_implementation=top_attn)
    lm = nn.Module()
    lm.config = SimpleNamespace(_attn_implementation=lm_attn)
    lm.layers = nn.ModuleList([nn.ModuleDict({"self_attn": attn_cls()}) for _ in range(n_layers)])
    vla.language_model = lm
    vla.vision_backbone = nn.ModuleDict({"attn": LlamaAttention()})  # e.g. timm ViT attention: must be ignored
    return vla


@pytest.fixture()
def flash_attn_importable(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(rg.importlib.util, "find_spec",
                        lambda name, *a, **k: SimpleNamespace(name=name) if name == "flash_attn" else real(name, *a, **k))


def test_accepts_real_flash_attention_2(flash_attn_importable):
    out = rg.require_flash_attention_2(fake_vla(), FA2)
    assert out == {"attn_implementation": FA2, "attention_classes": ["LlamaFlashAttention2"]}


@pytest.mark.parametrize("kwargs, requested, needle", [
    ({}, "sdpa", "--attn_implementation is 'sdpa'"),
    ({}, None, "--attn_implementation is None"),
    ({"top_attn": "sdpa"}, FA2, "model.config._attn_implementation is 'sdpa'"),
    ({"top_attn": "eager"}, FA2, "model.config._attn_implementation is 'eager'"),
    ({"lm_attn": "sdpa"}, FA2, "model.language_model.config._attn_implementation is 'sdpa'"),
    ({"attn_cls": LlamaSdpaAttention}, FA2, "not FlashAttention2: ['LlamaSdpaAttention']"),  # config lies
    ({"attn_cls": LlamaAttention}, FA2, "not FlashAttention2: ['LlamaAttention']"),
    ({"n_layers": 0}, FA2, "no attention modules found"),
])
def test_refuses_anything_but_flash_attention_2(flash_attn_importable, kwargs, requested, needle):
    with pytest.raises(rg.RunGuardError) as e:
        rg.require_flash_attention_2(fake_vla(**kwargs), requested)
    assert needle in str(e.value) and "no fallback" in str(e.value)


def test_refuses_when_flash_attn_package_missing(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(rg.importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "flash_attn" else real(name, *a, **k))
    with pytest.raises(rg.RunGuardError, match="flash_attn package is not importable"):
        rg.require_flash_attention_2(fake_vla(), FA2)


def test_run_guard_error_is_not_swallowed_as_success():
    # uncaught -> Python exits non-zero; it must not be a SystemExit(0)/Warning
    assert issubclass(rg.RunGuardError, RuntimeError)


# ---- real transformers 4.40.1 models (the attribute the guard reads is the one transformers actually sets) ----
transformers = pytest.importorskip("transformers")
needs_4401 = pytest.mark.skipif(transformers.__version__ != "4.40.1", reason="needs transformers==4.40.1")


def tiny_llama(impl):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=16, attn_implementation=impl)
    return LlamaForCausalLM(cfg)


def wrap(lm):
    vla = nn.Module()
    vla.config = SimpleNamespace(_attn_implementation=lm.config._attn_implementation)
    vla.language_model = lm
    return vla


@needs_4401
@pytest.mark.parametrize("impl, actual", [("eager", "eager"), ("sdpa", "sdpa"), (None, "sdpa")])
def test_real_llama_without_flash_attention_is_refused(flash_attn_importable, impl, actual):
    lm = tiny_llama(impl)
    assert lm.config._attn_implementation == actual  # None silently becomes sdpa in 4.40.1
    with pytest.raises(rg.RunGuardError):
        rg.require_flash_attention_2(wrap(lm), FA2)


@needs_4401
def test_real_llama_with_flash_attention_2_modules_is_accepted(flash_attn_importable):
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    lm = tiny_llama("eager")
    for i, layer in enumerate(lm.model.layers):  # flash-attn is not installable here: swap the module classes in
        layer.self_attn = LlamaFlashAttention2(lm.config, layer_idx=i)
    lm.config._attn_implementation = FA2
    out = rg.require_flash_attention_2(wrap(lm), FA2)
    assert out["attention_classes"] == ["LlamaFlashAttention2"]


# ---- fingerprint ----
def test_fingerprint_contents(monkeypatch):
    monkeypatch.setattr(rg.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rg.torch.cuda, "get_device_name", lambda i=0: "NVIDIA A10")
    fp = rg.build_run_fingerprint({"attn_implementation": FA2, "attention_classes": ["LlamaFlashAttention2"]},
                                  environ={"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "OTHER": "x"})
    assert fp["attn_implementation"] == FA2 and fp["gpu_name"] == "NVIDIA A10"
    assert fp["render_backend"] == {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    assert fp["versions"]["torch"] is not None
    assert fp["versions"]["transformers"] == transformers.__version__
    assert set(fp["versions"]) == {"torch", "transformers", "flash_attn"}
    json.dumps(fp)  # must be JSON-serialisable for the provenance file


def test_fingerprint_without_gpu_and_optional_pyopengl(monkeypatch):
    monkeypatch.setattr(rg.torch.cuda, "is_available", lambda: False)
    fp = rg.build_run_fingerprint({"attn_implementation": FA2, "attention_classes": []}, environ={"MUJOCO_GL": "osmesa"})
    assert fp["gpu_name"] is None and fp["render_backend"] == {"MUJOCO_GL": "osmesa"}
    assert rg.render_backend({}) == {"MUJOCO_GL": None}


def test_fingerprint_lands_in_provenance_and_layers_payload(tmp_path):
    from experiments.robot.libero.layer_cache import save_episode_layers, write_provenance
    fp = rg.build_run_fingerprint({"attn_implementation": FA2, "attention_classes": ["LlamaFlashAttention2"]},
                                  environ={"MUJOCO_GL": "egl"})
    prov = tmp_path / "provenance" / "provenance--tasks0-0--joblocal.json"
    write_provenance(prov, {"seed": 7})
    rg.add_fingerprint_to_provenance(prov, fp)
    info = json.loads(prov.read_text())
    assert info["run_fingerprint"] == fp and info["cfg"] == {"seed": 7}  # existing keys untouched
    assert not list(prov.parent.glob("*.tmp"))
    # same meta dict shape run_libero_eval.py passes, plus the ONE new key
    path = tmp_path / "task0--ep0--succ1.layers.pt"
    save_episode_layers(path, [torch.zeros(3, 33, 4, dtype=torch.bfloat16)], ["mean", "first", "last"],
                        {"task_id": 0, "episode_idx": 0, "episode_success": True, "seed": 7, "run_fingerprint": fp})
    payload = torch.load(path, weights_only=False)
    assert payload["run_fingerprint"] == fp and payload["features"].shape == (1, 3, 33, 4)


# ---- the call sites in run_libero_eval.py are only reached with --save_all_layers ----
def test_call_sites_are_gated_by_save_all_layers():
    src = (REPO / "experiments" / "robot" / "libero" / "run_libero_eval.py").read_text()
    tree = ast.parse(src)
    guarded = {"require_flash_attention_2", "build_run_fingerprint", "add_fingerprint_to_provenance"}
    seen = set()

    def visit(node, gated):
        if isinstance(node, ast.If) and "save_all_layers" in ast.unparse(node.test):
            for child in node.body:
                visit(child, True)
            for child in node.orelse:
                visit(child, gated)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in guarded:
            assert gated, f"{node.func.id} called outside an `if cfg.save_all_layers` block (line {node.lineno})"
            seen.add(node.func.id)
        if isinstance(node, ast.Name) and node.id == "run_fingerprint" and isinstance(node.ctx, ast.Load):
            assert gated, f"run_fingerprint used outside an `if cfg.save_all_layers` block (line {node.lineno})"
        for child in ast.iter_child_nodes(node):
            visit(child, gated)

    visit(tree, False)
    assert seen == guarded
    # the guard runs right after the model is loaded, before any rollout or provenance write
    assert src.index("require_flash_attention_2(model, cfg.attn_implementation)") < src.index("write_provenance(save_folder")
