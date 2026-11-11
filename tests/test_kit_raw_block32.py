"""Must-fix 5 (raw block 32 = layer index 33): CPU tests of layer_cache.py on a tiny random LlamaForCausalLM
(3 blocks, hidden 32, fp32), driven through greedy generate(max_new_tokens=7, output_hidden_states=True) with KV cache,
exactly as OpenVLA's predict_action does. Needs transformers==4.40.1 (the fork's pin); skipped otherwise.
The GPU proof on the real 7B model is Safe-Research tests/gpu/test_layer_mapping.py (T6)."""
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.40.1":
    pytest.skip(f"needs transformers==4.40.1, found {transformers.__version__}", allow_module_level=True)
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.robot.libero import layer_cache as lc  # noqa: E402

N_BLOCKS, D, N_TOK, POOLS = 3, 32, 7, ["mean", "first", "last"]


def _old_pool_all_layers(all_hidden_states, pools):
    """Verbatim copy of pool_all_layers from Rhea's patch commit b5cb926 (the 33-layer reference)."""
    per_token = [torch.stack([layer[0, -1, :] for layer in token_states], dim=0) for token_states in all_hidden_states]
    tok = torch.stack(per_token, dim=0)  # (n_tokens, L, D)
    out = []
    for p in pools:
        if p == "mean":
            out.append(tok.float().mean(dim=0))
        elif p == "first":
            out.append(tok[0].float())
        else:
            out.append(tok[-1].float())
    return torch.stack(out, dim=0).to(torch.bfloat16).cpu()


class FakeVLA(torch.nn.Module):
    """Mimics OpenVLA's attribute path: model.language_model = LlamaForCausalLM."""

    def __init__(self, lm):
        super().__init__()
        self.language_model = lm


@pytest.fixture(scope="module")
def model():
    cfg = LlamaConfig(vocab_size=64, hidden_size=D, intermediate_size=64, num_hidden_layers=N_BLOCKS,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).float().eval()


def generate(model, seed=0, n_prompt=9):
    ids = torch.randint(1, 64, (1, n_prompt), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        return model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=N_TOK, min_new_tokens=N_TOK,
                              do_sample=False, output_hidden_states=True, return_dict_in_generate=True,
                              use_cache=True, pad_token_id=0)


def test_capture_matches_block_output_and_norm(model):
    block_out = []
    h = model.model.layers[-1].register_forward_hook(lambda _m, _i, out: block_out.append(out[0][0, -1].clone()))
    try:
        out, cap = lc.call_with_raw_block32(True, FakeVLA(model), generate, model)
    finally:
        h.remove()
    hs = out["hidden_states"]
    assert len(hs) == N_TOK and all(len(s) == N_BLOCKS + 1 for s in hs)
    assert len(cap.states) == len(block_out) == N_TOK  # one capture per generated token
    assert cap.seq_lens == [9] + [1] * (N_TOK - 1)       # first forward = the prompt pass
    raw = cap.aligned_states(hs)
    for j in range(N_TOK):
        assert torch.equal(raw[j], block_out[j])                                      # pre-norm = last block's output
        assert torch.equal(model.model.norm(raw[j][None, None])[0, 0], hs[j][-1][0, -1])  # norm(raw) = hidden_states[-1]
        assert not torch.equal(raw[j], hs[j][-1][0, -1])
        assert not torch.equal(raw[j], hs[j][-2][0, -1])                             # hs[-2] is block N-1, not N
    assert not model.model.norm._forward_pre_hooks  # removed after generation


def test_tokens_unchanged_by_hook(model):
    ref = generate(model, seed=3)
    hooked, _ = lc.call_with_raw_block32(True, model, generate, model, seed=3)
    assert torch.equal(ref["sequences"], hooked["sequences"])


def test_pooled_has_extra_layer_and_old_layers_unchanged(model):
    out, cap = lc.call_with_raw_block32(True, model, generate, model, seed=1)
    hs = out["hidden_states"]
    new = lc.pool_all_layers(hs, POOLS, cap)
    old = _old_pool_all_layers(hs, POOLS)
    assert new.shape == (3, N_BLOCKS + 2, D) and new.dtype == torch.bfloat16
    assert torch.equal(new[:, : N_BLOCKS + 1], old)  # indices 0..N keep their exact meaning
    raw = torch.stack(cap.states)  # (7, D)
    assert torch.equal(new[0, -1], raw.float().mean(0).to(torch.bfloat16))
    assert torch.equal(new[1, -1], raw[0].to(torch.bfloat16)) and torch.equal(new[2, -1], raw[-1].to(torch.bfloat16))


def test_flag_off_reproduces_old_output(model, tmp_path):
    out, cap = lc.call_with_raw_block32(False, model, generate, model, seed=2)
    assert cap is None and not model.model.norm._forward_pre_hooks
    new = lc.pool_all_layers(out["hidden_states"], POOLS)
    assert torch.equal(new, _old_pool_all_layers(out["hidden_states"], POOLS))
    names = lc.layer_names_for(model, raw_block32=False)
    assert names == ["emb", "block01", "block02", "block03_postnorm"]
    lc.save_episode_layers(tmp_path / "a.layers.pt", [new, new], POOLS, {"task_id": 0, "layer_names": names})
    payload = torch.load(tmp_path / "a.layers.pt", weights_only=False)
    assert torch.equal(payload["features"], torch.stack([new, new])) and payload["layer_names"] == names


def test_hook_count_mismatch_fails_loudly(model):
    out, cap = lc.call_with_raw_block32(True, model, generate, model)
    with torch.no_grad():  # an extra forward while capturing -> 8 captures for 7 tokens
        with cap:
            model(torch.ones(1, 2, dtype=torch.long))
    with pytest.raises(RuntimeError, match="fired 8 times for 7 generated tokens"):
        lc.pool_all_layers(out["hidden_states"], POOLS, cap)
    cap.states, cap.seq_lens = cap.states[:N_TOK], [1] + cap.seq_lens[1:N_TOK]
    with pytest.raises(RuntimeError, match="out of order"):
        cap.aligned_states(out["hidden_states"])


def test_hook_removed_on_error(model):
    def boom(m):
        generate(m)
        raise ValueError("simulated failure inside get_action")
    with pytest.raises(ValueError, match="simulated"):
        lc.call_with_raw_block32(True, model, boom, model)
    assert not model.model.norm._forward_pre_hooks


def test_layer_names_and_save_validation(tmp_path):
    names = lc.make_layer_names(32, raw_block32=True)
    assert len(names) == 34 and names[0] == "emb" and names[1] == "block01" and names[31] == "block31"
    assert names[32] == "block32_postnorm" and names[33] == "block32_prenorm"
    assert lc.make_layer_names(32) == names[:33]
    f34 = torch.zeros(3, 34, 4).to(torch.bfloat16)
    lc.save_episode_layers(tmp_path / "b.layers.pt", [f34], POOLS, {"layer_names": names})
    assert torch.load(tmp_path / "b.layers.pt", weights_only=False)["layer_names"] == names
    with pytest.raises(ValueError, match="34 layers saved but 33 layer_names"):  # no names = legacy 33 only
        lc.save_episode_layers(tmp_path / "c.layers.pt", [f34], POOLS, {})
    lc.save_episode_layers(tmp_path / "d.layers.pt", [f34[:, :33]], POOLS, {})
    assert torch.load(tmp_path / "d.layers.pt", weights_only=False)["layer_names"] == names[:33]
    assert "33 (only if save_raw_block32" in lc.LAYER_CONVENTION
    with pytest.raises(TypeError, match="RMSNorm"):
        lc.find_final_norm(torch.nn.Linear(2, 2))
