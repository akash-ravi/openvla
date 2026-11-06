"""CPU tests for feature_capture.FeatureCapture on a tiny random LlamaForCausalLM (4 layers, hidden 64), driven through
generate(max_new_tokens=7, output_hidden_states=True, return_dict_in_generate=True) with KV cache.
Needs transformers==4.40.1 (the version OpenVLA's fork pins); skipped otherwise."""
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.40.1":
    pytest.skip(f"needs transformers==4.40.1, found {transformers.__version__}", allow_module_level=True)
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.robot.libero.feature_capture import FeatureCapture, FeatureCaptureConfig  # noqa: E402

L, D, NS = 4, 64, 7
BF = torch.bfloat16


def make_model(dtype, seed=0):
    cfg = LlamaConfig(vocab_size=100, hidden_size=D, intermediate_size=128, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg).to(dtype).eval()


def gen(model, ids, capture=None):
    if capture is not None:
        capture.begin_action()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=NS, min_new_tokens=NS, do_sample=False,
                             output_hidden_states=True, return_dict_in_generate=True, use_cache=True, pad_token_id=0)
    feats = capture.end_action(out) if capture is not None else None
    return out, feats


def prompt(seed, n=12):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 100, (1, n), generator=g)


def cfg(**kw):
    base = dict(blocks=(2, 4), final_norm=True, layer_mean_all=True, pooling=("first", "last", "mean"), n_states=NS)
    base.update(kw)
    return FeatureCaptureConfig(**base)


@pytest.fixture(params=[torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def model(request):
    return make_model(request.param)


def independent_hooks(model):
    """Reference capture written separately from FeatureCapture: raw output of every layer + norm pre-hook input."""
    layers, norm = model.model.layers, model.model.norm
    blocks = {k: [] for k in range(1, L + 1)}
    norm_in = []
    hs = [layers[k - 1].register_forward_hook(lambda m, i, o, k=k: blocks[k].append((o[0] if isinstance(o, tuple) else o)[0, -1].clone()))
          for k in range(1, L + 1)]
    hs.append(norm.register_forward_pre_hook(lambda m, i: norm_in.append(i[0][0, -1].clone())))
    return blocks, norm_in, hs


def test_hidden_states_structure_matches_openvla_assumptions(model):
    out, _ = gen(model, prompt(1))
    hs = out.hidden_states
    assert len(hs) == NS and all(len(s) == L + 1 for s in hs)
    assert hs[0][0].shape == (1, 12, D) and all(s[0].shape == (1, 1, D) for s in hs[1:])  # prefill, then KV-cache steps


def test_block_hooks_equal_hidden_states_exactly(model):
    cap = FeatureCapture(cfg(blocks=(1, 2, 3, 4)))
    cap.attach(model)
    out, _ = gen(model, prompt(1), cap)
    tl, keys = cap.last_token_level, cap.layer_keys
    assert keys == ["block_1", "block_2", "block_3", "block_4", "final_norm", "layer_mean_all"] and tl.shape == (NS, 6, D) and tl.dtype == BF
    for j in range(NS):
        for k in range(1, L):  # k = 1..L-1: layer hook == hidden_states[j][k]
            assert torch.equal(tl[j, keys.index(f"block_{k}")], out.hidden_states[j][k][0, -1].to(BF)), (j, k)
        assert torch.equal(tl[j, keys.index("final_norm")], out.hidden_states[j][L][0, -1].to(BF)), j
    cap.detach()


def test_norm_prehook_input_reapplied_norm_equals_final_hidden_state_exactly(model):
    cap = FeatureCapture(cfg())
    cap.attach(model)
    blocks, norm_in, hs = independent_hooks(model)
    out, _ = gen(model, prompt(2), cap)
    assert len(norm_in) == NS
    tl, keys = cap.last_token_level, cap.layer_keys
    for j in range(NS):
        recomputed = model.model.norm(norm_in[j][None, None, :])[0, 0]
        assert torch.equal(recomputed, out.hidden_states[j][L][0, -1]), f"state {j}"
        assert torch.equal(norm_in[j], blocks[L][j])  # raw block L via pre-hook == layers[L-1] output
        assert torch.equal(tl[j, keys.index("block_4")], norm_in[j].to(BF))  # captured block_L came from the pre-hook
    for h in hs:
        h.remove()
    cap.detach()


def test_pooling_recomputed_from_token_level_sample_matches(model):
    cap = FeatureCapture(cfg())
    cap.attach(model)
    _, feats = gen(model, prompt(3), cap)
    tl, keys = cap.last_token_level, cap.layer_keys
    assert list(feats) == keys
    for i, key in enumerate(keys):
        x = tl[:, i]  # [7, D] bf16
        assert feats[key].shape == (3, D) and feats[key].dtype == BF
        assert torch.equal(feats[key][0], x[0])                              # first = state 0
        assert torch.equal(feats[key][1], x[6])                              # last = state 6
        assert torch.equal(feats[key][2], x.float().mean(0).to(BF))          # mean over the 7 states
    cap.detach()


def test_layer_mean_all_is_per_token_mean_of_raw_blocks_then_pooled(model):
    cap = FeatureCapture(cfg())
    cap.attach(model)
    blocks, norm_in, hs = independent_hooks(model)
    _, feats = gen(model, prompt(4), cap)
    tl, keys = cap.last_token_level, cap.layer_keys
    ref_tok = torch.stack([torch.stack([blocks[k][j] for k in range(1, L + 1)]).float().mean(0).to(BF)
                           for j in range(NS)])  # [7, D]: mean of raw blocks 1..L per token
    assert torch.equal(tl[:, keys.index("layer_mean_all")], ref_tok)
    assert torch.equal(feats["layer_mean_all"][0], ref_tok[0]) and torch.equal(feats["layer_mean_all"][1], ref_tok[6])
    assert torch.equal(feats["layer_mean_all"][2], ref_tok.float().mean(0).to(BF))
    for h in hs:
        h.remove()
    cap.detach()


def test_custom_mean_blocks_and_describe(model):
    cap = FeatureCapture(cfg(blocks=(2,), mean_blocks=(2, 3), final_norm=False, pooling=("last", "first")))
    cap.attach(model)
    blocks, _, hs = independent_hooks(model)
    _, feats = gen(model, prompt(5), cap)
    assert cap.layer_keys == ["block_2", "layer_mean_all"] and feats["block_2"].shape == (2, D)
    ref = torch.stack([torch.stack([blocks[2][j], blocks[3][j]]).float().mean(0).to(BF) for j in range(NS)])
    assert torch.equal(feats["layer_mean_all"][0], ref[6]) and torch.equal(feats["layer_mean_all"][1], ref[0])  # last, first
    d = cap.describe()
    assert d["layer_keys"] == cap.layer_keys and d["pooling"] == ["last", "first"] and "[2, 3]" in d["layer_mean_def"]
    for h in hs:
        h.remove()
    cap.detach()


def test_generated_tokens_identical_with_and_without_hooks(model):
    ids = prompt(6)
    base, _ = gen(model, ids)
    cap = FeatureCapture(cfg())
    cap.attach(model)
    hooked, _ = gen(model, ids, cap)
    assert torch.equal(base.sequences, hooked.sequences)
    cap.detach()
    after, _ = gen(model, ids)  # detach restores the original behaviour
    assert torch.equal(base.sequences, after.sequences)
    assert not model.model.norm._forward_hooks and not model.model.norm._forward_pre_hooks
    assert all(not l._forward_hooks for l in model.model.layers)


def test_buffers_do_not_leak_across_actions(model):
    cap = FeatureCapture(cfg())
    cap.attach(model)
    ids_a, ids_b = prompt(7), prompt(8, n=15)
    _, fa1 = gen(model, ids_a, cap)
    assert cap._cnt_blk == {} and cap._blk == {}  # cleared by end_action
    _, fb = gen(model, ids_b, cap)
    _, fa2 = gen(model, ids_a, cap)
    for k in fa1:  # identical input -> identical features, regardless of what ran in between
        assert torch.equal(fa1[k], fa2[k]), k
    assert not all(torch.equal(fa1[k], fb[k]) for k in fa1)  # different prompt -> different features
    fresh = FeatureCapture(cfg())
    model2 = make_model(next(model.parameters()).dtype)
    fresh.attach(model2)
    _, fb_fresh = gen(model2, ids_b, fresh)
    for k in fb:  # second-in-sequence action equals a fresh capture of the same input
        assert torch.equal(fb[k], fb_fresh[k]), k
    # forgetting end_action() between actions is detected loudly, and begin_action() recovers
    cap.begin_action()
    gen_out, _ = gen(model, ids_a)  # hooks still attached: fires 7 times without end_action
    with pytest.raises(RuntimeError):
        gen(model, ids_a)  # would be a 14th call
    _, ok = gen(model, ids_a, cap)  # begin_action() clears everything
    assert torch.equal(ok["block_2"], fa1["block_2"])
    cap.detach()


def test_verify_catches_outputs_from_a_different_run(model):
    cap = FeatureCapture(cfg())
    cap.attach(model)
    other, _ = gen(model, prompt(10))
    cap.begin_action()
    with torch.no_grad():
        model.generate(prompt(9), attention_mask=torch.ones_like(prompt(9)), max_new_tokens=NS, min_new_tokens=NS, do_sample=False,
                       output_hidden_states=True, return_dict_in_generate=True, pad_token_id=0)
    with pytest.raises(RuntimeError):
        cap.end_action(other)  # hooks from run 9, hidden_states from run 10
    cap.detach()


def test_works_through_language_model_wrapper_like_openvla(model):
    class Wrapper(torch.nn.Module):  # OpenVLA exposes the LLM as model.language_model (a LlamaForCausalLM)
        def __init__(self, lm):
            super().__init__()
            self.language_model = lm
    w = Wrapper(model)
    cap = FeatureCapture(cfg())
    cap.attach(w)
    _, feats = gen(model, prompt(11), cap)  # generation drives the inner LM, as OpenVLA's forward does
    assert list(feats) == cap.layer_keys and cap.L == L
    cap.detach()


def test_invalid_config_is_rejected(model):
    with pytest.raises(AssertionError):
        FeatureCapture(cfg(blocks=(2, 8))).attach(model)  # block 8 > L
    with pytest.raises(AssertionError):
        FeatureCapture(cfg(pooling=("median",)))
