"""
feature_capture.py -- per-generation-state hidden-state capture for the collector (A1).

CPU-importable (torch only). Works on any HF Llama-style causal LM; OpenVLA is located at
`model.language_model.model` (verified in the fork), a plain LlamaForCausalLM at `model.model`.

What is captured, per action (= one `generate(max_new_tokens=7)` call = 7 generation states j = 0..6):
  * forward hooks on decoder layers: raw residual output of block k = layers[k-1], position [0, -1]
  * forward PRE-hook on the final norm: raw output of block L (the tuple in `hidden_states` only has norm(block L))
  * forward hook on the final norm: `final_norm`; asserted bit-equal to hidden_states[j][-1][0, -1]
  * `layer_mean_all` per token = float32 mean of raw blocks `mean_blocks` (default 1..L) cast to bf16
Token-level tensors are bf16. Pooling (first / last / mean over the 7 states) is applied AFTER per-token capture and is
a deterministic function of the bf16 token-level values (mean = float32 mean of bf16 values, cast to bf16), so pooled
features can be recomputed exactly from `token_level_sample`.

Everything is parameterised by FeatureCaptureConfig. Buffers live on the model's device as one [7, L, D] tensor per
action (no per-hook device->host sync) and are copied to CPU once in end_action(); buffers are rebuilt every action.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

POOLINGS = ("first", "last", "mean")


@dataclass
class FeatureCaptureConfig:
    blocks: Tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32)  # raw block_k keys (1-based, k <= L)
    final_norm: bool = True
    layer_mean_all: bool = True
    mean_blocks: Optional[Tuple[int, ...]] = None              # None => all blocks 1..L
    pooling: Tuple[str, ...] = POOLINGS
    n_states: int = 7                                          # generation states per action
    verify: bool = True                                        # assert hooks == hidden_states every action (1 sync)


def parse_int_list(s) -> Tuple[int, ...]:
    if isinstance(s, (list, tuple)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).replace(" ", "").split(",") if x != "")


def locate_llama(model):
    """(layers, norm). OpenVLA: model.language_model.model ; plain LlamaForCausalLM: model.model."""
    llama = model.language_model.model if hasattr(model, "language_model") else model.model
    layers, norm = llama.layers, llama.norm
    assert type(norm).__name__ == "LlamaRMSNorm", type(norm)
    return layers, norm


def pool(x: torch.Tensor, pooling: Sequence[str]) -> torch.Tensor:
    """x: [n_states, ..., D] bf16 token-level -> [len(pooling), ..., D] bf16. Deterministic function of x."""
    outs = []
    for p in pooling:
        if p == "first":
            outs.append(x[0])
        elif p == "last":
            outs.append(x[-1])
        elif p == "mean":
            outs.append(x.float().mean(0).to(x.dtype))
        else:
            raise ValueError(p)
    return torch.stack(outs, 0)


def layer_mean(blocks_tokens: torch.Tensor) -> torch.Tensor:
    """blocks_tokens: [n_states, n_blocks, D] (bf16) -> [n_states, D] bf16: float32 mean over blocks, cast to bf16."""
    return blocks_tokens.float().mean(1).to(blocks_tokens.dtype)


class FeatureCapture:
    def __init__(self, cfg: Optional[FeatureCaptureConfig] = None):
        self.cfg = cfg or FeatureCaptureConfig()
        for p in self.cfg.pooling:
            assert p in POOLINGS, p
        self.model = None
        self._handles: List = []
        self.last_token_level: Optional[torch.Tensor] = None  # [n_states, n_keys, D] bf16, CPU, set by end_action

    # ------------------------------------------------------------------ setup
    def attach(self, model) -> None:
        assert not self._handles, "already attached"
        self.model = model
        self.layers, self.norm = locate_llama(model)
        self.L = len(self.layers)
        c = self.cfg
        self.mean_blocks = tuple(range(1, self.L + 1)) if c.mean_blocks is None else tuple(c.mean_blocks)
        for k in tuple(c.blocks) + self.mean_blocks:
            assert 1 <= k <= self.L, f"block {k} outside 1..{self.L}"
        self.layer_keys: List[str] = [f"block_{k}" for k in c.blocks]
        if c.final_norm:
            self.layer_keys.append("final_norm")
        if c.layer_mean_all:
            self.layer_keys.append("layer_mean_all")
        assert self.layer_keys, "nothing to capture"
        need = set(c.blocks) | (set(self.mean_blocks) if c.layer_mean_all else set())
        # block L comes from the final-norm pre-hook; a layers[L-1] hook is added only to verify that equality
        self._hook_blocks = sorted(k for k in need if k < self.L)
        if c.verify and self.L in need:
            self._hook_blocks.append(self.L)
        self._need_norm_pre = self.L in need or c.verify
        self._need_norm_out = c.final_norm or c.verify
        self._reset()

        def make_layer_hook(k):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                self._write(self._blk, self._cnt_blk, k, h[0, -1])
            return hook

        for k in self._hook_blocks:
            self._handles.append(self.layers[k - 1].register_forward_hook(make_layer_hook(k)))
        if self._need_norm_pre:
            self._handles.append(self.norm.register_forward_pre_hook(
                lambda _m, inp: self._write(self._blk, self._cnt_blk, "norm_in", inp[0][0, -1])))
        if self._need_norm_out:
            self._handles.append(self.norm.register_forward_hook(
                lambda _m, _i, out: self._write(self._blk, self._cnt_blk, "norm_out", out[0, -1])))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------------ per action
    def _reset(self) -> None:
        self._blk: Dict = {}      # key -> [n_states, D] tensor on model device
        self._cnt_blk: Dict = {}  # key -> hook call count this action

    def _write(self, buf, cnt, key, vec) -> None:
        j = cnt.get(key, 0)
        if j >= self.cfg.n_states:
            raise RuntimeError(f"hook {key} fired more than {self.cfg.n_states} times in one action")
        if key not in buf:
            buf[key] = torch.empty(self.cfg.n_states, vec.shape[-1], dtype=vec.dtype, device=vec.device)
        buf[key][j] = vec.detach()
        cnt[key] = j + 1

    def begin_action(self) -> None:
        """Clear all buffers/counters. Call before every generate()."""
        self._reset()
        self.last_token_level = None

    def end_action(self, generated_outputs=None) -> Dict[str, torch.Tensor]:
        """Return {layer_key: [len(pooling), D] bf16 CPU}. Also sets self.last_token_level [n_states, n_keys, D]."""
        c, n = self.cfg, self.cfg.n_states
        for key, cnt in self._cnt_blk.items():
            if cnt != n:
                raise RuntimeError(f"hook {key} fired {cnt} times, expected {n}")
        expected = set(self._hook_blocks) | ({"norm_in"} if self._need_norm_pre else set()) | ({"norm_out"} if self._need_norm_out else set())
        if set(self._cnt_blk) != expected:
            raise RuntimeError(f"hooks missing: {expected - set(self._cnt_blk)}")
        raw = {k: self._blk[k] for k in self._hook_blocks if k < self.L}
        if self._need_norm_pre:
            raw[self.L] = self._blk["norm_in"]

        if c.verify:
            self._verify(generated_outputs)

        # one device->host copy per action
        dev = {k: v for k, v in raw.items()}
        tok: Dict[str, torch.Tensor] = {}
        for k in c.blocks:
            tok[f"block_{k}"] = dev[k]
        if c.final_norm:
            tok["final_norm"] = self._blk["norm_out"]
        if c.layer_mean_all:
            tok["layer_mean_all"] = layer_mean(torch.stack([dev[k] for k in self.mean_blocks], 1))
        tok = {k: v.detach().to("cpu", dtype=torch.bfloat16) for k, v in tok.items()}
        self._reset()  # buffers never survive an action
        assert list(tok) == self.layer_keys, (list(tok), self.layer_keys)
        self.last_token_level = torch.stack([tok[k] for k in self.layer_keys], 1)  # [n_states, n_keys, D]
        return {k: pool(v, c.pooling) for k, v in tok.items()}

    def _verify(self, go) -> None:
        """Hooks must reproduce HF's hidden_states exactly (one equality check per group -> few syncs)."""
        n, L = self.cfg.n_states, self.L
        if go is None or "hidden_states" not in go:
            raise RuntimeError("verify=True needs generated_outputs with output_hidden_states=True")
        hs = go["hidden_states"]
        assert len(hs) == n and all(len(s) == L + 1 for s in hs), "unexpected hidden_states structure"
        final = torch.stack([hs[j][L][0, -1] for j in range(n)])
        if not torch.equal(self._blk["norm_out"], final):
            raise RuntimeError("final_norm hook output != hidden_states[j][-1][0,-1]")
        blocks = [k for k in self._hook_blocks if k < self.L]
        if blocks:
            ref = torch.stack([torch.stack([hs[j][k][0, -1] for j in range(n)]) for k in blocks])
            got = torch.stack([self._blk[k] for k in blocks])
            if not torch.equal(ref, got):
                raise RuntimeError("layer hook outputs != hidden_states[j][k][0,-1]")
        if self._need_norm_pre and self.L in self._hook_blocks:
            if not torch.equal(self._blk["norm_in"], self._blk[self.L]):
                raise RuntimeError("norm pre-hook input != layers[L-1] output")
        # RMSNorm(raw block L) vs final_norm is reported by tests/gpu/test_layer_mapping.py (T6), not asserted here.

    # ------------------------------------------------------------------ metadata
    def describe(self) -> dict:
        c = self.cfg
        a, b = (min(self.mean_blocks), max(self.mean_blocks)) if c.layer_mean_all else (None, None)
        return {
            "features_captured": True, "layer_keys": list(self.layer_keys), "pooling": list(c.pooling),
            "num_blocks": self.L, "n_states": c.n_states,
            "layer_mean_def": (f"per token: float32 mean of raw blocks {list(self.mean_blocks)} (block_{a}..block_{b}), cast to bf16, then pooled"
                               if c.layer_mean_all else None),
            "pooling_def": "pooled after per-token capture, from bf16 token-level values; mean = float32 mean over the n_states tokens, cast to bf16",
            "final_norm_def": "norm module output at position [0,-1] per generation state (== hidden_states[j][-1][0,-1], asserted)",
            "feature_verify": c.verify,
        }
