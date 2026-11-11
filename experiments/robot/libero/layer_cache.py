"""Save every OpenVLA hidden-state layer during LIBERO rollouts (shared-cache extension).

Layer convention (HuggingFace `output_hidden_states`, LLaMA-2 backbone, 33 entries):
    index 0       = token embeddings (input to decoder layer 1)
    index i, 1-31 = output of decoder layer i (before the final RMSNorm)
    index 32      = output of decoder layer 32 AFTER the final RMSNorm
                    (this is exactly the "last layer" SAFE uses, `s[-1]`)
    index 33      = output of decoder layer 32 BEFORE the final RMSNorm ("raw block 32"); only present
                    with `save_raw_block32=True` (the default with --save_all_layers). It is not in
                    `hidden_states`, so it is captured by a forward PRE-hook on the final norm
                    (`model.language_model.model.norm`), once per generated token. Pending team approval
                    (review checklist item 3); `--save_raw_block32 False` gives the old 33-layer files.
So spreadsheet "L4" == index 4, ..., "L32" == index 32. Indices 0-32 mean the same with or without index 33.
Every `.layers.pt` stores `layer_names` (one string per index): "emb", "block01".."block31",
"block32_postnorm", and "block32_prenorm" for index 33. Files without `layer_names` predate this key and
follow the 33-layer convention.

Per generated action token, SAFE keeps the hidden state at the last sequence position
(`s[-1][0, -1, :]`). We do the same for every layer, giving (7 tokens, L layers, 4096) with L = 33 or 34,
then reduce over the 7 action tokens with the poolings SAFE uses via `token_idx_rel`:
    mean  -> token_idx_rel="mean"
    first -> token_idx_rel=0.0 (state before the first action token is generated)
    last  -> token_idx_rel=1.0
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

LAYER_CONVENTION = (
    "hf_hidden_states: 0=embeddings, i(1..31)=output of decoder layer i (pre-norm), "
    "32=output of decoder layer 32 after final RMSNorm (SAFE's last layer), "
    "33 (only if save_raw_block32; see layer_names)=output of decoder layer 32 BEFORE the final RMSNorm "
    "(forward pre-hook on the final norm)"
)
LEGACY_NUM_LAYERS = 33  # files without `layer_names`: 0=emb, 1..31 blocks, 32=block 32 after the final norm
VALID_POOLS = ("mean", "first", "last")


def parse_pools(spec: str) -> list[str]:
    pools = [p.strip() for p in spec.split(",") if p.strip()]
    bad = [p for p in pools if p not in VALID_POOLS]
    if bad or not pools:
        raise ValueError(f"all_layers_pooling must be a comma list of {VALID_POOLS}, got {spec!r}")
    return pools


def make_layer_names(n_blocks: int = 32, raw_block32: bool = False) -> list[str]:
    """Names of the saved layer indices: emb, block01.., block{N}_postnorm[, block{N}_prenorm]."""
    names = ["emb"] + [f"block{i:02d}" for i in range(1, n_blocks)] + [f"block{n_blocks:02d}_postnorm"]
    if raw_block32:
        names.append(f"block{n_blocks:02d}_prenorm")
    return names


def find_final_norm(model):
    """The language model's final RMSNorm. OpenVLA: model.language_model.model.norm (LlamaRMSNorm);
    a plain LlamaForCausalLM (tests): model.model.norm."""
    lm = getattr(model, "language_model", model)
    norm = getattr(getattr(lm, "model", None), "norm", None)
    if norm is None or "RMSNorm" not in type(norm).__name__:
        raise TypeError(f"save_raw_block32: expected <model>.language_model.model.norm to be an RMSNorm, got {type(norm)}")
    return norm


def layer_names_for(model, raw_block32: bool) -> list[str]:
    """Layer names for this model; also fails early if the final norm cannot be located."""
    if raw_block32:
        find_final_norm(model)
    lm = getattr(model, "language_model", model)
    return make_layer_names(len(lm.model.layers), raw_block32)


class RawBlock32Capture:
    """Forward PRE-hook on the final RMSNorm: records its input (the raw output of the last decoder block) at the
    last sequence position, once per language-model forward = once per generated token. Use as a context
    manager so the hook is removed after generation and on error."""

    def __init__(self, model):
        self.norm = find_final_norm(model)
        self.states: list[torch.Tensor] = []
        self.seq_lens: list[int] = []
        self._handle = None

    def _hook(self, _module, args):
        h = args[0]
        self.states.append(h[0, -1, :].detach().clone())
        self.seq_lens.append(int(h.shape[1]))

    def __enter__(self):
        self._handle = self.norm.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False

    def aligned_states(self, all_hidden_states) -> list[torch.Tensor]:
        """Captured states, after checking one capture per generated token in the order of `all_hidden_states`
        (first forward = the prompt pass, whose sequence length is the prompt length; then one token each)."""
        n = len(all_hidden_states)
        if len(self.states) != n:
            raise RuntimeError(f"raw block-32 hook fired {len(self.states)} times for {n} generated tokens")
        expected = [int(token_states[-1].shape[1]) for token_states in all_hidden_states]
        if self.seq_lens != expected:
            raise RuntimeError(f"raw block-32 captures out of order: seq lens {self.seq_lens} != hidden_states {expected}")
        return self.states


def call_with_raw_block32(enabled: bool, model, fn, *args, **kwargs):
    """Return (fn(*args, **kwargs), RawBlock32Capture or None). With enabled=False this is just fn(...)."""
    if not enabled:
        return fn(*args, **kwargs), None
    with RawBlock32Capture(model) as capture:
        result = fn(*args, **kwargs)
    return result, capture


def pool_all_layers(all_hidden_states, pools: list[str], raw_block32: RawBlock32Capture | None = None) -> torch.Tensor:
    """all_hidden_states: tuple over generated tokens, each a tuple over layers of (bs=1, seq, D).
    raw_block32: the capture from `call_with_raw_block32`; if given, it is appended as the last layer index.

    Returns a CPU bfloat16 tensor (P, L, D): P poolings, L layers (33 for OpenVLA-7B, 34 with raw block 32).
    """
    per_token = [[layer[0, -1, :] for layer in token_states] for token_states in all_hidden_states]
    if raw_block32 is not None:
        for layers, raw in zip(per_token, raw_block32.aligned_states(all_hidden_states)):
            layers.append(raw)
    per_token = [torch.stack(layers, dim=0) for layers in per_token]
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


def save_episode_layers(path: Path, steps: list[torch.Tensor], pools: list[str], meta: dict) -> None:
    """Write (T, P, L, D) bfloat16 tensor atomically (tmp file + rename).
    meta["layer_names"] (one name per layer index) is stored; if absent, L must be 33 (legacy convention)."""
    feats = torch.stack(steps, dim=0) if steps else torch.empty(0)
    names = list(meta.get("layer_names") or make_layer_names())
    if feats.dim() == 4 and feats.shape[2] != len(names):
        raise ValueError(f"{feats.shape[2]} layers saved but {len(names)} layer_names: {names[-2:]}")
    payload = {
        "features": feats,
        "pools": pools,
        "layer_convention": LAYER_CONVENTION,
        **meta,
        "layer_names": names,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _git_commit(repo_dir: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_provenance(path: Path, cfg, model=None) -> None:
    """Record everything needed to reproduce or audit the rollouts."""
    gpu = None
    if torch.cuda.is_available():
        gpu = {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
               "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    checkpoint_commit = None
    if model is not None:
        checkpoint_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    repo_dir = Path(__file__).resolve().parents[3]
    info = {
        "cfg": asdict(cfg) if is_dataclass(cfg) else dict(cfg),
        "checkpoint_commit_hash": checkpoint_commit,
        "openvla_repo_commit": _git_commit(repo_dir),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": gpu,
        "packages": {n: _pkg_version(n) for n in (
            "torch", "transformers", "tokenizers", "timm", "accelerate", "flash-attn", "numpy",
            "tensorflow", "robosuite", "mujoco", "libero", "draccus")},
        "layer_convention": LAYER_CONVENTION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str) + "\n")
