"""Save every OpenVLA hidden-state layer during LIBERO rollouts (shared-cache extension).

Layer convention (HuggingFace `output_hidden_states`, LLaMA-2 backbone, 33 entries):
    index 0       = token embeddings (input to decoder layer 1)
    index i, 1-31 = output of decoder layer i (before the final RMSNorm)
    index 32      = output of decoder layer 32 AFTER the final RMSNorm
                    (this is exactly the "last layer" SAFE uses, `s[-1]`)
So spreadsheet "L4" == index 4, ..., "L32" == index 32.

Per generated action token, SAFE keeps the hidden state at the last sequence position
(`s[-1][0, -1, :]`). We do the same for every layer, giving (7 tokens, 33 layers, 4096),
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
    "32=output of decoder layer 32 after final RMSNorm (SAFE's last layer)"
)
VALID_POOLS = ("mean", "first", "last")


def parse_pools(spec: str) -> list[str]:
    pools = [p.strip() for p in spec.split(",") if p.strip()]
    bad = [p for p in pools if p not in VALID_POOLS]
    if bad or not pools:
        raise ValueError(f"all_layers_pooling must be a comma list of {VALID_POOLS}, got {spec!r}")
    return pools


def pool_all_layers(all_hidden_states, pools: list[str]) -> torch.Tensor:
    """all_hidden_states: tuple over generated tokens, each a tuple over layers of (bs=1, seq, D).

    Returns a CPU bfloat16 tensor (P, L, D): P poolings, L layers (33 for OpenVLA-7B).
    """
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


def save_episode_layers(path: Path, steps: list[torch.Tensor], pools: list[str], meta: dict) -> None:
    """Write (T, P, L, D) bfloat16 tensor atomically (tmp file + rename)."""
    feats = torch.stack(steps, dim=0) if steps else torch.empty(0)
    payload = {
        "features": feats,
        "pools": pools,
        "layer_convention": LAYER_CONVENTION,
        **meta,
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
