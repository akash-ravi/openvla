"""Run-time guards for `run_libero_eval.py --save_all_layers` (shared-cache kit, must-fix 3 + dataset consistency).

Only used when `--save_all_layers` is on. With it off, SAFE's script never calls anything here.

1. `require_flash_attention_2(model, requested)` refuses to run unless the LOADED model really uses
   `flash_attention_2`. It checks what transformers 4.40.1 actually configured, not just the CLI string:
     - the requested `--attn_implementation`,
     - `model.config._attn_implementation` (a property; it reads "eager" when nothing was set, and transformers
       silently picks "sdpa" when `attn_implementation` is left as None),
     - `model.language_model.config._attn_implementation` (OpenVLA builds its Llama backbone from this),
     - the class of every attention module in the language model (must be `*FlashAttention2`),
     - that the `flash_attn` package is importable.
   Any mismatch raises `RunGuardError` (uncaught -> traceback + non-zero exit). There is no fallback to sdpa/eager.

2. `build_run_fingerprint(attention)` records what must be identical across the whole dataset: attention
   implementation, GPU name, MuJoCo render backend (`MUJOCO_GL`, plus `PYOPENGL_PLATFORM` when set; EGL vs OSMesa
   changes pixels, hence policy inputs) and the torch / transformers / flash-attn versions. The fingerprint is
   stored under the key `run_fingerprint` in the provenance JSON and in every `.layers.pt` payload;
   `collection/build_cache.py` refuses to build a cache from a mixed dataset.

This module imports only the standard library and torch, so importing it has no side effects on SAFE's script.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

REQUIRED_ATTN_IMPLEMENTATION = "flash_attention_2"
FINGERPRINT_KEY = "run_fingerprint"
FINGERPRINT_SCHEMA_VERSION = 1
RENDER_ENV_VARS = ("MUJOCO_GL", "PYOPENGL_PLATFORM")  # MUJOCO_GL always recorded; PYOPENGL_PLATFORM only if set


class RunGuardError(RuntimeError):
    """The run would produce data that cannot be mixed into the shared dataset; refuse to start."""


def _config_attn(config) -> str | None:
    # transformers 4.40.1: PretrainedConfig._attn_implementation is a property over `_attn_implementation_internal`
    return getattr(config, "_attn_implementation", None) if config is not None else None


def _attention_class_names(module) -> list[str]:
    names = set()
    for m in module.modules():
        name = type(m).__name__
        if name.endswith("Attention") or name.endswith("Attention2"):
            names.add(name)
    return sorted(names)


def require_flash_attention_2(model, requested: str | None) -> dict:
    """Raise RunGuardError unless `model` really runs flash_attention_2. Returns what was found (for the fingerprint)."""
    language_model = getattr(model, "language_model", None)
    found = {
        "requested": requested,
        "model_config": _config_attn(getattr(model, "config", None)),
        "language_model_config": _config_attn(getattr(language_model, "config", None)) if language_model is not None else None,
        "attention_classes": _attention_class_names(language_model if language_model is not None else model),
    }
    problems = []
    if requested != REQUIRED_ATTN_IMPLEMENTATION:
        problems.append(f"--attn_implementation is {requested!r}")
    if found["model_config"] != REQUIRED_ATTN_IMPLEMENTATION:
        problems.append(f"model.config._attn_implementation is {found['model_config']!r}")
    if language_model is not None and found["language_model_config"] != REQUIRED_ATTN_IMPLEMENTATION:
        problems.append(f"model.language_model.config._attn_implementation is {found['language_model_config']!r}")
    if not found["attention_classes"]:
        problems.append("no attention modules found in the language model (cannot verify the implementation)")
    non_fa2 = [n for n in found["attention_classes"] if "FlashAttention2" not in n]
    if non_fa2:
        problems.append(f"language-model attention modules are not FlashAttention2: {non_fa2}")
    if importlib.util.find_spec("flash_attn") is None:
        problems.append("the flash_attn package is not importable")
    if problems:
        raise RunGuardError(
            f"--save_all_layers requires attention implementation {REQUIRED_ATTN_IMPLEMENTATION!r} for the whole "
            "dataset; refusing to run (no fallback to sdpa/eager):\n  - " + "\n  - ".join(problems)
            + "\nInstall flash-attn==2.5.5 and run with --attn_implementation flash_attention_2.")
    return {"attn_implementation": found["model_config"], "attention_classes": found["attention_classes"]}


def _version(dist: str, module: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        mod = sys.modules.get(module)
        return getattr(mod, "__version__", None) if mod is not None else None


def gpu_name() -> str | None:
    try:
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def render_backend(environ=None) -> dict:
    env = os.environ if environ is None else environ
    backend = {"MUJOCO_GL": env.get("MUJOCO_GL")}
    if env.get("PYOPENGL_PLATFORM") is not None:
        backend["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM")
    return backend


def build_run_fingerprint(attention: dict, environ=None) -> dict:
    """`attention` is the dict returned by `require_flash_attention_2`."""
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "attn_implementation": attention["attn_implementation"],
        "attention_classes": list(attention["attention_classes"]),
        "gpu_name": gpu_name(),
        "render_backend": render_backend(environ),
        "versions": {
            "torch": _version("torch", "torch"),
            "transformers": _version("transformers", "transformers"),
            "flash_attn": _version("flash-attn", "flash_attn"),
        },
    }


def add_fingerprint_to_provenance(path: Path, fingerprint: dict) -> None:
    """Add `run_fingerprint` to an existing provenance JSON (written by layer_cache.write_provenance), atomically."""
    path = Path(path)
    info = json.loads(path.read_text())
    info[FINGERPRINT_KEY] = fingerprint
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(info, indent=2, default=str) + "\n")
    os.replace(tmp, path)
