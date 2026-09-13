"""Load ngxson's Fly-LLM classes from fly-llm-hf without renaming that clone."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_DIR = ROOT / "fly-llm-hf"
CONNECTOME_KEYS = ("w_offsets", "w_indices", "w_values", "in_index", "out_index")
_PKG = "fly_hf_src"


def load_fly_classes():
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(HF_DIR)]
        pkg.__file__ = str(HF_DIR / "__init__.py")
        sys.modules[_PKG] = pkg

    def _load(name: str):
        full = f"{_PKG}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, HF_DIR / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {name} from {HF_DIR}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        return module

    configuration_fly = _load("configuration_fly")
    modeling_fly = _load("modeling_fly")
    return configuration_fly.FlyConfig, modeling_fly.FlyForCausalLM


def load_published_config():
    FlyConfig, _ = load_fly_classes()
    return FlyConfig.from_json_file(str(HF_DIR / "config.json"))


def load_tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))


def encode_text(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text).ids)


def trainable_parameter_groups(model) -> dict[str, list]:
    fast = []
    slow = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.startswith("brain.wte")
            or name == "brain.in_proj"
            or name in {"brain.gain", "brain.rec_gain", "brain.bias"}
        ):
            fast.append(param)
        else:
            slow.append(param)
    return {"fast": fast, "slow": slow}


def copy_connectome(model, buffers: dict) -> None:
    missing = [key for key in CONNECTOME_KEYS if key not in buffers]
    if missing:
        raise KeyError(f"connectome checkpoint missing {missing}")
    brain = model.brain
    brain.w_offsets.copy_(buffers["w_offsets"].to(dtype=brain.w_offsets.dtype))
    brain.w_indices.copy_(buffers["w_indices"].to(dtype=brain.w_indices.dtype))
    brain.w_values.copy_(buffers["w_values"].to(dtype=brain.w_values.dtype))
    brain.in_index.copy_(buffers["in_index"].to(dtype=brain.in_index.dtype))
    brain.out_index.copy_(buffers["out_index"].to(dtype=brain.out_index.dtype))


def extract_connectome_tensors(state: dict) -> dict:
    stripped = {}
    for key, value in state.items():
        name = key.split(".")[-1]
        if name in CONNECTOME_KEYS:
            stripped[name] = value
    missing = [key for key in CONNECTOME_KEYS if key not in stripped]
    if missing:
        raise KeyError(f"could not find connectome tensors {missing} in {list(state)[:20]}")
    return stripped


def save_connectome(path: Path, buffers: dict, meta: dict | None = None) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: buffers[key].cpu() for key in CONNECTOME_KEYS}
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta or {}, indent=2), encoding="utf-8")
