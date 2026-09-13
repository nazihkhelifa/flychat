"""Export the trained checkpoint as a local browser model (ONNX step + ngxson edge files)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import hf_hub_download

from fly_llm.model_io import HF_DIR, ROOT, encode_text, load_tokenizer
from fly_llm.onnx_step import FlyOnnxStep, connectome_rec
from fly_llm.train_loop import build_fresh_model

EDGE_REPO = "ngxson/fly-llm-onnx"
EDGE_FILES = ("edges_offsets.i32", "edges_src.u16", "edges_val.i16", "edges.json")
DEFAULT_CKPT = ROOT / "artifacts" / "fly-llm" / "final.pt"
DEFAULT_OUT = ROOT / "fly-llm-demo" / "public" / "model"


def _load_trained(checkpoint: Path, device: torch.device):
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    connectome = torch.load(ROOT / "data" / "connectome.pt", map_location="cpu", weights_only=False)
    model = build_fresh_model(connectome, device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def _check_parity(model, step: FlyOnnxStep, device: torch.device) -> None:
    tok = load_tokenizer()
    prompt = encode_text(tok, "Once upon a time, there was a")
    ids = torch.tensor([[1] + prompt], dtype=torch.long, device=device)
    out = model(ids, use_cache=True)
    cache = out.cache_params
    next_id = int(out.logits[0, -1].argmax())
    follow = torch.tensor([[next_id]], dtype=torch.long, device=device)
    ref = model(follow, cache_params=cache, use_cache=True)

    window = torch.cat([cache.last_tokens, follow], dim=1)[:, -step.k :]
    rec = connectome_rec(model, cache.state)
    logits, new_state = step(window, cache.state, rec)
    logit_err = (logits - ref.logits[:, -1]).abs().max().item()
    state_err = (new_state - ref.cache_params.state).abs().max().item()
    print(f"parity max |d logits|={logit_err:.6g}  |d state|={state_err:.6g}  next={next_id!r}")
    if logit_err > 2e-3 or state_err > 2e-3:
        raise SystemExit("ONNX step does not match PyTorch; refusing to export")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.checkpoint} on {device}", flush=True)
    model = _load_trained(args.checkpoint, device)
    step = FlyOnnxStep(model).to(device).eval()
    if not args.skip_parity:
        _check_parity(model, step, device)

    args.out.mkdir(parents=True, exist_ok=True)
    cpu_step = step.to("cpu").eval()
    n = model.config.n_neurons
    k = int(model.config.delay_k)
    dummy = (
        torch.zeros(1, k, dtype=torch.int32),
        torch.zeros(1, n, dtype=torch.float32),
        torch.zeros(1, n, dtype=torch.float32),
    )
    onnx_path = args.out / "fly_step.onnx"
    print(f"exporting {onnx_path}", flush=True)
    torch.onnx.export(
        cpu_step,
        dummy,
        str(onnx_path),
        input_names=["window_ids", "state", "rec"],
        output_names=["logits", "new_state"],
        opset_version=18,
        dynamo=False,
    )
    print(f"onnx {onnx_path.stat().st_size / 1e6:.1f} MB", flush=True)

    print("copying tokenizer and connectome edge files", flush=True)
    shutil.copy2(HF_DIR / "tokenizer.json", args.out / "tokenizer.json")
    shutil.copy2(HF_DIR / "tokenizer_config.json", args.out / "tokenizer_config.json")
    shutil.copy2(HF_DIR / "config.json", args.out / "config.json")
    for name in EDGE_FILES:
        src = hf_hub_download(EDGE_REPO, name)
        shutil.copy2(src, args.out / name)

    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in args.out.iterdir() if p.is_file()}
    (args.out / "manifest.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "files_mb": sizes}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(sizes, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
