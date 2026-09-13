"""Download ngxson/fly-llm-hf and save only the frozen connectome buffers (~73 MB)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from fly_llm.model_io import ROOT, extract_connectome_tensors, load_published_config, save_connectome

DEFAULT_REPO = "ngxson/fly-llm-hf"
DEFAULT_OUT = ROOT / "data" / "connectome.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    config = load_published_config()
    print(f"downloading {args.repo}/model.safetensors (about 284 MB, cached by huggingface_hub)", flush=True)
    weights = hf_hub_download(args.repo, "model.safetensors")
    state = load_file(weights)
    buffers = extract_connectome_tensors(state)
    meta = {
        "repo": args.repo,
        "n_neurons": config.n_neurons,
        "n_edges": config.n_edges,
        "n_in": config.n_in,
        "n_out": config.n_out,
        "tensors": {key: list(buffers[key].shape) for key in buffers},
    }
    save_connectome(args.out, buffers, meta)
    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
