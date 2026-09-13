"""Train Fly-LLM from scratch on the prepared TinyStories subset. Connectome stays frozen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fly_llm.model_io import ROOT
from fly_llm.train_loop import PHASE1, PHASE2, build_fresh_model, run_training

DEFAULT_CONNECTOME = ROOT / "data" / "connectome.pt"
DEFAULT_DATA = ROOT / "data" / "tinystories_subset"
DEFAULT_OUT = ROOT / "artifacts" / "fly-llm"


def _load_jsonl_ids(path: Path) -> list[list[int]]:
    stories = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stories.append(json.loads(line)["ids"])
    return stories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectome", type=Path, default=DEFAULT_CONNECTOME)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs1", type=int, default=PHASE1["epochs"])
    parser.add_argument("--epochs2", type=int, default=PHASE2["epochs"])
    parser.add_argument("--limit-train", type=int, default=0, help="debug: use only N train stories")
    parser.add_argument("--limit-val", type=int, default=0, help="debug: use only N val stories")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}", flush=True)

    blob = torch.load(args.connectome, map_location="cpu", weights_only=False)
    train_ids = _load_jsonl_ids(args.data / "train.jsonl")
    val_ids = _load_jsonl_ids(args.data / "val.jsonl")
    if args.limit_train:
        train_ids = train_ids[: args.limit_train]
    if args.limit_val:
        val_ids = val_ids[: args.limit_val]
    print(f"{len(train_ids)} train stories, {len(val_ids)} val stories", flush=True)

    model = build_fresh_model(blob, device)
    run_training(
        model,
        train_ids,
        val_ids,
        device,
        args.out,
        phase1={"epochs": args.epochs1, "fast": PHASE1["fast"], "slow": PHASE1["slow"]},
        phase2={"epochs": args.epochs2, "fast": PHASE2["fast"], "slow": PHASE2["slow"]},
    )


if __name__ == "__main__":
    main()
