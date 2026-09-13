"""Build the 1,000 / 100 TinyStories subset with the published Fly-LLM tokenizer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from fly_llm.data import N_TRAIN, N_VAL, SPLIT_SEED, story_token_ids
from fly_llm.model_io import ROOT, encode_text, load_tokenizer

DEFAULT_OUT = ROOT / "data" / "tinystories_subset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    print("loading roneneldan/TinyStories train split", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train").shuffle(seed=args.seed)
    print(f"scanning shuffled split for {args.n_train}+{args.n_val} stories that fit in 320 tokens", flush=True)

    def encode(text: str):
        return encode_text(tokenizer, text)

    need = args.n_train + args.n_val
    kept: list[list[int]] = []
    for row in ds:
        ids = story_token_ids(row["text"], encode)
        if ids is None:
            continue
        kept.append(ids)
        if len(kept) >= need:
            break
    if len(kept) < need:
        raise SystemExit(f"only found {len(kept)} fitting stories, need {need}")
    train_ids, val_ids = kept[: args.n_train], kept[args.n_train :]
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": "roneneldan/TinyStories",
        "split": "train",
        "seed": args.seed,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "max_tokens_including_bos": 320,
        "note": (
            "ngxson did not publish the exact 1,100 row ids. This subset uses the published "
            "tokenizer, the 320-token cap, and seed 42."
        ),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_split(args.out / "train.jsonl", train_ids)
    _write_split(args.out / "val.jsonl", val_ids)
    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out}")


def _write_split(path: Path, stories: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for ids in stories:
            handle.write(json.dumps({"ids": ids, "n_tokens": len(ids)}) + "\n")


if __name__ == "__main__":
    main()
