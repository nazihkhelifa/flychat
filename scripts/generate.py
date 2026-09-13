"""Generate from a trained Fly-LLM checkpoint (or the published Hub weights)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fly_llm.model_io import HF_DIR, ROOT, encode_text, load_tokenizer
from fly_llm.train_loop import build_fresh_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="", help="path to artifacts/fly-llm/final.pt")
    parser.add_argument("--hub", action="store_true", help="run ngxson/fly-llm-hf instead of a local checkpoint")
    parser.add_argument("--prompt", default="Once upon a time, there was a")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.hub:
        tok = AutoTokenizer.from_pretrained(str(HF_DIR), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("ngxson/fly-llm-hf", trust_remote_code=True)
        model = model.to(device).eval()
        ids = tok(args.prompt, return_tensors="pt").input_ids
        ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], dim=1).to(device)
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_k=args.top_k,
            temperature=args.temperature,
            pad_token_id=tok.pad_token_id,
        )
        print(tok.decode(out[0], skip_special_tokens=True))
        return

    if not args.checkpoint:
        raise SystemExit("pass --checkpoint artifacts/fly-llm/final.pt or --hub")

    tok = load_tokenizer()
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    connectome = torch.load(ROOT / "data" / "connectome.pt", map_location="cpu", weights_only=False)
    model = build_fresh_model(connectome, device)
    model.load_state_dict(blob["model"])
    model.eval()
    body = encode_text(tok, args.prompt)
    ids = torch.tensor([[1] + body], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_k=args.top_k,
        temperature=args.temperature,
        pad_token_id=0,
    )
    # decode without specials
    pieces = tok.decode(out[0].tolist(), skip_special_tokens=True)
    print(pieces)


if __name__ == "__main__":
    main()
