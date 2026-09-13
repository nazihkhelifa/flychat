"""Compare PyTorch greedy vs exported ONNX (and print token ids)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import torch

from fly_llm.model_io import ROOT, encode_text, load_tokenizer
from fly_llm.onnx_step import FlyOnnxStep, connectome_rec
from fly_llm.train_loop import build_fresh_model

PROMPT = "Once upon a time, there was a"
CKPT = ROOT / "artifacts" / "fly-llm" / "final.pt"
ONNX = ROOT / "fly-llm-demo" / "public" / "model" / "fly_step.onnx"


def main() -> None:
    tok = load_tokenizer()
    ids = [1] + encode_text(tok, PROMPT)
    print("prompt ids", ids)
    print("pieces", [tok.decode([i]) for i in ids])

    device = torch.device("cpu")
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    connectome = torch.load(ROOT / "data" / "connectome.pt", map_location="cpu", weights_only=False)
    model = build_fresh_model(connectome, device)
    model.load_state_dict(blob["model"])
    model.eval()
    step = FlyOnnxStep(model).eval()

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    print("onnx inputs", [(i.name, i.type, i.shape) for i in sess.get_inputs()])

    state = torch.zeros(1, model.config.n_neurons)
    window = torch.zeros(1, step.k, dtype=torch.long)
    state_o = np.zeros((1, model.config.n_neurons), dtype=np.float32)
    window_o = np.zeros((1, step.k), dtype=np.int64)

    def one_token(token_id: int):
        nonlocal state, window, state_o, window_o
        window = torch.cat([window[:, 1:], torch.tensor([[token_id]])], dim=1)
        rec = connectome_rec(model, state)
        with torch.no_grad():
            logits, new_state = step(window, state, rec)
        window_o = np.concatenate([window_o[:, 1:], np.array([[token_id]], dtype=np.int64)], axis=1)
        rec_o = rec.detach().numpy().astype(np.float32)
        out = sess.run(None, {"window_ids": window_o, "state": state_o, "rec": rec_o})
        logits_o, new_state_o = out[0], out[1]
        dlog = float(np.max(np.abs(logits.numpy() - logits_o)))
        dst = float(np.max(np.abs(new_state.numpy() - new_state_o)))
        pt_next = int(logits[0].argmax())
        on_next = int(np.argmax(logits_o[0]))
        print(
            f"feed {token_id:>4} window={window[0].tolist()}  "
            f"pt={pt_next} ({tok.decode([pt_next])!r})  "
            f"onnx={on_next} ({tok.decode([on_next])!r})  "
            f"|dlog|={dlog:.3g} |dstate|={dst:.3g}  mean|x|={new_state.abs().mean():.4f}"
        )
        state = new_state
        state_o = new_state_o

    with torch.no_grad():
        for t in ids:
            one_token(t)
        print("greedy 12:")
        for _ in range(12):
            nxt = int(torch.argmax(step(window, state, connectome_rec(model, state))[0][0]))
            one_token(nxt)


if __name__ == "__main__":
    main()
