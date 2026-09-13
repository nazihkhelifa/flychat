"""One Fly-LLM token step with rec = W @ state supplied from outside (browser CSR)."""
from __future__ import annotations

import torch
from torch import nn


class FlyOnnxStep(nn.Module):
    def __init__(self, causal_lm):
        super().__init__()
        brain = causal_lm.brain
        self.wte = brain.wte
        self.in_proj = brain.in_proj
        self.gain = brain.gain
        self.rec_gain = brain.rec_gain
        self.bias = brain.bias
        self.register_buffer("in_index", brain.in_index.clone())
        self.register_buffer("out_index", brain.out_index.clone())
        self.ln = causal_lm.ln
        self.lm_head = causal_lm.lm_head
        self.k = int(brain.config.delay_k)
        self.a = float(brain.config.leak)

    def forward(self, window_ids: torch.Tensor, state: torch.Tensor, rec: torch.Tensor):
        # int32 is required: onnxruntime-web WebGPU silently mishandles int64 gather indices.
        window_ids = window_ids.to(dtype=torch.long)
        k = self.k
        emb = self.wte(window_ids)
        slots = [emb[:, k - 1 - j] @ self.in_proj[j] for j in range(k)]
        drive = torch.cat(slots, dim=-1)
        pre = rec * self.rec_gain
        pre = pre.index_add(1, self.in_index, drive)
        new_state = (1.0 - self.a) * state + self.a * torch.tanh(self.gain * pre + self.bias)
        hidden = new_state.index_select(1, self.out_index)
        logits = self.lm_head(self.ln(hidden))
        return logits, new_state


def connectome_rec(model, state: torch.Tensor) -> torch.Tensor:
    W, _ = model.brain.connectome()
    return torch.sparse.mm(W, state.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()
