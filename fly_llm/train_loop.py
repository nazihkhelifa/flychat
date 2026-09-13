"""Published Fly-LLM training loop: truncated BPTT, two-group AdamW, two cosine phases."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fly_llm.chunks import CHUNK_SIZE, iter_bptt_chunks
from fly_llm.model_io import copy_connectome, load_fly_classes, load_published_config, trainable_parameter_groups

PHASE1 = {"epochs": 30, "fast": 1e-3, "slow": 1e-4}
PHASE2 = {"epochs": 14, "fast": 3e-4, "slow": 3e-5}


def build_fresh_model(connectome: dict, device: torch.device):
    FlyConfig, FlyForCausalLM = load_fly_classes()
    config = load_published_config()
    model = FlyForCausalLM(config)
    copy_connectome(model, connectome)
    return model.to(device)


def make_optimizer(model, fast_lr: float, slow_lr: float) -> AdamW:
    groups = trainable_parameter_groups(model)
    return AdamW(
        [
            {"params": groups["fast"], "lr": fast_lr},
            {"params": groups["slow"], "lr": slow_lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )


def _detach_cache(cache):
    if cache is None:
        return None
    cache.state = cache.state.detach()
    return cache


@torch.no_grad()
def eval_loss(model, stories: list[list[int]], device: torch.device, chunk_size: int = CHUNK_SIZE) -> float:
    model.eval()
    total = 0.0
    count = 0
    for ids in stories:
        cache = None
        for chunk in iter_bptt_chunks(ids, chunk_size):
            batch = torch.tensor([chunk], dtype=torch.long, device=device)
            out = model(input_ids=batch, labels=batch, cache_params=cache, use_cache=True)
            total += float(out.loss)
            count += 1
            cache = _detach_cache(out.cache_params)
    model.train()
    return total / max(count, 1)


def train_epoch(model, optimizer, stories: list[list[int]], device: torch.device, chunk_size: int = CHUNK_SIZE) -> float:
    model.train()
    total = 0.0
    count = 0
    for ids in stories:
        cache = None
        for chunk in iter_bptt_chunks(ids, chunk_size):
            batch = torch.tensor([chunk], dtype=torch.long, device=device)
            out = model(input_ids=batch, labels=batch, cache_params=cache, use_cache=True)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(out.loss.detach())
            count += 1
            cache = _detach_cache(out.cache_params)
    return total / max(count, 1)


def save_checkpoint(model, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "extra": extra or {}}
    torch.save(payload, path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(extra or {}, indent=2), encoding="utf-8")


def run_training(
    model,
    train_ids: list[list[int]],
    val_ids: list[list[int]],
    device: torch.device,
    out_dir: Path,
    chunk_size: int = CHUNK_SIZE,
    phase1: dict | None = None,
    phase2: dict | None = None,
) -> None:
    phase1 = phase1 or PHASE1
    phase2 = phase2 or PHASE2
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    def run_phase(name: str, spec: dict) -> None:
        optimizer = make_optimizer(model, spec["fast"], spec["slow"])
        scheduler = CosineAnnealingLR(optimizer, T_max=max(spec["epochs"], 1))
        for epoch in range(1, spec["epochs"] + 1):
            t0 = time.time()
            train = train_epoch(model, optimizer, train_ids, device, chunk_size)
            val = eval_loss(model, val_ids, device, chunk_size)
            scheduler.step()
            row = {
                "phase": name,
                "epoch": epoch,
                "train_loss": train,
                "val_loss": val,
                "seconds": round(time.time() - t0, 1),
                "lr_fast": optimizer.param_groups[0]["lr"],
                "lr_slow": optimizer.param_groups[1]["lr"],
            }
            history.append(row)
            print(
                f"{name} {epoch}/{spec['epochs']}  train {train:.4f}  val {val:.4f}  "
                f"{row['seconds']}s  lr {row['lr_fast']:.2e}/{row['lr_slow']:.2e}",
                flush=True,
            )
            save_checkpoint(model, out_dir / f"{name}_epoch{epoch:02d}.pt", row)

    run_phase("phase1", phase1)
    run_phase("phase2", phase2)
    save_checkpoint(model, out_dir / "final.pt", {"history": history})
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
