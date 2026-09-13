"""Truncated BPTT windows used while training Fly-LLM."""
from __future__ import annotations

from collections.abc import Iterator, Sequence

CHUNK_SIZE = 32


def iter_bptt_chunks(ids: Sequence[int], chunk_size: int = CHUNK_SIZE) -> Iterator[list[int]]:
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least 2 so the causal shift has a target")
    tokens = [int(t) for t in ids]
    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start : start + chunk_size]
        if len(chunk) < 2:
            continue
        yield chunk
