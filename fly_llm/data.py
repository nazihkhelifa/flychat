"""TinyStories subset helpers matching ngxson's published Fly-LLM recipe.

ngxson did not publish the exact 1,100 row indices. This code uses the published
tokenizer, keeps stories that fit in 320 tokens including <s>, then draws 1,000
train + 100 val with a documented seed.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence

EncodeFn = Callable[[str], Sequence[int]]

BOS_ID = 1
MAX_STORY_TOKENS = 320
N_TRAIN = 1000
N_VAL = 100
SPLIT_SEED = 42


def story_token_ids(
    text: str,
    encode_fn: EncodeFn,
    bos_id: int = BOS_ID,
    max_tokens: int = MAX_STORY_TOKENS,
) -> list[int] | None:
    body = [int(t) for t in encode_fn(text)]
    if not body:
        return None
    ids = [bos_id] + body
    if len(ids) > max_tokens:
        return None
    return ids


def collect_fitting_stories(
    texts: Iterable[str],
    encode_fn: EncodeFn,
    n_train: int = N_TRAIN,
    n_val: int = N_VAL,
    seed: int = SPLIT_SEED,
    bos_id: int = BOS_ID,
    max_tokens: int = MAX_STORY_TOKENS,
) -> tuple[list[list[int]], list[list[int]]]:
    pool = list(texts)
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    kept: list[list[int]] = []
    need = n_train + n_val
    for i in order:
        ids = story_token_ids(pool[i], encode_fn, bos_id=bos_id, max_tokens=max_tokens)
        if ids is None:
            continue
        kept.append(ids)
        if len(kept) >= need:
            break
    if len(kept) < need:
        raise ValueError(f"only found {len(kept)} fitting stories, need {need}")
    return kept[:n_train], kept[n_train:need]
