"""Truncated BPTT chunking: 32-token windows, skip leftover single tokens."""
from fly_llm.chunks import iter_bptt_chunks


def test_iter_bptt_chunks_splits_into_non_overlapping_windows_of_32():
    ids = list(range(1, 97))  # 96 tokens -> three full chunks
    chunks = list(iter_bptt_chunks(ids, chunk_size=32))
    assert chunks == [list(range(1, 33)), list(range(33, 65)), list(range(65, 97))]


def test_iter_bptt_chunks_keeps_a_short_final_window_if_it_has_at_least_two_tokens():
    ids = list(range(10))  # 10 tokens, chunk 8 -> 8 + 2
    chunks = list(iter_bptt_chunks(ids, chunk_size=8))
    assert chunks == [list(range(8)), list(range(8, 10))]


def test_iter_bptt_chunks_drops_a_final_window_of_one_token():
    ids = list(range(9))  # 8 + 1
    chunks = list(iter_bptt_chunks(ids, chunk_size=8))
    assert chunks == [list(range(8))]
