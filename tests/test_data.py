"""Subset selection and story length filter for the Fly-LLM TinyStories recipe."""
from fly_llm.data import collect_fitting_stories, story_token_ids


def _encode(text: str) -> list[int]:
    return [3 + (ord(ch) % 50) for ch in text]


def test_story_token_ids_prepends_bos_and_keeps_short_stories():
    ids = story_token_ids("hi", _encode, bos_id=1, max_tokens=320)
    assert ids[0] == 1
    assert ids[1:] == _encode("hi")


def test_story_token_ids_rejects_empty_body():
    assert story_token_ids("", _encode, bos_id=1, max_tokens=320) is None


def test_story_token_ids_rejects_stories_longer_than_320_with_bos():
    body = "x" * 320
    assert story_token_ids(body, _encode, bos_id=1, max_tokens=320) is None
    assert story_token_ids("x" * 319, _encode, bos_id=1, max_tokens=320) is not None


def test_collect_fitting_stories_is_deterministic_and_sized():
    texts = [f"story {i:04d}" for i in range(80)]
    first = collect_fitting_stories(texts, _encode, n_train=10, n_val=3, seed=42)
    second = collect_fitting_stories(texts, _encode, n_train=10, n_val=3, seed=42)
    other = collect_fitting_stories(texts, _encode, n_train=10, n_val=3, seed=7)
    assert len(first[0]) == 10
    assert len(first[1]) == 3
    assert first == second
    assert first != other


def test_collect_fitting_stories_skips_stories_that_do_not_fit():
    texts = ["ok", "", "x" * 400, "also ok", "third"]
    train, val = collect_fitting_stories(texts, _encode, n_train=2, n_val=1, seed=0)
    kept = train + val
    assert len(kept) == 3
    for ids in kept:
        assert 2 <= len(ids) <= 320
        assert ids[0] == 1
