import pytest

torch = pytest.importorskip("torch", reason="torch is required for the tiny Fly-LLM freeze test")
pytest.importorskip("transformers", reason="transformers is required to construct FlyForCausalLM")

try:
    torch.zeros(1)
except OSError as exc:
    pytest.skip(f"torch is installed but cannot load: {exc}", allow_module_level=True)

from fly_llm.model_io import load_fly_classes, trainable_parameter_groups


def test_connectome_buffers_do_not_require_grad_on_a_tiny_model():
    FlyConfig, FlyForCausalLM = load_fly_classes()
    config = FlyConfig(
        vocab_size=8,
        n_neurons=4,
        n_edges=2,
        n_in=8,
        n_out=4,
        d_embed=4,
        delay_k=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = FlyForCausalLM(config)
    for name in ("w_offsets", "w_indices", "w_values", "in_index", "out_index"):
        buf = getattr(model.brain, name)
        assert buf.requires_grad is False, name


def test_trainable_parameter_groups_exclude_connectome_and_split_lrs():
    FlyConfig, FlyForCausalLM = load_fly_classes()
    config = FlyConfig(
        vocab_size=8,
        n_neurons=4,
        n_edges=2,
        n_in=8,
        n_out=4,
        d_embed=4,
        delay_k=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = FlyForCausalLM(config)
    groups = trainable_parameter_groups(model)
    named = {id(p): n for n, p in model.named_parameters()}
    fast_names = {named[id(p)] for p in groups["fast"]}
    slow_names = {named[id(p)] for p in groups["slow"]}
    all_names = fast_names | slow_names
    assert not any("w_values" in n or "w_offsets" in n or "w_indices" in n for n in all_names)
    assert any(n.endswith("wte.weight") or n.endswith("in_proj") or n.endswith("gain") for n in fast_names)
    assert any("lm_head" in n or n.endswith("ln.weight") for n in slow_names)
    fast_ids = {id(p) for p in groups["fast"]}
    slow_ids = {id(p) for p in groups["slow"]}
    assert fast_ids.isdisjoint(slow_ids)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in trainable} == fast_ids | slow_ids
