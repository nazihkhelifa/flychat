from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_fly import FlyConfig


class FlyCache:
    """Reservoir state carried between generate() steps: neuron activations and the last k token ids."""

    is_compileable = False

    def __init__(self, state, last_tokens, seq_len=0):
        self.state = state  # [B, N]
        self.last_tokens = last_tokens  # [B, k], most recent last
        self.seq_len = seq_len

    def get_seq_length(self, layer_idx=0):
        return self.seq_len

    def reorder_cache(self, beam_idx):
        self.state = self.state.index_select(0, beam_idx.to(self.state.device))
        self.last_tokens = self.last_tokens.index_select(0, beam_idx.to(self.last_tokens.device))


@dataclass
class FlyOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    cache_params: FlyCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class FlyCausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    cache_params: FlyCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


class _SpMM(torch.autograd.Function):
    """y = W @ x with W fixed sparse CSR; backward uses a precomputed W^T so no transpose per step."""

    @staticmethod
    def forward(ctx, x, W, WT):
        ctx.WT = WT
        return torch.sparse.mm(W, x)

    @staticmethod
    def backward(ctx, g):
        return torch.sparse.mm(ctx.WT, g), None, None


class FlyPreTrainedModel(PreTrainedModel):
    config: FlyConfig
    config_class = FlyConfig
    base_model_prefix = "brain"
    supports_gradient_checkpointing = False
    _is_stateful = True
    _no_split_modules = []

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # the model builds its own FlyCache in forward, generate() must not create a KV cache
        return False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=min(0.02, module.in_features**-0.5))  # readout can have 50k inputs
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=1.0)  # reservoir wants O(1) drive, not transformer-scale
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, FlyModel):
            nn.init.normal_(module.in_proj, std=module.config.input_scale / module.config.d_embed**0.5)
            nn.init.ones_(module.gain)
            nn.init.ones_(module.rec_gain)
            nn.init.zeros_(module.bias)


class FlyModel(FlyPreTrainedModel):
    def __init__(self, config: FlyConfig):
        super().__init__(config)
        N, E, k, d = config.n_neurons, config.n_edges, config.delay_k, config.d_embed
        self.n_in_slot = config.n_in // k
        self.wte = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_token_id)
        # slot j projects the embedding of token t-j into input neuron group j
        self.in_proj = nn.Parameter(torch.empty(k, d, self.n_in_slot))
        self.gain = nn.Parameter(torch.ones(N))
        self.rec_gain = nn.Parameter(torch.ones(N))  # per-neuron scale of connectome input (neuromodulation knob)
        self.bias = nn.Parameter(torch.zeros(N))
        # connectome, incoming CSR: row = target neuron, col = source neuron, value = signed scaled synapse count
        self.register_buffer("w_offsets", torch.zeros(N + 1, dtype=torch.int32))
        self.register_buffer("w_indices", torch.zeros(E, dtype=torch.int32))
        self.register_buffer("w_values", torch.zeros(E, dtype=torch.float32))
        self.register_buffer("in_index", torch.zeros(k * self.n_in_slot, dtype=torch.long))
        self.register_buffer("out_index", torch.zeros(config.n_out, dtype=torch.long))
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def connectome(self):
        N = self.config.n_neurons
        W = torch.sparse_csr_tensor(self.w_offsets, self.w_indices, self.w_values, size=(N, N))
        key = (self.w_values.data_ptr(), self.w_values._version, self.w_values.device)
        if getattr(self, "_wt_key", None) != key:
            self._wt = W.t().to_sparse_csr()
            self._wt_key = key
        return W, self._wt

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_params: FlyCache | None = None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        cfg = self.config
        use_cache = use_cache if use_cache is not None else cfg.use_cache
        return_dict = return_dict if return_dict is not None else getattr(cfg, "return_dict", True)
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        B, T, _ = inputs_embeds.shape
        dev, k, N = inputs_embeds.device, cfg.delay_k, cfg.n_neurons

        if cache_params is None:
            state = inputs_embeds.new_zeros(B, N)
            prev_ids = torch.full((B, k), cfg.pad_token_id, dtype=torch.long, device=dev)
            seq_len = 0
        else:
            state, prev_ids, seq_len = cache_params.state, cache_params.last_tokens, cache_params.seq_len

        # delay line: slot j sees the embedding of token t-j (pad embedding is zero)
        prev_emb = self.wte(prev_ids)  # [B, k, d]
        ext = torch.cat([prev_emb, inputs_embeds], dim=1)  # [B, k+T, d]
        drive = []
        for j in range(k):
            e_j = ext[:, k - j : k - j + T]  # embedding of token t-j for t in [0, T)
            drive.append(e_j @ self.in_proj[j])  # [B, T, n_in_slot]
        drive = torch.cat(drive, dim=-1)  # [B, T, k * n_in_slot]

        W, WT = self.connectome()
        a = cfg.leak
        # state kept as [N, B] so the sparse matmul output is contiguous
        state = state.t().contiguous()
        gain, bias, rec_gain = self.gain[:, None], self.bias[:, None], self.rec_gain[:, None]
        mask = attention_mask.to(state.dtype).t() if attention_mask is not None else None  # [T, B]
        drive = drive.permute(1, 2, 0).contiguous()  # [T, k * n_in_slot, B]
        outs = []
        for t in range(T):
            pre = rec_gain * _SpMM.apply(state, W, WT)  # recurrent input from the connectome
            pre = pre.index_add(0, self.in_index, drive[t])
            new = (1 - a) * state + a * torch.tanh(gain * pre + bias)
            if mask is not None:
                m = mask[t][None, :]
                new = m * new + (1 - m) * state
            state = new
            outs.append(state[self.out_index])
        hidden = torch.stack(outs, dim=0).permute(2, 0, 1)  # [B, T, n_out]
        state = state.t().contiguous()

        cache = None
        if use_cache:
            if input_ids is not None:
                ids = torch.cat([prev_ids, input_ids], dim=1)[:, -k:]
            else:
                ids = prev_ids
            cache = FlyCache(state, ids, seq_len + T)
        if not return_dict:
            return tuple(v for v in [hidden, cache, (hidden,) if output_hidden_states else None] if v is not None)
        return FlyOutput(last_hidden_state=hidden, cache_params=cache, hidden_states=(hidden,) if output_hidden_states else None)


class FlyForCausalLM(FlyPreTrainedModel, GenerationMixin):
    def __init__(self, config: FlyConfig):
        super().__init__(config)
        self.brain = FlyModel(config)
        self.ln = nn.LayerNorm(config.n_out)
        self.lm_head = nn.Linear(config.n_out, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.brain.wte

    def set_input_embeddings(self, value):
        self.brain.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(
        self, input_ids, inputs_embeds=None, use_cache=None, cache_params=None, attention_mask=None, is_first_iteration=False, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_params=cache_params,
            attention_mask=attention_mask,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if use_cache and not is_first_iteration:
            model_inputs["attention_mask"] = None
        return model_inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_params: FlyCache | None = None,
        labels=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep=0,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        out = self.brain(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        h = out.last_hidden_state
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(self.ln(h[:, sl]))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1).to(shift_logits.device)
            loss = F.cross_entropy(shift_logits.float(), shift_labels, ignore_index=-100)
        if not return_dict:
            return tuple(v for v in [loss, logits, out.cache_params, out.hidden_states] if v is not None)
        return FlyCausalLMOutput(loss=loss, logits=logits, cache_params=out.cache_params, hidden_states=out.hidden_states)
