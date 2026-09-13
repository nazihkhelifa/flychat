from transformers import PretrainedConfig


class FlyConfig(PretrainedConfig):
    """Connectome reservoir language model. The recurrent weights are the fruit fly connectome (fixed buffers)."""

    model_type = "fly"

    def __init__(
        self,
        vocab_size=1024,
        n_neurons=1,
        n_edges=0,
        n_in=1,
        n_out=1,
        d_embed=128,
        delay_k=8,
        leak=0.9,
        input_scale=1.0,
        connectome="MaleCNS v1.0 central brain (cb_sensory, visual_projection, cb_intrinsic, ascending_neuron, descending_neuron)",
        spectral_radius=0.99,
        rec_target=5.0,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_neurons = n_neurons
        self.n_edges = n_edges
        self.n_in = n_in
        self.n_out = n_out
        self.d_embed = d_embed
        self.delay_k = delay_k
        self.leak = leak
        self.input_scale = input_scale
        self.connectome = connectome
        self.spectral_radius = spectral_radius
        self.rec_target = rec_target  # init: per-neuron recurrent gain so sum |w_in| = rec_target
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def hidden_size(self):
        return self.n_out
