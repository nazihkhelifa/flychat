---
license: cc-by-4.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
datasets:
- roneneldan/TinyStories
tags:
- connectome
- reservoir-computing
- echo-state-network
- fruit-fly
- drosophila
- tinystories
- custom_code
---

# fly-hf: a fruit fly brain as a language model

**This is a toy model built for fun. It is NOT supported by llama.cpp and there is no GGUF for it: the recurrent layer is a 49k x 49k sparse matrix, which ggml has no operator for. It only runs through `transformers` with `trust_remote_code=True`.**

The transformer blocks of a GPT-style language model were replaced by the wiring diagram of a real fruit fly brain. The connectome is used as a fixed [echo state network](https://en.wikipedia.org/wiki/Echo_state_network) reservoir: the synaptic weights are never trained, only the token input projection, per-neuron gains, and the readout are. The model was then overfitted on 1000 TinyStories so that it produces bedtime-story-like text.

## What is inside

- **Connectome**: [MaleCNS v1.0](https://male-cns.janelia.org/) (FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research, CC BY 4.0), as packaged by the [Xenova/fruit-fly-simulation](https://huggingface.co/spaces/Xenova/fruit-fly-simulation) space. The central brain subset is used: superclasses `cb_sensory`, `visual_projection`, `cb_intrinsic`, `ascending_neuron`, `descending_neuron`. That is 49,393 neurons and 9,050,172 directed synaptic edges, stored as a CSR matrix in `model.safetensors` (frozen buffers `w_offsets`, `w_indices`, `w_values`).
- **Weights**: signed synapse counts. Sign comes from the presynaptic neuron's predicted neurotransmitter: acetylcholine +1, GABA and glutamate -1, others 0 (same convention as Shiu et al. 2024). The matrix is globally scaled to spectral radius 0.99 and then each neuron has a learnable recurrent gain, initialised so that the sum of absolute incoming weights is 5.
- **Input**: tokens enter through the 14,069 sensory-facing neurons (`cb_sensory` + `visual_projection`). These are split into 8 groups forming a delay line: group j receives the embedding of token t-j. There is no positional embedding, order comes from the recurrence.
- **Dynamics**: rate-based leaky tanh reservoir, `x_t = (1 - a) x_{t-1} + a * tanh(gain * (rec_gain * W x_{t-1} + P u_t) + bias)` with `a = 0.9`.
- **Readout**: LayerNorm + linear layer over all 49,393 neuron states to a 1024-token byte-level BPE vocabulary.
- **Trainable parameters**: 52.8M, of which 50.6M are the readout. The connectome itself (73 MB of buffers) is frozen.

The model is a stateful recurrent LM, integrated with `transformers` the same way Mamba is: a `cache_params` object carries the neuron state and the last 8 token ids between `generate()` steps.

## Training

- Data: 1000 stories from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), 100 held out. Stories were kept if they fit in 320 tokens. Byte-level BPE tokenizer with vocab 1024 trained on the same corpus.
- Objective: next-token cross entropy, truncated backprop through time in 32-token chunks (full BPTT over 300 steps of a strongly recurrent reservoir explodes).
- Optimiser: AdamW, lr 1e-3 for embedding / input projection / gains, 1e-4 for the readout, cosine schedule, 30 epochs, then 14 more epochs at lr 3e-4 / 3e-5.
- Result: train loss 0.86, validation loss 3.99. It is overfitted on purpose.

A control with the same graph but randomly permuted wiring (same in-degrees, same weight values) reached train loss 0.77 and validation loss 4.27 after 30 epochs, versus 1.02 / 3.84 for the real connectome at the same point. The real wiring fits the training set slower but generalises a bit better, which matches what Costi et al. 2025 reported for time-series prediction with this connectome.

A variant that only read out from the 1,314 descending neurons (the "motor output" of the brain) could not memorise the stories (train loss 4.7 after 12 epochs), so the full-state readout was used.

## Samples

Greedy decoding, prompt in bold:

**Once upon a time, there was a** little girl named Lily. She loved to play with her toys and eating her pains. One day, she found a shiny painate that Lily wanted to paint the push and push. "Let's paintents!" She said, and continued to par

**One day, a little girl named Lily found a needle** to explore the world. She saw a little monster and made it loud. When the mill was walking, the stick was very happy and began to play. She wanted to get closer. So, she saw a mouse and the cold. It was so much fun!

**Tom and his dog** were friends were bossy. They wanted to play on the face. They have fun to make a big ball. They loved to play together. One day, they saw a growy at the model. He said, "Let's go to have some leaves. "Oh no!" Jimmy

Sampling (top-k 50, temperature 0.7):

**Once upon a time, there was a** light. He was so happy and never wanted to go to what. So the hurto, Benny and played together. His shope yummy and goed, but they were very happy to stay. One day, the tank was feeling to the statch. The friends hugged sun and shiny the

Local grammar is fine, long-range coherence is not. The reservoir's memory is a few tokens beyond the 8-token delay line.

## How to run it locally

Requirements: Python 3.10+, `torch` (CUDA recommended, CPU works but is slow), `transformers>=5.0`. The sparse matmul runs on whatever device the model is on.

```bash
pip install torch transformers
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "ngxson/fly-hf"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

prompt = "Once upon a time, there was a"
ids = tok(prompt, return_tensors="pt").input_ids
ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], dim=1).to(model.device)  # stories start with <s>

out = model.generate(ids, max_new_tokens=80, do_sample=True, top_k=50, temperature=0.7, pad_token_id=tok.pad_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
```

Notes:

- `trust_remote_code=True` is required: the architecture lives in `modeling_fly.py` and `configuration_fly.py` in this repo.
- Prepend the `<s>` token as above, the model was trained with it at the start of every story.
- Batched generation works with left padding (`tok.padding_side = "left"`).
- Beam search and assisted generation are not supported (stateful model). Greedy, sampling, top-k and top-p are fine.
- The state is carried in `outputs.cache_params`. You can feed it back with `model(input_ids, cache_params=...)` to continue a sequence without recomputing the prefix.
- Expect roughly 20 tokens/s on a consumer GPU and much less on CPU. Each token is one sparse matvec over 9M edges.

## Limitations

- Toy model. It writes broken English about Lily and Tom and nothing else.
- It has an 8-token hard context plus a short, leaky recurrent memory. It cannot follow a plot.
- The rate-based tanh neuron is a big simplification of the spiking leaky integrate-and-fire model used in the biology literature. No spiking, no delays, no neuromodulation.
- Only the central brain is used. The optic lobes (89k neurons) and the ventral nerve cord (20k neurons) were dropped.

## Credits

- Connectome: MaleCNS v1.0, FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research. CC BY 4.0.
- Data packaging and the LIF parameter set that inspired the sign convention: [Xenova/fruit-fly-simulation](https://huggingface.co/spaces/Xenova/fruit-fly-simulation).
- Whole-brain LIF model: Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing", Nature 2024.
- Connectome as reservoir: Costi, Hadjiivanov, Dold, Hale, Izzo, "The Drosophila Connectome as a Computational Reservoir for Time-Series Prediction", 2025.
- Training data: Eldan and Li, TinyStories, 2023.
- Training recipe scaffold: Karpathy's build-nanogpt.

The model weights (the trained embedding, gains and readout) are released under CC BY 4.0 to match the connectome licence. The modeling code in this repo is MIT.
