# Fly-LLM — complete technical reference

This document describes **ngxson’s fruit-fly connectome language model**: how it works, how next-token generation is done, how it was trained, how to run the published weights, and how you would train a copy. It is the selected implementation for this workspace.

It is **not** a 499-neuron emotion classifier. That older experiment was removed from this workspace.

**This is a toy.** It was overfitted on 1,000 TinyStories so it rambles about Lily and Tom. It is not a chatbot, not a living fly, not a whole-brain simulation, and not evidence that Drosophila implements language.

Upstream sources (authoritative):

| What | Where | Local copy |
| --- | --- | --- |
| Python model, tokenizer, architecture | [ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf) (card also calls it `ngxson/fly-hf`) | `fly-llm-hf/` |
| Browser demo (Svelte + onnxruntime-web) | [ngxson/fly-llm-demo](https://huggingface.co/spaces/ngxson/fly-llm-demo) | `fly-llm-demo/` |
| ONNX one-step graph + connectome edges | [ngxson/fly-llm-onnx](https://huggingface.co/ngxson/fly-llm-onnx) | downloaded by the demo (~97 MB) |

Reviewed revisions in this workspace:

- Demo: `41a1483c05aacad99e6ac50febbddf2264f8497a`
- Python model files: `65c677b3d566a2e9793d5f72999cdb441c6c0a9f`

The Hugging Face model repo **does not include a training script**. This workspace now has one, reconstructed from the model card:

- `scripts/prepare_connectome.py` — frozen `W` from `ngxson/fly-llm-hf`
- `scripts/prepare_tinystories.py` — 1,000 / 100 stories, seed 42, 320-token cap
- `scripts/train_fly_llm.py` — truncated BPTT training loop
- `scripts/generate.py` — sample from a checkpoint or the Hub

ngxson did not publish the exact TinyStories row ids. The split here uses the published tokenizer and **seed 42**. Anything not stated on the card is marked as reconstruction (AdamW betas, weight decay 0.1, batch size 1, cosine per phase).

---

## 1. One-sentence idea

**Replace GPT transformer blocks with the wiring diagram of a real fruit-fly central brain. Keep that wiring frozen. Train only the ports that inject tokens and the linear map that reads the next token from all neuron activations.**

That is reservoir computing (an echo-state network) wrapped in the Hugging Face causal-LM API so `generate()` works like Mamba: a recurrent state is carried between steps instead of a KV cache.

```
token id
   │
   ▼
embedding (128-d, learned)
   │
   ▼
8-slot delay line → 14,069 sensory-facing neurons
   │
   ▼
frozen connectome mix: 49,393 neurons, 9,050,172 synapses
   │
   ▼
leaky tanh update of the whole brain state
   │
   ▼
LayerNorm + linear readout → 1,024 logits
   │
   ▼
sample / argmax → next token id → feed back
```

---

## 2. What the fly brain actually is in this model

### 2.1 Source reconstruction

The graph is **MaleCNS v1.0**, the electron-microscopy reconstruction of an adult male *Drosophila melanogaster* central nervous system (FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research, CC BY 4.0).

ngxson did not download the raw EM volume. The connectome used here is the packaging from [Xenova/fruit-fly-simulation](https://huggingface.co/spaces/Xenova/fruit-fly-simulation): neuron positions, superclasses, and synapse counts already prepared for a browser simulation.

### 2.2 Which neurons are in the language model

The full packaged fly has **166,700** cells with positions in the demo (`meta.n_neurons`). The **language model uses 49,393** of them (`meta.n_model` / `config.n_neurons`): the **central-brain subset**.

Included superclasses and counts (from `fly-llm-demo/public/data/meta.json`):

| Superclass | Count | Role in this model |
| --- | ---: | --- |
| `cb_intrinsic` | 32,164 | Internal central-brain cells; most of the reservoir |
| `visual_projection` | 9,201 | Input-facing (with sensory); delay-line targets |
| `cb_sensory` | 4,868 | Input-facing; delay-line targets |
| `ascending_neuron` | 1,846 | Body → brain |
| `descending_neuron` | 1,314 | Brain → body (“motor output”) |
| **Total in model** | **49,393** | |

Input-facing total: **14,069** (`cb_sensory` + `visual_projection`). That is `config.n_in`.

Dropped on purpose (shown grey in the 3D view, never updated by the LM):

- Optic lobes (~89k, e.g. `ol_intrinsic`, `ol_sensory`)
- Ventral nerve cord (~20k)
- Motor, endocrine, enteric, and miscellaneous classes

So “fruit fly brain as an LLM” here means **central brain only**.

### 2.3 Synapses

- **9,050,172 directed edges** (`config.n_edges`)
- Stored as **incoming CSR**: row = postsynaptic (target) neuron, column = presynaptic (source), value = signed scaled synapse count
- Buffers in `model.safetensors`: `w_offsets` (int32, length N+1), `w_indices` (int32, length E), `w_values` (float32, length E)
- About **73 MB** of frozen buffers

**Sign convention** (same idea as Shiu et al. 2024, via Xenova’s packaged transmitter predictions):

| Presynaptic transmitter | Sign |
| --- | --- |
| Acetylcholine | +1 (excitatory) |
| GABA | −1 (inhibitory) |
| Glutamate | −1 (inhibitory, Drosophila convention used here) |
| Other / unknown | 0 |

Raw values are **synapse counts**, not learned weights. Then:

1. Globally scale the sparse matrix so its **spectral radius is 0.99**. That keeps the linear recurrence below the chaos/explosion line.
2. Give each neuron a learnable **recurrent gain** `rec_gain`, initialised so the sum of absolute incoming weights is **5** (`config.rec_target`). That is a per-cell neuromodulation knob, not a new synapse.

### 2.4 Neuron model (what it is not)

Each unit is a **rate-based leaky tanh**. It is **not**:

- a leaky integrate-and-fire spike
- a conductance-based compartment
- a model with axonal delays, plasticity, or real neuromodulation

The 3D colour sweep in the demo is **illustrative**. Hop distance from input neurons is used to stagger the colour change so it *looks* like a wave. The mathematical update for token \(t\) is instantaneous: one sparse matvec, then tanh.

---

## 3. Architecture, piece by piece

All of this lives in `fly-llm-hf/modeling_fly.py` and `fly-llm-hf/configuration_fly.py`.

### 3.1 Config (published)

```text
vocab_size        1024
n_neurons         49393
n_edges           9050172
n_in              14069
n_out             49393          # full-state readout
d_embed           128
delay_k           8
leak              0.9            # a in the update equation
input_scale       1.0
spectral_radius   0.99
rec_target        5.0
pad / bos / eos   0 / 1 / 2
```

### 3.2 Token embeddings

`wte = Embedding(1024, 128, padding_idx=0)`

- Byte-level BPE, vocab 1,024, trained on the same TinyStories subset.
- Special tokens: `<pad>=0`, `<s>=1`, `</s>=2`.
- Pad embedding is **hard-zero**, so empty delay slots inject nothing.
- Init: normal with std **1.0** (reservoir-scale drive, not transformer-scale 0.02).

There is **no learned positional embedding**. Sequence order is the delay line plus the recurrent state.

### 3.3 Delay line (how text enters the fly)

Sensory-facing cells are split into **8 groups**. In code:

```text
n_in_slot = n_in // delay_k = 14069 // 8 = 1758
k * n_in_slot = 14064
```

The demo copy says each slot drives **1,758** neurons. Five of the 14,069 input-facing ids are leftover from integer division; the learned projection is `in_proj` with shape **`(8, 128, 1758)`**.

At time \(t\):

| Slot \(j\) | Embedding used | Meaning |
| ---: | --- | --- |
| 0 | token \(t\) | newest |
| 1 | token \(t-1\) | |
| … | … | |
| 7 | token \(t-7\) | oldest in the hard window |

Missing past tokens are `<pad>` → zero embedding → zero drive.

Learned map per slot:

```text
drive_j = embedding(token_{t-j}) @ in_proj[j]     # [1758]
```

Those 8 vectors are concatenated to length 14,064 and **scattered onto the real sensory neuron indices** (`in_index` buffer) with `index_add`. Cells that are not in those slots get no direct token current; they only feel the connectome.

This is why the model can see a few tokens of order **even if you ignore recurrence**. Recurrence is extra fading memory beyond those 8 tokens.

### 3.4 Frozen connectome multiply

```text
rec = rec_gain * (W @ state)
```

`W` is sparse CSR, **not an `nn.Parameter`**. It is a buffer. The custom autograd function `_SpMM` computes `y = W @ x` and in backward does `W^T @ grad` into **x only**. Gradients for `W` and `W^T` are explicitly `None`.

**Setting `requires_grad` on `w_values` would not train synapses.** You would need a different backward (and almost certainly a different optimizer / memory plan). That change is not in the published model.

### 3.5 State update

For each token in the current chunk:

```text
pre  = rec_gain * (W @ x) + token_drive_scattered
x    = (1 - a) * x + a * tanh(gain * pre + bias)
```

with `a = leak = 0.9`, so:

```text
x_t = 0.1 * x_{t-1} + 0.9 * tanh(...)
```

`gain` and `bias` are length-49,393 learned vectors. Sequences start from **x = 0**. Padding can freeze state via `attention_mask` (keep previous x on pad positions).

After the update, the hidden vector for the readout is `x[out_index]`. In the published config `n_out = n_neurons` and `out_index` is the full set, so this is the entire central-brain state.

### 3.6 Next-token head

```text
logits = Linear(LayerNorm(x))     # 49393 → 1024, no bias on the linear
```

Training loss (standard causal LM):

```text
shift_logits = logits[:, :-1]
shift_labels = labels[:, 1:]
loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

That is **next-token prediction**. The fly graph never outputs words. It outputs a 49,393-vector of activations. The linear layer is a classifier over those activations.

### 3.7 Trainable parameter budget (approx.)

| Tensor | Shape | Parameters |
| --- | --- | ---: |
| `wte` | 1024 × 128 | 131,072 |
| `in_proj` | 8 × 128 × 1758 | 1,800,192 |
| `gain` | 49393 | 49,393 |
| `rec_gain` | 49393 | 49,393 |
| `bias` | 49393 | 49,393 |
| `ln` (weight+bias) | 2 × 49393 | 98,786 |
| `lm_head` | 49393 × 1024 | **50,578,432** |
| **Total trainable** | | **~52.8M** |

About **96% of learning capacity is the readout**. The connectome is a frozen nonlinear feature expander. That is why a biological graph can be “used for next token” without teaching each synapse English.

---

## 4. How next-token generation works

This is a **stateful RNN**, not a transformer.

### 4.1 State you must carry

`FlyCache` holds:

- `state`: `[batch, 49393]` neuron activations
- `last_tokens`: `[batch, 8]` most recent token ids (pad-filled at the start)
- `seq_len`: how many tokens have been consumed

Hugging Face `generate()` is told not to build a DynamicCache (`_supports_default_dynamic_cache` → false; `_is_stateful` → true). The model builds `FlyCache` itself, same pattern as Mamba.

### 4.2 Algorithm (Python / transformers)

1. Prepend `<s>` (id 1). Training always started stories that way.
2. Run the prompt through `forward` once. That walks the delay line and connectome over the prompt tokens. You get logits for every position and a filled cache.
3. Take the **last** logit vector (1,024 numbers).
4. Choose the next id:
   - greedy: `argmax`
   - sampling: softmax with temperature, optionally top-k / top-p
5. Feed **only that new id** on the next call, with `cache_params` from the previous step. Do **not** replay the whole prompt if the cache is valid.
6. Repeat until `max_new_tokens` or `</s>`.

Beam search and assisted generation are unsupported. Greedy, sampling, top-k, top-p are fine. Batched generation wants **left padding**.

Python (published):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "ngxson/fly-llm-hf"  # also referred to as ngxson/fly-hf on the card
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

prompt = "Once upon a time, there was a"
ids = tok(prompt, return_tensors="pt").input_ids
ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], dim=1).to(model.device)

out = model.generate(
    ids,
    max_new_tokens=80,
    do_sample=True,
    top_k=50,
    temperature=0.7,
    pad_token_id=tok.pad_token_id,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

`trust_remote_code=True` is required: the classes are in the repo, not in stock transformers. Expect ~**20 tokens/s** on a consumer GPU; CPU is much slower. Each token is one sparse matvec over 9M edges.

Local Python files are in `fly-llm-hf/` but **`model.safetensors` (~284 MB) is not cloned** in the sparse checkout. `from_pretrained("ngxson/fly-llm-hf")` downloads it.

### 4.3 Algorithm (browser demo)

The demo never runs PyTorch. It runs **one ONNX graph per token** plus a **JavaScript CSR loop** for `W @ x`.

Worker (`fly-llm-demo/src/lib/worker.js`):

1. `spmv(state)`: for each neuron j, sum `val[e] * state[src[e]]` over incoming edges, times `val_scale`.
2. ONNX session inputs: `window_ids` `[1,8]` int64, `state` `[1,49393]` float32, `rec` `[1,49393]` float32.
3. ONNX outputs: `logits` `[1,1024]`, `new_state` `[1,49393]`.

UI (`App.svelte`):

1. User types e.g. `Once upon`.
2. **Reset** brain to zeros if the text is not exactly the prefix already fed. Editing invalidates state.
3. Encode: `[bos] + tokenizer.encode(text)`.
4. **Feed** each prompt token through `model.step(id)` (updates window, state, logits; animates).
5. For generation: `sample(lastLogits)` → `step(new_id)` → decode `fed[1:]` back into the textarea.
6. Stop on `</s>`.

Sampling in JS:

- greedy: argmax
- sample: softmax at temperature T, then sample among the top-k ids

Animation speed (100 ms / 1 s / 3 s / off) only delays the colour interpolation. It does not change the math.

### 4.4 Why this can emit English-like tokens at all

After the prompt `Once upon a time, there was a`, the 49,393-vector sits in some region of state space. The readout was trained so that, on TinyStories, that kind of state maps to tokens like ` little`, ` girl`, ` named`, ` Lily`. The connectome’s job is to make those states **nonlinear functions of the last few tokens**. The classifier’s job is to turn the state into a vocabulary distribution.

If you scramble `W` (same weights, random targets, same in-degrees), the mixer is different; the same readout recipe still works, but validation is a bit worse (see §7). If you only read the 1,314 descending neurons, the mixer’s features are too few / too filtered: **train loss stuck at 4.7** after 12 epochs — it could not even memorize. Full-state readout was required.

---

## 5. Tokenizer

File: `fly-llm-hf/tokenizer.json`

- Hugging Face `tokenizers` **byte-level BPE**
- Vocab size **1024** (bytes + merges + 3 specials)
- `add_prefix_space`: pre-tokenizer false, post-processor true
- No Unicode normalizer
- Trained **on the same 1,000-story corpus**, not on a general web crawl

Consequence: the model can only emit this tiny vocabulary. It cannot learn a large English lexicon. That is acceptable for a TinyStories toy (short words, lots of repetition).

Always prepend `<s>` at inference. Do not add `</s>` to the prompt unless you want the story to try to end.

---

## 6. How to train this

ngxson did not ship `train.py` in `fly-llm-hf`. The card says the **scaffold is Karpathy’s build-nanogpt**. The scripts under `scripts/` and `fly_llm/` implement that recipe. **Hyperparameters in bold are published.** Batch size 1, AdamW betas `(0.9, 0.95)`, and weight decay `0.1` are reconstructions.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe scripts/prepare_connectome.py
.\.venv\Scripts\python.exe scripts/prepare_tinystories.py
.\.venv\Scripts\python.exe scripts/train_fly_llm.py
```

Checkpoints go to `artifacts/fly-llm/`. Use `--limit-train` / `--epochs1 1 --epochs2 0` for a smoke run.

### 6.1 Hardware and software

Published runtime for inference: Python 3.10+, PyTorch, `transformers>=5.0`, CUDA recommended.

Training reconstruction (not on the card, but required in practice):

- A GPU with enough RAM for:
  - sparse CSR `W` (73 MB) plus activations `[chunk, 49393]`
  - readout `49393 × 1024` (~200 MB fp32)
  - truncated BPTT graph over **32** steps
- Mixed precision is not described; the published weights are **float32**.
- CPU training is theoretically possible and would be extremely slow (9M sparse matvec × 32 × batch × steps).

Install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

### 6.2 Frozen connectome

This repo does **not** rebuild MaleCNS from Xenova. `scripts/prepare_connectome.py` downloads `ngxson/fly-llm-hf` and writes only the frozen CSR buffers to `data/connectome.pt`. Trainable tensors stay randomly initialized.

How those buffers were originally built (for reading, not required to train):

1. Take MaleCNS / Xenova packaged neurons: positions, superclass, directed synapse counts, predicted transmitters.
2. Keep neurons whose superclass is one of:
   `cb_sensory`, `visual_projection`, `cb_intrinsic`, `ascending_neuron`, `descending_neuron`.
3. Relabel them to **0 … 49392**. Record:
   - `in_index`: the 14,064 delay-line targets (sensory + visual_projection, split into 8 contiguous groups of 1,758)
   - `out_index`: either all 49,393 (published) or the 1,314 descending cells (failed variant)
4. Build incoming CSR from remaining edges that both start and end inside the subset. You should get **9,050,172** edges.
5. Sign each edge from the **presynaptic** transmitter (ACh +1, GABA/Glu −1, else 0).
6. Compute the spectral radius of this sparse signed matrix (power iteration). Scale all values so radius = **0.99**.
7. For each neuron j, set initial `rec_gain[j]` so `rec_gain[j] * sum_i |W_ji| = 5`. In the published `FlyModel._init_weights`, `rec_gain` is then filled with ones; the **card** says the init target is 5. If you reimplement, prefer the card: fold the per-row scale into `rec_gain` (or into `W`) so incoming L1 is 5, then let `rec_gain` remain trainable.
8. Save `w_offsets`, `w_indices`, `w_values` as non-trainable buffers. Do **not** register them as parameters.

### 6.3 Data: TinyStories subset

Published:

- Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023).
- **1,000** stories for training, **100** held out.
- Keep a story only if it **fits in 320 tokens** after the tokenizer exists (chicken-and-egg: train the tokenizer on candidate text first, then drop long stories, or iterate).
- Every training sequence starts with `<s>`.

Reconstruction for loading:

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train")
# take a deterministic 1100 stories, later split 1000 / 100
```

The card does not name the random seed or which 1,100 rows. A reproduction will not bit-match the published run unless that split is recovered.

### 6.4 Train the tokenizer

Published: byte-level BPE, vocab **1024**, on the same corpus.

Reconstruction with Hugging Face `tokenizers`:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProc
from tokenizers.decoders import ByteLevel as ByteLevelDec

tok = Tokenizer(BPE(unk_token=None))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
trainer = BpeTrainer(
    vocab_size=1024,
    special_tokens=["<pad>", "<s>", "</s>"],
)
tok.train_from_iterator(story_texts, trainer)
tok.post_processor = ByteLevelProc(trim_offsets=False)
tok.decoder = ByteLevelDec()
tok.enable_padding(pad_id=0, pad_token="<pad>")
tok.save("tokenizer.json")
```

Match published ids: pad=0, bos=1, eos=2. Then drop stories with `len(ids) > 320` (or `> 319` plus bos, depending how you count). The card says “fit in 320 tokens.”

### 6.5 Initialise FlyForCausalLM

```python
from configuration_fly import FlyConfig
from modeling_fly import FlyForCausalLM

config = FlyConfig(
    vocab_size=1024,
    n_neurons=49393,
    n_edges=9050172,
    n_in=14069,
    n_out=49393,
    d_embed=128,
    delay_k=8,
    leak=0.9,
    input_scale=1.0,
    spectral_radius=0.99,
    rec_target=5.0,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
model = FlyForCausalLM(config)
# copy CSR buffers + in_index + out_index into model.brain.*
```

Weight init from code (`_init_weights`):

- Linear (readout): N(0, min(0.02, in_features^{-0.5})), bias 0
- Embedding: N(0, 1.0), pad row 0
- `in_proj`: N(0, input_scale / sqrt(d_embed))
- `gain` = 1, `rec_gain` = 1, `bias` = 0

Confirm after load: `w_values.requires_grad == False`.

### 6.6 Objective and truncated BPTT

**Problem:** a story can be ~300 tokens. Backpropagating through 300 steps of a 49k recurrent reservoir explodes (unstable gradients, huge compute).

**Published solution:** truncated BPTT in **32-token chunks**.

Reconstruction of one training step on a story of length T (including bos):

1. `state = 0`, `last_tokens = [pad]*8`.
2. Split token ids into chunks of 32. For each chunk:
   - `labels = chunk` (or aligned next-token labels; the model’s `forward` already shifts by one)
   - `out = model(input_ids=chunk, cache_params=cache, labels=labels)`
   - `loss.backward()`
   - optimizer step
   - **detach** `cache.state` (and do not backprop into the previous chunk)
   - keep `cache.last_tokens` for the delay line
3. That trains the model to predict token \(t+1\) from the current 32-token window plus a **stop-gradient** reservoir state inherited from earlier chunks.

This is why long-range plot fails: gradients never travel 300 steps through W (and W has no gradient anyway). The readout and input projection only see 32-step traces plus a frozen leftover state.

Loss: next-token cross entropy, `ignore_index=-100` on pad.

### 6.7 Optimiser (published)

Two parameter groups, **AdamW**:

| Group | Tensors | Phase 1 lr | Phase 2 lr |
| --- | --- | ---: | ---: |
| Fast | embedding, `in_proj`, `gain`, `rec_gain`, `bias` | **1e-3** | **3e-4** |
| Slow | readout (`lm_head` + `ln`) | **1e-4** | **3e-5** |

- Schedule: **cosine** (per phase; card does not specify warmup).
- Phase 1: **30 epochs**
- Phase 2: **14 more epochs** at the lower lrs
- Batch size: **not published**. Reconstruction: start with batch 1–4 stories, or pack 32-token chunks. Memory is dominated by `[batch, 32, 49393]` and the backward of the sparse matvec.

Do **not** put `w_offsets` / `w_indices` / `w_values` in the optimizer.

### 6.8 Training loop sketch (reconstruction)

```python
# pseudo-code aligned with FlyForCausalLM.forward
for epoch in range(30):
    for story_ids in train_loader:          # ids include leading bos
        cache = None
        for chunk, labels in chunks_of(story_ids, 32):
            out = model(
                input_ids=chunk,
                labels=labels,
                cache_params=cache,
                use_cache=True,
            )
            out.loss.backward()
            opt.step(); opt.zero_grad()
            cache = out.cache_params
            cache.state = cache.state.detach()
    # cosine lr step; evaluate val loss on 100 stories, same chunking, no grad

# then 14 epochs with lrs * 0.3
```

Validation: same 100 held-out stories, teacher-forced loss (not generation). Published **val loss 3.99** vs **train 0.86** after the full run.

### 6.9 What “overfit on purpose” means

TinyStories stories share templates (“Once upon a time”, “little girl named Lily”). A 50M readout on 1,000 short stories can memorize local n-grams. Train loss 0.86 is low for a 1024-way softmax; val 3.99 means it did **not** learn a general story writer. Greedy decoding still sounds like TinyStories because that is what was memorized.

If you want a *better* LM, this recipe is the wrong scale of data and the wrong memory (8-token delay + leaky tanh). If you want a *faithful copy of ngxson’s toy*, keep the 1,000-story overfit.

### 6.10 Saving in Hugging Face format

```python
model.save_pretrained("fly-hf-out")
tok.save("fly-hf-out/tokenizer.json")
# plus tokenizer_config.json, config.json, configuration_fly.py, modeling_fly.py
```

`config.json` must keep `auto_map` pointing at `FlyConfig` and `FlyForCausalLM` so `trust_remote_code` works.

### 6.11 Optional: ONNX export (what the demo uses)

Published in `ngxson/fly-llm-onnx`, not required to train.

- **v1** `fly_step_q8.onnx` (~150 MB): whole step including connectome matvec as Gather + ScatterElements. Easier in Python.
- **v2** `fly_step_v2_dq8.onnx` (~60 MB) + `edges_*.i32/u16/i16` (~36 MB): graph **does not** include `W @ x`. The host computes CSR spmv. This is what `fly-llm-demo` uses, because generic ONNX scatter is slow in WASM.

I/O of one step:

| Name | Shape | Role |
| --- | --- | --- |
| `window_ids` | `[B, 8]` int64 | delay line, oldest first, current token last |
| `state` | `[B, 49393]` f32 | activations in |
| `rec` | `[B, 49393]` f32 | v2 only: `W @ state` |
| `logits` | `[B, 1024]` f32 | next-token scores |
| `new_state` | `[B, 49393]` f32 | activations out |

v1 greedy matched fp32 PyTorch on tested prompts; max logit deviation **0.07**. Readout is int8 per-column quantized with in-graph `DequantizeLinear`. Synapse counts in the edge files are int16.

---

## 7. Published numbers and controls

### 7.1 Main run

| Split | Loss |
| --- | ---: |
| Train (1,000 stories) | **0.86** |
| Validation (100 stories) | **3.99** |

### 7.2 Scrambled-wiring control (after 30 epochs)

Same in-degrees, same weight values, random targets, then presumably re-scaled.

| Wiring | Train | Val |
| --- | ---: | ---: |
| Real connectome (30 ep) | 1.02 | **3.84** |
| Scrambled | **0.77** | 4.27 |

Real wiring **fits slower, generalises a bit better**. ngxson notes this matches Costi et al. 2025 (connectome as reservoir for time-series: topology/weights help overfitting resilience). It is a small gap on a toy LM, not a claim of biological language.

### 7.3 Motor-only readout

Read only 1,314 `descending_neuron` cells: train loss **4.7 after 12 epochs** — failed to memorize. Full 49,393 readout was kept.

### 7.4 Greedy samples (from the card)

Prompt in bold:

- **Once upon a time, there was a** little girl named Lily. She loved to play with her toys and eating her pains. …
- **One day, a little girl named Lily found a needle** to explore the world. She saw a little monster and made it loud. …
- **Tom and his dog** were friends were bossy. They wanted to play on the face. …

Local grammar can look fine. Long-range coherence does not. Memory is “a few tokens beyond the 8-token delay line.”

---

## 8. How to run what is already trained

### 8.1 Browser demo (this repo)

```powershell
cd fly-llm-demo
npm ci
npm run dev -- --host 127.0.0.1 --port 4173 --strictPort
```

Open http://127.0.0.1:4173/

1. Click **Download and start** (~97 MB from `ngxson/fly-llm-onnx`, cached in the Cache API as `fly-llm-onnx-v3`).
2. WebGPU if available, else WASM.
3. Prompt defaults to `Once upon`. Use **Next token** / **Next 10 tokens**.
4. Editing the box **resets** the brain.

Windows note: upstream `package.json` had `vite build && rm -f dist/assets/*.wasm`. The local demo script is `vite build` only so it works on Windows.

Cross-origin isolation: `public/coi-serviceworker.min.js` is loaded so threaded WASM can use SharedArrayBuffer when needed.

### 8.2 Python inference

See §4.2. Requires downloading `model.safetensors` (~284 MB).

### 8.3 Files map

```text
fly-llm-demo/
  src/App.svelte          UI, prompt feed, generate loop
  src/lib/model.js        tokenizer, sampling, worker RPC
  src/lib/worker.js       JS CSR spmv + ONNX step
  src/lib/brain.js        three.js point cloud (visual only)
  src/lib/config.js       ONNX / HF URLs, 97 MB file list
  public/data/            anatomy, hop, slots, superclasses
fly-llm-hf/
  modeling_fly.py         architecture (train + infer)
  configuration_fly.py    FlyConfig
  config.json             published hyperparameters
  tokenizer.json          1024 BPE
  model.safetensors       not in sparse clone; 284 MB on the Hub
```

---

## 9. Implementation details worth not missing

### 9.1 Why `_SpMM` exists

`torch.sparse.mm` on CSR is used in forward. Backward needs `W^T @ grad_y`. Transposing a 9M-edge CSR every step is expensive, so `W^T` is cached (`connectome()` rebuilds when the data pointer, version, or device changes). Backward still does **not** return a gradient for W.

### 9.2 Layout of the recurrent loop

State is stored `[N, batch]` inside the token loop so the sparse result is contiguous. Drive is `[T, n_in_slots, batch]`. This is a performance layout, not a scientific one.

### 9.3 Delay line during `generate()`

`prepare_inputs_for_generation` clears `attention_mask` after the first iteration when cache is on, so later single-token steps are not treated as padded prefixes.

`last_tokens` is always the last **8 ids**, including generated ones. Slot 0 of the next step sees the token you just sampled.

### 9.4 Browser window vs Python cache

JS: `window` is a `BigInt64Array(8)` shifted left, new id in the last slot (oldest first).  
Python: `last_tokens` is the same idea, concatenated with `input_ids` and cropped to `[:, -k:]`.

Both start as all pad (0).

### 9.5 Colour modes in the demo

| Mode | Meaning |
| --- | --- |
| activation | orange +, blue − current x |
| delta | change this step |
| class | superclass colour |
| hop | synapses from input (0/1/2/3+) |

Grey points: cells **not in the 49,393**. They never receive `x`.

---

## 10. Limits (scientific and practical)

- Toy English about Lily/Tom only.
- Hard context **8 tokens**; recurrent memory short and leaky (`a=0.9` mostly overwrites).
- Rate tanh ≠ spiking LIF used in the biology papers.
- Central brain only; optic lobes and VNC dropped.
- No llama.cpp / GGUF: ggml has no operator for a 49k×49k sparse recurrent layer.
- Dataset licence and TinyStories domain: children’s synthetic stories, not chat.
- Connectome licence **CC BY 4.0**; modeling code **MIT**; trained embedding/gains/readout released CC BY 4.0 to match the connectome.
- Training code for this model is `scripts/train_fly_llm.py`. It does not train synapses.

---

## 11. Conceptual FAQ

**Did they train the fly synapses to write stories?**  
No. Synapses are frozen signed, scaled counts.

**Then where does next-token ability come from?**  
From a linear classifier on the 49,393 activations, plus learned input embeddings and sensory projections, trained with teacher forcing on TinyStories.

**Is the delay line “the context window”?**  
It is the *hard* context. The state `x` is extra fading memory. Empirically that extra is only a few tokens.

**Why not read motor neurons only?**  
That experiment failed to memorize. Language readout here is an engineering choice, not a claim that descending neurons are a speech nerve.

**Why 32-token BPTT?**  
Full backprop through a 300-step strongly recurrent 49k reservoir is unstable and costly. Truncation is why this overfits local patterns.

**Does the animation show real conduction delays?**  
No.

**Did this repo keep the old 499-cell emotion classifier?**  
No. Those files were removed. This workspace is Fly-LLM only.

---

## 12. Credits (as declared upstream)

- Connectome: MaleCNS v1.0, FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research. CC BY 4.0.
- Packaging / transmitter-inspired signs: Xenova/fruit-fly-simulation.
- Whole-brain LIF reference: Shiu et al., *Nature* 2024, “A Drosophila computational brain model reveals sensorimotor processing.”
- Connectome as reservoir: Costi, Hadjiivanov, Dold, Hale, Izzo, 2025, “The Drosophila Connectome as a Computational Reservoir for Time-Series Prediction.”
- Data: Eldan and Li, TinyStories, 2023.
- Training scaffold: Karpathy, build-nanogpt.
- App and modeling code: ngxson (Xuan-Son Nguyen), MIT as declared upstream.

---

## 13. If you only remember four facts

1. The fly graph is a **frozen mixer**, not a trained transformer.
2. Next token = **linear readout of all 49k activations**.
3. Tokens enter through an **8-step sensory delay line**.
4. Training is **truncated BPTT (32) on 1,000 TinyStories**, with synapses blocked from learning in autograd.
