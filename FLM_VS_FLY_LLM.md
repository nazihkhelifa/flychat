# FLM vs Fly-LLM — idea comparison

This note compares **nftechie/flm** ([github.com/nftechie/flm](https://github.com/nftechie/flm), paper: [Flies Are All You Need](https://artificialscientific.com/papers/flies-are-all-you-need)) with **this repo’s Fly-LLM** (ngxson’s connectome reservoir LM). Both use MaleCNS v1.0. They are not the same idea.

FLM was cloned only to read the source, then removed. Nothing from that tree is vendored here.

---

## One sentence each

| | Idea |
| --- | --- |
| **Ours (Fly-LLM)** | The fly central brain **is** the language model: a frozen 49k-neuron reservoir writes next-token logits over a 1,024-token TinyStories vocabulary. |
| **Theirs (FLM)** | A **pretrained 1.2B chat model** already speaks; the full 166,700-node fly graph only supplies a small, bounded residual that nudges those logits. |

If you strip the fly out of ours, there is no model. If you strip the fly out of theirs, you still have Liquid AI LFM2.5-1.2B-Instruct.

---

## Side by side

| | **Fly-LLM (this repo)** | **FLM (nftechie)** |
| --- | --- | --- |
| Role of the fly | Recurrent core / the LM itself | Feature extractor for a residual adapter |
| Language backbone | None (1024 BPE, trained from scratch) | Frozen **LFM2.5-1.2B-Instruct** (~1.2B params) |
| What is trained | Embeddings, 8-slot sensory projections, per-neuron gains/bias, LayerNorm, linear readout | **278,528** adapter weights only (`128 → 128 → hidden`, bias-free) |
| Graph size | **49,393** central-brain neurons, **9,050,172** edges | **166,700** retained neurons, **25,582,938** edges |
| Graph source | Xenova/fruit-fly-simulation packaging (same subset ngxson uses) | Official MaleCNS v1.0 feather files (Janelia GCS), hashed |
| Dropped cells | Optic lobes, VNC, glia, motor/endocrine, etc. (shown grey in the 3D view) | Only empty superclass and `status == Glia` |
| Synapse weights | Signed (ACh +, GABA/Glu −), scaled to spectral radius **0.99**, then per-cell `rec_gain` | **Unsigned** contact counts, **row-normalized** (incoming sum = 1). No transmitter signs |
| Neuron update | `x ← 0.1 x + 0.9 tanh(g (W x + sensory) + b)` | `x ← tanh(W @ (0.6 x + 0.4 input))` |
| How tokens enter the graph | Learned embedding → **anatomical** sensory/visual-projection neurons (14,069 cells, 8 delay slots) | Frozen LM embedding → **random** seeded bins/signs onto all nodes (`B` is not anatomy) |
| How tokens leave the graph | Linear readout over **all** 49,393 activations → 1024 logits | Random bin-pool to 128-d features → adapter → hidden residual → `lm_head`; delta RMS capped at **0.25** |
| Task | Overfit TinyStories (Lily/Tom bedtime prose) | Chat: 64 SmolTalk conversations + 32 synthetic style turns |
| Output | Next TinyStories token; browser 3D brain | Chat replies; CLI `scripts/chat.py` |
| Controls | Frozen `W` unit tests | **Direct-input** adapter (no fly matvec), **shuffled** node labels, **no-edges** (exact zero residual) |
| Their own finding | — | Matched direct-input control **slightly beat** the fly graph. Anatomy is **not** shown to help. |

---

## Architecture, in pictures

### Fly-LLM (ours)

```
token id (vocab 1024)
        │
        ▼
 learned embedding (128-d)
        │
        ▼
 8-slot delay line → 14,069 sensory-facing fly neurons
        │
        ▼
 frozen MaleCNS central-brain mix (49,393 × 9.05M signed synapses)
        │
        ▼
 leaky tanh state
        │
        ▼
 LayerNorm + linear → 1024 logits → next token
```

The connectome **replaces** transformer blocks. `generate()` carries a 49k activation vector instead of a KV cache (same API shape as Mamba).

### FLM (theirs)

```
user chat tokens
        │
        ├──────────────────────────────┐
        ▼                              ▼
 frozen LFM2.5-1.2B  (hidden, KV)     frozen LM embedding
        │                              │
        │                              ▼
        │                     random B: embed → 166,700 nodes
        │                              │
        │                              ▼
        │                     x = tanh(W (0.6x + 0.4 drive))
        │                     full MaleCNS, unsigned, in-normalized
        │                              │
        │                              ▼
        │                     random P: pool → 128-d features
        │                              │
        │                              ▼
        │                     FlyAdapter (278,528 params)
        │                     bounded Δh  (RMS ≤ 0.25)
        │                              │
        ▼                              ▼
   baseline logits  +  residual  =  next-token scores
```

Language ability is the 1.2B model. The fly is a **side channel**. Disconnect the edges and the residual is exactly zero (bias-free adapter + RMS-normalized features). The backbone still talks.

---

## Graph: same species, different object

Both say “MaleCNS v1.0”. They do not use the same matrix.

**Ours** takes the Xenova browser packaging and keeps the **central brain** used as a 3D toy: `cb_intrinsic`, `visual_projection`, `cb_sensory`, `ascending_neuron`, `descending_neuron`. Edges are synapse counts with a **sign from predicted transmitter**, then a **global spectral-radius** scale so the linear recurrence stays below explosion.

**Theirs** streams the public Janelia feather files (`minconf-0.5`), keeps every annotated non-glia body with a superclass (**optic lobes and VNC included**), and stores `W[post, pre] = contacts / incoming_total`. They refuse to invent excitation/inhibition from anatomy. That is a scientific choice, not an omission: their README says the model does **not** infer transmitter signs, dopamine, or biological time.

So: we use a **smaller, signed, spectrally scaled** reservoir that we pretend is a tiny LM. They use a **larger, unsigned, row-stochastic** graph as a frozen nonlinear filter on someone else’s embeddings.

---

## What “the fly computes language” would require — and who claims it

Neither project claims a biological fly understands language. The honesty level differs.

| Claim | Fly-LLM | FLM |
| --- | --- | --- |
| Fly wiring **is** the sequence model | Yes (toy, overfit) | No |
| Fly wiring **improves** a real LM | Not tested | Tested; **direct-input control won slightly** |
| Random relabeling of neurons | Not a published control here | Explicit `shuffled` mode: same `W` topology, scrambled alignment to `B`/`P`. They call this an **interface** control, not a win for anatomy |
| Chat quality | Not a chatbot | Small development split (24 test conversations). Corpus may already be in LFM pretraining |

FLM’s paper and repo are careful: a stronger frozen LM plus a tiny readout of fly state is **not** evidence that Drosophila implements language, and **not** evidence that this topology beats an arbitrary graph of the same size.

Our demo is a different kind of toy: you can **watch** 49k cells update while a 1024-token model rambles about Lily. That is visualization plus reservoir LM, not a chat system.

---

## Training recipes

**Fly-LLM (reconstructed ngxson recipe, this repo)**

- Data: 1,000 / 100 TinyStories, seed 42, 320-token cap, published 1024 BPE
- Truncated BPTT 32, batch 1, two-group AdamW, 30 + 14 epochs
- Trainable: embedding, `in_proj`, `gain` / `rec_gain` / `bias`, `ln`, `lm_head` (readout is ~50M of ~51M trainable params; `W` stays frozen)
- Local run: train loss **2.71**, val **3.60** (ngxson reported 0.86 / 3.99 on an unpublished story split)

**FLM conversation recipe**

- Backbone frozen at revision `0f604ada…` of `LiquidAI/LFM2.5-1.2B-Instruct`
- Offline: run the 25.6M-edge graph (optional C kernel) to cache 128-d features
- Fit two adapters in parallel: fly (`intact`) and **direct-input** (embedding → 128-d, no `W @ x`)
- 3 epochs, AdamW 3e-4, batch 32, CE + 0.5 × KL back to the frozen LM (keeps the residual conservative)
- Checkpoint picked on 16 val conversations; 24 test conversations untouched
- Adapter init: output projection **zeros**, so training starts as “pure LFM”

---

## Dynamics, compared as equations

Ours (leaky rate unit, learned per-cell knobs, anatomical injection):

```
pre = rec_gain ⊙ (W x)  +  sensory_drive(token_{t} … token_{t-7})
x   ← (1 − 0.9) x + 0.9 tanh(gain ⊙ pre + bias)
logits = W_out LayerNorm(x)
```

Theirs (no leak in the usual sense; mix then one tanh; random injection):

```
drive = 0.6 x + 0.4 (B · embedding_t)
x     ← tanh(W drive)
f     = normalize(P x)          # 128-d
Δh    = 0.03 tanh(W2 gelu(W1 f))
logits = lm_head(h) + bound(lm_head(Δh))   # conversation profile
```

`B` and `P` are **seeded sparse random** maps (seed 7301). They are not sensory or descending neuron lists. The fly graph mixes a random code; a second random code reads it out.

---

## What you can run today

| | Fly-LLM | FLM |
| --- | --- | --- |
| In this workspace | Yes: train, generate, browser demo at http://127.0.0.1:4173/ | No (intentionally not vendored) |
| Browser / 3D brain | Yes (WASM ONNX + JS CSR) | No |
| Chat CLI | No (TinyStories toy only) | Yes, after local train (`scripts/chat.py`) |
| OS they document | Windows-friendly here | Python 3.12, **macOS or Linux** |
| Disk | Connectome ~73 MB + ~212 MB ONNX | Several GB (LFM weights + MaleCNS feathers), ≥16 GB RAM preferred |

---

## Bottom line

These are two different research toys that share a connectome name.

- **Use Fly-LLM** if the question is: *can a frozen fly wiring diagram be the recurrent core of a tiny next-token model, and can we watch it?* That is our method. Language is weak and domain-narrow on purpose.
- **Use FLM** if the question is: *can fly state add a small residual on top of a real instruct model, and does anatomy beat a matched control?* Their own control says **not really** — the fly is optional seasoning on LFM2.5, not the cook.

They should not be scored against each other on “which fly talks better.” One barely talks (TinyStories). The other talks because a 1.2B LM talks, with or without *Drosophila*.
