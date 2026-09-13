# Fly-LLM

A toy language model whose recurrent core is the **MaleCNS fruit-fly connectome** (49,393 central-brain neurons, 9 million frozen synapses). It is ngxson’s [fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf) architecture. You can run the published demo, or train the learned ports yourself on a TinyStories subset.

This is not a chatbot and not a living brain. The synapses do not learn. Only embeddings, sensory projections, neuron gains, and the readout are trained.

## Browser demo

```powershell
cd fly-llm-demo
npm ci
npm run dev -- --host 127.0.0.1 --port 4173 --strictPort
```

Open http://127.0.0.1:4173/, click **Download and start** (~97 MB ONNX, cached in the browser).

## Train it yourself

The Hugging Face repo did not ship `train.py`. This workspace reconstructs the published recipe: same tokenizer, stories that fit in 320 tokens, **1,000 train / 100 val**, truncated BPTT of 32, two-group AdamW, 30 + 14 epochs. ngxson did not publish the exact row indices, so the split uses **seed 42**.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe scripts/prepare_connectome.py
.\.venv\Scripts\python.exe scripts/prepare_tinystories.py
.\.venv\Scripts\python.exe scripts/train_fly_llm.py
```

`prepare_connectome.py` downloads the 284 MB Hub file once and writes `data/connectome.pt` (~73 MB of frozen `W`). Trainable weights start random. CUDA is strongly recommended; CPU will be very slow.

Smoke test with a few stories:

```powershell
.\.venv\Scripts\python.exe scripts/train_fly_llm.py --limit-train 2 --limit-val 1 --epochs1 1 --epochs2 0
```

Generate from a checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts/generate.py --checkpoint artifacts/fly-llm/final.pt --prompt "Once upon a time, there was a"
```

Or from the published weights: `--hub`.

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

How the model works, next-token generation, and the training recipe: [`FLY_LLM.md`](FLY_LLM.md).
