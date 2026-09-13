---
title: Fly brain LLM
emoji: 🪰
colorFrom: yellow
colorTo: red
sdk: static
app_build_command: npm run build
app_file: dist/index.html
pinned: false
license: mit
models:
- ngxson/fly-llm-hf
- ngxson/fly-llm-onnx
short_description: Watch a real fruit fly connectome write bedtime stories
---

# Fly brain LLM

**This is a toy model built for fun.**

A language model whose transformer blocks were replaced by the wiring diagram of a real fruit fly brain (49,393 central-brain neurons and 9 million synapses from the MaleCNS connectome), running entirely in your browser with onnxruntime-web. The left panel shows the brain activity while it generates, the right panel is the text.

- Model card and training details: [ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf)
- ONNX + connectome edges used by this demo (about 97 MB, downloaded once after you accept, cached by the browser): [ngxson/fly-llm-onnx](https://huggingface.co/ngxson/fly-llm-onnx)
- Inspired by [Xenova/fruit-fly-simulation](https://huggingface.co/spaces/Xenova/fruit-fly-simulation), which also provided the packaged connectome data

How it works: each token is embedded and injected into 14,069 sensory neurons (8 delay slots, one per recent token). The frozen connectome mixes the 49k-neuron state once per token, `x = 0.1 x + 0.9 tanh(g (W x + input) + b)`, and a linear readout over all neurons gives the next-token logits. The brain animation is visual only: the real update is instant, the colour sweep is ordered by the number of synapses from the input neurons so you can see where the signal enters and where it goes. Grey points are neurons the model does not use (optic lobes, ventral nerve cord).

Runs with WebGPU when available, WebAssembly otherwise. The model is trained to overfit 1000 TinyStories and only knows how to ramble about Lily and Tom.

Connectome: MaleCNS v1.0, FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research (CC BY 4.0). App code MIT.

## Development

```bash
npm install
npm run dev     # http://localhost:5173
npm run build   # dist/
```
