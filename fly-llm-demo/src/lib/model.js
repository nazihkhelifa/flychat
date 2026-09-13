// The fly brain step function (onnxruntime-web in a Web Worker) plus tokenizer and sampling.
import { AutoTokenizer, env } from '@huggingface/transformers';
import { LOCAL_MODEL, TOKENIZER_ID } from './config.js';

env.allowLocalModels = LOCAL_MODEL;
env.useBrowserCache = !LOCAL_MODEL;

export class FlyModel {
  constructor(meta) {
    this.N = meta.n_model;
    this.k = meta.delay_k;
    this.V = 1024;
    this.worker = null;
    this.backend = null;
    this.tok = null;
    this.pending = new Map();
    this.nextId = 1;
    this.reset();
  }

  async loadTokenizer() {
    this.tok = await AutoTokenizer.from_pretrained(TOKENIZER_ID);
    this.bos = this.tok.model.tokens_to_ids.get('<s>') ?? 1;
    this.eos = this.tok.model.tokens_to_ids.get('</s>') ?? 2;
  }

  encode(text) {
    return this.tok.encode(text, { add_special_tokens: false });
  }

  decode(ids) {
    return this.tok.decode(ids, { skip_special_tokens: true });
  }

  init(files, { preferWebGPU = true, onStatus = () => {} } = {}) {
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'status') onStatus(m.text);
        else if (m.type === 'ready') { this.backend = m.backend; this.initMs = m.initMs; resolve(m.backend); }
        else if (m.type === 'step') { const p = this.pending.get(m.id); this.pending.delete(m.id); p?.resolve(m); }
        else if (m.type === 'error') {
          if (m.id != null) { const p = this.pending.get(m.id); this.pending.delete(m.id); p?.reject(new Error(m.message)); }
          else reject(new Error(m.message));
        }
      };
      this.worker.onerror = (e) => reject(new Error(e.message || 'worker error'));
      this.worker.postMessage({ type: 'init', files, preferWebGPU, N: this.N, K: this.k }, Object.values(files));
    });
  }

  reset() {
    this.state = new Float32Array(this.N);
    this.window = new BigInt64Array(this.k); // zeros = <pad>
    this.fed = [];
    this.lastLogits = null;
    this.lastMs = 0;
  }

  /** Feed one token. Returns previous state, new state and logits. The compute is instant; the animation is separate. */
  async step(tokenId) {
    this.window.copyWithin(0, 1);
    this.window[this.k - 1] = BigInt(tokenId);
    const prev = this.state;
    const id = this.nextId++;
    const res = await new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'step', id, window: this.window.slice(), state: Float32Array.from(prev) });
    });
    this.lastMs = res.ms;
    this.lastMsSpmv = res.msSpmv;
    this.state = res.state;
    this.lastLogits = res.logits;
    this.fed.push(tokenId);
    return { prev, next: this.state, logits: this.lastLogits, ms: this.lastMs };
  }

  /** Softmax with temperature, returns {probs, top: [{id, p}]} */
  distribution(logits, temperature = 1, topN = 10) {
    const V = logits.length;
    const t = Math.max(temperature, 1e-4);
    let mx = -Infinity;
    for (let i = 0; i < V; i++) mx = Math.max(mx, logits[i]);
    const probs = new Float32Array(V);
    let s = 0;
    for (let i = 0; i < V; i++) {
      probs[i] = Math.exp((logits[i] - mx) / t);
      s += probs[i];
    }
    for (let i = 0; i < V; i++) probs[i] /= s;
    const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, topN);
    return { probs, top: idx.map((id) => ({ id, p: probs[id] })) };
  }

  sample(logits, { mode = 'greedy', temperature = 0.8, topK = 50 } = {}) {
    if (mode === 'greedy') {
      let best = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
      return best;
    }
    const { probs } = this.distribution(logits, temperature, 1);
    const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, topK);
    let s = 0;
    for (const i of idx) s += probs[i];
    let r = Math.random() * s;
    for (const i of idx) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return idx[idx.length - 1];
  }
}
