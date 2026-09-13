// Runs the model off the main thread. The 9M-edge connectome matvec is a plain CSR loop here (fast in JS,
// slow as generic ONNX gather/scatter on wasm); ONNX does the input projection, nonlinearity and readout.
import * as ort from 'onnxruntime-web/all';
import { ORT_WASM_CDN } from './config.js';

let session = null;
let N = 0, K = 0;
let offsets, src, val, scale;
let rec;

function spmv(state) {
  // rec[j] = scale * sum_e val[e] * state[src[e]] for e in [offsets[j], offsets[j+1])
  for (let j = 0; j < N; j++) {
    let acc = 0;
    const end = offsets[j + 1];
    for (let e = offsets[j]; e < end; e++) acc += val[e] * state[src[e]];
    rec[j] = acc * scale;
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      ort.env.wasm.wasmPaths = ORT_WASM_CDN;
      N = m.N; K = m.K;
      offsets = new Int32Array(m.files.offsets);
      src = new Uint16Array(m.files.src);
      val = new Int16Array(m.files.val);
      scale = JSON.parse(new TextDecoder().decode(m.files.edges)).val_scale;
      rec = new Float32Array(N);
      const tries = [];
      if (m.preferWebGPU && self.navigator?.gpu) tries.push('webgpu');
      tries.push('wasm');
      let lastErr = null;
      for (const ep of tries) {
        try {
          self.postMessage({ type: 'status', text: `creating ${ep} session` });
          const t0 = performance.now();
          session = await ort.InferenceSession.create(m.files.onnx, { executionProviders: [ep], graphOptimizationLevel: 'basic' });
          self.postMessage({ type: 'ready', backend: ep + (ep === 'wasm' ? ` x${ort.env.wasm.numThreads || 1}` : ''), initMs: performance.now() - t0 });
          return;
        } catch (err) {
          console.warn(`execution provider ${ep} failed`, err);
          lastErr = err;
        }
      }
      throw lastErr;
    }
    if (m.type === 'step') {
      const t0 = performance.now();
      spmv(m.state);
      const t1 = performance.now();
      const window = m.window instanceof Int32Array ? m.window : Int32Array.from(m.window, Number);
      const out = await session.run({
        window_ids: new ort.Tensor('int32', window, [1, K]),
        state: new ort.Tensor('float32', m.state, [1, N]),
        rec: new ort.Tensor('float32', rec, [1, N]),
      });
      const logits = Float32Array.from(out.logits.data);
      const state = Float32Array.from(out.new_state.data);
      self.postMessage({ type: 'step', id: m.id, logits, state, ms: performance.now() - t0, msSpmv: t1 - t0 }, [logits.buffer, state.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err?.message || err) });
  }
};
