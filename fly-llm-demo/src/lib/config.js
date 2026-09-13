export const ONNX_REPO = 'ngxson/fly-llm-onnx';
export const MODEL_REPO = 'ngxson/fly-llm-hf';
export const LOCAL_MODEL = true;
const base = import.meta.env.BASE_URL.replace(/\/?$/, '/');
const R = LOCAL_MODEL ? `${base}model/` : `https://huggingface.co/${ONNX_REPO}/resolve/main/`;
export const TOKENIZER_ID = LOCAL_MODEL ? `${base}model` : ONNX_REPO;
export const FILES = LOCAL_MODEL
  ? {
      onnx: { url: R + 'fly_step.onnx', mb: 212 },
      offsets: { url: R + 'edges_offsets.i32', mb: 0.2 },
      src: { url: R + 'edges_src.u16', mb: 18 },
      val: { url: R + 'edges_val.i16', mb: 18 },
      edges: { url: R + 'edges.json', mb: 0 },
    }
  : {
      onnx: { url: R + 'fly_step_v2_dq8.onnx', mb: 60 },
      offsets: { url: R + 'edges_offsets.i32', mb: 0.2 },
      src: { url: R + 'edges_src.u16', mb: 18 },
      val: { url: R + 'edges_val.i16', mb: 18 },
      edges: { url: R + 'edges.json', mb: 0 },
    };
export const TOTAL_MB = Object.values(FILES).reduce((s, f) => s + f.mb, 0);
export const CACHE_NAME = LOCAL_MODEL ? 'fly-llm-local-v2' : 'fly-llm-onnx-v3';
export const ORT_VERSION = '1.29.0';
export const ORT_WASM_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
