// Static visualisation assets (public/data) and the model download with consent + Cache API.
import { FILES, CACHE_NAME } from './config.js';

const base = import.meta.env.BASE_URL.replace(/\/?$/, '/');

async function bytes(path) {
  const r = await fetch(base + 'data/' + path);
  if (!r.ok) throw new Error(`failed to load ${path}: ${r.status}`);
  return await r.arrayBuffer();
}

export async function loadAssets() {
  const [meta, positions, superclass, sub2global, inSlot, hop, types] = await Promise.all([
    fetch(base + 'data/meta.json').then((r) => r.json()),
    bytes('positions.u16'),
    bytes('superclass.u8'),
    bytes('sub2global.i32'),
    bytes('in_slot.u8'),
    bytes('hop.u8'),
    fetch(base + 'data/types.json').then((r) => r.json()),
  ]);
  return {
    meta,
    positions: new Uint16Array(positions),
    superclass: new Uint8Array(superclass),
    sub2global: new Int32Array(sub2global),
    inSlot: new Uint8Array(inSlot),
    hop: new Uint8Array(hop),
    types,
  };
}

/** Returns {onnx, offsets, src, val, edges} ArrayBuffers from the Cache API, or null if any is missing. */
export async function cachedModel() {
  try {
    const cache = await caches.open(CACHE_NAME);
    const out = {};
    for (const [k, f] of Object.entries(FILES)) {
      const hit = await cache.match(f.url);
      if (!hit) return null;
      out[k] = await hit.arrayBuffer();
      if (out[k].byteLength < 16) { // corrupt / empty entry: drop the cache and download again
        await caches.delete(CACHE_NAME);
        return null;
      }
    }
    return out;
  } catch {
    return null;
  }
}

async function fetchWithProgress(url, onChunk) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`download failed: HTTP ${r.status} for ${url}`);
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onChunk(value.length);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf.buffer;
}

/** Downloads all model files, reports progress in [0,1], stores them in the Cache API in the background. */
export async function downloadModel(onProgress = () => {}) {
  const total = Object.values(FILES).reduce((s, f) => s + f.mb, 0) * 1e6;
  let got = 0;
  const out = {};
  await Promise.all(
    Object.entries(FILES).map(async ([k, f]) => {
      out[k] = await fetchWithProgress(f.url, (n) => { got += n; onProgress(Math.min(1, got / total), got); });
    }),
  );
  // snapshot into Blobs now: the ArrayBuffers get transferred to the worker right after we return
  const blobs = Object.fromEntries(Object.keys(FILES).map((k) => [k, new Blob([out[k].slice(0)])]));
  caches.open(CACHE_NAME)
    .then((cache) => Promise.all(Object.entries(FILES).map(([k, f]) => cache.put(f.url, new Response(blobs[k])))))
    .catch(() => { /* private mode or quota: run without caching */ });
  return out;
}

export async function clearModelCache() {
  try {
    await caches.delete(CACHE_NAME);
  } catch {}
}
