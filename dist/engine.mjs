// Independent text inference implementation; same arithmetic and pooling as training.
export function decodeModel(manifest, buffer) {
  if (manifest.schema !== 1 || manifest.neurons !== 499 || manifest.labels.length !== 6) throw Error('Unsupported model schema.');
  const constructors = {'<u4': Uint32Array, '<u2': Uint16Array, '<f4': Float32Array, '<f8': Float64Array};
  const arrays = {};
  for (const [key, spec] of Object.entries(manifest.arrays)) {
    const Type = constructors[spec.dtype];
    if (!Type || spec.offset < 0 || spec.offset + spec.count * Type.BYTES_PER_ELEMENT > buffer.byteLength) throw Error('Invalid model array: ' + key);
    arrays[key] = new Type(buffer, spec.offset, spec.count);
  }
  const {indptr, indices, weights, coef, bias, ranked, token_drive} = arrays;
  if (indptr.length !== 500 || indptr[499] !== manifest.edges || indices.length !== weights.length ||
      coef.length !== 6 * 4 * 499 || bias.length !== 6 || ranked.length !== 499 || token_drive.length % 499) throw Error('Incomplete model.');
  if (indices.some(v => v >= 499) || weights.some(v => !Number.isFinite(v))) throw Error('Invalid connectivity.');
  return {manifest, ...arrays};
}

export function tokenize(text) {
  return text.toLowerCase().replaceAll('\u2019', "'").match(/[a-z]+(?:'[a-z]+)?/g) || [];
}

export function simulate(model, text, silenced = 0) {
  if (typeof text !== 'string' || !text.trim() || text.length > 4000) throw Error('Enter 1–4,000 characters of English text.');
  if (!Number.isInteger(silenced) || silenced < 0 || silenced > 200) throw Error('Silenced count must be 0–200.');
  const allTokens = tokenize(text);
  if (!allTokens.length) throw Error('Enter at least one word.');
  const {manifest: m, indptr, indices, weights, token_drive: drive, coef, bias, ranked} = model;
  const n = m.neurons, tokens = allTokens.slice(0, m.max_tokens);
  const ids = tokens.map(token => Object.hasOwn(m.vocabulary, token) ? m.vocabulary[token] : 0);
  const known = ids.filter(Boolean).length;
  while (ids.length < m.pools) ids.push(ids.at(-1));
  const frames = ids.length, mask = new Uint8Array(n).fill(1);
  for (let i = 0; i < silenced; i++) mask[ranked[i]] = 0;
  let previous = new Float32Array(n), next = new Float32Array(n);
  const states = new Float32Array(frames * n), activity = new Float32Array(frames), f = Math.fround;
  for (let t = 0; t < frames; t++) {
    for (let post = 0; post < n; post++) {
      if (!mask[post]) { next[post] = 0; continue; }
      let recurrent = 0;
      for (let edge = indptr[post]; edge < indptr[post + 1]; edge++) recurrent = f(recurrent + f(weights[edge] * previous[indices[edge]]));
      const activated = f(Math.tanh(f(recurrent + drive[ids[t] * n + post])));
      next[post] = f(f((1 - m.leak) * previous[post]) + f(m.leak * activated));
      activity[t] += Math.abs(next[post]) / n;
    }
    states.set(next, t * n);
    [previous, next] = [next, previous];
  }
  const pooled = new Float32Array(m.pools * n);
  for (let part = 0; part < m.pools; part++) {
    const start = Math.floor(part * frames / m.pools), end = Math.floor((part + 1) * frames / m.pools);
    for (let t = start; t < end; t++) for (let i = 0; i < n; i++) pooled[part * n + i] += states[t * n + i];
    for (let i = 0; i < n; i++) pooled[part * n + i] /= end - start;
  }
  const scores = new Float64Array(bias);
  for (let c = 0; c < m.labels.length; c++) for (let i = 0; i < pooled.length; i++) scores[c] += coef[c * pooled.length + i] * pooled[i];
  if (scores.some(v => !Number.isFinite(v))) throw Error('Non-finite model output.');
  return {scores, states, activity, frames, tokens, known, silenced, mask, truncated: allTokens.length > m.max_tokens};
}
