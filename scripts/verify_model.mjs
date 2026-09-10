import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {decodeModel, simulate} from '../dist/engine.mjs';
const manifest = JSON.parse(fs.readFileSync('dist/model/manifest.json'));
const raw = fs.readFileSync('dist/model/weights.bin');
assert.equal(crypto.createHash('sha256').update(raw).digest('hex'), manifest.weights_sha256);
const model = decodeModel(manifest, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const fixtures = JSON.parse(fs.readFileSync('dist/model/verification.json'));
let maxError = 0;
for (const fixture of fixtures) {
  for (const count of [0, 50]) {
    const result = simulate(model, fixture.text, count);
    const expected = count ? fixture.silenced_50_scores : fixture.scores;
    const error = Math.max(...result.scores.map((v, i) => Math.abs(v - expected[i])));
    maxError = Math.max(maxError, error);
    assert.ok(error < 0.002, `Python/JS score mismatch ${error}`);
    if (!count) {
      const last = result.states.slice(-499);
      assert.ok(Math.max(...last.map((v,i) => Math.abs(v-fixture.last_state[i]))) < 1e-5);
      assert.ok(Math.max(...result.activity.map((v,i) => Math.abs(v-fixture.mean_abs_activity[i]))) < 1e-5);
    } else {
      for (let t = 0; t < result.frames; t++) for (let k = 0; k < count; k++) assert.equal(result.states[t * 499 + model.ranked[k]], 0);
    }
  }
}
assert.throws(() => simulate(model, ''));
assert.throws(() => simulate(model, 'Hello', -1));
assert.throws(() => simulate(model, 'Hello', 499));
assert.throws(() => simulate(model, 'x'.repeat(4001)));
assert.equal(simulate(model, 'zzzxqv').known, 0);
assert.equal(simulate(model, 'hello '.repeat(200)).frames, 96);
const a = simulate(model, 'I feel happy.'), b = simulate(model, 'I feel happy.');
assert.deepEqual(a.scores, b.scores);
console.log(JSON.stringify({fixtures: fixtures.length, intact_and_silenced: true, python_js_max_score_error: maxError, validation: 'passed'}, null, 2));
