import test from 'node:test';
import assert from 'node:assert/strict';
import { visualInputGroups, summarizeVisualInput, activityColor } from './visual-input.js';

test('uses current visual-input neurons and anatomical side metadata', () => {
  const groups = visualInputGroups({
    meta: { classes: ['cb_sensory', 'visual_projection'], side: 'RL?RL' },
    sub2global: [3, 2, 1, 0, 4], superclass: [1, 1, 1, 1, 0], inSlot: [0, 0, 0, 1, 0],
  });
  assert.deepEqual(groups, { R: [0], L: [1], unknown: [2] });
});

test('keeps signed activity, common scaling and actual per-step changes', () => {
  const groups = { R: [0, 1], L: [2], unknown: [3] };
  const state = new Float32Array([2, -4, 1, 999]);
  const previous = new Float32Array([1, -2, 1, 999]);
  const result = summarizeVisualInput(groups, state, previous, 1);
  assert.equal(result.peak, 4);
  assert.deepEqual(result.sides[0], { side: 'R', count: 2, bins: [-4], mean: 3, change: 1.5 });
  assert.equal(result.sides[1].change, 0);
  assert.notEqual(activityColor(1, 4), activityColor(-1, 4));
  assert.deepEqual(summarizeVisualInput(groups, null, null).sides[0].bins, [0, 0]);
  assert.equal(summarizeVisualInput(visualInputGroups(null), null, null).peak, 0);
});
