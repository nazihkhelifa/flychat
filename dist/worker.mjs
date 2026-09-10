import {decodeModel, simulate} from './engine.mjs';
let model;
self.onmessage = ({data}) => {
  try {
    if (data.kind === 'init') {
      model = decodeModel(data.manifest, data.buffer);
      self.postMessage({kind: 'ready'});
    } else if (data.kind === 'classify') {
      if (!model) throw Error('The model is still loading.');
      const result = simulate(model, data.text, data.silenced);
      self.postMessage({kind: 'result', id: data.id, result}, [result.scores.buffer, result.states.buffer, result.activity.buffer, result.mask.buffer]);
    }
  } catch (error) { self.postMessage({kind: 'error', id: data.id, message: error.message}); }
};
