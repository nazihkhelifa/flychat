// Input-neuron activity, grouped by anatomical side; never a retinal pixel map.
export function visualInputGroups(assets) {
  const groups = { R: [], L: [], unknown: [] };
  if (!assets) return groups;
  const visualClass = assets.meta.classes.indexOf('visual_projection');
  for (let i = 0; i < assets.sub2global.length; i++) {
    if (assets.inSlot[i] !== 0 || assets.superclass[assets.sub2global[i]] !== visualClass) continue;
    const side = assets.meta.side?.[i];
    groups[side === 'R' || side === 'L' ? side : 'unknown'].push(i);
  }
  return groups;
}

export function summarizeVisualInput(groups, state, previous, binCount = 48) {
  let peak = 0;
  for (const side of ['R', 'L']) {
    for (const i of groups[side]) peak = Math.max(peak, Math.abs(state?.[i] || 0));
  }
  const sides = ['R', 'L'].map(side => {
    const ids = groups[side];
    const bins = Array(Math.min(binCount, ids.length)).fill(0);
    let magnitude = 0, change = 0;
    ids.forEach((id, j) => {
      const value = state?.[id] || 0;
      magnitude += Math.abs(value);
      change += Math.abs(value - (previous?.[id] || 0));
      const bin = Math.floor(j * bins.length / ids.length);
      if (Math.abs(value) > Math.abs(bins[bin])) bins[bin] = value;
    });
    return { side, count: ids.length, bins, mean: ids.length ? magnitude / ids.length : 0, change: ids.length ? change / ids.length : 0 };
  });
  return { sides, peak, unknown: groups.unknown.length };
}

export function activityColor(value, peak) {
  if (!value || !peak) return '#dce2e8';
  const amount = Math.sqrt(Math.min(1, Math.abs(value) / peak));
  return value > 0 ? `hsl(38 98% ${90 - amount * 44}%)` : `hsl(199 95% ${92 - amount * 47}%)`;
}
