<script>
  import VisualCortex from './VisualCortex.svelte';
  let { backend, ms, model, hopCounts, colorMode = 'activation', assets = null, tokenText = null, visualState = null, previousVisualState = null, windowTokens = [], busy = false } = $props();
  const css = (c) => `rgb(${c.map((x) => Math.round(x * 255)).join(',')})`;
  const LEGEND = {
    activation: [
      { c: [0.8, 0.82, 0.86], l: 'quiet' },
      { c: [1, 0.72, 0.12], l: 'positive' },
      { c: [0.15, 0.75, 1], l: 'negative' },
    ],
    synapseChange: [
      { c: [1.0, 0.92, 0.55], l: 'this token' },
      { c: [0.85, 0.5, 0.22], l: 'recent' },
      { c: [0.5, 0.28, 0.12], l: 'earlier tokens' },
      { c: [0.8, 0.82, 0.86], l: 'quiet' },
    ],
  };
</script>

<div class="dbg">
  <div class="legend">
    {#each LEGEND[colorMode] || LEGEND.activation as item}
      <span class="chip"><i class="sw" style={'background:' + css(item.c)}></i>{item.l}</span>
    {/each}
  </div>

  <div class="h">What the eyes see</div>
  <VisualCortex {tokenText} {assets} {visualState} {previousVisualState} {windowTokens} {busy} />

  <div class="foot muted">{backend ?? '-'} · {ms.toFixed(0)} ms · {model?.N ?? 0} model / {assets?.meta?.n_neurons ?? 0} fly cells · {hopCounts}</div>
</div>

<style>
  .dbg { font-size: 12px; display: grid; gap: 5px; padding: 8px 14px 12px; border-top: 1px solid var(--line); background: var(--panel); }
  .legend { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 2px; }
  .chip { display: inline-flex; gap: 5px; align-items: center; color: var(--muted); }
  .h { font-weight: 600; color: var(--text); margin-top: 4px; }
  .sw { width: 8px; height: 8px; border-radius: 99px; flex-shrink: 0; display: inline-block; }
  .foot { line-height: 1.35; margin-top: 4px; }
</style>
