<script>
  import { onMount } from 'svelte';
  import { FlyHeadView } from './fly-head.js';
  import { visualInputGroups, summarizeVisualInput, activityColor } from './visual-input.js';
  let { tokenText = null, assets = null, visualState = null, previousVisualState = null, windowTokens = [], busy = false } = $props();
  let canvas;
  let headView = $state.raw(null);
  let overlay = $state(true);
  let renderError = $state('');
  onMount(() => {
    let view;
    try {
      view = new FlyHeadView(canvas);
      headView = view;
    } catch (error) {
      renderError = 'The 3D head requires WebGL. The live activity values remain available below.';
      console.error('3D head initialization failed', error);
      return;
    }
    const observer = new ResizeObserver(() => view.resize());
    observer.observe(canvas);
    return () => { observer.disconnect(); view.dispose(); };
  });
  let groups = $derived(visualInputGroups(assets));
  let activity = $derived(summarizeVisualInput(groups, visualState, previousVisualState));
  $effect(() => headView?.setActivity(activity, overlay));
  const number = value => value === 0 ? '0' : value.toPrecision(3);
</script>

<figure class="eyes">
  <div class="input-line">
    <span class:live={busy} class="status-dot"></span>
    <span>{busy ? 'Processing' : visualState ? 'Inspecting' : 'Waiting for text'}</span>
    {#if tokenText != null}<code class="current-token">{JSON.stringify(tokenText)}</code>{/if}
  </div>
  <div class="context" aria-label="Input context, oldest to newest">
    <span class="context-label">Context →</span>
    {#each windowTokens as token, i}
      <code class:newest={i === windowTokens.length - 1} title={`Context ${i + 1}: ${JSON.stringify(token)}`}>{token === '<pad>' ? '·' : token}</code>
    {/each}
  </div>
  <div class="anatomy">
    <canvas bind:this={canvas} aria-label="Interactive 3D fruit-fly head. Drag to rotate and scroll to zoom. Use the rotation and front-view buttons below."></canvas>
    {#if renderError}<p class="render-error">{renderError}</p>{/if}
    <span class="view-label">3D HEAD · Drag to rotate · Scroll to zoom</span>
  </div>
  <div class="head-tools">
    <div class="angles">
      <button onclick={() => headView?.turn(-0.35)} aria-label="Rotate head left">↶</button>
      <button onclick={() => headView?.front()}>Front view</button>
      <button onclick={() => headView?.turn(0.35)} aria-label="Rotate head right">↷</button>
    </div>
    <label><input type="checkbox" bind:checked={overlay} /> Activity on eyes</label>
  </div>
  <div class="activity-pair">
    {#each activity.sides as side}
      <section class="channel" aria-label={`${side.side === 'R' ? 'Right' : 'Left'} visual-input activity`}>
        <div class="channel-heading"><strong>{side.side === 'R' ? 'Right' : 'Left'} visual input</strong><span>{side.count.toLocaleString()} neurons</span></div>
        <div class="heatmap" role="img" aria-label={`${side.side === 'R' ? 'Right' : 'Left'} grouped neuron activations; mean magnitude ${number(side.mean)}, mean change ${number(side.change)}`}>
          {#each side.bins as value, i}
            <i style:background={activityColor(value, activity.peak)} title={`Group ${i + 1}: strongest activation ${number(value)}`}></i>
          {/each}
        </div>
        <div class="readings"><span>Mean |a| <b>{number(side.mean)}</b></span><span>Change <b>{number(side.change)}</b></span></div>
      </section>
    {/each}
  </div>
  <div class="scale"><span><i class="positive"></i>positive</span><span><i class="negative"></i>negative</span><span><i class="quiet"></i>zero</span><span>Shared peak ±{number(activity.peak)}</span></div>
  {#if activity.unknown}<div class="unassigned">{activity.unknown} inputs have unknown side and are excluded.</div>{/if}
  <figcaption>Head: anatomical 3D approximation. Eye colors show grouped model activity, not measured retinal locations. The model receives text. Right/left refer to the fly’s sides.</figcaption>
</figure>

<style>
  .eyes { margin: 3px 0; padding: 10px 12px; border: 1px solid var(--line, #e2e5ea); border-radius: 12px; background: #fff; color: #334155; }
  .input-line { display: flex; align-items: center; gap: 7px; min-height: 24px; font-size: 11px; }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: #94a3b8; flex-shrink: 0; }
  .status-dot.live { background: #10b981; box-shadow: 0 0 0 3px #d1fae5; }
  .current-token { margin-left: auto; max-width: 60%; overflow: hidden; text-overflow: ellipsis; white-space: pre; color: #0f172a; background: #fff0cd; padding: 3px 7px; border-radius: 5px; font-size: 12px; }
  .context { display: flex; gap: 4px; align-items: center; overflow-x: auto; padding: 5px 0; font-size: 10px; }
  .context-label { white-space: nowrap; color: #64748b; }
  .context code { background: #f1f5f9; border-radius: 3px; padding: 2px 5px; white-space: pre; }
  .context code.newest { background: #fff0cd; color: #854d0e; }
  .anatomy { position: relative; height: 220px; margin: 4px 0 0; border-radius: 8px; overflow: hidden; background: #f7f8fa; }
  .anatomy canvas { width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
  .anatomy canvas:active { cursor: grabbing; }
  .view-label { position: absolute; bottom: 6px; left: 9px; font-size: 9px; color: #64748b; pointer-events: none; }
  .head-tools { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 6px; margin: 6px 0 8px; font-size: 10px; }
  .angles { display: flex; gap: 4px; }
  .angles button { border: 1px solid #e2e8f0; border-radius: 5px; padding: 4px 8px; background: #f8fafc; }
  .angles button:hover { background: #e2e8f0; }
  .head-tools label { display: inline-flex; align-items: center; gap: 4px; }
  .render-error { position: absolute; inset: 30px 15px; color: #9f1239; }
  .activity-pair { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 10px; }
  .channel { padding: 7px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px; min-width: 0; }
  .channel-heading, .readings { display: flex; justify-content: space-between; gap: 4px; flex-wrap: wrap; font-size: 10px; }
  .channel-heading span, .readings { color: #64748b; }
  .heatmap { display: grid; grid-template-columns: repeat(24, 1fr); gap: 2px; margin: 6px 0; min-height: 18px; }
  .heatmap i { height: 7px; border-radius: 1px; }
  .readings b { color: #334155; font-variant-numeric: tabular-nums; }
  .scale { display: flex; gap: 9px; flex-wrap: wrap; font-size: 9px; color: #64748b; margin: 7px 0 4px; }
  .scale span { display: inline-flex; align-items: center; gap: 3px; }
  .scale i { width: 6px; height: 6px; border-radius: 2px; }
  .positive { background: #e99d03; } .negative { background: #08a2ed; } .quiet { background: #dce2e8; }
  figcaption, .unassigned { font-size: 10px; line-height: 1.4; color: #64748b; }
  @media (max-width: 400px) { .eyes { padding: 8px; } .anatomy { height: 190px; } .activity-pair { gap: 6px; } }
</style>
