<script>
  import { onMount } from 'svelte';
  import { BrainView } from './brain.js';
  let { assets, onReady, colorMode = 'activation', autoRotate = true } = $props();
  let canvas;
  let view = $state.raw(null);
  onMount(() => {
    view = new BrainView(canvas, assets);
    const ro = new ResizeObserver(() => view.resize());
    ro.observe(canvas);
    onReady(view);
    return () => { ro.disconnect(); view.dispose(); };
  });
  $effect(() => { view?.setColorMode(colorMode); });
  $effect(() => { view?.setAutoRotate(autoRotate); });
</script>

<canvas bind:this={canvas}></canvas>

<style>
  canvas { width: 100%; height: 100%; display: block; min-height: 0; }
</style>
