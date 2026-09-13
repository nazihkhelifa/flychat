<script>
  import { LOCAL_MODEL, ONNX_REPO, TOTAL_MB } from './config.js';
  let { onAccept, progress = null, error = null } = $props();
  let preferWebGPU = $state(LOCAL_MODEL ? false : !!navigator.gpu);
</script>

<div class="overlay">
  <div class="card">
    <div class="mark">F</div>
    <h2>Fly brain</h2>
    <p>
      A language model whose transformer is a real fruit-fly connectome — 49,393 neurons,
      9 million synapses — running in your browser.
    </p>
    {#if LOCAL_MODEL}
      <p class="muted">Loads your trained TinyStories checkpoint (~{TOTAL_MB.toFixed(0)} MB) plus the frozen connectome.</p>
    {:else}
      <p class="muted">
        Downloads ~{TOTAL_MB.toFixed(0)} MB once from
        <a href={'https://huggingface.co/' + ONNX_REPO} target="_blank" rel="noreferrer">{ONNX_REPO}</a>, then caches it.
      </p>
    {/if}
    {#if progress == null}
      {#if !LOCAL_MODEL}
        <label class="row muted">
          <input type="checkbox" bind:checked={preferWebGPU} disabled={!navigator.gpu} />
          Use WebGPU {navigator.gpu ? '' : '(unavailable)'}
        </label>
      {/if}
      <button class="cta" onclick={() => onAccept({ preferWebGPU })}>
        {LOCAL_MODEL ? 'Start chat' : 'Download and start'}
      </button>
    {:else}
      <div class="bar"><i style={'width:' + Math.round(progress.frac * 100) + '%'}></i></div>
      <p class="muted">{progress.label}</p>
    {/if}
    {#if error}<p class="err">{error}</p>{/if}
  </div>
</div>

<style>
  .overlay {
    position: fixed; inset: 0; z-index: 20;
    background: rgba(246, 247, 249, 0.82);
    backdrop-filter: blur(10px);
    display: grid; place-items: center; padding: 20px;
  }
  .card {
    width: min(440px, 100%);
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 28px 26px 24px;
    text-align: center;
    box-shadow: 0 18px 50px rgba(17, 24, 39, 0.08);
  }
  .mark {
    width: 44px; height: 44px; border-radius: 50%; margin: 0 auto 12px;
    display: grid; place-items: center; background: #17181b; color: #fff; font-weight: 700;
  }
  h2 { margin: 0 0 10px; font-size: 22px; font-weight: 600; letter-spacing: -0.03em; }
  p { margin: 0 0 12px; line-height: 1.55; }
  .cta {
    margin-top: 8px;
    width: 100%;
    background: #17181b; color: #fff;
    border-radius: 999px; padding: 12px 16px; font-weight: 600;
  }
  .cta:hover { background: #2a2b30; }
  .bar { margin: 16px 0 8px; }
  .err { color: var(--danger); }
</style>
