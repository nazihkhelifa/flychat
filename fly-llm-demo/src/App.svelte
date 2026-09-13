<script>
  import { onMount, tick } from 'svelte';
  import Consent from './lib/Consent.svelte';
  import Brain from './lib/Brain.svelte';
  import Chat from './lib/Chat.svelte';
  import Debug from './lib/Debug.svelte';
  import { loadAssets, cachedModel, downloadModel, clearModelCache } from './lib/data.js';
  import { FlyModel } from './lib/model.js';
  import { LOCAL_MODEL, MODEL_REPO, ONNX_REPO } from './lib/config.js';

  const MODEL_CLASSES = ['cb_sensory', 'visual_projection', 'cb_intrinsic', 'ascending_neuron', 'descending_neuron'];
  const LIVE_STEP_MS = 320;

  let assets = $state(null);
  let model = $state(null);
  let view = $state.raw(null);
  let phase = $state('loading');
  let progress = $state(null);
  let error = $state(null);
  let busy = $state(false);
  let draft = $state('');
  let userPrompt = $state('');
  let text = $state('');
  let fedText = '';
  let speed = $state(0);
  let sampling = $state({ mode: 'greedy', temperature: 0.8, topK: 50 });
  let colorMode = $state('activation');
  let autoRotate = $state(true);
  $effect(() => { if (colorMode === 'hop') colorMode = 'synapseChange'; });
  let status = $state('');
  let lastTokenText = $state(null);
  let visualState = $state.raw(null);
  let previousVisualState = $state.raw(null);
  let stats = $state({ classes: MODEL_CLASSES.map((name) => ({ name, mean: 0 })), saturated: 0, meanAbs: 0, changed: 0, topNeurons: [] });
  let windowTokens = $state(Array(8).fill('<pad>'));
  let top = $state([]);
  let ms = $state(0);
  let hopCounts = $state('');
  let classOfModel = null;
  let generatedTokens = $state([]);
  let tokenStates = [];
  let promptState = null;
  let selectedToken = $state(null);

  onMount(async () => {
    try {
      assets = await loadAssets();
      classOfModel = new Uint8Array(assets.meta.n_model);
      for (let i = 0; i < classOfModel.length; i++) classOfModel[i] = assets.superclass[assets.sub2global[i]];
      const hc = [0, 0, 0, 0];
      for (const h of assets.hop) hc[Math.min(h, 3)]++;
      hopCounts = `${hc[0]} in · ${hc[1]} hop1 · ${hc[2]} hop2`;
      model = new FlyModel(assets.meta);
      const cached = await cachedModel();
      if (cached) {
        try {
          await start(cached, { preferWebGPU: LOCAL_MODEL ? false : !!navigator.gpu });
        } catch (e) {
          console.warn('cached model failed to load, downloading again', e);
          await clearModelCache();
          phase = 'consent';
        }
      } else phase = 'consent';
    } catch (e) {
      error = String(e);
      phase = 'error';
    }
  });

  onMount(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') closeBrain();
      if (selectedToken == null) return;
      if (e.key === 'ArrowLeft') shiftToken(-1);
      if (e.key === 'ArrowRight') shiftToken(1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  async function onAccept(opts) {
    phase = 'downloading';
    progress = { frac: 0, label: 'starting download' };
    try {
      const tokP = model.loadTokenizer();
      const bytes = await downloadModel((frac, got) => {
        progress = { frac, label: `loading model: ${(got / 1e6).toFixed(0)} MB` };
      });
      await tokP;
      await start(bytes, { ...opts, preferWebGPU: LOCAL_MODEL ? false : opts.preferWebGPU });
    } catch (e) {
      error = 'Download failed: ' + e;
      progress = null;
      phase = 'consent';
    }
  }

  async function start(bytes, opts) {
    phase = 'init';
    progress = { frac: 1, label: 'loading tokenizer' };
    if (!model.tok) await model.loadTokenizer();
    await model.init(bytes, { ...opts, onStatus: (s) => (progress = { frac: 1, label: s }) });
    phase = 'ready';
    status = `Ready · ${model.backend} · ${(model.initMs / 1000).toFixed(1)}s`;
  }

  function updateStats(prev, next) {
    previousVisualState = prev;
    visualState = next;
    const nm = next.length;
    const sums = new Float64Array(assets.meta.classes.length), cnt = new Int32Array(assets.meta.classes.length);
    let abs = 0, maxAbs = 1e-12, maxD = 1e-12;
    for (let i = 0; i < nm; i++) {
      const a = Math.abs(next[i]);
      const d = Math.abs(next[i] - prev[i]);
      sums[classOfModel[i]] += a; cnt[classOfModel[i]]++;
      abs += a;
      if (a > maxAbs) maxAbs = a;
      if (d > maxD) maxD = d;
    }
    let sat = 0, changed = 0;
    const cand = [];
    const hot = 0.35 * maxAbs, moved = 0.15 * maxD;
    for (let i = 0; i < nm; i++) {
      const a = Math.abs(next[i]);
      if (a > 0.85 * maxAbs) sat++;
      if (Math.abs(next[i] - prev[i]) > moved) changed++;
      if (a >= hot) cand.push(i);
    }
    cand.sort((x, y) => Math.abs(next[y]) - Math.abs(next[x]));
    const idx = assets.meta.classes;
    stats = {
      classes: MODEL_CLASSES.map((name) => { const c = idx.indexOf(name); return { name, mean: cnt[c] ? sums[c] / cnt[c] : 0 }; }),
      saturated: sat / nm, meanAbs: abs / nm, changed: changed / nm,
      topNeurons: cand.slice(0, 8).map((i) => ({ type: assets.types[i] || '?', side: assets.meta.side?.[i] || '', x: next[i] })),
    };
    windowTokens = Array.from(model.window, (b) => Number(b)).map((id) => (id === 0 ? '<pad>' : model.decode([id])));
    ms = model.lastMs;
  }

  function updateTop(logits) {
    const d = model.distribution(logits, sampling.mode === 'sample' ? sampling.temperature : 1, 10);
    top = d.top.map((t) => ({ ...t, text: t.id === model.eos ? '</s>' : model.decode([t.id]) }));
  }

  async function feed(id, durationMs, paint = true) {
    const r = await model.step(id);
    lastTokenText = id === model.bos ? '<start>' : model.decode([id]);
    updateStats(r.prev, r.next);
    updateTop(r.logits);
    if (paint && view) {
      await view.showStep(r.prev, r.next, durationMs);
    }
    return r;
  }

  async function ensurePrompt() {
    if (text === fedText && model.fed.length) return;
    model.reset();
    view?.showStep(model.state, model.state, 0);
    const ids = [model.bos, ...model.encode(text)];
    for (let i = 0; i < ids.length; i++) {
      status = `Reading prompt ${i + 1}/${ids.length}`;
      await feed(ids[i], 0, false);
    }
    fedText = text;
    promptState = Float32Array.from(model.state);
    view?.showStep(promptState, promptState, 0);
  }

  async function onGenerate(n) {
    if (busy || phase !== 'ready' || !text) return;
    busy = true;
    selectedToken = null;
    view?.resize();
    try {
      await ensurePrompt();
      for (let i = 0; i < n; i++) {
        const id = model.sample(model.lastLogits, sampling);
        if (id === model.eos) { status = 'The story ended.'; break; }
        status = `Writing ${i + 1}/${n}: ${JSON.stringify(model.decode([id]))}`;
        const animMs = speed > 0 ? speed : LIVE_STEP_MS;
        await feed(id, animMs, true);
        text = model.decode(model.fed.slice(1));
        fedText = text;
        const snap = new Float32Array(model.state.length);
        snap.set(model.state);
        tokenStates.push(snap);
        generatedTokens = [
          ...generatedTokens,
          {
            text: lastTokenText,
            window: Array.from(model.window, (b) => Number(b)).map((tid) => (tid === 0 ? '<pad>' : model.decode([tid]))),
            top: top.slice(0, 5),
          },
        ];
      }
      status = `${model.fed.length - 1} tokens · ${model.lastMs.toFixed(0)} ms`;
    } catch (e) {
      status = 'Error: ' + e;
      console.error(e);
    } finally {
      busy = false;
    }
  }

  function resetBrain() {
    model?.reset();
    fedText = '';
    lastTokenText = null;
    visualState = null;
    previousVisualState = null;
    top = [];
    windowTokens = Array(8).fill('<pad>');
    generatedTokens = [];
    tokenStates = [];
    promptState = null;
    selectedToken = null;
    view?.clearTrail?.();
    view?.showStep(model.state, model.state, 0);
  }

  function onReset() {
    resetBrain();
    userPrompt = '';
    text = '';
    draft = '';
    status = '';
  }

  async function onSend(preset) {
    const p = (typeof preset === 'string' ? preset : draft).trim();
    if (!p || busy || phase !== 'ready') return;
    resetBrain();
    userPrompt = p;
    text = p;
    draft = '';
    await onGenerate(40);
  }

  $effect(() => {
    const i = selectedToken;
    const v = view;
    const live = busy;
    const cur = i == null ? null : generatedTokens[i];
    const next = i == null ? null : tokenStates[i];
    if (live || i == null || !v || !cur || !next) return;
    const prev = i > 0 ? tokenStates[i - 1] : (promptState ?? next);
    let cancelled = false;
    (async () => {
      await tick();
      await new Promise((r) => requestAnimationFrame(r));
      if (cancelled) return;
      v.resize();
      v.setHighlightSlot(null);
      await v.showStep(prev, next, 420);
      if (cancelled) return;
      updateStats(prev, next);
      windowTokens = cur.window;
      lastTokenText = cur.text;
      if (cur.top) top = cur.top;
      status = `Token ${i + 1}/${generatedTokens.length}: ${JSON.stringify(cur.text)}`;
    })();
    return () => { cancelled = true; };
  });

  function onSelectToken(i) {
    if (busy) return;
    if (i == null) {
      selectedToken = null;
      view?.setHighlightSlot(null);
      return;
    }
    selectedToken = i;
  }

  function onBrainReady(v) {
    view = v;
  }

  function closeBrain() {
    onSelectToken(null);
  }

  function shiftToken(d) {
    if (selectedToken == null || !generatedTokens.length) return;
    const n = generatedTokens.length;
    onSelectToken((selectedToken + d + n) % n);
  }

  async function forget() {
    await clearModelCache();
    location.reload();
  }
</script>

<div class="page">
  {#if phase === 'consent' || phase === 'downloading' || phase === 'init'}
    <Consent {onAccept} progress={phase === 'consent' ? null : progress} {error} />
  {/if}

  <aside class="brain-pane" aria-label="Brain at selected token">
    <div class="sheet-head">
      <div>
        <div class="sheet-kicker muted">
          {#if busy && generatedTokens.length}
            Writing token {generatedTokens.length}
          {:else if selectedToken != null}
            Brain at token {selectedToken + 1} / {generatedTokens.length || 0}
          {:else if generatedTokens.length}
            Last token {generatedTokens.length}
          {:else}
            Fly brain
          {/if}
        </div>
        <div class="sheet-title">
          {#if selectedToken != null}
            {JSON.stringify(generatedTokens[selectedToken]?.text ?? '')}
          {:else if lastTokenText}
            {JSON.stringify(lastTokenText)}
          {:else}
            Click a generated word
          {/if}
        </div>
      </div>
      <div class="sheet-nav">
        <button class="icon" onclick={() => shiftToken(-1)} aria-label="Previous token" disabled={selectedToken == null}>←</button>
        <button class="icon" onclick={() => shiftToken(1)} aria-label="Next token" disabled={selectedToken == null}>→</button>
        <button class="icon" onclick={closeBrain} aria-label="Clear selection" disabled={selectedToken == null}>✕</button>
      </div>
    </div>
    <div class="brain-canvas">
      {#if assets}
        <Brain {assets} onReady={onBrainReady} {colorMode} {autoRotate} />
      {:else}
        <div class="brain-placeholder muted">Loading connectome…</div>
      {/if}
    </div>
    <div class="sheet-tools">
      <label>Colour
        <select bind:value={colorMode}>
          <option value="activation">activation</option>
          <option value="synapseChange">synapse changes</option>
        </select>
      </label>
      <label><input type="checkbox" bind:checked={autoRotate} /> Rotate</label>
    </div>
    <div class="activity">
      <Debug
        backend={model?.backend}
        {ms}
        {hopCounts}
        {model}
        {colorMode}
        {assets}
        tokenText={lastTokenText}
        {visualState}
        {previousVisualState}
        {windowTokens}
        {busy}
      />
    </div>
  </aside>

  <section class="main">
    <header>
      <div class="brand">
        <span class="logo">F</span>
        <div class="model-name">
          Fly brain
          <span class="muted">{phase === 'ready' ? (model?.backend ?? '') : phase} · {LOCAL_MODEL ? 'Local TinyStories' : 'Demo checkpoint'}</span>
        </div>
      </div>
      <div class="head-actions muted">
        <button class="new" onclick={onReset} disabled={busy}>+ New chat</button>
        <a href={'https://huggingface.co/' + MODEL_REPO} target="_blank" rel="noreferrer">Model</a>
        <a href={'https://huggingface.co/' + ONNX_REPO} target="_blank" rel="noreferrer">ONNX</a>
        {#if phase === 'ready'}
          <button class="link" onclick={forget}>Clear cache</button>
        {/if}
      </div>
    </header>
    {#if phase === 'error'}
      <p class="fatal">{error}</p>
    {:else}
      <Chat
        bind:draft
        bind:sampling
        {userPrompt}
        tokens={generatedTokens}
        selected={selectedToken}
        {busy}
        {status}
        {onSend}
        {onSelectToken}
        onContinue={onGenerate}
        {onReset}
      />
    {/if}
  </section>
</div>

<style>
  .page { height: 100%; display: flex; min-height: 0; background: var(--bg); }
  .brain-pane {
    width: min(48vw, 640px);
    flex: 1 1 46%;
    min-width: 280px;
    min-height: 0;
    display: flex;
    flex-direction: column;
    background: var(--panel);
    border-right: 1px solid var(--line);
  }
  .sheet-head {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 14px 16px; border-bottom: 1px solid var(--line); flex-shrink: 0;
  }
  .sheet-kicker { font-size: 12px; }
  .sheet-title { font-size: 18px; font-weight: 600; }
  .sheet-nav { display: flex; gap: 6px; }
  .icon {
    width: 36px; height: 36px; border-radius: 10px; border: 1px solid var(--line); font-size: 16px;
    background: var(--raised);
  }
  .icon:hover:not(:disabled) { background: var(--hover); }
  .brain-canvas { flex: 1; min-height: 240px; background: var(--brain-bg); position: relative; display: flex; }
  .brain-canvas :global(canvas) { flex: 1; width: 100%; height: 100%; min-height: 0; }
  .brain-placeholder { position: absolute; inset: 0; display: grid; place-items: center; }
  .sheet-tools {
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
    padding: 8px 16px; border-top: 1px solid var(--line); font-size: 13px; color: var(--muted);
    flex-shrink: 0; background: var(--panel);
  }
  .sheet-tools label { display: inline-flex; gap: 6px; align-items: center; }
  .activity { flex-shrink: 0; max-height: 58%; overflow: auto; }
  .main {
    flex: 1 1 54%;
    min-width: 0; min-height: 0;
    display: flex; flex-direction: column;
    background: var(--bg);
  }
  header {
    flex-shrink: 0; min-height: 56px; display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 8px 18px; border-bottom: 1px solid var(--line); background: var(--panel);
  }
  .brand { display: flex; gap: 10px; align-items: center; min-width: 0; }
  .logo {
    width: 32px; height: 32px; border-radius: 50%;
    background: #17181b; color: #fff; display: grid; place-items: center; font-weight: 700;
    flex-shrink: 0;
  }
  .model-name { font-weight: 600; font-size: 15px; display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
  .head-actions { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  .new {
    border: 1px solid var(--line); border-radius: 999px;
    padding: 6px 12px; font-weight: 500; background: var(--raised);
  }
  .new:hover:not(:disabled) { background: var(--hover); }
  .link { color: inherit; font-size: 13px; padding: 0; }
  .link:hover { text-decoration: underline; }
  .fatal { color: var(--danger); padding: 24px; }
  @media (max-width: 860px) {
    .page { flex-direction: column; height: auto; min-height: 100%; }
    .brain-pane { width: 100%; min-width: 0; flex: none; height: min(72vh, 740px); border-right: 0; border-bottom: 1px solid var(--line); }
    .main { min-height: 54vh; }
  }
</style>
