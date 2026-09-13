<script>
  const stats = [
    ['49,393', 'fly neurons in the reservoir'],
    ['9.05M', 'frozen connectome synapses'],
    ['8', 'token delay slots'],
    ['1,024', 'TinyStories vocabulary logits'],
  ];

  const dot = (i) => ({
    left: `${8 + ((i * 37) % 84)}%`,
    top: `${12 + ((i * 61) % 72)}%`,
    delay: `${(i % 7) * 0.08}s`,
  });
</script>

<section class="method" aria-labelledby="method-title">
  <div class="method-hero">
    <div>
      <p class="eyebrow">The method behind the demo</p>
      <h1 id="method-title">A fruit-fly brain used as an echo-state language model</h1>
      <p class="lead">The connectome supplies a fixed, recurrent circuit. A small trainable input port writes tokens into that circuit, and a linear readout translates the fly-brain state into the next word.</p>
      <a class="try-button" href="/demo">Try it now <span aria-hidden="true">→</span></a>
    </div>
    <div class="stat-grid" aria-label="Model scale">
      {#each stats as stat}
        <div class="stat"><strong>{stat[0]}</strong><span>{stat[1]}</span></div>
      {/each}
    </div>
  </div>

  <div class="method-grid">
    <article class="copy-card">
      <p class="eyebrow">Echo-state method</p>
      <h2>Let the wiring remember</h2>
      <p>Each token is embedded into 128 numbers, then placed in the newest slot of a hard eight-token delay line. The eight slots are projected into sensory-facing fly neurons. Slot 0 is the newest token; slot 7 is the oldest, so order is represented by the delay itself.</p>
      <p>The fruit-fly wiring is frozen. At every step, activity is mixed through the connectome, passed through a leaky tanh update, and carried forward as a recurrent state:</p>
      <div class="equation"><code>xₜ = 0.1 xₜ₋₁ + 0.9 tanh(g(Wxₜ₋₁ + inputₜ) + b)</code></div>
      <p>This is the “echo”: a useful trace of recent tokens persists in the reservoir without adding transformer attention or learned synapses.</p>
    </article>

    <article class="flow-card">
      <p class="eyebrow">One generated token</p>
      <h2>From prompt to readout</h2>
      <div class="flow" aria-label="Token flow diagram">
        <div class="flow-node token"><b>token id</b><span>“girl”</span></div>
        <div class="arrow">→</div>
        <div class="flow-node"><b>8-slot delay</b><span>t … t−7</span></div>
        <div class="arrow">→</div>
        <div class="flow-node brain"><b>fly reservoir</b><span>49,393 states</span></div>
        <div class="arrow">→</div>
        <div class="flow-node readout"><b>linear readout</b><span>1,024 logits</span></div>
      </div>
      <div class="flow-note"><span class="dot blue"></span><b>Input:</b> token ids enter through sensory-facing ports. <span class="dot orange"></span><b>Output:</b> the readout chooses the next token; the fly cells themselves do not spell words.</div>
    </article>
  </div>

  <div class="screens">
    <div class="section-heading">
      <div><p class="eyebrow">Screenshots from the live interface</p><h2>What to watch while it writes</h2></div>
      <p class="muted">Click any generated word to replay its state.</p>
    </div>
    <div class="shot-grid">
      <figure class="shot">
        <div class="shot-bar"><span class="window-dot red"></span><span class="window-dot yellow"></span><span class="window-dot green"></span><span class="shot-label">Fly brain · activation</span></div>
        <div class="shot-body brain-shot">
          <div class="mini-connectome">
            {#each Array(34) as _, i}
              {@const p = dot(i)}
              <i style={`left:${p.left};top:${p.top};animation-delay:${p.delay}`}></i>
            {/each}
            <span class="mini-brain-label">49,393 fly cells</span>
          </div>
          <div class="shot-side"><b>“girl”</b><span class="pulse-line"></span><small>state update<br />12.4 ms</small></div>
        </div>
        <figcaption><b>01 · Brain state</b> — orange and blue activity shows which neurons changed for the selected token. The moving sweep is a visual guide; the state update is measured from the model.</figcaption>
      </figure>

      <figure class="shot">
        <div class="shot-bar"><span class="window-dot red"></span><span class="window-dot yellow"></span><span class="window-dot green"></span><span class="shot-label">What the eyes see · 3D head</span></div>
        <div class="shot-body eyes-shot">
          <div class="eye-head"><div class="eye left"></div><div class="face"></div><div class="eye right"></div></div>
          <div class="eye-readout"><b>RIGHT EYE</b><b>LEFT EYE</b><span>sensory activity</span><span>sensory activity</span></div>
          <div class="window-strip">&lt;pad&gt; · once · upon · a · time · there · was · a</div>
        </div>
        <figcaption><b>02 · Sensory view</b> — the 3D fruit-fly head shows both compound eyes and their live right/left activity summaries beside the eight-token context window.</figcaption>
      </figure>
    </div>
  </div>

  <div class="scope"><b>Biological scope.</b> In this language demo, text is the controlled sensory input and the full central-brain state feeds the readout. The same ports are designed to accept visual or optic-flow features, with descending neurons providing a path toward leg and motor controllers in an embodied version.</div>
  <div class="end-cta"><div><b>Ready to inspect a live token?</b><span>Open the browser demo and click any generated word.</span></div><a class="try-button" href="/demo">Open the live demo <span aria-hidden="true">→</span></a></div>
</section>

<style>
  .method { padding: 28px clamp(18px, 4vw, 52px) 34px; display: grid; gap: 24px; max-width: 1180px; margin: 0 auto; }
  .method-hero { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(300px, .8fr); gap: 28px; align-items: end; }
  .eyebrow { margin: 0 0 7px; color: #7a5c2e; text-transform: uppercase; letter-spacing: .11em; font-size: 10px; font-weight: 750; }
  h1, h2, p { margin-top: 0; }
  h1 { margin-bottom: 12px; font-size: clamp(27px, 3.2vw, 43px); line-height: 1.03; letter-spacing: -.045em; max-width: 680px; }
  h2 { margin-bottom: 10px; font-size: 19px; letter-spacing: -.025em; }
  .lead { max-width: 650px; color: #4b5563; line-height: 1.62; font-size: 15px; margin-bottom: 0; }
  .try-button { display: inline-flex; align-items: center; gap: 9px; width: fit-content; margin-top: 18px; padding: 10px 16px; border-radius: 999px; background: #17181b; color: #fff; font-size: 13px; font-weight: 650; text-decoration: none; box-shadow: 0 5px 12px rgba(23,24,27,.16); }.try-button:hover { background: #34363b; text-decoration: none; transform: translateY(-1px); }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
  .stat { border: 1px solid var(--line); background: rgba(255,255,255,.7); border-radius: 14px; padding: 12px 13px; display: grid; gap: 3px; }
  .stat strong { font-size: 20px; letter-spacing: -.03em; }
  .stat span { color: var(--muted); line-height: 1.25; font-size: 11px; }
  .method-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .copy-card, .flow-card { border: 1px solid var(--line); border-radius: 18px; padding: 20px; background: var(--panel); box-shadow: 0 8px 24px rgba(31,41,55,.04); }
  .copy-card p:not(.eyebrow), .flow-card p:not(.eyebrow) { color: #4b5563; font-size: 13px; line-height: 1.58; }
  .equation { background: #191a1d; color: #f9fafb; border-radius: 11px; padding: 13px 14px; overflow-x: auto; margin: 14px 0; font-size: 12px; }
  .flow { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin: 20px 0 15px; }
  .flow-node { flex: 1 1 82px; min-width: 80px; border: 1px solid #dbe1e8; border-radius: 12px; padding: 10px 8px; background: #f8fafc; display: grid; gap: 4px; text-align: center; }
  .flow-node b { font-size: 11px; }.flow-node span { font-size: 10px; color: var(--muted); }.flow-node.token { background: #fff7ed; border-color: #fed7aa; }.flow-node.brain { background: #eff6ff; border-color: #bfdbfe; }.flow-node.readout { background: #f0fdf4; border-color: #bbf7d0; }.arrow { color: #9ca3af; font-size: 18px; }
  .flow-note { color: #4b5563; font-size: 12px; line-height: 1.55; }.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin: 0 4px 0 2px; }.dot.blue { background: #3b82f6; }.dot.orange { background: #f97316; }
  .screens { display: grid; gap: 12px; }.section-heading { display: flex; align-items: end; justify-content: space-between; gap: 15px; }.section-heading h2 { margin-bottom: 0; }.section-heading .muted { margin: 0; font-size: 12px; }
  .shot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }.shot { margin: 0; min-width: 0; }.shot-bar { height: 30px; display: flex; align-items: center; gap: 5px; padding: 0 11px; background: #202226; color: #d1d5db; border-radius: 14px 14px 0 0; font-size: 10px; }.window-dot { width: 7px; height: 7px; border-radius: 50%; }.window-dot.red { background: #fb7185; }.window-dot.yellow { background: #fbbf24; }.window-dot.green { background: #34d399; }.shot-label { margin-left: 5px; opacity: .85; }.shot-body { height: 190px; border: 1px solid var(--line); border-top: 0; background: #f8fafc; overflow: hidden; position: relative; }.brain-shot { display: grid; grid-template-columns: 1fr 100px; gap: 8px; padding: 14px; }.mini-connectome { position: relative; border-radius: 12px; background: radial-gradient(circle at 50% 50%, #25364b, #101722 72%); overflow: hidden; }.mini-connectome::before { content: ''; position: absolute; inset: 18% 20%; border: 1px solid rgba(147,197,253,.28); border-radius: 50% 45% 52% 48%; transform: rotate(-12deg); }.mini-connectome i { position: absolute; width: 5px; height: 5px; border-radius: 50%; background: #fb923c; box-shadow: 0 0 9px #fb923c; animation: glow 2.2s ease-in-out infinite; }.mini-connectome i:nth-child(3n) { background: #38bdf8; box-shadow: 0 0 9px #38bdf8; }.mini-brain-label { position: absolute; left: 10px; bottom: 9px; color: rgba(255,255,255,.7); font-size: 9px; }.shot-side { display: grid; align-content: center; gap: 9px; color: #1f2937; font-size: 13px; }.shot-side small { color: var(--muted); line-height: 1.35; font-size: 10px; }.pulse-line { height: 3px; border-radius: 5px; background: linear-gradient(90deg,#3b82f6,#f97316,#38bdf8); }
  .eyes-shot { padding: 13px 13px 0; background: radial-gradient(circle at 50% 30%,#fff,#edf0f4); }.eye-head { display: flex; align-items: center; justify-content: center; height: 130px; }.eye { width: 106px; height: 110px; border-radius: 50%; background: radial-gradient(circle at 35% 28%, #f7a36f 0 8%, #a83e25 36%, #501b18 78%); border: 3px solid #331512; box-shadow: inset -10px -8px 18px rgba(0,0,0,.35), 0 8px 12px rgba(31,41,55,.2); position: relative; }.eye::after { content: ''; position: absolute; inset: 7px; border-radius: 50%; opacity: .7; background-image: radial-gradient(circle, rgba(39,12,8,.8) 0 2px, transparent 2.5px); background-size: 11px 11px; }.face { width: 60px; height: 104px; margin: 0 -8px; border-radius: 45% 45% 36% 36%; background: linear-gradient(90deg,#876848,#c2a27c 45%,#7d5c40); box-shadow: inset 7px 0 9px rgba(255,255,255,.16), inset -8px 0 11px rgba(0,0,0,.2); }.eye-readout { display: grid; grid-template-columns: 1fr 1fr; text-align: center; gap: 2px; font-size: 8px; color: #6b4c36; }.eye-readout span { color: #9ca3af; font-size: 9px; }.window-strip { margin: 12px -13px 0; padding: 8px 10px; border-top: 1px solid var(--line); background: rgba(255,255,255,.75); color: #6b7280; font: 9px ui-monospace,monospace; white-space: nowrap; overflow: hidden; }
  figcaption { padding: 10px 2px 0; color: #6b7280; font-size: 11px; line-height: 1.45; }.shot b { color: #374151; }.scope { border-left: 3px solid #d97706; padding: 11px 14px; background: #fffbeb; color: #5b4a2f; font-size: 12px; line-height: 1.55; border-radius: 0 10px 10px 0; }.end-cta { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 16px 18px; border: 1px solid var(--line); border-radius: 15px; background: var(--panel); }.end-cta div { display: grid; gap: 3px; }.end-cta span { color: var(--muted); font-size: 12px; }.end-cta .try-button { margin-top: 0; flex-shrink: 0; }
  @keyframes glow { 50% { opacity: .35; transform: scale(.72); } }
  @media (max-width: 980px) { .method-hero, .method-grid, .shot-grid { grid-template-columns: 1fr; } }
  @media (max-width: 560px) { .end-cta { align-items: flex-start; flex-direction: column; } }
</style>
