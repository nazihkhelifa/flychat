<script>
  let {
    draft = $bindable(),
    userPrompt = '',
    tokens = [],
    selected = null,
    busy = false,
    status = '',
    sampling = $bindable(),
    onSend,
    onContinue,
    onReset,
    onSelectToken,
  } = $props();

  let threadEl;

  $effect(() => {
    tokens.length;
    busy;
    queueMicrotask(() => threadEl?.scrollTo({ top: threadEl.scrollHeight }));
  });

  function keydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!busy) onSend();
    }
  }

  function tokenKey(e, i) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelectToken?.(i);
    }
  }
</script>

<div class="chat">
  <div class="thread" bind:this={threadEl}>
    {#if !userPrompt}
      <div class="empty">
        <div class="mark">F</div>
        <h2>What can I help with?</h2>
        <p class="muted">A fruit-fly connectome writing TinyStories. Watch the left brain while it writes, then click a word to inspect that step.</p>
        <div class="chips">
          <button type="button" onclick={() => onSend('Once upon a time, there was a')}>Once upon a time, there was a</button>
          <button type="button" onclick={() => onSend('Lily and Tom went to the')}>Lily and Tom went to the</button>
        </div>
      </div>
    {:else}
      <div class="col">
        <div class="msg user"><div class="bubble">{userPrompt}</div></div>
        <div class="msg assistant">
          <div class="avatar" aria-hidden="true">F</div>
          <div class="body" role="region" aria-label="Generated tokens. Click a word to open the 3D brain.">
            {#if tokens.length}
              {#each tokens as t, i}
                <span
                  class="tok"
                  class:on={selected === i || (busy && i === tokens.length - 1)}
                  role="button"
                  tabindex="0"
                  title={'Open brain at ' + JSON.stringify(t.text)}
                  onclick={() => onSelectToken?.(i)}
                  onkeydown={(e) => tokenKey(e, i)}
                >{t.text || '·'}</span>
              {/each}
              {#if busy}<span class="cursor">▋</span>{/if}
            {:else if busy}
              <span class="cursor">▋</span>
            {:else}
              <span class="muted">Waiting for tokens…</span>
            {/if}
          </div>
        </div>
      </div>
    {/if}
  </div>

  <div class="dock">
    {#if userPrompt}
      <div class="actions">
        <button class="pill" disabled={busy} onclick={() => onContinue(1)}>Continue</button>
        <button class="pill" disabled={busy} onclick={() => onContinue(10)}>Continue 10</button>
        <button class="pill" disabled={busy} onclick={onReset}>New chat</button>
        <label class="muted">
          <select bind:value={sampling.mode}>
            <option value="greedy">Greedy</option>
            <option value="sample">Sample</option>
          </select>
        </label>
        {#if sampling.mode === 'sample'}
          <label class="muted">T <input type="number" min="0.1" max="2" step="0.1" bind:value={sampling.temperature} /></label>
        {/if}
      </div>
    {/if}
    <div class="composer">
      <textarea
        rows="1"
        placeholder="Message Fly brain"
        bind:value={draft}
        disabled={busy}
        onkeydown={keydown}
      ></textarea>
      <button class="send" disabled={busy || !draft.trim()} onclick={onSend} aria-label="Send">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M11 20V7.8L6.4 12.4 5 11l7-7 7 7-1.4 1.4L13 7.8V20h-2z"/></svg>
      </button>
    </div>
    {#if status}<div class="hint muted">{status}</div>{/if}
    <div class="hint muted">The left brain follows each new token. After it finishes, click a word to inspect that step.</div>
  </div>
</div>

<style>
  .chat { height: 100%; min-height: 0; flex: 1; display: flex; flex-direction: column; }
  .thread { flex: 1; min-height: 0; overflow: auto; }
  .empty {
    min-height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 48px 20px 24px;
    gap: 10px;
  }
  .empty h2 { margin: 8px 0 0; font-size: 28px; font-weight: 600; letter-spacing: -0.03em; }
  .empty p { margin: 0; max-width: 420px; }
  .mark, .avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: grid; place-items: center;
    background: #17181b; color: #fff; font-weight: 700; font-size: 15px;
    flex-shrink: 0;
  }
  .empty .mark { width: 48px; height: 48px; font-size: 20px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 10px; }
  .chips button, .pill {
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 8px 14px;
    font-size: 13px;
    color: var(--text);
  }
  .chips button:hover, .pill:hover:not(:disabled) { background: var(--hover); }
  .col { width: min(768px, 100%); margin: 0 auto; padding: 24px 20px 12px; display: grid; gap: 22px; }
  .msg { display: flex; gap: 14px; }
  .msg.user { justify-content: flex-end; }
  .bubble {
    background: var(--user);
    border-radius: 22px;
    padding: 10px 18px;
    max-width: min(70%, 560px);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.6;
  }
  .msg.assistant { align-items: flex-start; }
  .body { flex: 1; font-size: 16px; line-height: 1.75; white-space: pre-wrap; overflow-wrap: anywhere; padding-top: 6px; }
  .tok {
    border-radius: 4px;
    cursor: pointer;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
  }
  .tok:hover { background: rgba(23, 24, 27, 0.06); }
  .tok.on { background: rgba(37, 99, 235, 0.16); box-shadow: inset 0 -2px 0 #2563eb; }
  .cursor { display: inline-block; margin-left: 1px; animation: blink 1s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  .dock { padding: 8px 16px 16px; }
  .actions { width: min(768px, 100%); margin: 0 auto 8px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .actions input[type=number] { width: 56px; }
  .composer {
    width: min(768px, 100%);
    margin: 0 auto;
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: var(--raised);
    border: 1px solid var(--line);
    border-radius: 28px;
    padding: 10px 10px 10px 18px;
  }
  .composer textarea { min-height: 28px; max-height: 160px; padding: 6px 0; }
  .send {
    width: 36px; height: 36px; border-radius: 50%;
    background: var(--send); color: var(--send-fg);
    display: grid; place-items: center; flex-shrink: 0;
  }
  .send:disabled { background: #d1d5db; color: #fff; }
  .hint { width: min(768px, 100%); margin: 8px auto 0; text-align: center; font-size: 12px; }
</style>
