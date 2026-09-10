const $ = selector => document.querySelector(selector);
const message = $('#message'), error = $('#error'), status = $('#status');
const COLORS = ['#b8aeff', '#a8dc84', '#edb573', '#ff8b78', '#82baff', '#70e0dc'];
const MODEL_NAMES = {fly:'Recorded fly wiring', scrambled:'Scrambled wiring', no_recurrence:'No recurrent connections', text_encoder_only:'Same text encoder only', tfidf_text_only:'TF-IDF text-only baseline'};
let manifest, worker, anatomy, result, currentText = '', intactScores, currentFrame = 0;
let playing = false, playStart = 0, playFrame = 0, requestId = 0, lastAcceptedId = 0;
const pending = new Map();
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const signed = value => (value >= 0 ? '+' : '−') + Math.abs(value).toFixed(3);
const percent = value => (value * 100).toFixed(1) + '%';

function fail(reason) {
  error.hidden = false;
  error.textContent = reason;
  status.textContent = 'Action needed';
}
function clearError() { error.hidden = true; error.textContent = ''; }

async function checkedBuffer(url, expected) {
  const response = await fetch(url);
  if (!response.ok) throw Error('Could not load ' + url + '. Reload to retry.');
  const buffer = await response.arrayBuffer();
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', buffer));
  const hash = [...digest].map(byte => byte.toString(16).padStart(2, '0')).join('');
  if (hash !== expected) throw Error('Model data verification failed. Reload to fetch a matching version.');
  return buffer;
}

function initWorker(buffer) {
  return new Promise((resolve, reject) => {
    worker = new Worker(new URL('./worker.mjs', import.meta.url), {type:'module'});
    const timeout = setTimeout(() => reject(Error('Model initialization timed out. Reload to retry.')), 15000);
    worker.onerror = () => {
      clearTimeout(timeout);
      const err = Error('The inference worker stopped. Reload to retry.');
      reject(err); pending.forEach(p => p.reject(err)); pending.clear(); fail(err.message);
    };
    worker.onmessage = ({data}) => {
      if (data.kind === 'ready') { clearTimeout(timeout); resolve(); return; }
      if (data.kind === 'error' && data.id === undefined) { clearTimeout(timeout); reject(Error(data.message)); return; }
      const promise = pending.get(data.id);
      if (!promise) return;
      pending.delete(data.id);
      if (data.kind === 'error') promise.reject(Error(data.message));
      else promise.resolve({id:data.id, result:data.result});
    };
    worker.postMessage({kind:'init', manifest, buffer}, [buffer]);
  });
}

function infer(text, silenced) {
  const id = ++requestId;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve,reject});
    worker.postMessage({kind:'classify', id, text, silenced});
  });
}

function renderScores() {
  const ordering = [...result.scores.keys()].sort((a,b) => result.scores[b] - result.scores[a]);
  const top = ordering[0];
  const enough = result.known > 0;
  $('#prediction').textContent = enough ? manifest.labels[top] : 'Unknown words';
  $('#prediction').style.color = enough ? COLORS[top] : '#a0b1bd';
  $('#top-score').textContent = enough ? signed(result.scores[top]) : '—';
  const margin = result.scores[top] - result.scores[ordering[1]];
  $('#result-note').textContent = !enough ? 'None of these words are in the learned vocabulary. No supported classification is available.' :
    result.silenced && intactScores ? `${result.silenced} neurons silenced. ${manifest.labels[top]} score changed from ${signed(intactScores[top])} to ${signed(result.scores[top])}.` :
    `Highest of six learned scores; margin ${margin.toFixed(3)} over ${manifest.labels[ordering[1]]}.`;
  const wrap = $('#scores'); wrap.replaceChildren();
  const scale = Math.max(.1, ...Array.from(result.scores, Math.abs));
  manifest.labels.forEach((label,i) => {
    const row = document.createElement('div'); row.className = 'score-row' + (i === top && enough ? ' is-top' : '');
    const text = document.createElement('span'); text.textContent = label;
    const track = document.createElement('span'); track.className = 'score-track'; track.setAttribute('aria-hidden','true');
    const fill = document.createElement('i'); fill.className = 'score-fill'; fill.style.background = COLORS[i];
    const width = enough ? Math.abs(result.scores[i]) / scale * 50 : 0;
    fill.style.width = width + '%'; fill.style.left = (result.scores[i] >= 0 ? 50 : 50-width) + '%';
    track.append(fill);
    const value = document.createElement('output'); value.textContent = enough ? signed(result.scores[i]) : '—';
    row.append(text,track,value); wrap.append(row);
  });
  const warnings = [];
  if (result.tokens.length < 3) warnings.push('Very short text; interpret cautiously.');
  if (result.known < result.tokens.length) warnings.push(`${result.tokens.length-result.known} of ${result.tokens.length} words are outside the vocabulary.`);
  if (result.truncated) warnings.push('Only the first 96 tokens were analyzed.');
  if (message.value.trim() !== currentText) warnings.push('Input has changed. Results still describe the last analyzed text.');
  $('#warning').textContent = warnings.join(' ');
}

function renderHeatmap() {
  const canvas = $('#heatmap'), ctx = canvas.getContext('2d');
  // Average neuron rows when displaying fewer than 499 pixels vertically.
  const rows = 120; canvas.width = result.frames; canvas.height = rows;
  const img = ctx.createImageData(result.frames, rows);
  for (let y = 0; y < rows; y++) {
    const start = Math.floor(y * 499 / rows), end = Math.floor((y+1) * 499 / rows);
    for (let x = 0; x < result.frames; x++) {
      let value = 0; for(let n=start;n<end;n++) value += Math.abs(result.states[x*499+n])/(end-start);
      const rgb = colorFor(value);
      const index = (y * result.frames + x) * 4;
      img.data.set([...rgb,255], index);
    }
  }
  ctx.putImageData(img,0,0);
}

function renderTokens() {
  const wrap = $('#tokens'); wrap.replaceChildren();
  result.tokens.forEach(token => { const span=document.createElement('span'); span.textContent=token; wrap.append(span); });
}
function setFrame(frame) {
  if (!result) return;
  currentFrame = Math.max(0,Math.min(result.frames-1,frame));
  $('#progress').value = String(currentFrame);
  $('#token-label').textContent = `${currentFrame+1} / ${result.frames}`;
  const actualToken = Math.min(currentFrame,result.tokens.length-1);
  [...$('#tokens').children].forEach((node,i) => node.classList.toggle('current',i===actualToken));
  $('#activity-state').textContent = `Mean |activity| ${result.activity[currentFrame].toFixed(3)}`;
  anatomy?.draw(result.states.subarray(currentFrame*499,(currentFrame+1)*499));
}
function stopPlayback() { playing=false; cancelAnimationFrame(playFrame); $('#play').textContent='Play trace'; }
function startPlayback() {
  if (!result) return;
  stopPlayback(); playing=true; $('#play').textContent='Pause'; playStart=performance.now();
  const duration=Math.max(2000,result.frames*200);
  function frame(now) {
    if (!playing) return;
    const fraction=Math.min(1,(now-playStart)/duration);
    setFrame(Math.min(result.frames-1,Math.floor(fraction*result.frames)));
    if (fraction < 1) playFrame=requestAnimationFrame(frame); else stopPlayback();
  }
  setFrame(0); playFrame=requestAnimationFrame(frame);
}

async function classify(text=message.value.trim(), silence=Number($('#silence').value)) {
  if (!worker) throw Error('The model is still loading.');
  clearError(); stopPlayback(); status.textContent='Computing circuit activity…';
  const isNew = text !== currentText;
  // When intervening in a new sentence, compute its intact reference first.
  let baseline;
  if (silence && (isNew || !intactScores)) baseline = (await infer(text,0)).result.scores;
  const response=await infer(text,silence);
  if (response.id < requestId || response.id < lastAcceptedId) return null;
  lastAcceptedId=response.id;
  result=response.result; currentText=text;
  if (!silence) intactScores=new Float64Array(result.scores); else if (baseline) intactScores=baseline;
  $('#silence').value=String(silence); $('#silence-label').textContent=`${silence} / 499`;
  renderScores(); renderTokens(); renderHeatmap();
  $('#progress').max=String(result.frames-1); $('#progress').disabled=false; $('#play').disabled=false;
  setFrame(result.frames-1);
  status.textContent='Trained model · running locally';
  return {label:result.known ? manifest.labels[result.scores.indexOf(Math.max(...result.scores))] : null,
    scores:Object.fromEntries(manifest.labels.map((label,i)=>[label,result.scores[i]])),
    silenced:silence, neurons:499, known_tokens:result.known, tokens:result.tokens.length, score_type:'raw ridge score, not probability'};
}

function colorFor(activity) {
  const value=Math.max(0,Math.min(1,activity/.8));
  const stops=[[27,47,68],[72,179,184],[255,207,112]];
  const i=value<.5?0:1, frac=(value-i*.5)*2;
  return stops[i].map((v,c)=>Math.round(v+(stops[i+1][c]-v)*frac));
}

class Anatomy {
  constructor(canvas, buffer) {
    this.canvas=canvas; this.ctx=canvas.getContext('2d'); this.raw=new Int16Array(buffer);
    this.yaw=.15; this.pitch=-.35; this.zoom=1; this.state=new Float32Array(499);
    let drag=null;
    canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY}; canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove',e=>{if(!drag)return;this.yaw+=(e.clientX-drag.x)*.007;this.pitch+=(e.clientY-drag.y)*.007;drag={x:e.clientX,y:e.clientY};this.draw();});
    canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
    canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoom=Math.max(.5,Math.min(3,this.zoom*Math.exp(-e.deltaY*.001)));this.draw();},{passive:false});
    canvas.addEventListener('keydown',e=>{const controls=['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-'];if(!controls.includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')this.yaw-=.1;if(e.key==='ArrowRight')this.yaw+=.1;if(e.key==='ArrowUp')this.pitch-=.1;if(e.key==='ArrowDown')this.pitch+=.1;if(['+','='].includes(e.key))this.zoom=Math.min(3,this.zoom*1.1);if(e.key==='-')this.zoom=Math.max(.5,this.zoom/1.1);this.draw();});
    $('#reset-view').addEventListener('click',()=>{this.yaw=.15;this.pitch=-.35;this.zoom=1;this.draw();});
    this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(canvas);
    this.draw();
  }
  draw(state) {
    if (state) this.state=state;
    const canvas=this.canvas,ctx=this.ctx,rect=canvas.getBoundingClientRect();
    if(!rect.width||!rect.height)return;
    const dpr=Math.min(2,devicePixelRatio||1);
    if(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr)){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
    const raw=this.raw,points=new Float32Array(raw.length/2),cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
    for(let v=0;v<raw.length/4;v++){
      const x=raw[v*4]/32767*2,y=raw[v*4+1]/32767*2,z=raw[v*4+2]/32767*2;
      const px=x*cy+z*sy,pz=-x*sy+z*cy,py=y*cp-pz*sp;
      points[v*2]=px;points[v*2+1]=py;
      minX=Math.min(minX,px);maxX=Math.max(maxX,px);minY=Math.min(minY,py);maxY=Math.max(maxY,py);
    }
    const scale=Math.min(rect.width/(maxX-minX+1e-8), (rect.height-64)/(maxY-minY+1e-8))*.85*this.zoom;
    const centerX=(minX+maxX)/2,centerY=(minY+maxY)/2;
    const paths=Array.from({length:24},()=>new Path2D());
    for(let v=0;v<raw.length/4;v+=2){
      const owner=raw[v*4+3],value=Math.min(1,Math.abs(this.state[owner]||0)/.8),bin=Math.min(23,Math.floor(value*24));
      const p=paths[bin];
      p.moveTo(rect.width/2+(points[v*2]-centerX)*scale,rect.height/2-(points[v*2+1]-centerY)*scale);
      p.lineTo(rect.width/2+(points[(v+1)*2]-centerX)*scale,rect.height/2-(points[(v+1)*2+1]-centerY)*scale);
    }
    for(let i=0;i<24;i++){ctx.strokeStyle='rgb('+colorFor((i+.5)/24*.8).join(',')+')';ctx.globalAlpha=.28+i/23*.65;ctx.lineWidth=.65+i/23*.6;ctx.stroke(paths[i]);}
    ctx.globalAlpha=1;
  }
}

function showBenchmark() {
  const report=manifest.report,counts=report.split.class_counts;
  const total=name=>counts[name].reduce((a,b)=>a+b,0).toLocaleString();
  $('#split-info').textContent=`${total('train')} training · ${total('validation')} validation · ${total('test')} held-out test sentences. Six English emotion labels.`;
  for(const [key,label] of Object.entries(MODEL_NAMES)){
    const row=document.createElement('tr');if(key==='fly')row.className='fly';
    const values=[label,percent(report.models[key].test.accuracy),percent(report.models[key].test.macro_f1)];
    for(const value of values){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}
    $('#benchmark').append(row);
  }
  const interval=report.fly_minus_scrambled_macro_f1_ci95;
  $('#benchmark-note').textContent=`Majority-label baseline: ${percent(report.majority_accuracy)} accuracy. Fly minus scrambled macro-F1: conditional 95% paired bootstrap interval ${(interval[0]*100).toFixed(1)} to ${(interval[1]*100).toFixed(1)} percentage points. The TF-IDF baseline and no-recurrence control score higher than this fly model.`;
}

async function registerAgentTools() {
  const context=document.modelContext;
  if(!context?.registerTool)return;
  const lifecycle=new AbortController();
  addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
  try {
    await context.registerTool({name:'classify_fly_text',title:'Classify text with the trained fly circuit',
      description:'Analyze English text using the fixed MaleCNS circuit and trained readout; update the visible prediction and neural activity.',
      inputSchema:{type:'object',properties:{text:{type:'string',minLength:1,maxLength:4000},silenced:{type:'integer',minimum:0,maximum:200}},required:['text'],additionalProperties:false},
      annotations:{readOnlyHint:false,untrustedContentHint:true},
      execute:async input=>{
        if(!input||typeof input.text!=='string'||!input.text.trim()||input.text.length>4000||Object.keys(input).some(k=>!['text','silenced'].includes(k)))throw Error('Invalid text input.');
        const count=input.silenced??0;if(!Number.isInteger(count)||count<0||count>200)throw Error('Invalid silenced count.');
        const output=await classify(input.text.trim(),count);message.value=input.text;$('#char-count').textContent=`${message.value.length.toLocaleString()} / 4,000`;renderScores();return output;
      }},{signal:lifecycle.signal});
  } catch { /* Optional proposed API: visible controls remain fully functional. */ }
}

message.addEventListener('input',()=>{
  $('#char-count').textContent=`${message.value.length.toLocaleString()} / 4,000`;
  if(result)renderScores();
});
$('#reader-form').addEventListener('submit',e=>{e.preventDefault();classify().catch(e=>fail(e.message));});
message.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();$('#reader-form').requestSubmit();}});
document.querySelectorAll('[data-sample]').forEach(button=>button.addEventListener('click',()=>{
  message.value=button.dataset.sample;message.dispatchEvent(new Event('input'));$('#silence').value='0';
  if(worker)classify(message.value,0).catch(e=>fail(e.message));
}));
$('#play').addEventListener('click',()=>playing?stopPlayback():startPlayback());
$('#progress').addEventListener('input',()=>{stopPlayback();setFrame(Number($('#progress').value));});
$('#silence').addEventListener('input',()=>{$('#silence-label').textContent=`${$('#silence').value} / 499`;});
$('#silence').addEventListener('change',()=>classify(currentText||message.value.trim()).catch(e=>fail(e.message)));
$('#restore').addEventListener('click',()=>classify(currentText||message.value.trim(),0).catch(e=>fail(e.message)));
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopPlayback();});
message.dispatchEvent(new Event('input'));

async function boot() {
  const response=await fetch('./model/manifest.json');if(!response.ok)throw Error('Trained model is missing. Reload after training completes.');
  manifest=await response.json(); showBenchmark();
  const [weights,geometry]=await Promise.all([checkedBuffer('./model/weights.bin',manifest.weights_sha256),checkedBuffer('./model/anatomy.bin',manifest.anatomy_sha256)]);
  anatomy=new Anatomy($('#brain'),geometry);$('#canvas-message').hidden=true;
  await initWorker(weights);
  for(const id of ['#classify','#silence','#restore'])$(id).disabled=false;
  await classify(message.value.trim(),0);
  await registerAgentTools();
  if(!reduced)startPlayback();
}
boot().catch(e=>{fail(e.message);$('#canvas-message').textContent='Model unavailable — see the error message.';});
