// three.js point cloud of the whole fly. Model neurons show LM state; optic lobes and VNC light from input drive.
// Animation is visual only: the real state update is instant, the colours sweep from prev to next ordered by
// synaptic hop distance from the input neurons, so it looks like the signal travels through the brain.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export const CLASS_COLORS = {
  cb_sensory: [0.35, 0.9, 0.5],
  visual_projection: [0.3, 0.75, 1.0],
  cb_intrinsic: [1.0, 0.7, 0.25],
  ascending_neuron: [0.85, 0.5, 1.0],
  descending_neuron: [1.0, 0.35, 0.35],
  ol_intrinsic: [0.25, 0.55, 0.95],
  ol_sensory: [0.45, 0.85, 1.0],
  visual_centrifugal: [0.2, 0.7, 0.9],
  vnc_intrinsic: [0.95, 0.5, 0.55],
  vnc_sensory: [0.65, 0.85, 0.4],
  vnc_motor: [1.0, 0.28, 0.32],
  cb_motor: [1.0, 0.45, 0.25],
};
export const SLOT_COLORS = [
  [1.0, 1.0, 1.0], [1.0, 0.9, 0.5], [1.0, 0.75, 0.4], [1.0, 0.6, 0.4],
  [0.95, 0.5, 0.55], [0.85, 0.45, 0.7], [0.7, 0.45, 0.85], [0.55, 0.5, 0.95],
];
const GREY = [0.52, 0.56, 0.62];
const GREY_QUIET = [0.8, 0.82, 0.86];

function copyF32(x) {
  const n = x?.length ?? 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = x[i];
  return out;
}

const vert = /* glsl */ `
  attribute float size;
  attribute vec3 color;
  attribute float slot;
  attribute float isModel;
  attribute float act;
  uniform float uScale;
  uniform float uHighlightSlot;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float s = size;
    vColor = color;
    vAlpha = act;
    if (uHighlightSlot >= 0.0) {
      if (abs(slot - uHighlightSlot) < 0.5) { vColor = vec3(1.0); s *= 2.5; vAlpha = 1.0; }
      else { vColor *= 0.35; }
    }
    gl_PointSize = s * uScale / -mv.z;
    gl_Position = projectionMatrix * mv;
  }`;
const frag = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = dot(c, c);
    if (d > 0.25) discard;
    if (vAlpha < 0.02) discard;
    float a = vAlpha * smoothstep(0.25, 0.12, d);
    gl_FragColor = vec4(vColor, a);
  }`;

export class BrainView {
  constructor(canvas, assets) {
    this.assets = assets;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.6;
    this.colorMode = 'activation'; // activation | synapseChange
    this.anim = null;
    this.buildGeometry();
    this.resize();
    this.camera.position.set(0, 0.35, 3.35);
    this.controls.target.set(0, 0, 0);
    this.animate = this.animate.bind(this);
    this.raf = requestAnimationFrame(this.animate);
  }

  buildGeometry() {
    const { positions, superclass, sub2global, inSlot, hop, meta } = this.assets;
    const n = meta.n_neurons;
    const nm = meta.n_model;
    const lo = meta.bbox_lo, hi = meta.bbox_hi;
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    // vertex list: every neuron with a soma position. modelVertex[i] = vertex index of model neuron i (or -1)
    const vertOf = new Int32Array(n).fill(-1);
    let count = 0;
    for (let g = 0; g < n; g++) if (positions[g * 3] !== 65535) vertOf[g] = count++;
    this.count = count;
    const pos = new Float32Array(count * 3);
    const color = new Float32Array(count * 3);
    const size = new Float32Array(count);
    const slot = new Float32Array(count).fill(-1);
    const isModel = new Float32Array(count);
    for (let g = 0; g < n; g++) {
      const v = vertOf[g];
      if (v < 0) continue;
      for (let a = 0; a < 3; a++) pos[v * 3 + a] = ((positions[g * 3 + a] / 65534) * (hi[a] - lo[a]) + lo[a] - (lo[a] + hi[a]) / 2) / span * 2;
      color.set(GREY, v * 3);
      size[v] = 1.0;
    }
    // MaleCNS axes: x left-right, y dorsal-ventral, z anterior-posterior. Put y up, flip so brain is on top.
    for (let v = 0; v < count; v++) { pos[v * 3 + 1] = -pos[v * 3 + 1]; }
    this.modelVertex = new Int32Array(nm);
    this.modelClass = new Uint8Array(nm);
    for (let i = 0; i < nm; i++) {
      const g = sub2global[i];
      this.modelVertex[i] = vertOf[g];
      this.modelClass[i] = superclass[g];
      const v = vertOf[g];
      if (v >= 0) { isModel[v] = 1; size[v] = 1.6; slot[v] = inSlot[i] === 255 ? -1 : inSlot[i]; }
    }
    this.classNames = meta.classes;
    this.hop = hop;
    this.vertexModel = new Int32Array(count).fill(-1);
    this.vertexClass = new Uint8Array(count);
    this.posX = new Float32Array(count);
    for (let g = 0; g < n; g++) {
      const v = vertOf[g];
      if (v < 0) continue;
      this.vertexClass[v] = superclass[g];
      this.posX[v] = pos[v * 3];
    }
    for (let i = 0; i < nm; i++) {
      const v = this.modelVertex[i];
      if (v >= 0) this.vertexModel[v] = i;
    }
    this.visualL = 0;
    this.visualR = 0;
    this.sensoryDrive = 0;
    this.bodyDrive = 0;
    this.somaPos = pos;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(color, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(size, 1));
    geo.setAttribute('slot', new THREE.BufferAttribute(slot, 1));
    geo.setAttribute('isModel', new THREE.BufferAttribute(isModel, 1));
    const act = new Float32Array(count);
    geo.setAttribute('act', new THREE.BufferAttribute(act, 1));
    this.geo = geo;
    this.mat = new THREE.ShaderMaterial({
      vertexShader: vert, fragmentShader: frag, transparent: true, depthWrite: false,
      blending: THREE.NormalBlending,
      uniforms: { uScale: { value: 300 }, uHighlightSlot: { value: -1 } },
    });
    this.points = new THREE.Points(geo, this.mat);
    this.scene.add(this.points);
    this.colorAttr = geo.getAttribute('color');
    this.sizeAttr = geo.getAttribute('size');
    this.actAttr = geo.getAttribute('act');
    this.buildNeurites(pos);
    this.display = new Float32Array(nm); // activation currently displayed per model neuron
    this.actScale = 1;
    this.deltaScale = 1;
    this.lastDelta = null;
    this.actTrail = [];
    this.paintAll();
  }

  resize() {
    const c = this.renderer.domElement;
    const w = c.clientWidth || 600, h = c.clientHeight || 500;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.mat.uniforms.uScale.value = h / 170;
  }

  setHighlightSlot(s) { this.mat.uniforms.uHighlightSlot.value = s == null ? -1 : s; }
  setAutoRotate(v) { this.controls.autoRotate = v; }
  setColorMode(m) {
    this.colorMode = m === 'synapseChange' || m === 'hop' ? 'synapseChange' : 'activation';
    const syn = this.colorMode === 'synapseChange';
    if (this.neurites) this.neurites.visible = syn;
    if (this.neuriteDots) this.neuriteDots.visible = syn;
    if (this.points) this.points.visible = true;
    this.paintAll(this.lastDelta);
  }

  clearTrail() {
    this.actTrail = [];
  }

  pushTrail(state) {
    if (!state || !state.length) return;
    const n = state.length;
    let mx = 1e-8;
    for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(state[i]));
    if (mx < 1e-7) return;
    const snap = new Float32Array(n);
    for (let i = 0; i < n; i++) snap[i] = Math.abs(state[i]) / mx;
    this.actTrail.push(snap);
  }

  neuronAge(mi) {
    const th = 0.12;
    const cur = Math.abs(this.display[mi] || 0) / Math.max(this.actScale, 1e-6);
    if (cur >= th) return 0;
    const trail = this.actTrail;
    for (let k = 1; k <= trail.length; k++) {
      if (trail[trail.length - k][mi] >= th) return k;
    }
    return -1;
  }

  refreshRegionUse() {
    const n = this.classNames.length;
    const sums = new Float64Array(n), cnt = new Int32Array(n);
    for (let i = 0; i < this.display.length; i++) {
      const c = this.modelClass[i];
      sums[c] += Math.abs(this.display[i]);
      cnt[c]++;
    }
    let peak = 1e-12;
    if (!this.regionUse || this.regionUse.length !== n) this.regionUse = new Float32Array(n);
    for (let c = 0; c < n; c++) {
      const m = cnt[c] ? sums[c] / cnt[c] : 0;
      this.regionUse[c] = m;
      if (m > peak) peak = m;
    }
    this.regionPeak = peak;
    if (this.visualL || this.visualR) {
      const vis = Math.max(this.visualL, this.visualR);
      const set = (name, val) => {
        const i = this.classNames.indexOf(name);
        if (i >= 0) {
          this.regionUse[i] = Math.max(this.regionUse[i], val * peak);
          if (this.regionUse[i] > this.regionPeak) this.regionPeak = this.regionUse[i];
        }
      };
      set('ol_intrinsic', vis);
      set('ol_sensory', vis);
      set('visual_centrifugal', vis);
      set('vnc_intrinsic', this.bodyDrive);
      set('vnc_sensory', this.sensoryDrive);
      set('vnc_motor', this.bodyDrive);
    }
  }

  gatherDrives(delta) {
    const vis = this.classNames.indexOf('visual_projection');
    const sens = this.classNames.indexOf('cb_sensory');
    const desc = this.classNames.indexOf('descending_neuron');
    const asc = this.classNames.indexOf('ascending_neuron');
    const inSlot = this.assets.inSlot;
    const src = delta || this.display;
    const scale = delta ? Math.max(this.deltaScale, 1e-6) : Math.max(this.actScale, 1e-6);
    let vL = 0, vR = 0, nL = 0, nR = 0, s = 0, ns = 0, b = 0, nb = 0;
    for (let i = 0; i < this.display.length; i++) {
      const mag = Math.abs(src[i] || 0);
      const c = this.modelClass[i];
      if (inSlot[i] === 0 && c === vis) {
        const v = this.modelVertex[i];
        if (v >= 0 && this.posX[v] < 0) { vL += mag; nL++; }
        else { vR += mag; nR++; }
      }
      if (inSlot[i] === 0 && c === sens) { s += mag; ns++; }
      if (c === desc || c === asc) { b += mag; nb++; }
    }
    this.visualL = Math.min(1, (nL ? vL / nL : 0) / scale * 1.8);
    this.visualR = Math.min(1, (nR ? vR / nR : 0) / scale * 1.8);
    this.sensoryDrive = Math.min(1, (ns ? s / ns : 0) / scale * 1.8);
    this.bodyDrive = Math.min(1, (nb ? b / nb : 0) / scale * 1.8);
  }

  bodyDriveFor(name, v) {
    if (name.startsWith('ol_') || name.startsWith('visual_')) return this.posX[v] < 0 ? this.visualL : this.visualR;
    if (name.startsWith('vnc_') || name.includes('motor') || name.includes('efferent') || name.includes('endocrine') || name === 'ENS') {
      return this.bodyDrive;
    }
    if (name.includes('sensory')) return this.sensoryDrive;
    return 0.22 * Math.max(this.visualL, this.visualR, this.sensoryDrive, this.bodyDrive);
  }

  paintBody(v) {
    this.colorAttr.setXYZ(v, GREY_QUIET[0], GREY_QUIET[1], GREY_QUIET[2]);
    this.sizeAttr.setX(v, 0.35);
    this.actAttr.setX(v, 0.07);
  }

  neuriteTint(name) {
    if (name.startsWith('vnc_') || name.includes('efferent') || name.includes('motor')) return [0.96, 0.93, 0.82];
    if (name.startsWith('ol_') || name.startsWith('visual_')) return [1.0, 0.68, 0.2];
    return [0.95, 0.52, 0.16];
  }

  buildNeurites(pos) {
    const count = this.count;
    const want = 8000;
    const stride = Math.max(1, Math.floor(count / want));
    const segsPer = 18;
    const maxSeg = Math.ceil(count / stride) * segsPer;
    const linePos = new Float32Array(maxSeg * 6);
    const lineCol = new Float32Array(maxSeg * 6);
    const soma = new Int32Array(maxSeg * 2);
    const tint = new Float32Array(maxSeg * 6);
    let s = 0;
    const u = (i) => {
      let x = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b);
      x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
      return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
    };
    const nrm = (x, y, z) => {
      const n = Math.hypot(x, y, z) || 1;
      return [x / n, y / n, z / n];
    };
    for (let v = 0; v < count; v += stride) {
      const name = this.classNames[this.vertexClass[v]] || '';
      const t = this.neuriteTint(name);
      const ox = pos[v * 3], oy = pos[v * 3 + 1], oz = pos[v * 3 + 2];
      const lateral = ox >= 0 ? 1 : -1;
      let bx = 0, by = 0, bz = 0;
      if (name.startsWith('ol_') || name.startsWith('visual_')) { bx = 0.55 * lateral; by = 0.08; bz = 0.2; }
      else if (name.startsWith('vnc_') || name.includes('efferent')) { bx = 0.12 * lateral; by = 0.02; bz = 0.75; }
      else { bx = 0.08 * lateral; by = 0.05; bz = 0.12; }
      const branches = 2 + (u(v + 17) > 0.55 ? 1 : 0);
      const steps = 7 + Math.floor(u(v + 3) * 4);
      const stepLen = name.startsWith('vnc_') ? 0.048 : 0.032;
      for (let b = 0; b < branches; b++) {
        let px = ox, py = oy, pz = oz;
        let [dx, dy, dz] = nrm(
          bx + (u(v * 13 + b * 5 + 1) - 0.5) * 1.4,
          by + (u(v * 13 + b * 5 + 2) - 0.5) * 1.4,
          bz + (u(v * 13 + b * 5 + 3) - 0.5) * 1.4,
        );
        for (let k = 0; k < steps && s < maxSeg; k++) {
          [dx, dy, dz] = nrm(
            dx * 0.78 + bx * 0.14 + (u(v * 31 + b * 19 + k * 3) - 0.5) * 0.7,
            dy * 0.78 + by * 0.14 + (u(v * 31 + b * 19 + k * 3 + 1) - 0.5) * 0.7,
            dz * 0.78 + bz * 0.14 + (u(v * 31 + b * 19 + k * 3 + 2) - 0.5) * 0.7,
          );
          let nx = px + dx * stepLen, ny = py + dy * stepLen, nz = pz + dz * stepLen;
          const rad = Math.hypot(nx, ny, nz);
          if (rad > 1.15) {
            nx *= 1.15 / rad; ny *= 1.15 / rad; nz *= 1.15 / rad;
            [dx, dy, dz] = nrm(-px, -py, -pz);
          }
          const i = s * 6;
          linePos[i] = px; linePos[i + 1] = py; linePos[i + 2] = pz;
          linePos[i + 3] = nx; linePos[i + 4] = ny; linePos[i + 5] = nz;
          tint[i] = tint[i + 3] = t[0];
          tint[i + 1] = tint[i + 4] = t[1];
          tint[i + 2] = tint[i + 5] = t[2];
          soma[s * 2] = soma[s * 2 + 1] = v;
          px = nx; py = ny; pz = nz;
          s++;
        }
      }
    }
    const nVert = s * 2;
    this.arborSoma = soma.subarray(0, nVert);
    this.arborTint = tint.subarray(0, nVert * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(linePos.subarray(0, nVert * 3), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(lineCol.subarray(0, nVert * 3), 3));
    geo.setAttribute('alpha', new THREE.BufferAttribute(new Float32Array(nVert), 1));
    geo.setAttribute('size', new THREE.BufferAttribute(new Float32Array(nVert), 1));
    this.neuriteGeo = geo;
    this.neuriteColor = geo.getAttribute('color');
    this.neuriteAlpha = geo.getAttribute('alpha');
    this.neuriteSize = geo.getAttribute('size');
    this.neuriteMat = new THREE.ShaderMaterial({
      vertexShader: /* glsl */ `
        attribute vec3 color;
        attribute float alpha;
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          vColor = color;
          vAlpha = alpha;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          if (vAlpha < 0.02) discard;
          gl_FragColor = vec4(vColor, vAlpha);
        }`,
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });
    this.neuriteDotMat = new THREE.ShaderMaterial({
      vertexShader: /* glsl */ `
        attribute float size;
        attribute vec3 color;
        attribute float alpha;
        uniform float uScale;
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vColor = color;
          vAlpha = alpha;
          gl_PointSize = size * uScale / -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: frag,
      transparent: true,
      depthWrite: false,
      uniforms: { uScale: this.mat.uniforms.uScale },
    });
    this.neurites = new THREE.LineSegments(geo, this.neuriteMat);
    this.neuriteDots = new THREE.Points(geo, this.neuriteDotMat);
    this.neurites.visible = false;
    this.neuriteDots.visible = false;
    this.scene.add(this.neurites);
    this.scene.add(this.neuriteDots);
  }

  paintNeurites() {
    if (!this.neuriteColor) return;
    const col = this.neuriteColor;
    const al = this.neuriteAlpha;
    const soma = this.arborSoma;
    const tint = this.arborTint;
    const sz = this.neuriteSize;
    const n = soma.length;
    const trail = this.colorMode === 'synapseChange';
    for (let i = 0; i < n; i++) {
      const v = soma[i];
      const mi = this.vertexModel[v];
      if (trail) {
        const age = mi >= 0 ? this.neuronAge(mi) : (this.bodyDriveFor(this.classNames[this.vertexClass[v]] || '', v) >= 0.12 ? 0 : -1);
        if (age < 0) {
          col.setXYZ(i, GREY_QUIET[0], GREY_QUIET[1], GREY_QUIET[2]);
          al.setX(i, 0.05);
          sz.setX(i, 0);
          continue;
        }
        const span = Math.max(this.actTrail.length, 1);
        const lum = Math.pow(1 - age / span, 2.2);
        col.setXYZ(i, 0.35 + 0.65 * lum, 0.18 + 0.74 * lum, 0.05 + 0.5 * lum);
        al.setX(i, 0.12 + 0.88 * lum);
        sz.setX(i, 0);
        continue;
      }
      let a = 0;
      if (mi >= 0) a = Math.min(1, Math.abs(this.display[mi]) / Math.max(this.actScale, 1e-6));
      else a = this.bodyDriveFor(this.classNames[this.vertexClass[v]] || '', v);
      if (a < 0.12) {
        col.setXYZ(i, GREY_QUIET[0], GREY_QUIET[1], GREY_QUIET[2]);
        al.setX(i, 0.05);
        sz.setX(i, 0);
        continue;
      }
      const lit = 0.55 + 0.45 * Math.pow(a, 0.45);
      col.setXYZ(i, Math.min(1, tint[i * 3] * lit + 0.25 * lit), Math.min(1, tint[i * 3 + 1] * lit + 0.15 * lit), tint[i * 3 + 2] * lit);
      al.setX(i, Math.min(1, 0.45 + 0.55 * a));
      sz.setX(i, 2.5 + 14 * a);
    }
    col.needsUpdate = true;
    al.needsUpdate = true;
    sz.needsUpdate = true;
  }

  fitScale(prev, next, delta) {
    let a = 1e-8, d = 1e-8;
    const n = next.length;
    for (let i = 0; i < n; i++) {
      a = Math.max(a, Math.abs(next[i]), Math.abs(prev[i]));
      if (delta) d = Math.max(d, Math.abs(delta[i]));
    }
    this.actScale = a;
    this.deltaScale = d;
  }

  /** colour + size of one model neuron for the current display value */
  paintNeuron(i, x) {
    const v = this.modelVertex[i];
    if (v < 0) return;
    if (this.colorMode === 'synapseChange') {
      const age = this.neuronAge(i);
      if (age < 0) {
        this.colorAttr.setXYZ(v, GREY_QUIET[0], GREY_QUIET[1], GREY_QUIET[2]);
        this.sizeAttr.setX(v, 0.4);
        this.actAttr.setX(v, 0.07);
        return;
      }
      const span = Math.max(this.actTrail.length, 1);
      const lum = Math.pow(1 - age / span, 2.2);
      this.colorAttr.setXYZ(v, 0.35 + 0.65 * lum, 0.18 + 0.74 * lum, 0.05 + 0.5 * lum);
      this.sizeAttr.setX(v, 0.45 + 1.1 * lum);
      this.actAttr.setX(v, 0.4 + 0.6 * lum);
      return;
    }
    const a = Math.min(1, Math.abs(x) / Math.max(this.actScale, 1e-6));
    if (a < 0.06) {
      this.colorAttr.setXYZ(v, GREY_QUIET[0], GREY_QUIET[1], GREY_QUIET[2]);
      this.sizeAttr.setX(v, 0.4);
      this.actAttr.setX(v, 0.08);
      return;
    }
    const bright = 0.35 + 0.65 * Math.pow(a, 0.45);
    let r, g, b;
    if (x >= 0) { r = Math.min(1, bright * 1.15); g = Math.min(1, bright * 0.72); b = bright * 0.12; }
    else { r = bright * 0.12; g = Math.min(1, bright * 0.75); b = Math.min(1, bright * 1.15); }
    this.colorAttr.setXYZ(v, r, g, b);
    this.sizeAttr.setX(v, 2.6 + 12 * a);
    this.actAttr.setX(v, 0.55 + 0.45 * a);
  }

  paintAll(delta = null) {
    this.gatherDrives(delta);
    const syn = this.colorMode === 'synapseChange';
    if (this.neurites) this.neurites.visible = syn;
    if (this.neuriteDots) this.neuriteDots.visible = syn;
    if (this.points) this.points.visible = true;
    if (syn) {
      for (let v = 0; v < this.count; v++) {
        const i = this.vertexModel[v];
        if (i >= 0) this.paintNeuron(i, this.display[i]);
        else this.paintBody(v);
      }
      this.colorAttr.needsUpdate = true;
      this.sizeAttr.needsUpdate = true;
      this.actAttr.needsUpdate = true;
      this.paintNeurites();
      return;
    }
    for (let v = 0; v < this.count; v++) {
      const i = this.vertexModel[v];
      if (i >= 0) this.paintNeuron(i, this.display[i]);
      else this.paintBody(v);
    }
    this.colorAttr.needsUpdate = true;
    this.sizeAttr.needsUpdate = true;
    this.actAttr.needsUpdate = true;
  }

  /** Flash the input neurons of a slot with the slot colour (used at the start of each step) */
  flashInputs(strength) {
    const { inSlot } = this.assets;
    for (let i = 0; i < inSlot.length; i++) {
      const s = inSlot[i];
      if (s === 255) continue;
      const v = this.modelVertex[i];
      if (v < 0) continue;
      const c = SLOT_COLORS[s];
      const w = strength * (1 - s / 8) ;
      const x = this.display[i];
      const a = Math.min(1, Math.abs(x) / Math.max(this.actScale, 1e-6));
      const base = 0.08 + 0.92 * Math.pow(a, 0.6);
      this.colorAttr.setXYZ(v, base * 0.5 + c[0] * w, base * 0.3 + c[1] * w, base * 0.1 + c[2] * w);
      this.sizeAttr.setX(v, 1.4 + 3.2 * a + 3 * w);
      this.actAttr.setX(v, Math.max(this.actAttr.getX(v), w));
    }
  }

  /** Visual-only propagation: prev -> next over durationMs, ordered by hop distance. */
  showStep(prev, next, durationMs) {
    if (this.anim) this.finishAnim();
    prev = copyF32(prev);
    next = copyF32(next);
    const delta = new Float32Array(next.length);
    for (let i = 0; i < next.length; i++) delta[i] = next[i] - prev[i];
    this.fitScale(prev, next, delta);
    if (prev !== next) {
      let same = prev.length === next.length;
      if (same) for (let i = 0; i < prev.length; i++) if (prev[i] !== next[i]) { same = false; break; }
      if (!same) this.pushTrail(prev);
    }
    if (durationMs <= 0) {
      this.display.set(next);
      this.lastDelta = delta;
      this.paintAll(delta);
      return Promise.resolve();
    }
    const from = Float32Array.from(this.display);
    return new Promise((resolve) => {
      this.anim = { from, next, delta, t0: performance.now(), dur: durationMs, resolve };
    });
  }

  finishAnim() {
    const a = this.anim;
    this.anim = null;
    this.display.set(a.next);
    this.lastDelta = a.delta;
    this.paintAll(a.delta);
    a.resolve();
  }

  tickAnim(now) {
    const a = this.anim;
    const u = (now - a.t0) / a.dur;
    if (u >= 1) return this.finishAnim();
    // timeline: 0-0.25 inputs flash; hop h sweeps during [0.15+0.2h, 0.45+0.2h]
    const hop = this.hop;
    for (let i = 0; i < this.display.length; i++) {
      const h = Math.min(hop[i], 3);
      const s = 0.15 + 0.2 * h, e = s + 0.3;
      const w = u <= s ? 0 : u >= e ? 1 : (u - s) / (e - s);
      const ws = w * w * (3 - 2 * w);
      this.display[i] = a.from[i] + (a.next[i] - a.from[i]) * ws;
    }
    this.paintAll(null);
    const flash = u < 0.3 ? Math.sin((u / 0.3) * Math.PI) : 0;
    if (flash > 0) this.flashInputs(flash * 0.9);
    this.colorAttr.needsUpdate = true;
    this.sizeAttr.needsUpdate = true;
  }

  animate(now) {
    this.raf = requestAnimationFrame(this.animate);
    if (this.anim) this.tickAnim(now);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.geo.dispose();
    this.mat.dispose();
    this.neuriteGeo?.dispose();
    this.neuriteMat?.dispose();
    this.neuriteDotMat?.dispose();
    this.renderer.dispose();
  }
}
