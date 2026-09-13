import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Procedural anatomical reconstruction, not a scan. Every visible part is geometry.
export class FlyHeadView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0xf7f8fa);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(33, 1, 0.05, 30);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enablePan = false;
    this.controls.minDistance = 2.8;
    this.controls.maxDistance = 10;
    this.controls.addEventListener('change', () => this.render());
    this.scene.add(new THREE.HemisphereLight(0xe5f1ff, 0x60452c, 2.1));
    for (const [color, power, position] of [[0xffead4, 3.6, [-3, 4, 5]], [0xe2eeff, 2.5, [3, 1, 2]], [0xffffff, 3, [0, 3, -3]]]) {
      const light = new THREE.DirectionalLight(color, power);
      light.position.set(...position);
      this.scene.add(light);
    }
    this.head = new THREE.Group();
    this.scene.add(this.head);
    this.eyes = [];
    this.buildHead();
    this.front();
    this.resize();
  }

  ellipsoid(position, scale, material, rotation = 0) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 28), material);
    mesh.position.set(...position);
    mesh.scale.set(...scale);
    mesh.rotation.z = rotation;
    this.head.add(mesh);
    return mesh;
  }

  hair(points, radius = 0.006, material = this.bristle) {
    const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)));
    // Tapered mesh: bristles end in a fine point rather than a blunt tube.
    const frames = curve.computeFrenetFrames(8, false);
    const positions = [], indices = [];
    for (let j = 0; j <= 8; j++) {
      const p = curve.getPoint(j / 8);
      const r = radius * (1 - j / 8) + 0.0003;
      for (let k = 0; k < 5; k++) {
        const a = k * Math.PI * 2 / 5;
        const v = p.clone().addScaledVector(frames.normals[j], Math.cos(a) * r).addScaledVector(frames.binormals[j], Math.sin(a) * r);
        positions.push(v.x, v.y, v.z);
        if (j < 8) {
          const n = j * 5 + k, next = j * 5 + (k + 1) % 5;
          indices.push(n, next, n + 5, next, next + 5, n + 5);
        }
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    this.head.add(new THREE.Mesh(geometry, material));
  }

  buildHead() {
    const grain = new Uint8Array(128 * 128 * 4);
    let seed = 71;
    for (let i = 0; i < grain.length; i += 4) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      grain[i] = grain[i + 1] = grain[i + 2] = seed >>> 24;
      grain[i + 3] = 255;
    }
    this.cuticleTexture = new THREE.DataTexture(grain, 128, 128);
    this.cuticleTexture.wrapS = this.cuticleTexture.wrapT = THREE.RepeatWrapping;
    this.cuticleTexture.repeat.set(3, 3);
    this.cuticleTexture.needsUpdate = true;
    const cuticle = new THREE.MeshStandardMaterial({ color: 0x9c723d, roughness: 0.76, bumpMap: this.cuticleTexture, bumpScale: 0.013 });
    const face = cuticle;
    const antenna = new THREE.MeshStandardMaterial({ color: 0x805b32, roughness: 0.85, bumpMap: this.cuticleTexture, bumpScale: 0.008 });
    this.bristle = new THREE.MeshStandardMaterial({ color: 0x30251c, roughness: 0.7 });
    const ocellus = new THREE.MeshPhysicalMaterial({ color: 0x4f1d10, roughness: 0.17, clearcoat: 0.8 });
    this.ellipsoid([0, 0.05, 0], [0.99, 0.99, 0.62], cuticle);
    this.ellipsoid([0, -0.16, 0.32], [0.58, 0.73, 0.33], face);
    this.ellipsoid([0, -0.2, 0.64], [0.11, 0.4, 0.08], face);
    // Ocellar triangle on the dorsal frons.
    for (const [x, y, z] of [[0, 0.84, 0.37], [-0.14, 0.75, 0.46], [0.14, 0.75, 0.46]]) {
      this.ellipsoid([x, y, z], [0.065, 0.056, 0.041], ocellus);
    }
    for (const sign of [-1, 1]) {
      this.buildEye(sign);
      this.ellipsoid([sign * 0.19, -0.08, 0.68], [0.11, 0.12, 0.10], cuticle);
      this.ellipsoid([sign * 0.23, -0.22, 0.74], [0.115, 0.13, 0.11], face);
      this.ellipsoid([sign * 0.27, -0.39, 0.79], [0.12, 0.23, 0.115], antenna, sign * -0.2);
      // Feathered arista, attached to the third antennal segment.
      this.hair([[sign * 0.29, -0.32, 0.87], [sign * 0.64, -0.03, 0.87], [sign * 1.01, 0.14, 0.73]], 0.009);
      for (let i = 0; i < 7; i++) {
        const x = 0.38 + i * 0.077, y = -0.25 + i * 0.045;
        for (const direction of [-1, 1]) {
          this.hair([[sign * x, y, 0.88], [sign * (x + 0.06), y + direction * 0.10, 0.89], [sign * (x + 0.075), y + direction * (0.19 - i * 0.013), 0.83]], 0.003);
        }
      }
      this.ellipsoid([sign * 0.18, -0.81, 0.66], [0.095, 0.20, 0.09], antenna, -sign * 0.24);
      // Orbital and vertical bristles, in paired rows on the head capsule.
      for (let i = 0; i < 8; i++) {
        const x = sign * (0.39 + 0.04 * Math.sin(i));
        const y = 0.72 - i * 0.16;
        const z = 0.5 + 0.09 * Math.sin(i * 0.5);
        this.hair([[x, y, z], [x + sign * 0.10, y + 0.1, z + 0.13], [x + sign * 0.18, y + 0.27, z + 0.18]], i < 3 ? 0.009 : 0.005);
      }
      for (let i = 0; i < 4; i++) {
        const x = sign * (0.16 + i * 0.17);
        this.hair([[x, 0.87 - i * 0.06, 0.25], [x + sign * 0.08, 1.08, 0.23], [x + sign * 0.16, 1.27 - i * 0.03, 0.14]], 0.008);
      }
    }
    this.ellipsoid([0, -0.91, 0.42], [0.16, 0.26, 0.17], antenna);
    for (const s of [-1, 1]) this.ellipsoid([s * 0.085, -1.07, 0.49], [0.09, 0.12, 0.14], face);
    // Fine cuticular hairs, deterministic surface samples, fully three-dimensional.
    for (let i = 0; i < 150; i++) {
      const y = -0.7 + ((i * 0.618034) % 1) * 1.5;
      const x = (((i * 0.754877) % 1) - 0.5) * 0.72;
      const z = Math.sqrt(Math.max(0.02, 1 - x * x - y * y)) * 0.63;
      this.hair([[x, y, z], [x + 0.006, y + 0.035, z + 0.028], [x + 0.009, y + 0.055, z + 0.04]], 0.0017);
    }
  }

  buildEye(sign) {
    const center = new THREE.Vector3(sign * 0.82, 0.08, 0.025);
    const radii = new THREE.Vector3(0.55, 0.83, 0.57);
    const orientation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(sign * 0.61, 0, 0.79).normalize());
    const material = new THREE.MeshPhysicalMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.34, clearcoat: 0.65, clearcoatRoughness: 0.22 });
    const shellMaterial = new THREE.MeshStandardMaterial({ color: 0x400d08, roughness: 0.48 });
    this.ellipsoid(center.toArray(), radii.toArray(), shellMaterial);
    const positions = [], indices = [], colors = [], facets = [];
    const base = new THREE.Color();
    const radius = 0.075;
    const point = (u, v, lift = 0) => {
      const angle = Math.hypot(u, v), scale = angle ? Math.sin(angle) / angle : 1;
      const d = new THREE.Vector3(u * scale, v * scale, Math.cos(angle)).applyQuaternion(orientation);
      const p = d.clone().multiply(radii);
      const normal = d.clone().divide(radii).normalize();
      return p.addScaledVector(normal, lift).add(center);
    };
    for (let q = -17; q <= 17; q++) for (let r = -19; r <= 19; r++) {
      const u = 1.5 * radius * q;
      const v = Math.sqrt(3) * radius * (r + q / 2);
      if (Math.hypot(u, v) > 1.82) continue;
      const start = positions.length / 3;
      const shade = 0.92 + Math.sin(q * 32 + r * 73) * 0.08;
      base.setRGB(0.36 * shade, 0.036 * shade, 0.012 * shade);
      const push = p => { positions.push(p.x, p.y, p.z); colors.push(base.r, base.g, base.b); };
      push(point(u, v, 0.012));
      for (const ring of [0.53, 0.97]) for (let k = 0; k < 6; k++) {
        const a = k * Math.PI / 3;
        push(point(u + Math.cos(a) * radius * ring, v + Math.sin(a) * radius * ring, ring < 0.6 ? 0.009 : 0.0015));
      }
      for (let k = 0; k < 6; k++) {
        const a = start + 1 + k, b = start + 1 + (k + 1) % 6;
        indices.push(start, a, b, a, a + 6, b + 6, a, b + 6, b);
      }
      facets.push({ start, shade });
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const eye = new THREE.Mesh(geometry, material);
    this.head.add(eye);
    this.eyes.push({ mesh: eye, facets, side: sign < 0 ? 'R' : 'L' });
  }

  setActivity(activity, enabled) {
    const color = new THREE.Color();
    for (const eye of this.eyes) {
      const side = activity.sides.find(s => s.side === eye.side);
      const attr = eye.mesh.geometry.getAttribute('color');
      eye.facets.forEach((facet, i) => {
        const value = side?.bins[Math.floor(i * side.bins.length / eye.facets.length)] || 0;
        if (enabled && value && activity.peak) {
          const strength = Math.sqrt(Math.min(1, Math.abs(value) / activity.peak));
          color.setHSL(value > 0 ? 0.105 : 0.55, 0.94, 0.12 + strength * 0.19);
        }
        else color.setRGB(0.36 * facet.shade, 0.036 * facet.shade, 0.012 * facet.shade);
        for (let j = 0; j < 13; j++) attr.setXYZ(facet.start + j, color.r, color.g, color.b);
      });
      attr.needsUpdate = true;
    }
    this.render();
  }

  front() {
    this.camera.position.set(0, 0.18, 6.3);
    this.controls.target.set(0, 0.03, 0);
    this.controls.update();
    this.render();
  }
  turn(amount) {
    const offset = this.camera.position.clone().sub(this.controls.target);
    offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), amount);
    this.camera.position.copy(this.controls.target).add(offset);
    this.controls.update();
  }
  resize() {
    const width = this.canvas.clientWidth, height = this.canvas.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.render();
  }
  render() { this.renderer.render(this.scene, this.camera); }
  dispose() {
    this.controls.dispose();
    const materials = new Set();
    this.scene.traverse(obj => {
      obj.geometry?.dispose();
      if (obj.material) materials.add(obj.material);
    });
    materials.forEach(material => material.dispose());
    this.cuticleTexture.dispose();
    this.renderer.dispose();
  }
}
