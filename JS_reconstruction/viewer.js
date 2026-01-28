// viewer_multi_synced.js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * Multi-dataset 3D marker viewer for synced datasets.
 * Each JSON: { positions: [F][M][3] }
 */
export class MarkerViewer {
  constructor(opts) {
    const defaults = {
      sceneEl: '#scene',
      hudEl: '#hud',
      datasets: null,
      scale: 1.0,
      fps: 30,
      loop: true,
      pointRadiusWorld: 20,
      onFrameChange: null,
    };
    this.opts = Object.assign({}, defaults, opts || {});

    // fallback single-dataset mode
    if (!this.opts.datasets) {
      this.opts.datasets = [{
        label: 'Dataset',
        dataUrl: this.opts.dataUrl || 'freemocap_data.json',
        color: this.opts.pointColor || 0x2f6efc,
        scale: this.opts.scale,
        visible: true,
      }];
    }

    // DOM
    this.container = this._elt(this.opts.sceneEl);
    this.hud = this._elt(this.opts.hudEl);

    // state
    this.k = 0;
    this.playing = false;
    this.lastT = 0;
    this.F = 0;

    // datasets array
    this.datasets = this.opts.datasets.map(ds => ({
      label: ds.label ?? 'Dataset',
      color: ds.color ?? 0x888888,
      scale: ds.scale ?? this.opts.scale,
      visible: ds.visible ?? true,
      dataUrl: ds.dataUrl,
      positions: null,
      F: 0,
      M: 0,
      mesh: null,
    }));

    this._initThree();
    this._initUI();
    this._loadAllData();
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  // helpers
  _elt(sel) {
    const el = typeof sel === 'string' ? document.querySelector(sel) : sel;
    if (!el) throw new Error(`Element not found: ${sel}`);
    return el;
  }

  _initThree() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(3000, 2500, 3500);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.AmbientLight(0xffffff, 1));

    this.grid = new THREE.GridHelper(4000, 40)
    this.grid.rotation.x = Math.PI / 2;
    this.grid.material.opacity = 0.50;
    this.grid.material.transparent = true;
    this.scene.add(this.grid);

    const resize = () => {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || (window.innerHeight - 80);
      this.renderer.setSize(w, h);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(this.container);
    resize();

    window.addEventListener('keydown', e => {
      if (e.code === 'Space') {
        e.preventDefault();
        this.toggle();
      }
    });
  }

  _initUI() {
    this.hud.style.display = 'flex';
    this.hud.style.flexWrap = 'wrap';
    this.hud.style.gap = '8px';
    this.hud.style.alignItems = 'center';

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '0';
    slider.step = '1';
    slider.value = '0';
    slider.style.width = '300px';

    const label = document.createElement('span');
    label.textContent = '0';

    const playBtn = document.createElement('button');
    playBtn.textContent = 'Play';

    const fpsWrap = document.createElement('label');
    fpsWrap.textContent = 'FPS ';
    fpsWrap.style.display = 'flex';
    fpsWrap.style.alignItems = 'center';
    fpsWrap.style.gap = '4px';
    const fpsInput = document.createElement('input');
    fpsInput.type = 'number';
    fpsInput.value = String(this.opts.fps);
    fpsInput.min = '1';
    fpsInput.max = '120';
    fpsInput.step = '1';
    fpsInput.style.width = '60px';
    fpsWrap.appendChild(fpsInput);

    const loopWrap = document.createElement('label');
    loopWrap.textContent = 'Loop ';
    loopWrap.style.display = 'flex';
    loopWrap.style.alignItems = 'center';
    loopWrap.style.gap = '4px';
    const loopCb = document.createElement('input');
    loopCb.type = 'checkbox';
    loopCb.checked = !!this.opts.loop;
    loopWrap.appendChild(loopCb);

    const legend = document.createElement('div');
    legend.style.display = 'flex';
    legend.style.flexWrap = 'wrap';
    legend.style.gap = '8px';
    legend.style.alignItems = 'center';

    this.datasets.forEach(ds => {
      const row = document.createElement('label');
      row.style.display = 'inline-flex';
      row.style.alignItems = 'center';
      row.style.gap = '6px';

      const swatch = document.createElement('span');
      swatch.style.width = '14px';
      swatch.style.height = '14px';
      swatch.style.borderRadius = '999px';
      swatch.style.border = '1px solid #0002';
      swatch.style.background = `#${ds.color.toString(16).padStart(6, '0')}`;

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = ds.visible;
    cb.addEventListener('change', () => {
      ds.visible = cb.checked;
      if (ds.mesh) ds.mesh.visible = ds.visible;
    });

    const name = document.createElement('span');
    name.textContent = ds.label;
    row.append(cb, swatch, name);
    legend.appendChild(row);
    });

    this.hud.append(slider, label, playBtn, fpsWrap, loopWrap, legend);

    slider.addEventListener('input', e => this.setFrame(Number(e.target.value)));
    playBtn.addEventListener('click', () => this.toggle());
    loopCb.addEventListener('change', () => (this.opts.loop = loopCb.checked));

    this.slider = slider;
    this.label = label;
    this.playBtn = playBtn;
    this.fpsInput = fpsInput;
  }

  async _loadAllData() {
    const results = await Promise.all(
      this.datasets.map(async ds => {
        const res = await fetch(ds.dataUrl);
        const json = await res.json();
        ds.positions = json.positions;
        ds.F = ds.positions.length;
        ds.M = ds.positions[0]?.length ?? 0;
        return ds;
      })
    );

    const F0 = results[0].F;
    const mismatch = results.find(d => d.F !== F0);
    if (mismatch)
      throw new Error('Datasets must have equal frame counts for synced playback.');
    this.F = F0;
    this.slider.max = String(this.F - 1);


    results.forEach(ds => {
      const geom = new THREE.SphereGeometry(this.opts.pointRadiusWorld * ds.scale);
      const mat = new THREE.MeshStandardMaterial({ color: ds.color, emissive: ds.color});
      const mesh = new THREE.InstancedMesh(geom, mat, ds.M);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.visible = ds.visible;
      this.scene.add(mesh);
      ds.mesh = mesh;
    });

    this.setFrame(0);
    const S0 = this.datasets[0].scale;
    this.camera.position.set(0, -2000 * S0, 1500 * S0);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  setFrame(k) {
    if (!this.F) return;
    this.k = Math.max(0, Math.min(this.F - 1, k));

    const tmp = new THREE.Matrix4();
    this.datasets.forEach(ds => {
      const pts = ds.positions[this.k];
      for (let i = 0; i < ds.M; i++) {
        const [x, y, z] = pts[i];
        tmp.makeTranslation(x * ds.scale, y * ds.scale, z * ds.scale);
        ds.mesh.setMatrixAt(i, tmp);
      }
      ds.mesh.instanceMatrix.needsUpdate = true;
    });

    this.slider.value = String(this.k);
    this.label.textContent = String(this.k);
    if (typeof this.opts.onFrameChange === 'function')
      this.opts.onFrameChange(this.k);
  }

  play() {
    if (!this.playing) {
      this.playing = true;
      this.lastT = 0;
    }
  }
  pause() {
    this.playing = false;
  }
  toggle() {
    this.playing = !this.playing;
    this.playBtn.textContent = this.playing ? 'Pause' : 'Play';
    this.lastT = 0;
  }

  _animate(t) {
    if (this.playing && this.F > 0) {
      const fps = Math.max(1, Math.min(120, Number(this.fpsInput.value) || this.opts.fps));
      const frameTime = 1000 / fps;
      if (!this.lastT) this.lastT = t;
      if (t - this.lastT >= frameTime) {
        this.lastT += frameTime;
        let k = this.k + 1;
        if (k >= this.F) {
          if (this.opts.loop) k = 0;
          else {
            this.pause();
            this.playBtn.textContent = 'Play';
          }
        }
        this.setFrame(k);
      }
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this._animate);
  }

  dispose() {
    this.pause();
    this.datasets.forEach(ds => {
      if (ds.mesh) {
        this.scene.remove(ds.mesh);
        ds.mesh.geometry.dispose();
        ds.mesh.material.dispose();
      }
    });
    this.renderer.dispose();
    this.container.innerHTML = '';
    this.hud.innerHTML = '';
  }
}
