// plots.js
// Marker selector uses ONLY dataset[0] (FreeMoCap) landmark names.
// Qualisys (dataset[1]) is shown only if it has a matching name.
// Public API:
//   initWindowedXYZSelectable(datasets, { windowHalf?, defaultMarkerName? })
//   updateXYZWindow(k)

let _divX = null, _divY = null, _divZ = null;
let _F = 0;
let _windowHalf = 150;

let _datasets = []; // [{label, positions, color, landmarks?, nameToIndex:Map}]
let _fmcNames = []; // names from dataset[0] only (or "Marker i" fallback)
let _activeName = null;

const AX = ['X','Y','Z'];

export function initWindowedXYZSelectable(inDatasets, opts = {}) {
  _divX = document.getElementById('plot_x');
  _divY = document.getElementById('plot_y');
  _divZ = document.getElementById('plot_z');
  if (!_divX || !_divY || !_divZ) { console.error('plots: missing plot divs'); return; }

  if (!Array.isArray(inDatasets) || inDatasets.length === 0) {
    console.warn('plots: no datasets'); return;
  }

  _windowHalf = Math.max(1, Number(opts.windowHalf ?? 150));

  // Build internal ds objects with name→index map
  _datasets = inDatasets.map((ds, i) => {
    const nameToIndex = new Map();
    if (Array.isArray(ds.landmarks)) {
      ds.landmarks.forEach((nm, idx) => nameToIndex.set(String(nm), idx));
    }

    let color = ds.color;
    if (typeof color === 'number') color = '#' + color.toString(16).padStart(6, '0');

    return {
      label: ds.label ?? `Dataset ${i+1}`,
      positions: ds.positions,
      color,
      landmarks: Array.isArray(ds.landmarks) ? ds.landmarks.slice() : null,
      nameToIndex,
    };
  });

  _F = Math.max(..._datasets.map(ds => Array.isArray(ds.positions) ? ds.positions.length : 0));
  if (!_F || _F < 2) { console.warn('plots: datasets have no frames'); return; }
  const nameSet = new Set();

  // 1) Prefer explicit landmark names if provided
  _datasets.forEach(ds => {
    if (Array.isArray(ds.landmarks) && ds.landmarks.length) {
      ds.landmarks.forEach(nm => nameSet.add(String(nm)));
    }
  });

  // 2) If nobody had landmark names, fall back to Marker i up to max M across datasets
  if (nameSet.size === 0) {
    const maxM = Math.max(..._datasets.map(ds => ds.positions?.[0]?.length ?? 0));
    for (let i = 0; i < maxM; i++) nameSet.add(`Marker ${i}`);

    // and map those marker labels to indices for each dataset (by index)
    _datasets.forEach(ds => {
      const M = ds.positions?.[0]?.length ?? 0;
      for (let j = 0; j < M; j++) ds.nameToIndex.set(`Marker ${j}`, j);
    });
  }

  const ref =
    _datasets.find(d => (d.label || '').toLowerCase().includes('mediapipe') && Array.isArray(d.landmarks) && d.landmarks.length) ||
    _datasets.find(d => Array.isArray(d.landmarks) && d.landmarks.length);

  const ordered = [];
  const used = new Set();

  if (ref && Array.isArray(ref.landmarks)) {
    ref.landmarks.forEach(nm => {
      nm = String(nm);
      if (nameSet.has(nm) && !used.has(nm)) { ordered.push(nm); used.add(nm); }
    });
  }

  // Append any remaining names (these are "extra" or differently-named markers)
  const leftovers = Array.from(nameSet).filter(nm => !used.has(nm));
  leftovers.sort((a, b) => a.localeCompare(b));

  _fmcNames = ordered.concat(leftovers);

  _activeName = (opts.defaultMarkerName && _fmcNames.includes(opts.defaultMarkerName))
    ? opts.defaultMarkerName
    : _fmcNames[0];
    
  buildSelector();     // dropdown from FMC names only
  drawAll();           // three plots
  updateXYZWindow(0);  // center start
  attachSharedRelayout();
}

function idxFor(ds, name) {
  if (!ds || !name) return null;
  if (ds.nameToIndex.size === 0) return null;
  const hit = ds.nameToIndex.get(name);
  return (hit === 0 || typeof hit === 'number') ? hit : null;
}

function seriesFor(ds, name, axisIdx) {
  const j = idxFor(ds, name);
  if (j == null) return { y: Array(_F).fill(null), visible: false };
  const pos = ds.positions;
  const y = new Array(_F);
  for (let k = 0; k < _F; k++) {
    const row = pos[k]; const p = row && row[j];
    y[k] = (p && typeof p[axisIdx] === 'number') ? p[axisIdx] : null;
  }
  const any = y.some(v => v !== null);
  return { y, visible: any };
}

function makeTraces(axisIdx) {
  const x = Array.from({ length: _F }, (_, i) => i);
  return _datasets.map((ds, i) => {
    const { y, visible } = seriesFor(ds, _activeName, axisIdx);
    return {
      name: ds.label,
      x, y,
      visible: visible ? true : 'legendonly',
      mode: 'lines',
      line: { width: 2, color: ds.color || ['#2f6efc', '#d62728'][i % 2], simplify: false },
      hovertemplate: `${AX[axisIdx]}=%{y:.3f}<br>frame=%{x}<extra>${ds.label}</extra>`
    };
  });
}

function layoutFor(axisLabel, showLegend) {
  return {
    margin: { l: 40, r: 10, t: 20, b: 30 },
    title: { text: `${axisLabel} (${_activeName})`, font: { size: 14 } },
    xaxis: { title: 'Frame', range: [-_windowHalf, _windowHalf], showgrid: false },
    yaxis: { title: `${axisLabel} (mm)`, automargin: true, showgrid: true, zeroline: false },
    showlegend: !!showLegend,
    legend: { orientation: 'h', x: 1, y: 1.1, xanchor: 'right', yanchor: 'bottom',
              bgcolor: 'rgba(255,255,255,0.8)', font: { size: 12 } },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    shapes: [{ type:'line', xref:'paper', yref:'paper', x0:0.5, x1:0.5, y0:0, y1:1,
               line:{ color:'#000', width: 2 } }]
  };
}

function drawAll() {
  const cfg = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d','lasso2d'] };
  Plotly.newPlot(_divX, makeTraces(0), layoutFor('X', true),  cfg);  // legend only on X
  Plotly.newPlot(_divY, makeTraces(1), layoutFor('Y', false), cfg);
  Plotly.newPlot(_divZ, makeTraces(2), layoutFor('Z', false), cfg);
}

export function updateXYZWindow(k) {
  const kk = Math.max(0, Math.min(_F - 1, Number(k) || 0));
  const range = [kk - _windowHalf, kk + _windowHalf];
  Plotly.relayout(_divX, { 'xaxis.range': range });
  Plotly.relayout(_divY, { 'xaxis.range': range });
  Plotly.relayout(_divZ, { 'xaxis.range': range });
}

function setMarkerName(name) {
  if (!name || name === _activeName) return;
  _activeName = name;

  [0,1,2].forEach(axisIdx => {
    _datasets.forEach((ds, i) => {
      const { y, visible } = seriesFor(ds, _activeName, axisIdx);
      Plotly.restyle(getDiv(axisIdx), { y: [y], visible: visible ? true : 'legendonly' }, [i]);
    });
    Plotly.relayout(getDiv(axisIdx), { 'title.text': `${AX[axisIdx]} (${_activeName})` });
  });

  const sel = document.getElementById('markerSelect');
  if (sel && sel.value !== _activeName) sel.value = _activeName;
}

function getDiv(axisIdx) { return axisIdx === 0 ? _divX : axisIdx === 1 ? _divY : _divZ; }

function buildSelector() {
  const host = document.getElementById('marker-select-wrap');
  if (!host) return;
  host.innerHTML = '';

  const label = document.createElement('label');
  label.textContent = 'Marker: ';
  label.style.marginRight = '6px';

  const sel = document.createElement('select');
  sel.id = 'markerSelect';
  _fmcNames.forEach(nm => {
    const opt = document.createElement('option');
    opt.value = nm; opt.textContent = nm;
    if (nm === _activeName) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => setMarkerName(sel.value));
  host.append(label, sel);
}

function attachSharedRelayout() {
  const sync = (src, e) => {
    if ('xaxis.range[0]' in e && 'xaxis.range[1]' in e) {
      const range = [e['xaxis.range[0]'], e['xaxis.range[1]']];
      [_divX,_divY,_divZ].forEach(div => { if (div !== src) Plotly.relayout(div, { 'xaxis.range': range }); });
    }
  };
  [_divX,_divY,_divZ].forEach(div => div.on('plotly_relayout', e => sync(div, e)));
}
