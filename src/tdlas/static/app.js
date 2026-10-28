/**
 * TDLAS Methane Spectroscopy Interactive Cockpit
 * Pure client-side JavaScript with zero external CDN dependencies.
 * Implements SVG chart rendering, real-time REST API interaction, and preset switching.
 */

const STATE = {
  x_ppm: 50.0,
  T_K: 296.0,
  P_atm: 1.0,
  noise_std_v: 0.0015,
  etalon_amplitude: 0.005,
  activeTab: 'pane-spectroscopy'
};

const PRESETS = {
  nominal: { x_ppm: 50.0, T_K: 296.0, P_atm: 1.0, noise_std_v: 0.0015, etalon_amplitude: 0.005 },
  detection_limit: { x_ppm: 1.0, T_K: 296.0, P_atm: 1.0, noise_std_v: 0.0035, etalon_amplitude: 0.002 },
  etalon_interference: { x_ppm: 40.0, T_K: 296.0, P_atm: 1.0, noise_std_v: 0.001, etalon_amplitude: 0.035 },
  high_tp: { x_ppm: 60.0, T_K: 350.0, P_atm: 1.5, noise_std_v: 0.002, etalon_amplitude: 0.005 }
};

document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  fetchActiveTabData();
});

function initEventListeners() {
  // Preset buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      const target = e.currentTarget;
      target.classList.add('active');
      const presetKey = target.dataset.preset;
      if (PRESETS[presetKey]) {
        applyPreset(PRESETS[presetKey]);
      }
    });
  });

  // Sliders
  bindSlider('slider-concentration', 'val-concentration', (v) => { STATE.x_ppm = parseFloat(v); return `${parseFloat(v).toFixed(1)} ppm`; });
  bindSlider('slider-temperature', 'val-temperature', (v) => { STATE.T_K = parseFloat(v); return `${parseFloat(v).toFixed(1)} K`; });
  bindSlider('slider-pressure', 'val-pressure', (v) => { STATE.P_atm = parseFloat(v); return `${parseFloat(v).toFixed(2)} atm`; });
  bindSlider('slider-noise', 'val-noise', (v) => { STATE.noise_std_v = parseFloat(v); return `${(parseFloat(v) * 1000).toFixed(1)} mV`; });
  bindSlider('slider-etalon', 'val-etalon', (v) => { STATE.etalon_amplitude = parseFloat(v); return `${(parseFloat(v) * 100).toFixed(1)} %`; });

  // Navigation Tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      
      const target = e.currentTarget;
      target.classList.add('active');
      const paneId = target.dataset.tab;
      document.getElementById(paneId).classList.add('active');
      STATE.activeTab = paneId;
      fetchActiveTabData();
    });
  });

  // Action Buttons
  document.getElementById('btn-run-simulation').addEventListener('click', () => {
    fetchActiveTabData();
  });

  document.getElementById('btn-reset-params').addEventListener('click', () => {
    applyPreset(PRESETS.nominal);
  });

  document.getElementById('btn-export-figures').addEventListener('click', () => {
    const btn = document.getElementById('btn-export-figures');
    const origHtml = btn.innerHTML;
    btn.innerHTML = '<span>Exporting...</span>';
    setTimeout(() => {
      alert('Publication Figures (300 DPI PNG & vector SVG) exported to docs/figures/!\nRun `python scripts/export_figures.py` to regenerate all files.');
      btn.innerHTML = origHtml;
    }, 400);
  });
}

function bindSlider(sliderId, labelId, formatFn) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);
  slider.addEventListener('input', (e) => {
    label.textContent = formatFn(e.target.value);
  });
  slider.addEventListener('change', () => {
    fetchActiveTabData();
  });
}

function applyPreset(preset) {
  STATE.x_ppm = preset.x_ppm;
  STATE.T_K = preset.T_K;
  STATE.P_atm = preset.P_atm;
  STATE.noise_std_v = preset.noise_std_v;
  STATE.etalon_amplitude = preset.etalon_amplitude;

  document.getElementById('slider-concentration').value = STATE.x_ppm;
  document.getElementById('val-concentration').textContent = `${STATE.x_ppm.toFixed(1)} ppm`;

  document.getElementById('slider-temperature').value = STATE.T_K;
  document.getElementById('val-temperature').textContent = `${STATE.T_K.toFixed(1)} K`;

  document.getElementById('slider-pressure').value = STATE.P_atm;
  document.getElementById('val-pressure').textContent = `${STATE.P_atm.toFixed(2)} atm`;

  document.getElementById('slider-noise').value = STATE.noise_std_v;
  document.getElementById('val-noise').textContent = `${(STATE.noise_std_v * 1000).toFixed(1)} mV`;

  document.getElementById('slider-etalon').value = STATE.etalon_amplitude;
  document.getElementById('val-etalon').textContent = `${(STATE.etalon_amplitude * 100).toFixed(1)} %`;

  fetchActiveTabData();
}

function fetchActiveTabData() {
  if (STATE.activeTab === 'pane-spectroscopy') {
    fetchSpectroscopy();
  } else if (STATE.activeTab === 'pane-comparison') {
    fetchComparison();
  } else if (STATE.activeTab === 'pane-allan') {
    fetchAllan();
  } else if (STATE.activeTab === 'pane-lockin') {
    fetchLockin();
  }
}

// ==========================================
// Tab 1: Spectroscopy & Voigt Fit
// ==========================================
async function fetchSpectroscopy() {
  try {
    const payload = JSON.stringify({
      x_ppm: STATE.x_ppm,
      T_K: STATE.T_K,
      P_atm: STATE.P_atm,
      noise_std_v: STATE.noise_std_v,
      etalon_amplitude: STATE.etalon_amplitude
    });

    // 1. Simulate scan
    const simResp = await fetch('/api/simulate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload });
    const simData = await simResp.json();

    // 2. Run Levenberg-Marquardt fit
    const fitResp = await fetch('/api/fit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload });
    const fitData = await fitResp.json();

    // Update Ribbon Metrics
    document.getElementById('spec-fitted-x').textContent = `${fitData.fitted_x_ppm.toFixed(2)} ppm`;
    document.getElementById('spec-true-x').textContent = `Ground Truth: ${STATE.x_ppm.toFixed(1)} ppm`;
    document.getElementById('spec-crb').textContent = `± ${fitData.crb_std_x.toFixed(3)} ppm`;
    document.getElementById('spec-iter').textContent = `${fitData.iterations} Iters`;
    document.getElementById('spec-latency').textContent = `Latency: ${fitData.latency_ms.toFixed(1)} ms`;

    const resStd = stdDev(fitData.residuals);
    document.getElementById('spec-residual-std').textContent = `${(resStd * 1000).toFixed(2)} mV`;
    document.getElementById('spec-status').textContent = fitData.converged ? '✓ Converged (LM)' : 'Not Converged';

    // Chart 1: Transmission V(nu) vs Baseline
    renderChart('chart-transmission', [
      { x: simData.nu_axis_cm1, y: simData.V_measured, color: '#06b6d4', width: 1.5, name: 'V_meas' },
      { x: simData.nu_axis_cm1, y: simData.baseline, color: '#f59e0b', width: 1.5, strokeDash: '4,4', name: 'Baseline' }
    ], { xLabel: 'Wavenumber ν (cm⁻¹)', yLabel: 'Voltage (V)' });

    // Chart 2: Absorbance A(nu) vs Fitted Voigt
    const baseline = simData.baseline;
    const fittedAbs = simData.absorbance.map(a => a * (fitData.fitted_x_ppm / Math.max(STATE.x_ppm, 1e-4)));
    renderChart('chart-absorbance', [
      { x: simData.nu_axis_cm1, y: simData.absorbance, color: '#a855f7', width: 1.5, name: 'Measured Abs' },
      { x: simData.nu_axis_cm1, y: fittedAbs, color: '#10b981', width: 2.0, name: 'Voigt Fit' }
    ], { xLabel: 'Wavenumber ν (cm⁻¹)', yLabel: 'Absorbance' });

    // Chart 3: Fit Residuals
    const resMv = fitData.residuals.map(r => r * 1000.0);
    renderChart('chart-residuals', [
      { x: simData.nu_axis_cm1, y: resMv, color: '#f43f5e', width: 1.2, name: 'Residuals' }
    ], { xLabel: 'Wavenumber ν (cm⁻¹)', yLabel: 'Error (mV)', zeroLine: true });

  } catch (err) {
    console.error('Error fetching spectroscopy data:', err);
  }
}

// ==========================================
// Tab 2: Physics vs ML Side-by-Side
// ==========================================
async function fetchComparison() {
  try {
    const payload = JSON.stringify({
      x_ppm: STATE.x_ppm,
      T_K: STATE.T_K,
      P_atm: STATE.P_atm,
      noise_std_v: STATE.noise_std_v,
      etalon_amplitude: STATE.etalon_amplitude
    });

    const resp = await fetch('/api/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload });
    const data = await resp.json();

    // Physics
    document.getElementById('comp-phys-val').textContent = data.physics.x_fitted.toFixed(2);
    document.getElementById('comp-phys-ci').textContent = `[${data.physics.ci95_low.toFixed(2)}, ${data.physics.ci95_high.toFixed(2)}] ppm`;
    document.getElementById('comp-phys-err').textContent = `${data.physics.error_ppm.toFixed(3)} ppm`;
    document.getElementById('comp-phys-lat').textContent = `${data.physics.latency_ms.toFixed(1)} ms`;

    // ML (trained deep ensemble; fields are null when the weights are not available)
    const ml = data.ml;
    const fmt = (v, d) => (v === null || v === undefined) ? '--' : v.toFixed(d);
    document.getElementById('comp-ml-val').textContent = fmt(ml.x_pred, 2);
    document.getElementById('comp-ml-ci').textContent = ml.available ? `[${fmt(ml.ci95_low, 2)}, ${fmt(ml.ci95_high, 2)}] ppm` : '[--, --] ppm';
    document.getElementById('comp-ml-err').textContent = ml.available ? `${fmt(ml.error_ppm, 3)} ppm` : '-- ppm';
    document.getElementById('comp-ml-lat').textContent = ml.available ? `${fmt(ml.latency_ms, 2)} ms` : '-- ms';

    const mlBadge = document.getElementById('comp-ml-badge');
    const mlOodText = document.getElementById('comp-ml-ood-text');

    if (!ml.available) {
      mlBadge.className = 'method-pill pill-warning';
      mlBadge.textContent = 'Weights missing';
      mlOodText.className = 'text-rose';
    } else if (ml.in_training_range) {
      mlBadge.className = 'method-pill pill-success';
      mlBadge.textContent = 'Inside training range';
      mlOodText.className = 'text-emerald';
    } else {
      mlBadge.className = 'method-pill pill-warning';
      mlBadge.textContent = 'Outside training range';
      mlOodText.className = 'text-rose';
    }
    mlOodText.textContent = ml.status;

    // Comparison Bar Chart
    renderBarComparison('chart-comparison-bars', [
      { name: 'Ground Truth', val: data.ground_truth_ppm, color: '#94a3b8' },
      { name: 'Physics LM', val: data.physics.x_fitted, color: '#06b6d4' },
      { name: 'Deep Ensemble ML', val: ml.available ? ml.x_pred : 0, color: '#a855f7' }
    ]);

  } catch (err) {
    console.error('Error fetching comparison data:', err);
  }
}

// ==========================================
// Tab 3: Allan Variance & Metrology
// ==========================================
async function fetchAllan() {
  try {
    const payload = JSON.stringify({
      n_samples: 4000,
      fs_hz: 100.0,
      noise_std_v: STATE.noise_std_v
    });

    const resp = await fetch('/api/allan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload });
    const data = await resp.json();

    document.getElementById('allan-slope').textContent = data.white_noise_slope.toFixed(2);
    document.getElementById('allan-tau-opt').textContent = `~ ${data.optimum_tau_s.toFixed(1)} s`;
    document.getElementById('allan-min-lod').textContent = `${(data.min_adev * 10.0).toFixed(3)} ppm`;

    // Log-Log Allan Chart
    const logTau = data.tau_s.map(t => Math.log10(t));
    const logAdev = data.adev.map(a => Math.log10(a));

    // Theoretical -0.5 slope guide line
    const t0 = logTau[0];
    const a0 = logAdev[0];
    const fitGuide = logTau.map(t => a0 - 0.5 * (t - t0));

    renderChart('chart-allan-plot', [
      { x: logTau, y: logAdev, color: '#06b6d4', width: 2.0, name: 'Measured σ_A(τ)' },
      { x: logTau, y: fitGuide, color: '#64748b', width: 1.5, strokeDash: '4,4', name: 'τ^-0.5 White Noise' }
    ], { xLabel: 'Log₁₀(Integration Time τ / s)', yLabel: 'Log₁₀(Allan Deviation σ_A)' });

  } catch (err) {
    console.error('Error fetching Allan data:', err);
  }
}

// ==========================================
// Tab 4: Harmonic Lock-In Demodulation
// ==========================================
async function fetchLockin() {
  try {
    const payload = JSON.stringify({
      x_ppm: STATE.x_ppm,
      T_K: STATE.T_K,
      P_atm: STATE.P_atm,
      f_mod_hz: 1000.0,
      fs_hz: 50000.0
    });

    const resp = await fetch('/api/lockin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload });
    const data = await resp.json();

    document.getElementById('lockin-r1f').textContent = `${data.r_1f.toFixed(3)} V`;
    document.getElementById('lockin-r2f').textContent = `${(data.r_2f * 1000).toFixed(2)} mV`;
    document.getElementById('lockin-phase').textContent = `${data.phase_deg.toFixed(1)} °`;
    document.getElementById('lockin-ratio').textContent = `${data.harmonic_ratio.toFixed(4)}`;

    // Raw Waveform & 2f peak chart
    renderChart('chart-lockin-plot', [
      { x: data.t_ms, y: data.raw_signal, color: '#06b6d4', width: 1.2, name: 'Modulated Photodiode Signal' }
    ], { xLabel: 'Time (ms)', yLabel: 'Voltage (V)' });

  } catch (err) {
    console.error('Error fetching lockin data:', err);
  }
}

// ==========================================
// Custom Pure SVG Chart Rendering Engine
// ==========================================
function renderChart(containerId, datasets, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const width = container.clientWidth || 600;
  const height = container.clientHeight || 240;
  const margin = { top: 20, right: 30, bottom: 35, left: 55 };

  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  // Determine global min/max
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  datasets.forEach(ds => {
    ds.x.forEach(v => { if (v < minX) minX = v; if (v > maxX) maxX = v; });
    ds.y.forEach(v => { if (v < minY) minY = v; if (v > maxY) maxY = v; });
  });

  if (options.zeroLine) {
    minY = Math.min(minY, 0);
    maxY = Math.max(maxY, 0);
  }

  // Padding
  const padY = (maxY - minY) * 0.08 || 0.1;
  minY -= padY;
  maxY += padY;

  const scaleX = (x) => margin.left + ((x - minX) / (maxX - minX || 1)) * plotWidth;
  const scaleY = (y) => margin.top + plotHeight - ((y - minY) / (maxY - minY || 1)) * plotHeight;

  let svgHtml = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="background: transparent;">`;

  // Grid lines
  const nGridY = 4;
  for (let i = 0; i <= nGridY; i++) {
    const gy = minY + (i / nGridY) * (maxY - minY);
    const sy = scaleY(gy);
    svgHtml += `<line x1="${margin.left}" y1="${sy}" x2="${width - margin.right}" y2="${sy}" stroke="rgba(255,255,255,0.06)" stroke-width="1" />`;
    svgHtml += `<text x="${margin.left - 8}" y="${sy + 4}" fill="#64748b" font-size="10" font-family="monospace" text-anchor="end">${gy.toFixed(2)}</text>`;
  }

  const nGridX = 5;
  for (let i = 0; i <= nGridX; i++) {
    const gx = minX + (i / nGridX) * (maxX - minX);
    const sx = scaleX(gx);
    svgHtml += `<line x1="${sx}" y1="${margin.top}" x2="${sx}" y2="${height - margin.bottom}" stroke="rgba(255,255,255,0.06)" stroke-width="1" />`;
    svgHtml += `<text x="${sx}" y="${height - margin.bottom + 16}" fill="#64748b" font-size="10" font-family="monospace" text-anchor="middle">${gx.toFixed(2)}</text>`;
  }

  // Zero line if applicable
  if (minY <= 0 && maxY >= 0) {
    const zy = scaleY(0);
    svgHtml += `<line x1="${margin.left}" y1="${zy}" x2="${width - margin.right}" y2="${zy}" stroke="rgba(255,255,255,0.2)" stroke-width="1" stroke-dasharray="2,2" />`;
  }

  // Datasets polylines
  datasets.forEach(ds => {
    let points = [];
    for (let i = 0; i < ds.x.length; i++) {
      points.push(`${scaleX(ds.x[i]).toFixed(1)},${scaleY(ds.y[i]).toFixed(1)}`);
    }
    const dash = ds.strokeDash ? `stroke-dasharray="${ds.strokeDash}"` : '';
    svgHtml += `<polyline points="${points.join(' ')}" fill="none" stroke="${ds.color}" stroke-width="${ds.width || 1.5}" ${dash} stroke-linecap="round" stroke-linejoin="round" />`;
  });

  // Axis Labels
  if (options.xLabel) {
    svgHtml += `<text x="${margin.left + plotWidth / 2}" y="${height - 6}" fill="#94a3b8" font-size="11" font-weight="500" text-anchor="middle">${options.xLabel}</text>`;
  }
  if (options.yLabel) {
    svgHtml += `<text x="14" y="${margin.top + plotHeight / 2}" fill="#94a3b8" font-size="11" font-weight="500" text-anchor="middle" transform="rotate(-90 14 ${margin.top + plotHeight / 2})">${options.yLabel}</text>`;
  }

  svgHtml += `</svg>`;
  container.innerHTML = svgHtml;
}

function renderBarComparison(containerId, items) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const width = container.clientWidth || 600;
  const height = container.clientHeight || 180;
  const margin = { top: 20, right: 30, bottom: 40, left: 120 };

  const plotWidth = width - margin.left - margin.right;
  const maxVal = Math.max(...items.map(i => i.val)) * 1.2 || 10;

  const barHeight = 26;
  const barGap = 16;

  let svgHtml = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;

  items.forEach((item, idx) => {
    const y = margin.top + idx * (barHeight + barGap);
    const barW = Math.max(2, (item.val / maxVal) * plotWidth);

    svgHtml += `<text x="${margin.left - 10}" y="${y + 17}" fill="#f1f5f9" font-size="12" font-weight="500" text-anchor="end">${item.name}</text>`;
    svgHtml += `<rect x="${margin.left}" y="${y}" width="${barW}" height="${barHeight}" rx="4" fill="${item.color}" opacity="0.85" />`;
    svgHtml += `<text x="${margin.left + barW + 8}" y="${y + 17}" fill="${item.color}" font-size="12" font-weight="700" font-family="monospace">${item.val.toFixed(2)} ppm</text>`;
  });

  svgHtml += `</svg>`;
  container.innerHTML = svgHtml;
}

function stdDev(arr) {
  if (!arr || arr.length === 0) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const variance = arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length;
  return Math.sqrt(variance);
}
