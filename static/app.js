/* OpenBCI × BioProxal flow validation - browser frontend.
 * Vanilla JS + Canvas 2D, no external resources. Contract: docs/API.md.
 *
 * Structure:
 *   store      - all state that arrives from the server (hello/status/analysis/band/marks/history)
 *   helpers    - DOM, formatting, canvas DPR fitting, tooltip, POST /api/cmd
 *   widgets    - TimeSeries, FFTPlot, BandPower, HeadPlot, FlowChart, LagPlot, StatsTiles,
 *                BandPanel, FeltFlow, Sessions, NavBar - each with update(...) and/or draw()
 *   sse        - EventSource with reconnect backoff
 */
'use strict';

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------
const COLOR_EEG = '#C77A00';
const COLOR_BAND = '#0B9C8F';
const INK = '#222';
const INK2 = '#666';
const INK3 = '#999';
const GRID = '#E6E6E6';
const DEFAULT_COLORS = ['#818181', '#7C4B8D', '#36579E', '#317159', '#DDB20D', '#FD5E34', '#E0382D', '#A25231'];
const QUALITY_COLORS = { good: '#2E8B57', near: '#D9A400', railed: '#C8362B', flat: '#999999', unknown: '#999999' };
const BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma'];
const RAMP_LO = [0xD3, 0xE9, 0xE2]; // light mint
const RAMP_HI = [0x0A, 0x58, 0x53]; // dark teal
const MAX_WINDOW_S = 20;
const HISTORY_MAX_S = 4 * 3600;

// ----------------------------------------------------------------------------
// Store
// ----------------------------------------------------------------------------
const store = {
  connected: false,
  hello: null,
  fs: 250,
  nCh: 8,
  labels: [],
  colors: DEFAULT_COLORS.slice(),
  positions: {},
  enabled: [],          // from analysis.enabled (default all true)
  analysis: null,
  status: null,
  band: null,           // latest band sample object
  eeg: [],              // [t, flow]
  bandHist: [],         // [t, flow, held]
  marks: [],            // [t, kind, value]
  pairs: [],            // [t, band, eeg]
  ring: null,           // Ring buffer of filtered samples
  ringDirty: false,
  windowS: 10,
  scale: 'auto',
  flowRange: '30',
  wearers: [],
  sessions: [],
  lastAnalysisT: null,
};

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const fmt = (v, d = 2) => (isNum(v) ? v.toFixed(d) : '—');
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const pad2 = (n) => String(n).padStart(2, '0');

function fmtClock(tSec, withSec = false) {
  if (!isNum(tSec)) return '—';
  const d = new Date(tSec * 1000);
  const s = pad2(d.getHours()) + ':' + pad2(d.getMinutes());
  return withSec ? s + ':' + pad2(d.getSeconds()) : s;
}
function fmtAge(s) {
  if (!isNum(s)) return '—';
  if (s < 60) return Math.round(s) + ' s';
  if (s < 3600) return Math.round(s / 60) + ' min';
  return (s / 3600).toFixed(1) + ' h';
}
function fmtMMSS(s) {
  if (!isNum(s)) return '00:00';
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return pad2(m) + ':' + pad2(r);
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function chColor(i) { return store.colors[i % store.colors.length] || DEFAULT_COLORS[i % 8]; }
function chLabel(i) { return store.labels[i] || ('ch' + (i + 1)); }
function chEnabled(i) { return store.enabled.length > i ? !!store.enabled[i] : true; }
function lerpColor(a, b, t) {
  const c = a.map((x, i) => Math.round(x + (b[i] - x) * clamp(t, 0, 1)));
  return 'rgb(' + c.join(',') + ')';
}

/** Fit a canvas to its CSS box using devicePixelRatio; returns css size + ctx with DPR transform. */
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(10, Math.round(rect.width));
  const h = Math.max(10, Math.round(rect.height));
  const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
  if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = '11px "Segoe UI", system-ui, sans-serif';
  return { ctx, w, h };
}

/** Rounded-top rectangle path (bars). */
function roundTopRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, Math.max(0, h));
  ctx.beginPath();
  ctx.moveTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h);
  ctx.closePath();
}

/** Shared tooltip (absolutely positioned div). */
const tooltip = {
  el: null,
  show(x, y, html) {
    if (!this.el) this.el = $('tooltip');
    this.el.innerHTML = html;
    this.el.hidden = false;
    const r = this.el.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 10;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 10;
    this.el.style.left = left + 'px';
    this.el.style.top = top + 'px';
  },
  hide() { if (!this.el) this.el = $('tooltip'); this.el.hidden = true; },
};
function ttRow(label, value, color) {
  const sw = color ? '<i class="sw" style="background:' + color + '"></i>' : '';
  return '<div class="tt-row">' + sw + '<span class="tt-k">' + esc(label) + '</span><span>' + esc(value) + '</span></div>';
}

/** POST /api/cmd */
async function cmd(action, fields) {
  const body = Object.assign({ action }, fields || {});
  try {
    const r = await fetch('/api/cmd', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({ ok: false, error: 'bad json' }));
    if (!j.ok) console.warn('cmd failed', action, j.error);
    return j;
  } catch (e) {
    console.warn('cmd error', action, e);
    return { ok: false, error: String(e) };
  }
}
async function getJSON(url) {
  try { const r = await fetch(url); return await r.json(); } catch (e) { console.warn('GET failed', url, e); return null; }
}

// ----------------------------------------------------------------------------
// Ring buffer of filtered samples (per channel)
// ----------------------------------------------------------------------------
class Ring {
  constructor(nCh, cap) {
    this.nCh = nCh; this.cap = cap; this.pos = 0; this.count = 0;
    this.buf = [];
    for (let c = 0; c < nCh; c++) this.buf.push(new Float32Array(cap));
  }
  push(ch) {
    const n = ch && ch[0] ? ch[0].length : 0;
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < this.nCh; c++) {
        const arr = ch[c];
        this.buf[c][this.pos] = arr && isNum(arr[i]) ? arr[i] : 0;
      }
      this.pos = (this.pos + 1) % this.cap;
    }
    this.count = Math.min(this.count + n, this.cap);
  }
  /** value of channel c, k samples before the newest (k = 0 is newest) */
  back(c, k) { return this.buf[c][(this.pos - 1 - k + this.cap * 2) % this.cap]; }
}

// ----------------------------------------------------------------------------
// Widget: TIME SERIES
// ----------------------------------------------------------------------------
class TimeSeries {
  constructor() {
    this.canvas = $('cvTs');
    this.lastDraw = 0;
    this.gutter = 64;   // label gutter (px)
    this.right = 70;    // quality pill gutter (px)
    this.canvas.addEventListener('click', (e) => this.onClick(e));
    this.canvas.addEventListener('mousemove', (e) => this.onMove(e));
    this.canvas.addEventListener('mouseleave', () => tooltip.hide());
  }
  stripIndex(e) {
    const rect = this.canvas.getBoundingClientRect();
    const n = store.nCh || 1;
    const i = Math.floor((e.clientY - rect.top) / (rect.height / n));
    return { i: clamp(i, 0, n - 1), x: e.clientX - rect.left };
  }
  onClick(e) {
    const { i, x } = this.stripIndex(e);
    if (x > this.gutter) return;
    cmd('set_channel', { index: i, enabled: !chEnabled(i) });
  }
  onMove(e) {
    const { i, x } = this.stripIndex(e);
    const rect = this.canvas.getBoundingClientRect();
    if (x > this.gutter && x < rect.width - this.right) { tooltip.hide(); return; }   // only the label / pill gutters explain themselves
    const q = store.analysis && store.analysis.quality ? store.analysis.quality[i] : null;
    const peak = store.lastRawPeak ? store.lastRawPeak[i] : null;
    const html = ttRow('channel', (i + 1) + ' ' + chLabel(i), chColor(i))
      + ttRow('quality', q ? (q.state + ' · ' + fmt(q.pct, 1) + '% · σ ' + fmt(q.std, 1) + ' µV') : '—')
      + ttRow('raw peak', isNum(peak) ? fmt(peak, 1) + ' µV' : '—')
      + (x <= this.gutter ? '<div class="tt-k">click label to ' + (chEnabled(i) ? 'disable' : 'enable') + '</div>' : '');
    tooltip.show(e.clientX, e.clientY, html);
  }
  tick(now) {
    if (now - this.lastDraw < 33) return;   // ≤ 30 fps
    if (!store.ringDirty && !this.forceDraw) return;
    this.lastDraw = now; store.ringDirty = false; this.forceDraw = false;
    this.draw();
  }
  draw() {
    const { ctx, w, h } = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const n = store.nCh || 8;
    const stripH = h / n;
    const plotX = this.gutter, plotW = Math.max(10, w - this.gutter - this.right);
    const ring = store.ring;
    const W = Math.round(store.windowS * store.fs);   // samples in the window
    const N = ring ? Math.min(ring.count, W) : 0;
    const q = store.analysis && store.analysis.quality ? store.analysis.quality : null;

    for (let c = 0; c < n; c++) {
      const y0 = c * stripH, mid = y0 + stripH / 2;
      const on = chEnabled(c);
      // strip separator + midline
      ctx.strokeStyle = GRID; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, y0 + stripH - 0.5); ctx.lineTo(w, y0 + stripH - 0.5); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(plotX, mid + 0.5); ctx.lineTo(plotX + plotW, mid + 0.5); ctx.stroke();
      if (!on) { ctx.fillStyle = 'rgba(0,0,0,0.05)'; ctx.fillRect(plotX, y0, plotW, stripH); }

      // scale
      let scale = 10;
      if (store.scale === 'auto') {
        let mx = 0;
        if (ring) for (let k = 0; k < N; k++) { const v = Math.abs(ring.back(c, k)); if (v > mx) mx = v; }
        scale = Math.max(10, mx * 1.05);
      } else scale = parseFloat(store.scale) || 100;
      const amp = (stripH / 2 - 4) / scale;

      // label: number + name in the channel colour (the one place series colour touches text, as in the GUI)
      ctx.textBaseline = 'middle';
      ctx.fillStyle = on ? chColor(c) : INK3;
      ctx.font = 'bold 12px "Segoe UI", system-ui, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(String(c + 1), 6, mid - 7);
      ctx.font = '11px "Segoe UI", system-ui, sans-serif';
      ctx.fillStyle = on ? INK : INK3;
      ctx.fillText(chLabel(c), 22, mid - 7);
      ctx.fillStyle = INK2; ctx.font = '10px "Segoe UI", system-ui, sans-serif';
      ctx.fillText('±' + (scale >= 100 ? Math.round(scale) : scale.toFixed(0)) + ' µV', 6, mid + 8);

      // trace: per-pixel min/max
      if (ring && N > 1) {
        ctx.strokeStyle = on ? chColor(c) : '#BBBBBB';
        ctx.lineWidth = 1;
        ctx.beginPath();
        let started = false;
        const sPerPx = W / plotW;
        for (let px = 0; px < plotW; px++) {
          // window index i (0 = oldest in window), k = W-1-i samples back
          const i0 = Math.floor(px * sPerPx), i1 = Math.max(i0 + 1, Math.floor((px + 1) * sPerPx));
          let mn = Infinity, mx = -Infinity;
          for (let i = i0; i < i1 && i < W; i++) {
            const k = W - 1 - i;
            if (k >= N) continue;
            const v = ring.back(c, k);
            if (v < mn) mn = v; if (v > mx) mx = v;
          }
          if (mn === Infinity) continue;
          const yA = clamp(mid - mx * amp, y0 + 1, y0 + stripH - 1);
          const yB = clamp(mid - mn * amp, y0 + 1, y0 + stripH - 1);
          const x = plotX + px + 0.5;
          if (!started) { ctx.moveTo(x, yA); started = true; } else ctx.lineTo(x, yA);
          if (yB !== yA) ctx.lineTo(x, yB);
        }
        ctx.stroke();
      }

      // quality pill at right edge
      const qi = q ? q[c] : null;
      const state = qi && qi.state ? qi.state : 'unknown';
      const pillText = state + (qi && isNum(qi.pct) ? ' ' + qi.pct.toFixed(1) + '%' : '');
      ctx.font = '10px "Segoe UI", system-ui, sans-serif';
      const tw = ctx.measureText(pillText).width + 10;
      const pw = Math.min(this.right - 6, tw), px0 = w - pw - 3, py0 = mid - 8;
      ctx.fillStyle = on ? (QUALITY_COLORS[state] || INK3) : '#CCCCCC';
      ctx.beginPath();
      ctx.moveTo(px0 + 8, py0); ctx.arcTo(px0 + pw, py0, px0 + pw, py0 + 16, 8); ctx.arcTo(px0 + pw, py0 + 16, px0, py0 + 16, 8);
      ctx.arcTo(px0, py0 + 16, px0, py0, 8); ctx.arcTo(px0, py0, px0 + pw, py0, 8); ctx.closePath(); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.textAlign = 'center';
      ctx.fillText(pillText, px0 + pw / 2, py0 + 8.5);
      ctx.textAlign = 'left';
    }
    // time axis hint
    ctx.fillStyle = INK2; ctx.font = '10px "Segoe UI", system-ui, sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(store.windowS + ' s window · ' + (store.fs | 0) + ' Hz', w - 4, h - 7);
    ctx.textAlign = 'left';
    $('tsInfo').textContent = store.windowS + ' s · scale ' + (store.scale === 'auto' ? 'auto' : store.scale + ' µV');
  }
}

// ----------------------------------------------------------------------------
// Widget: FFT PLOT
// ----------------------------------------------------------------------------
class FFTPlot {
  constructor() {
    this.canvas = $('cvFft');
    this.pad = { l: 44, r: 10, t: 8, b: 22 };
    this.hoverX = null;
    this.canvas.addEventListener('mousemove', (e) => this.onMove(e));
    this.canvas.addEventListener('mouseleave', () => { this.hoverX = null; tooltip.hide(); this.draw(); });
  }
  onMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    this.hoverX = e.clientX - rect.left;
    this.hoverClient = [e.clientX, e.clientY];
    this.draw();
  }
  draw() {
    const { ctx, w, h } = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const p = this.pad, pw = w - p.l - p.r, ph = h - p.t - p.b;
    const a = store.analysis, fft = a && a.fft;
    const freqs = fft && Array.isArray(fft.freqs) ? fft.freqs : [];
    const psd = fft && Array.isArray(fft.psd) ? fft.psd : [];
    // y domain in log10
    let vmin = Infinity, vmax = -Infinity;
    for (let c = 0; c < psd.length; c++) {
      if (!chEnabled(c) || !psd[c]) continue;
      for (let i = 0; i < freqs.length; i++) { const v = psd[c][i]; if (isNum(v) && v > 0 && freqs[i] <= 60) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; } }
    }
    let lo = -2, hi = 2;
    if (vmin < vmax) { lo = Math.max(-2, Math.floor(Math.log10(vmin))); hi = Math.max(lo + 2, Math.ceil(Math.log10(vmax))); }
    const xOf = (f) => p.l + (f / 60) * pw;
    const yOf = (v) => p.t + (1 - (clamp(Math.log10(Math.max(v, 1e-12)), lo, hi) - lo) / (hi - lo)) * ph;
    // grid
    ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.fillStyle = INK2; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let d = lo; d <= hi; d++) {
      const y = Math.round(p.t + (1 - (d - lo) / (hi - lo)) * ph) + 0.5;
      ctx.beginPath(); ctx.moveTo(p.l, y); ctx.lineTo(p.l + pw, y); ctx.stroke();
      ctx.fillText(d >= 0 ? String(Math.pow(10, d)) : Math.pow(10, d).toFixed(-d), p.l - 4, y);
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let f = 0; f <= 60; f += 10) {
      const x = Math.round(xOf(f)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, p.t); ctx.lineTo(x, p.t + ph); ctx.stroke();
      ctx.fillText(String(f), x, p.t + ph + 4);
    }
    ctx.fillText('Hz', p.l + pw - 10, p.t + ph + 4);
    if (!freqs.length) { ctx.fillStyle = INK3; ctx.textBaseline = 'middle'; ctx.fillText(store.connected ? 'waiting for analysis…' : 'connecting to server…', p.l + pw / 2, p.t + ph / 2); return; }
    // lines
    ctx.lineWidth = 2; ctx.lineJoin = 'round';
    for (let c = 0; c < psd.length; c++) {
      if (!chEnabled(c) || !psd[c]) continue;
      ctx.strokeStyle = chColor(c); ctx.beginPath(); let st = false;
      for (let i = 0; i < freqs.length; i++) {
        if (freqs[i] > 60 || !isNum(psd[c][i])) continue;
        const x = xOf(freqs[i]), y = yOf(psd[c][i]);
        if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    // crosshair + tooltip
    if (this.hoverX !== null && this.hoverX >= p.l && this.hoverX <= p.l + pw) {
      const f = ((this.hoverX - p.l) / pw) * 60;
      let bi = 0, bd = Infinity;
      for (let i = 0; i < freqs.length; i++) { const d = Math.abs(freqs[i] - f); if (d < bd) { bd = d; bi = i; } }
      const x = Math.round(xOf(freqs[bi])) + 0.5;
      ctx.strokeStyle = INK3; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, p.t); ctx.lineTo(x, p.t + ph); ctx.stroke(); ctx.setLineDash([]);
      let html = ttRow('freq', fmt(freqs[bi], 1) + ' Hz');
      for (let c = 0; c < psd.length; c++) {
        if (!chEnabled(c) || !psd[c]) continue;
        const v = psd[c][bi];
        ctx.fillStyle = chColor(c); ctx.beginPath(); ctx.arc(xOf(freqs[bi]), yOf(v), 4, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
        html += ttRow(chLabel(c), isNum(v) ? (v >= 10 ? v.toFixed(1) : v.toPrecision(3)) + ' µV²/Hz' : '—', chColor(c));
      }
      if (this.hoverClient) tooltip.show(this.hoverClient[0], this.hoverClient[1], html);
    }
  }
}

// ----------------------------------------------------------------------------
// Widget: BAND POWER
// ----------------------------------------------------------------------------
class BandPower {
  constructor() {
    this.canvas = $('cvBand');
    this.pad = { l: 44, r: 8, t: 8, b: 22 };
    this.log = false;
    this.bars = [];     // hit rects
    this.hover = null;
    $('chkBandLog').addEventListener('change', (e) => { this.log = e.target.checked; this.draw(); });
    this.canvas.addEventListener('mousemove', (e) => this.onMove(e));
    this.canvas.addEventListener('mouseleave', () => { this.hover = null; tooltip.hide(); this.draw(); });
  }
  onMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const b = this.bars.find((r) => x >= r.x - 1 && x <= r.x + r.w + 1 && y >= Math.min(r.y, r.y0) - 4 && y <= Math.max(r.y, r.y0) + 2);
    this.hover = b || null;
    if (b) tooltip.show(e.clientX, e.clientY, ttRow('band', b.band) + ttRow(chLabel(b.c), fmt(b.v, 2) + ' µV²', chColor(b.c)));
    else tooltip.hide();
    this.draw();
  }
  draw() {
    const { ctx, w, h } = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const p = this.pad, pw = w - p.l - p.r, ph = h - p.t - p.b;
    const a = store.analysis, bands = a && a.bands;
    const order = bands && Array.isArray(bands.order) && bands.order.length ? bands.order : BAND_ORDER;
    const abs = bands && bands.abs ? bands.abs : null;
    const nCh = store.nCh || 8;
    const enabledIdx = [];
    for (let c = 0; c < nCh; c++) if (chEnabled(c)) enabledIdx.push(c);
    // domain
    let vmax = 0, vmin = Infinity;
    if (abs) for (const b of order) { const arr = abs[b] || []; for (const c of enabledIdx) { const v = arr[c]; if (isNum(v)) { if (v > vmax) vmax = v; if (v > 0 && v < vmin) vmin = v; } } }
    let yOf, ticks = [];
    if (this.log) {
      const lo = Math.max(-2, Math.floor(Math.log10(vmin < Infinity ? vmin : 0.01))), hi = Math.max(lo + 2, Math.ceil(Math.log10(Math.max(vmax, 0.1))));
      yOf = (v) => p.t + (1 - (clamp(Math.log10(Math.max(v, 1e-12)), lo, hi) - lo) / (hi - lo)) * ph;
      for (let d = lo; d <= hi; d++) ticks.push({ v: Math.pow(10, d), label: d >= 0 ? String(Math.pow(10, d)) : Math.pow(10, d).toFixed(-d) });
    } else {
      const top = niceCeil(vmax > 0 ? vmax : 10);
      yOf = (v) => p.t + (1 - clamp(v, 0, top) / top) * ph;
      for (let i = 0; i <= 4; i++) { const v = (top * i) / 4; ticks.push({ v, label: v >= 100 ? String(Math.round(v)) : String(+v.toFixed(1)) }); }
    }
    // grid
    ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.fillStyle = INK2; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const t of ticks) { const y = Math.round(yOf(t.v)) + 0.5; ctx.beginPath(); ctx.moveTo(p.l, y); ctx.lineTo(p.l + pw, y); ctx.stroke(); ctx.fillText(t.label, p.l - 4, y); }
    ctx.save(); ctx.translate(10, p.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('µV²', 0, 0); ctx.restore();
    // bars
    this.bars = [];
    const groupW = pw / order.length, gap = 2, inner = groupW * 0.8;
    const n = Math.max(1, enabledIdx.length);
    const barW = Math.max(1, (inner - gap * (n - 1)) / n);
    const base = yOf(this.log ? ticks[0].v : 0);   // bars grow from the lowest tick (log) or zero (linear)
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    order.forEach((band, gi) => {
      const gx = p.l + gi * groupW + (groupW - inner) / 2;
      ctx.fillStyle = INK2; ctx.fillText(band, p.l + gi * groupW + groupW / 2, p.t + ph + 4);
      if (!abs) return;
      enabledIdx.forEach((c, j) => {
        const v = abs[band] ? abs[band][c] : null;
        if (!isNum(v)) return;
        const x = gx + j * (barW + gap), y = yOf(v), hh = Math.max(0, base - y);
        const hov = this.hover && this.hover.c === c && this.hover.band === band;
        ctx.fillStyle = chColor(c); ctx.globalAlpha = this.hover && !hov ? 0.55 : 1;
        roundTopRect(ctx, x, y, barW, hh, 4); ctx.fill(); ctx.globalAlpha = 1;
        this.bars.push({ x, y, y0: base, w: barW, c, band, v });
      });
    });
    if (!abs) { ctx.fillStyle = INK3; ctx.textBaseline = 'middle'; ctx.fillText(store.connected ? 'waiting for analysis…' : 'connecting to server…', p.l + pw / 2, p.t + ph / 2); }
  }
}
function niceCeil(v) {
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / e;
  const nm = m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10;
  return nm * e;
}

// ----------------------------------------------------------------------------
// Widget: HEAD PLOT (inline SVG)
// ----------------------------------------------------------------------------
class HeadPlot {
  constructor() {
    this.wrap = $('headSvgWrap');
    this.band = 'alpha';
    this.nodes = [];
    $('selHeadBand').addEventListener('change', (e) => { this.band = e.target.value; this.draw(); });
    this.build();
  }
  build() {
    const NS = 'http://www.w3.org/2000/svg';
    const mk = (tag, attrs) => { const el = document.createElementNS(NS, tag); for (const k in attrs) el.setAttribute(k, attrs[k]); return el; };
    const svg = mk('svg', { viewBox: '0 0 300 280', role: 'img', 'aria-label': 'head plot' });
    const cx = 150, cy = 145, R = 108;
    // ears, nose, head
    svg.appendChild(mk('ellipse', { cx: cx - R - 6, cy, rx: 9, ry: 20, fill: '#fff', stroke: '#444', 'stroke-width': 2 }));
    svg.appendChild(mk('ellipse', { cx: cx + R + 6, cy, rx: 9, ry: 20, fill: '#fff', stroke: '#444', 'stroke-width': 2 }));
    svg.appendChild(mk('path', { d: 'M' + (cx - 16) + ' ' + (cy - R + 3) + ' L' + cx + ' ' + (cy - R - 20) + ' L' + (cx + 16) + ' ' + (cy - R + 3), fill: '#fff', stroke: '#444', 'stroke-width': 2, 'stroke-linejoin': 'round' }));
    svg.appendChild(mk('circle', { cx, cy, r: R, fill: '#fff', stroke: '#444', 'stroke-width': 2 }));
    this.g = mk('g', {});
    svg.appendChild(this.g);
    this.wrap.innerHTML = '';
    this.wrap.appendChild(svg);
    this.geom = { cx, cy, R };
    this.mk = mk;
    this.layout();
  }
  /** (Re)create electrode nodes from hello.positions + labels. */
  layout() {
    const { cx, cy, R } = this.geom;
    this.g.innerHTML = '';
    this.nodes = [];
    const n = store.nCh || 8;
    for (let c = 0; c < n; c++) {
      const lab = chLabel(c);
      const pos = store.positions && store.positions[lab];
      let x, y;
      if (Array.isArray(pos) && isNum(pos[0]) && isNum(pos[1])) { x = cx + pos[0] * R * 0.86; y = cy - pos[1] * R * 0.86; }
      else { const ang = -Math.PI / 2 + (c / n) * Math.PI * 2; x = cx + Math.cos(ang) * R * 0.6; y = cy + Math.sin(ang) * R * 0.6; }
      const circ = this.mk('circle', { cx: x, cy: y, r: 15, fill: '#fff', stroke: '#444', 'stroke-width': 1.5 });
      // label centred inside the electrode (labels under the circle collided near the bottom rim)
      const txt = this.mk('text', { x, y, 'text-anchor': 'middle', 'dominant-baseline': 'central', 'font-size': 11, fill: INK, 'font-family': 'Segoe UI, system-ui, sans-serif', 'pointer-events': 'none' });
      txt.textContent = lab;
      const title = this.mk('title', {});
      circ.appendChild(title);
      this.g.appendChild(circ); this.g.appendChild(txt);
      this.nodes.push({ circ, txt, title, c });
    }
  }
  draw() {
    if (this.nodes.length !== (store.nCh || 8)) this.layout();
    const a = store.analysis, rel = a && a.bands && a.bands.rel ? a.bands.rel[this.band] : null;
    let vmin = Infinity, vmax = -Infinity;
    if (rel) for (const nd of this.nodes) { const v = rel[nd.c]; if (chEnabled(nd.c) && isNum(v)) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; } }
    const ok = vmin < vmax;
    for (const nd of this.nodes) {
      const v = rel ? rel[nd.c] : null;
      const on = chEnabled(nd.c);
      let textFill = INK3;
      if (!on) { nd.circ.setAttribute('fill', 'none'); nd.circ.setAttribute('stroke', INK3); nd.circ.setAttribute('stroke-dasharray', '3 2'); }
      else {
        nd.circ.setAttribute('stroke', '#444'); nd.circ.removeAttribute('stroke-dasharray');
        const t = ok && isNum(v) ? (v - vmin) / (vmax - vmin) : (isNum(v) ? 0.5 : 0);
        nd.circ.setAttribute('fill', isNum(v) ? lerpColor(RAMP_LO, RAMP_HI, t) : '#fff');
        // label ink by the fill's lightness: dark ink on light mint, white on dark teal
        const rgb = isNum(v) ? RAMP_LO.map((x, i) => x + (RAMP_HI[i] - x) * clamp(t, 0, 1)) : [255, 255, 255];
        const lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
        textFill = lum > 0.55 ? INK : '#fff';
      }
      nd.txt.setAttribute('fill', textFill);
      nd.title.textContent = chLabel(nd.c) + ': ' + (isNum(v) ? (v * 100).toFixed(1) + ' % ' + this.band + ' (relative)' : '—') + (on ? '' : ' · disabled');
    }
    $('headMin').textContent = ok ? (vmin * 100).toFixed(0) + ' %' : '—';
    $('headMax').textContent = ok ? (vmax * 100).toFixed(0) + ' %' : '—';
  }
}

// ----------------------------------------------------------------------------
// Widget: BIOPROXAL FLOW chart (one y-axis 0-1; EEG line + band step line + marks)
// ----------------------------------------------------------------------------
class FlowChart {
  constructor() {
    this.canvas = $('cvFlow');
    this.pad = { l: 36, r: 12, t: 10, b: 22 };
    this.hoverX = null;
    this.canvas.addEventListener('mousemove', (e) => { const r = this.canvas.getBoundingClientRect(); this.hoverX = e.clientX - r.left; this.hoverClient = [e.clientX, e.clientY]; this.draw(); });
    this.canvas.addEventListener('mouseleave', () => { this.hoverX = null; tooltip.hide(); this.draw(); });
    document.querySelectorAll('input[name=flowRange]').forEach((el) => el.addEventListener('change', (e) => { store.flowRange = e.target.value; this.draw(); }));
  }
  /** x-range anchored on the latest timestamp seen in either series (robust to clock skew). */
  range() {
    const le = store.eeg.length ? store.eeg[store.eeg.length - 1][0] : null;
    const lb = store.bandHist.length ? store.bandHist[store.bandHist.length - 1][0] : null;
    let tEnd = Math.max(le || -Infinity, lb || -Infinity);
    if (!isNum(tEnd) || tEnd === -Infinity) tEnd = store.status && isNum(store.status.clock) ? store.status.clock : Date.now() / 1000;
    let tStart;
    if (store.flowRange === 'all') {
      const fe = store.eeg.length ? store.eeg[0][0] : Infinity;
      const fb = store.bandHist.length ? store.bandHist[0][0] : Infinity;
      const fm = store.marks.length ? store.marks[0][0] : Infinity;
      tStart = Math.min(fe, fb, fm);
      if (!isNum(tStart) || tStart === Infinity || tEnd - tStart < 120) tStart = tEnd - 120;
    } else tStart = tEnd - 1800;
    return { tStart, tEnd: tEnd + (tEnd - tStart) * 0.01 };
  }
  draw() {
    const { ctx, w, h } = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const p = this.pad, pw = w - p.l - p.r, ph = h - p.t - p.b;
    const { tStart, tEnd } = this.range();
    const xOf = (t) => p.l + ((t - tStart) / (tEnd - tStart)) * pw;
    const yOf = (v) => p.t + (1 - clamp(v, 0, 1)) * ph;
    // grid y
    ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.fillStyle = INK2; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) { const v = i / 4, y = Math.round(yOf(v)) + 0.5; ctx.beginPath(); ctx.moveTo(p.l, y); ctx.lineTo(p.l + pw, y); ctx.stroke(); ctx.fillText(v.toFixed(2), p.l - 4, y); }
    // grid x (time ticks)
    const span = tEnd - tStart;
    const step = span <= 600 ? 60 : span <= 1900 ? 300 : span <= 7200 ? 900 : 1800;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let t = Math.ceil(tStart / step) * step; t <= tEnd; t += step) {
      const x = Math.round(xOf(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, p.t); ctx.lineTo(x, p.t + ph); ctx.stroke();
      ctx.fillText(fmtClock(t), x, p.t + ph + 4);
    }
    // EEG line
    ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.strokeStyle = COLOR_EEG; ctx.beginPath();
    let st = false;
    for (const pt of store.eeg) {
      const t = pt[0], v = pt[1];
      if (t < tStart - 10 || t > tEnd) continue;
      if (!isNum(v)) { st = false; continue; }
      const x = xOf(t), y = yOf(v);
      if (!st) { ctx.moveTo(x, y); st = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // band step line (held segments dashed)
    const bh = store.bandHist;
    ctx.strokeStyle = COLOR_BAND;
    for (let i = 0; i < bh.length; i++) {
      const t0 = bh[i][0], v0 = bh[i][1], held = !!bh[i][2];
      const t1 = i + 1 < bh.length ? bh[i + 1][0] : tEnd;
      if (t1 < tStart || t0 > tEnd || !isNum(v0)) continue;
      ctx.setLineDash(held ? [4, 4] : []);
      ctx.beginPath();
      ctx.moveTo(xOf(Math.max(t0, tStart)), yOf(v0)); ctx.lineTo(xOf(Math.min(t1, tEnd)), yOf(v0));
      const v1 = i + 1 < bh.length ? bh[i + 1][1] : null;
      if (isNum(v1) && t1 <= tEnd && t1 >= tStart) ctx.lineTo(xOf(t1), yOf(v1));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    // marks
    ctx.lineWidth = 2;
    for (const m of store.marks) {
      const t = m[0], kind = m[1], val = m[2];
      if (t < tStart || t > tEnd) continue;
      const x = Math.round(xOf(t)) + 0.5;
      if (kind === 'felt_flow') {
        ctx.strokeStyle = '#555'; ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(x, p.t); ctx.lineTo(x, p.t + ph); ctx.stroke();
        const v = typeof val === 'number' ? val : parseFloat(val);
        if (isNum(v)) { ctx.fillStyle = '#555'; ctx.beginPath(); ctx.arc(x, yOf(v), 4, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 2; }
      } else {
        ctx.strokeStyle = '#777'; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x, p.t); ctx.lineTo(x, p.t + ph); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = INK2; ctx.font = '9px "Segoe UI", system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        const lab = kind === 'flow_start' ? 'start' : kind === 'flow_end' ? 'end' : kind === 'distracted' ? 'distr.' : (kind === 'note' ? 'note' : String(kind));
        ctx.fillText(lab, x + 2, p.t + 1);
        ctx.font = '11px "Segoe UI", system-ui, sans-serif';
      }
    }
    // empty hint
    if (!store.eeg.length && !bh.length) { ctx.fillStyle = INK3; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(store.connected ? 'no flow data yet' : 'connecting to server…', p.l + pw / 2, p.t + ph / 2); }
    // crosshair + tooltip
    if (this.hoverX !== null && this.hoverX >= p.l && this.hoverX <= p.l + pw) {
      const t = tStart + ((this.hoverX - p.l) / pw) * (tEnd - tStart);
      const x = Math.round(this.hoverX) + 0.5;
      ctx.strokeStyle = INK3; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, p.t); ctx.lineTo(x, p.t + ph); ctx.stroke(); ctx.setLineDash([]);
      // nearest EEG
      let eBest = null, eD = Infinity;
      for (const pt of store.eeg) { const d = Math.abs(pt[0] - t); if (d < eD) { eD = d; eBest = pt; } }
      if (eBest && eD > 30) eBest = null;
      // band value in effect (last sample at or before t)
      let bBest = null;
      for (let i = bh.length - 1; i >= 0; i--) if (bh[i][0] <= t) { bBest = bh[i]; break; }
      if (bBest && t - bBest[0] > 900) bBest = null;
      if (eBest && isNum(eBest[1])) { ctx.fillStyle = COLOR_EEG; ctx.beginPath(); ctx.arc(xOf(eBest[0]), yOf(eBest[1]), 4, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); }
      if (bBest && isNum(bBest[1])) { ctx.fillStyle = COLOR_BAND; ctx.beginPath(); ctx.arc(x, yOf(bBest[1]), 4, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); }
      const html = ttRow('time', fmtClock(t, true))
        + ttRow('EEG flow', eBest ? fmt(eBest[1]) : '—', COLOR_EEG)
        + ttRow('band flow', bBest ? fmt(bBest[1]) + (bBest[2] ? ' (held)' : '') + ' @ ' + fmtClock(bBest[0], true) : '—', COLOR_BAND);
      if (this.hoverClient) tooltip.show(this.hoverClient[0], this.hoverClient[1], html);
    }
  }
}

// ----------------------------------------------------------------------------
// Widget: LAG PLOT (r vs lag, small)
// ----------------------------------------------------------------------------
class LagPlot {
  constructor() { this.canvas = $('cvLag'); }
  draw() {
    const { ctx, w, h } = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const s = store.analysis && store.analysis.stats;
    const lags = s && Array.isArray(s.lags) ? s.lags.filter((l) => Array.isArray(l) && isNum(l[0]) && isNum(l[1])) : [];
    const p = { l: 16, r: 6, t: 3, b: 10 }, pw = w - p.l - p.r, ph = h - p.t - p.b;
    const xOf = (lag) => p.l + ((lag + 180) / 360) * pw;
    const yOf = (r) => p.t + (1 - (clamp(r, -1, 1) + 1) / 2) * ph;
    ctx.strokeStyle = GRID; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(p.l, Math.round(yOf(0)) + 0.5); ctx.lineTo(p.l + pw, Math.round(yOf(0)) + 0.5); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(Math.round(xOf(0)) + 0.5, p.t); ctx.lineTo(Math.round(xOf(0)) + 0.5, p.t + ph); ctx.stroke();
    ctx.fillStyle = INK2; ctx.font = '8px "Segoe UI", system-ui, sans-serif'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
    ctx.fillText('−180', p.l, p.t + ph + 1); ctx.textAlign = 'right'; ctx.fillText('+180', p.l + pw, p.t + ph + 1);
    ctx.textBaseline = 'middle'; ctx.fillText('+1', p.l - 2, p.t + 3); ctx.fillText('−1', p.l - 2, p.t + ph - 3);
    if (!lags.length) { ctx.textAlign = 'center'; ctx.fillStyle = INK3; ctx.fillText('n/a', p.l + pw / 2, p.t + ph / 2); return; }
    ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.beginPath();
    lags.forEach((l, i) => { const x = xOf(l[0]), y = yOf(l[1]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.stroke();
    if (isNum(s.best_lag_s) && isNum(s.r_at_best_lag)) {
      ctx.fillStyle = COLOR_BAND; ctx.beginPath(); ctx.arc(xOf(s.best_lag_s), yOf(s.r_at_best_lag), 4, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
}

// ----------------------------------------------------------------------------
// Stats tiles + big flow tiles + validation table
// ----------------------------------------------------------------------------
const statsTiles = {
  update() {
    const a = store.analysis, s = a && a.stats ? a.stats : {};
    $('stN').textContent = isNum(s.n_pairs) ? String(s.n_pairs) : '—';
    $('stR').textContent = isNum(s.r) ? fmt(s.r) + (Array.isArray(s.r_ci) && isNum(s.r_ci[0]) ? ' [' + fmt(s.r_ci[0]) + ', ' + fmt(s.r_ci[1]) + ']' : '') : '—';
    $('stR5').textContent = fmt(s.r_last_5min);
    $('stMae').textContent = fmt(s.mae);
    $('stBias').textContent = isNum(s.band_minus_eeg_mean) ? (s.band_minus_eeg_mean >= 0 ? '+' : '') + fmt(s.band_minus_eeg_mean) : '—';
    $('stLag').textContent = isNum(s.best_lag_s) ? (s.best_lag_s > 0 ? '+' : '') + s.best_lag_s + ' s · r ' + fmt(s.r_at_best_lag) : '—';
    const f = s.felt || {};
    $('stFelt').textContent = isNum(f.n) ? f.n + ' · ' + fmt(f.r_band) + ' · ' + fmt(f.r_eeg) : '—';
    // big tiles
    const fe = a && a.flow_eeg;
    $('tileEeg').textContent = fe && isNum(fe.flow_eeg) ? fe.flow_eeg.toFixed(2) : '—';
    $('tileEegSub').textContent = fe ? (fe.baseline_ready === false ? 'learning baseline…' : (isNum(fe.n_good) ? fe.n_good + ' good channels' : '')) : (store.connected ? 'no epoch yet' : '');
    const b = store.band;
    $('tileBand').textContent = b && isNum(b.flow) ? b.flow.toFixed(2) : '—';
    $('tileBandSub').textContent = b ? [b.held ? 'held' : '', b.settling ? 'settling' : '', isNum(b.age_s) ? fmtAge(b.age_s) + ' ago' : ''].filter(Boolean).join(' · ') : 'no band sample';
  },
};

const validationTable = {
  open: false,
  init() {
    $('btnTable').addEventListener('click', () => { this.open = !this.open; $('tableWrap').hidden = !this.open; if (this.open) this.refresh(); });
    $('btnTableRefresh').addEventListener('click', () => this.refresh());
  },
  async refresh() {
    const j = await getJSON('/api/history');
    if (j && Array.isArray(j.pairs)) store.pairs = j.pairs;
    this.render();
  },
  render() {
    const tb = $('pairsTable').querySelector('tbody');
    const rows = store.pairs.slice(-500).reverse();
    tb.innerHTML = rows.map((r) => '<tr><td>' + fmtClock(r[0], true) + '</td><td class="num">' + fmt(r[1]) + '</td><td class="num">' + fmt(r[2]) + '</td><td class="num">' + (isNum(r[1]) && isNum(r[2]) ? fmt(r[1] - r[2]) : '—') + '</td></tr>').join('');
    $('tableInfo').textContent = store.pairs.length + ' pairs (newest first' + (store.pairs.length > 500 ? ', last 500 shown' : '') + ')';
  },
};

// ----------------------------------------------------------------------------
// Widget: BAND PANEL
// ----------------------------------------------------------------------------
const bandPanel = {
  init() {
    $('selSource').addEventListener('change', (e) => cmd('set_band_source', { source: e.target.value }));
    $('selWearer').addEventListener('change', (e) => {
      const w = store.wearers.find((x) => x.user_id === e.target.value);
      if (w) cmd('set_band_source', { source: 'supabase', user_id: w.user_id, band: w.band });
    });
    $('btnWearers').addEventListener('click', () => this.loadWearers());
  },
  async loadWearers() {
    const j = await getJSON('/api/wearers');
    store.wearers = j && Array.isArray(j.wearers) ? j.wearers : [];
    const sel = $('selWearer');
    const cur = sel.value;
    const short = store.status && store.status.band ? store.status.band.user_id_short : null;
    sel.innerHTML = '<option value="">— select wearer —</option>' + store.wearers.map((w) =>
      '<option value="' + esc(w.user_id) + '">' + esc((w.short || '') + ' · ' + (w.band || '—') + ' · ' + fmtAge(w.age_s) + ' ago · ' + (isNum(w.rows_last_hour) ? w.rows_last_hour : '?') + '/h') + '</option>').join('');
    const match = store.wearers.find((w) => short && w.short === short);
    sel.value = cur && store.wearers.some((w) => w.user_id === cur) ? cur : (match ? match.user_id : '');
  },
  tag(text, cls) { return '<span class="tag' + (cls ? ' ' + cls : '') + '">' + esc(text) + '</span>'; },
  update() {
    const b = store.band, sb = store.status && store.status.band ? store.status.band : null;
    $('bpFlow').textContent = b && isNum(b.flow) ? b.flow.toFixed(2) : '—';
    let tags = '';
    if (b) {
      if (b.held) tags += this.tag('held', 'warn');
      if (b.settling) tags += this.tag('settling');
      if (b.poor_contact) tags += this.tag('poor contact', 'bad');
      if (b.asleep) tags += this.tag('asleep');
      if (b.low_battery) tags += this.tag('low battery', 'warn');
    }
    $('bpTags').innerHTML = tags;
    $('bpMode').textContent = b ? (b.mode === 0 ? 'Standard monitoring' : b.mode === 1 ? 'Enhanced monitoring' : (b.mode == null ? '—' : String(b.mode))) : '—';
    $('bpWindow').textContent = b && b.window != null ? String(b.window) + (isNum(b.flow_state) ? ' · state ' + b.flow_state : '') : '—';
    $('bpHr').textContent = b && isNum(b.hr) ? b.hr.toFixed(0) + ' bpm' : '—';
    $('bpAnom').textContent = b && isNum(b.anom) ? b.anom.toFixed(2) : '—';
    $('bpName').textContent = (b && b.band) || (sb && sb.unit) || '—';
    $('bpSource').textContent = (b && b.source) || (sb && sb.source) || '—';
    $('bpAge').textContent = b && isNum(b.age_s) ? 'sample ' + fmtAge(b.age_s) + ' ago' : (sb && isNum(sb.last_line_age_s) ? 'last line ' + fmtAge(sb.last_line_age_s) + ' ago' : '—');
    if (sb) {
      $('bpLink').textContent = (sb.connected ? 'connected' : 'not connected') + (sb.error ? ' · ' + sb.error : '') + (isNum(sb.lines) ? ' · ' + sb.lines + ' lines' : '') + (sb.mac ? ' · ' + sb.mac : '') + (sb.port && sb.source === 'usb' ? ' · ' + sb.port : '');
      const sel = $('selSource');
      if (document.activeElement !== sel && sb.source && sel.value !== sb.source) sel.value = sb.source;
      const isSb = sb.source === 'supabase';
      $('wearerRow').hidden = !isSb;
      if (isSb && !this._wearersLoadedFor) { this._wearersLoadedFor = true; this.loadWearers(); }
      if (!isSb) this._wearersLoadedFor = false;
      const seen = Array.isArray(sb.seen) ? sb.seen : [];
      const showSeen = sb.source === 'ble' && !sb.connected && seen.length > 0;
      $('seenList').hidden = !showSeen;
      if (showSeen) {
        const html = '<div class="muted">BLE devices seen:</div>' + seen.map((s) => '<div class="seen-item"><span>' + esc(s) + '</span><button class="btn btn-mini" data-addr="' + esc(s) + '">connect</button></div>').join('');
        if (html !== this._seenHtml) {   // only rebuild when the list changes (status ticks every second)
          this._seenHtml = html;
          $('seenList').innerHTML = html;
          $('seenList').querySelectorAll('button').forEach((btn) => btn.addEventListener('click', () => cmd('set_band_source', { source: 'ble', address: btn.dataset.addr })));
        }
      } else this._seenHtml = null;
    } else $('bpLink').textContent = '—';
  },
};

// ----------------------------------------------------------------------------
// Widget: FELT FLOW
// ----------------------------------------------------------------------------
const feltFlow = {
  init() {
    const sl = $('feltSlider');
    sl.addEventListener('input', () => { $('feltVal').textContent = parseFloat(sl.value).toFixed(2); });
    $('btnFelt').addEventListener('click', () => cmd('mark', { kind: 'felt_flow', value: parseFloat(sl.value) }));
    document.querySelectorAll('#wFelt button[data-mark]').forEach((b) => b.addEventListener('click', () => cmd('mark', { kind: b.dataset.mark })));
    $('btnNote').addEventListener('click', () => {
      const txt = $('noteText').value.trim();
      if (!txt) return;
      cmd('mark', { kind: 'note', value: txt, note: txt }).then(() => { $('noteText').value = ''; });
    });
    $('noteText').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('btnNote').click(); });
  },
  log(m) {
    const el = $('markLog');
    const line = document.createElement('div');
    line.textContent = fmtClock(m.t, true) + ' · ' + m.kind + (m.value != null && m.value !== '' ? ' ' + (isNum(m.value) ? m.value.toFixed(2) : m.value) : '') + (m.note ? ' · ' + m.note : '');
    el.prepend(line);
    while (el.children.length > 20) el.removeChild(el.lastChild);
  },
};

// ----------------------------------------------------------------------------
// Widget: SESSIONS (collapsible footer) + report modal
// ----------------------------------------------------------------------------
const sessions = {
  init() {
    $('wSessions').addEventListener('toggle', (e) => { if (e.target.open) this.refresh(); });
    $('btnSessionsRefresh').addEventListener('click', () => this.refresh());
    $('modalClose').addEventListener('click', () => { $('modal').hidden = true; });
    $('modal').addEventListener('click', (e) => { if (e.target === $('modal')) $('modal').hidden = true; });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('modal').hidden = true; });
  },
  async refresh() {
    const j = await getJSON('/api/sessions');
    store.sessions = j && Array.isArray(j.sessions) ? j.sessions : [];
    $('sessCount').textContent = store.sessions.length ? store.sessions.length + ' sessions' : '';
    const tb = $('sessionsTable').querySelector('tbody');
    tb.innerHTML = store.sessions.map((s, i) => '<tr><td title="' + esc(s.dir || '') + '">' + esc(s.name || '') + '</td><td class="num">' + (isNum(s.epochs) ? s.epochs : '—') + '</td><td class="num">' + (isNum(s.band) ? s.band : '—') + '</td><td class="num">' + (isNum(s.marks) ? s.marks : '—') + '</td><td>' + (s.has_report ? 'yes' : '—') + '</td><td><button class="btn btn-mini" data-i="' + i + '">Report</button></td></tr>').join('')
      || '<tr><td colspan="6" class="muted">no sessions recorded yet</td></tr>';
    tb.querySelectorAll('button[data-i]').forEach((b) => b.addEventListener('click', () => this.report(store.sessions[+b.dataset.i])));
  },
  async report(s) {
    if (!s) return;
    $('modalTitle').textContent = 'Report · ' + s.name;
    $('modalBody').innerHTML = '<p class="muted">running report.py…</p>';
    $('modal').hidden = false;
    const j = await getJSON('/api/report?session=' + encodeURIComponent(s.name));
    if (!j || j.ok === false) { $('modalBody').innerHTML = '<p>report failed: ' + esc(j && j.error ? j.error : 'no response') + '</p>'; return; }
    $('modalBody').innerHTML = renderMarkdown(j.markdown || '');
    this.refresh();
  },
};

/** Tiny markdown -> HTML: headings, bold, inline code, bullets, tables, fenced code, paragraphs. */
function renderMarkdown(md) {
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`([^`]+)`/g, '<code>$1</code>');
  const lines = String(md).replace(/\r/g, '').split('\n');
  let out = '', i = 0;
  while (i < lines.length) {
    const ln = lines[i];
    if (/^```/.test(ln)) { let buf = []; i++; while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]); i++; out += '<pre>' + esc(buf.join('\n')) + '</pre>'; continue; }
    const hm = /^(#{1,3})\s+(.*)/.exec(ln);
    if (hm) { out += '<h' + hm[1].length + '>' + inline(hm[2]) + '</h' + hm[1].length + '>'; i++; continue; }
    if (/^\s*[-*]\s+/.test(ln)) { out += '<ul>'; while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) { out += '<li>' + inline(lines[i].replace(/^\s*[-*]\s+/, '')) + '</li>'; i++; } out += '</ul>'; continue; }
    if (/^\s*\|/.test(ln)) {
      const rows = []; while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      out += '<table>';
      rows.forEach((r, ri) => { if (/^\s*\|?\s*:?-{2,}/.test(r)) return; const tag = ri === 0 ? 'th' : 'td'; out += '<tr>' + cells(r).map((c) => '<' + tag + '>' + inline(c) + '</' + tag + '>').join('') + '</tr>'; });
      out += '</table>'; continue;
    }
    if (ln.trim() === '') { i++; continue; }
    let buf = [ln]; i++;
    while (i < lines.length && lines[i].trim() !== '' && !/^(#|```|\s*[-*]\s|\s*\|)/.test(lines[i])) buf.push(lines[i++]);
    out += '<p>' + inline(buf.join(' ')) + '</p>';
  }
  return out;
}

// ----------------------------------------------------------------------------
// Nav bar: stream/record buttons, filter controls, status chips
// ----------------------------------------------------------------------------
const navBar = {
  init() {
    $('btnStream').addEventListener('click', () => {
      const streaming = store.status && store.status.board && store.status.board.streaming;
      cmd(streaming ? 'stop' : 'start');
    });
    $('btnRecord').addEventListener('click', () => {
      const active = store.status && store.status.recorder && store.status.recorder.active;
      if (active) cmd('record_stop').then(() => sessions.refresh());
      else {
        const label = ($('recLabel').value || '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 32);
        cmd('record_start', { label });
      }
    });
    const applyFilter = () => cmd('set_filter', { notch: parseInt($('selNotch').value, 10), bp_lo: parseFloat($('bpLo').value), bp_hi: parseFloat($('bpHi').value) });
    $('selNotch').addEventListener('change', applyFilter);
    $('bpLo').addEventListener('change', applyFilter);
    $('bpHi').addEventListener('change', applyFilter);
    $('selWindow').addEventListener('change', (e) => { store.windowS = parseInt(e.target.value, 10) || 10; store.ringDirty = true; });
    $('selScale').addEventListener('change', (e) => { store.scale = e.target.value; store.ringDirty = true; });
  },
  setChip(id, text, cls) { const el = $(id); el.textContent = text; el.className = 'chip' + (cls ? ' ' + cls : ''); },
  update() {
    const s = store.status;
    if (!s) return;
    const b = s.board || {}, bd = s.band || {}, r = s.recorder || {}, f = s.filter || {};
    // stream button
    const btn = $('btnStream');
    btn.textContent = b.streaming ? 'STOP STREAM' : 'START STREAM';
    btn.className = 'btn ' + (b.streaming ? 'btn-stop' : 'btn-blue');
    // record
    const rb = $('btnRecord');
    rb.textContent = r.active ? 'STOP REC' : 'RECORD';
    rb.className = 'btn ' + (r.active ? 'btn-stop' : 'btn-blue');
    $('recStatus').hidden = !r.active;
    $('recLabel').hidden = !!r.active;
    if (r.active) {
      $('recElapsed').textContent = fmtMMSS(r.since_s);
      const c = r.counts || {};
      $('recCounts').textContent = 'eeg ' + (c.eeg || 0) + ' · epochs ' + (c.epochs || 0) + ' · band ' + (c.band || 0) + ' · marks ' + (c.marks || 0);
    }
    // filter controls (don't fight the user while editing)
    const setIf = (id, v) => { const el = $(id); if (document.activeElement !== el && v != null && String(el.value) !== String(v)) el.value = String(v); };
    if (isNum(f.notch)) setIf('selNotch', Math.round(f.notch));
    if (isNum(f.bp_lo)) setIf('bpLo', +f.bp_lo);
    if (isNum(f.bp_hi)) setIf('bpHi', +f.bp_hi);
    // chips
    const noData = !b.streaming || !isNum(b.last_packet_age_s) || b.last_packet_age_s > 2;
    this.setChip('chipBoard', 'board ' + (b.kind || '—') + ' · ' + (b.port || '—') + ' · ' + (b.n_ch || '—') + ' ch · ' + (isNum(b.fs) ? Math.round(b.fs) : '—') + ' Hz · ' + (b.packets || 0) + ' pkts · ' + (b.dropped || 0) + ' dropped' + (noData ? ' · no data' : ''),
      noData ? (b.streaming ? 'chip-bad' : 'chip-warn') : 'chip-ok');
    const age = store.band && isNum(store.band.age_s) ? store.band.age_s : bd.last_line_age_s;
    this.setChip('chipBand', 'band ' + (bd.source || 'none') + ' · ' + (bd.unit || '—') + ' · ' + (bd.connected ? 'connected' : (bd.error ? 'error: ' + bd.error : 'not connected')) + (isNum(age) ? ' · ' + fmtAge(age) + ' ago' : ''),
      bd.error ? 'chip-bad' : (bd.connected ? 'chip-ok' : 'chip-warn'));
    this.setChip('chipRec', r.active ? 'rec ' + fmtMMSS(r.since_s) + (r.dir ? ' · ' + String(r.dir).split(/[\\/]/).pop() : '') : 'rec idle', r.active ? 'chip-bad' : '');
  },
};

// ----------------------------------------------------------------------------
// Legends (FFT + band power share the channel legend)
// ----------------------------------------------------------------------------
function buildLegends() {
  const n = store.nCh || 8;
  const html = Array.from({ length: n }, (_, c) => '<span class="lg' + (chEnabled(c) ? '' : ' off') + '" data-c="' + c + '"><i class="sw" style="background:' + chColor(c) + '"></i>' + esc((c + 1) + ' ' + chLabel(c)) + '</span>').join('');
  $('legendFft').innerHTML = html;
  $('legendBand').innerHTML = html;
}

// ----------------------------------------------------------------------------
// Event handling
// ----------------------------------------------------------------------------
const widgets = {};

function onHello(d) {
  store.hello = d;
  store.fs = isNum(d.fs) && d.fs > 0 ? d.fs : 250;
  store.nCh = isNum(d.n_ch) && d.n_ch > 0 ? d.n_ch : (Array.isArray(d.labels) ? d.labels.length : 8);
  store.labels = Array.isArray(d.labels) ? d.labels : [];
  store.colors = Array.isArray(d.channel_colors) && d.channel_colors.length ? d.channel_colors : DEFAULT_COLORS.slice();
  store.positions = d.positions || {};
  store.ring = new Ring(store.nCh, Math.round(MAX_WINDOW_S * store.fs));
  store.ringDirty = true;
  const h = d.history || {};
  store.eeg = Array.isArray(h.eeg) ? h.eeg.filter((p) => Array.isArray(p) && isNum(p[0])) : [];
  store.bandHist = Array.isArray(h.band) ? h.band.filter((p) => Array.isArray(p) && isNum(p[0])) : [];
  store.marks = Array.isArray(h.marks) ? h.marks.filter((p) => Array.isArray(p) && isNum(p[0])) : [];
  store.pairs = Array.isArray(h.pairs) ? h.pairs : [];
  store.lastAnalysisT = store.eeg.length ? store.eeg[store.eeg.length - 1][0] : null;
  buildLegends();
  widgets.head.layout();
  redrawAnalysisWidgets();
  widgets.flow.draw();
  widgets.lag.draw();
  statsTiles.update();
  if (validationTable.open) validationTable.render();
  // fill status/analysis immediately rather than waiting for the next tick
  getJSON('/api/state').then((j) => { if (j && j.status) onStatus(j.status); if (j && j.analysis) onAnalysis(j.analysis); });
}

function onSamples(d) {
  if (!store.ring) store.ring = new Ring(store.nCh, Math.round(MAX_WINDOW_S * store.fs));
  if (isNum(d.fs) && d.fs > 0 && d.fs !== store.fs) { store.fs = d.fs; store.ring = new Ring(store.nCh, Math.round(MAX_WINDOW_S * store.fs)); }
  if (Array.isArray(d.ch) && d.ch.length) store.ring.push(d.ch);
  if (Array.isArray(d.raw_peak)) store.lastRawPeak = d.raw_peak;
  store.ringDirty = true;
}

function trimHistory() {
  const cut = (arr) => { if (!arr.length) return arr; const tEnd = arr[arr.length - 1][0]; let k = 0; while (k < arr.length && arr[k][0] < tEnd - HISTORY_MAX_S) k++; return k ? arr.slice(k) : arr; };
  store.eeg = cut(store.eeg); store.bandHist = cut(store.bandHist); store.marks = cut(store.marks);
}

function onAnalysis(d) {
  store.analysis = d;
  if (Array.isArray(d.enabled)) {
    const changed = d.enabled.length !== store.enabled.length || d.enabled.some((v, i) => !!v !== !!store.enabled[i]);
    store.enabled = d.enabled.map((v) => !!v);
    if (changed) { buildLegends(); store.ringDirty = true; }
  }
  if (d.band) store.band = d.band;
  // append the EEG flow point (dedupe on t)
  const fe = d.flow_eeg;
  if (fe && isNum(d.t) && d.t !== store.lastAnalysisT) {
    store.eeg.push([d.t, isNum(fe.flow_eeg) ? fe.flow_eeg : null]);
    store.lastAnalysisT = d.t;
    if (store.eeg.length % 200 === 0) trimHistory();
  }
  redrawAnalysisWidgets();
  widgets.flow.draw();
  widgets.lag.draw();
  statsTiles.update();
  bandPanel.update();
}

function onStatus(d) {
  store.status = d;
  navBar.update();
  bandPanel.update();
}

function onBand(d) {
  if (!d) return;
  store.band = d;
  if (isNum(d.t) && isNum(d.flow)) {
    const last = store.bandHist[store.bandHist.length - 1];
    if (!last || last[0] !== d.t) store.bandHist.push([d.t, d.flow, !!d.held]);
  }
  statsTiles.update();
  bandPanel.update();
  widgets.flow.draw();
}

function onMark(d) {
  if (!d || !isNum(d.t)) return;
  store.marks.push([d.t, d.kind, d.value != null ? d.value : (d.note || null)]);
  feltFlow.log(d);
  widgets.flow.draw();
}

function redrawAnalysisWidgets() {
  widgets.fft.draw();
  widgets.band.draw();
  widgets.head.draw();
}

// ----------------------------------------------------------------------------
// SSE connection with reconnect backoff
// ----------------------------------------------------------------------------
let es = null, backoff = 1000, reconnectTimer = null;
function connect() {
  if (es) { try { es.close(); } catch (e) { /* ignore */ } }
  navBar.setChip('chipConn', 'connecting to server…', 'chip-warn');
  es = new EventSource('/events');
  const parse = (ev) => { try { return JSON.parse(ev.data); } catch (e) { console.warn('bad event json', ev.type); return null; } };
  es.addEventListener('open', () => { store.connected = true; backoff = 1000; navBar.setChip('chipConn', 'connected', 'chip-ok'); });
  es.addEventListener('hello', (ev) => { const d = parse(ev); if (d) onHello(d); });
  es.addEventListener('samples', (ev) => { const d = parse(ev); if (d) onSamples(d); });
  es.addEventListener('analysis', (ev) => { const d = parse(ev); if (d) onAnalysis(d); });
  es.addEventListener('status', (ev) => { const d = parse(ev); if (d) onStatus(d); });
  es.addEventListener('band', (ev) => { const d = parse(ev); if (d) onBand(d); });
  es.addEventListener('mark', (ev) => { const d = parse(ev); if (d) onMark(d); });
  es.addEventListener('error', () => {
    store.connected = false;
    navBar.setChip('chipConn', 'disconnected · retry in ' + Math.round(backoff / 1000) + ' s', 'chip-bad');
    try { es.close(); } catch (e) { /* ignore */ }
    es = null;
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 15000);
  });
}

// ----------------------------------------------------------------------------
// Boot
// ----------------------------------------------------------------------------
function boot() {
  widgets.ts = new TimeSeries();
  widgets.fft = new FFTPlot();
  widgets.band = new BandPower();
  widgets.head = new HeadPlot();
  widgets.flow = new FlowChart();
  widgets.lag = new LagPlot();
  navBar.init();
  bandPanel.init();
  feltFlow.init();
  sessions.init();
  validationTable.init();
  buildLegends();
  redrawAnalysisWidgets();
  widgets.flow.draw();
  widgets.lag.draw();
  widgets.ts.forceDraw = true;

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { redrawAnalysisWidgets(); widgets.flow.draw(); widgets.lag.draw(); widgets.ts.forceDraw = true; }, 80);
  });

  const loop = (now) => { widgets.ts.tick(now); requestAnimationFrame(loop); };
  requestAnimationFrame(loop);

  // age counters tick once a second even if the server is quiet
  setInterval(() => { if (store.status) navBar.update(); }, 1000);

  connect();
}

document.addEventListener('DOMContentLoaded', boot);
