'use strict';

// ---------- state ----------
const KEY = 'impulse-morph-ui';
const state = (() => {
  try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { return {}; }
})();
if (state.dark === undefined) state.dark = true;
if (state.ann === undefined) state.ann = true;
if (state.hmax === undefined) state.hmax = 512;
if (state.frame === undefined) state.frame = 128;
function save(patch) {
  Object.assign(state, patch);
  localStorage.setItem(KEY, JSON.stringify(state));
}

// ---------- FFT (shared by synthesis and analysis) ----------
const WIN = 2048, STRIDE = 2048, NFRAMES = 257, NH = WIN / 2 + 1;
const AUDIO_SR = 88200;
const FUND = AUDIO_SR / WIN; // 43.066 Hz

function fftInPlace(re, im) {
  const N = re.length;
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < N; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = i + k + half;
        const vr = re[b] * cr - im[b] * ci;
        const vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr; im[a] += vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

// ---------- DIY synthesis ----------
// Each frame is one 2048-sample cycle built additively from a harmonic
// amplitude recipe A_h(t), t = frame/256. Phases follow a log down-TSP
// (ESS style: equal group delay per octave, high harmonics first): the
// pulse is dispersed over TSP_SPAN samples, which lowers the crest factor
// to ~3-5 so per-cycle peak normalization yields a much higher RMS
// (+7..+21 dB over zero phase depending on the frame).
function smoothstep(a, b, x) {
  const u = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return u * u * (3 - 2 * u);
}
const tilt = t => -4.8 * smoothstep(0, 0.25, t) + 1.8 * smoothstep(0.6, 1, t); // dB/oct
const log2c1 = t => 2 + 3.585 * t;       // F1 center: h4 -> h48 (rising)
const log2c2 = t => 8.966 - 4.966 * t;   // F2 center: h500 -> h16 (falling)

function harmDb(h, t) {
  const l = Math.log2(h);
  let d = tilt(t) * l;
  const gate = smoothstep(0.02, 0.12, t);
  const g1 = l - log2c1(t), g2 = l - log2c2(t);
  d += 18 * gate * Math.exp(-g1 * g1 / (2 * 0.35 * 0.35));
  d += 14 * gate * Math.exp(-g2 * g2 / (2 * 0.5 * 0.5));
  const s = smoothstep(0.15, 0.2, t) * (1 - smoothstep(0.7, 0.75, t));
  if (s > 0 && h === 2 + Math.floor(8 * t)) d += 10 * s;
  if (h % 2 === 0) d -= 26 * smoothstep(0.62, 0.95, t);
  return d;
}

// log down-TSP phase table: group delay decreases linearly in log2(h)
// (equal time per octave), so the cycle is a centered descending sweep
// spanning TSP_SPAN samples — highest harmonics first, fundamental last
const TSP_SPAN = 1600; // dispersion length in samples (78% of the cycle)
const PHI = new Float64Array(WIN / 2);
{
  const H = WIN / 2, t0 = (WIN - TSP_SPAN) / 2;
  let acc = 0;
  for (let h = 1; h < H; h++) {
    acc += t0 + TSP_SPAN * (1 - Math.log2(h) / Math.log2(H));
    PHI[h] = 2 * Math.PI / WIN * acc;
  }
}

const NSAMP = NFRAMES * WIN;
const audio = new Float32Array(NSAMP);
{
  const re = new Float64Array(WIN), im = new Float64Array(WIN);
  for (let p = 0; p < NFRAMES; p++) {
    const t = p / (NFRAMES - 1);
    re.fill(0); im.fill(0);
    for (let h = 1; h < WIN / 2; h++) {
      const a = Math.pow(10, harmDb(h, t) / 20);
      re[h] = a * Math.cos(PHI[h]); im[h] = a * Math.sin(PHI[h]);
      re[WIN - h] = re[h]; im[WIN - h] = -im[h];
    }
    fftInPlace(re, im);
    let peak = 0;
    for (let i = 0; i < WIN; i++) { const v = Math.abs(re[i]); if (v > peak) peak = v; }
    const sc = 0.95 / (peak || 1);
    const off = p * STRIDE;
    for (let i = 0; i < WIN; i++) audio[off + i] = re[i] * sc;
  }
}

// ---------- per-frame FFT analysis ----------
function fftMag(src) {
  const re = Float64Array.from(src), im = new Float64Array(WIN);
  fftInPlace(re, im);
  const m = new Float32Array(NH);
  for (let k = 0; k < NH; k++) m[k] = Math.hypot(re[k], im[k]) / (WIN / 2);
  return m;
}

const dbArr = new Float32Array(NFRAMES * NH);
for (let p = 0; p < NFRAMES; p++) {
  const m = fftMag(audio.subarray(p * STRIDE, p * STRIDE + WIN));
  let fmax = 0;
  for (let k = 1; k < NH; k++) if (m[k] > fmax) fmax = m[k];
  for (let k = 0; k < NH; k++) {
    const v = 20 * Math.log10(m[k] / (fmax || 1) + 1e-12);
    dbArr[p * NH + k] = v < -60 ? -60 : (v > 0 ? 0 : v);
  }
}
const db = (p, h) => dbArr[p * NH + h];

// ---------- color ----------
const RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
  '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
function hex2rgb(h) {
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
}
let LUT = null; // Uint8Array 256*3
function buildLUT() {
  const dark = state.dark;
  const stops = dark
    ? ['#1a1a19'].concat(RAMP.slice().reverse())
    : ['#ffffff'].concat(RAMP);
  const rgb = stops.map(hex2rgb);
  LUT = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    let u = i / 255;
    if (dark) u = Math.pow(u, 1.6); // keep quiet cells near the dark surface
    const t = u * (rgb.length - 1);
    const a = Math.min(Math.floor(t), rgb.length - 2);
    const f = t - a;
    for (let c = 0; c < 3; c++)
      LUT[i * 3 + c] = Math.round(rgb[a][c] * (1 - f) + rgb[a + 1][c] * f);
  }
}
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---------- layout ----------
const ML = 46, MR = 8, MT = 8, MB = 24;
const heat = document.getElementById('heat');
const heatOv = document.getElementById('heatOv');
const cbar = document.getElementById('cbar');
const waveCv = document.getElementById('wave');
const specCv = document.getElementById('spec');
const tooltip = document.getElementById('tooltip');
const dpr = window.devicePixelRatio || 1;

let heatW = 0, heatH = 430;

function setupCanvas(cv, w, h) {
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.style.width = w + 'px';
  cv.style.height = h + 'px';
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

const xPix = f => ML + (heatW - ML - MR) * f / (NFRAMES - 1);
const yPix = h => MT + (heatH - MT - MB) * (1 - Math.log2(h) / Math.log2(state.hmax));
const pix2frame = x => Math.max(0, Math.min(NFRAMES - 1,
  Math.round((x - ML) / (heatW - ML - MR) * (NFRAMES - 1))));
const pix2harm = y => Math.pow(2, (1 - (y - MT) / (heatH - MT - MB)) * Math.log2(state.hmax));

// ---------- heatmap ----------
function drawHeat() {
  const ctx = setupCanvas(heat, heatW, heatH);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const pw = Math.round((heatW - ML - MR) * dpr);
  const ph = Math.round((heatH - MT - MB) * dpr);
  const img = ctx.createImageData(pw, ph);
  const d = img.data;
  const log2max = Math.log2(state.hmax);
  const hOfRow = new Int32Array(ph);
  for (let j = 0; j < ph; j++) {
    let h = Math.round(Math.pow(2, (1 - j / (ph - 1)) * log2max));
    hOfRow[j] = Math.max(1, Math.min(state.hmax, Math.min(h, NH - 1)));
  }
  const fOfCol = new Int32Array(pw);
  for (let i = 0; i < pw; i++)
    fOfCol[i] = Math.min(NFRAMES - 1, Math.floor(i / pw * NFRAMES));
  for (let j = 0; j < ph; j++) {
    const hrow = hOfRow[j];
    let o = j * pw * 4;
    for (let i = 0; i < pw; i++, o += 4) {
      const v = dbArr[fOfCol[i] * NH + hrow];
      const li = Math.round((v + 60) / 60 * 255) * 3;
      d[o] = LUT[li]; d[o + 1] = LUT[li + 1]; d[o + 2] = LUT[li + 2]; d[o + 3] = 255;
    }
  }
  ctx.putImageData(img, Math.round(ML * dpr), Math.round(MT * dpr));

  // axes
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = '10px system-ui';
  ctx.fillStyle = cssVar('--ink-muted');
  ctx.strokeStyle = cssVar('--border');
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let h = 1; h <= state.hmax; h *= 2)
    ctx.fillText(String(h), ML - 5, yPix(h));
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let f = 0; f <= 256; f += 32)
    ctx.fillText(String(f), xPix(f), heatH - MB + 6);
  ctx.save();
  ctx.translate(11, (MT + heatH - MB) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('倍音番号', 0, 0);
  ctx.restore();

  if (state.ann) drawAnnotations(ctx);
}

const ANNS = [
  { ax: 1, ay: 80, lx: 8, ly: 180, t: 'log down TSP: 全倍音 0 dB' },
  { ax: 24, ay: 2, lx: 16, ly: 1.35, t: 'ティルト 0→−4.8 dB/oct' },
  { ax: 128, ay: 14, lx: 48, ly: 40, t: 'F1: h4→h48 上昇 (+18 dB)' },
  { ax: 64, ay: 211, lx: 90, ly: 420, t: 'F2: h500→h16 下降 (+14 dB)' },
  { ax: 208, ay: 30, lx: 170, ly: 95, t: 'F1×F2 交差 (f≈208)' },
  { ax: 96, ay: 5, lx: 120, ly: 2.2, t: '階段ブースト h3→h7 (+10 dB)' },
  { ax: 230, ay: 90, lx: 175, ly: 280, t: '偶数次 −26 dB（奇数次のみへ）' },
];
function drawAnnotations(ctx) {
  ctx.font = '11px system-ui';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (const a of ANNS) {
    if (a.ay > state.hmax || a.ly > state.hmax) continue;
    const x1 = xPix(a.ax), y1 = yPix(a.ay);
    const x2 = xPix(a.lx), y2 = yPix(a.ly);
    ctx.strokeStyle = cssVar('--ink-muted');
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    const w = ctx.measureText(a.t).width;
    ctx.fillStyle = cssVar('--surface');
    ctx.globalAlpha = 0.88;
    ctx.fillRect(x2 - 3, y2 - 9, w + 8, 18);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = cssVar('--border');
    ctx.strokeRect(x2 - 3, y2 - 9, w + 8, 18);
    ctx.fillStyle = cssVar('--ink-2');
    ctx.fillText(a.t, x2 + 1, y2 + 1);
  }
}

function drawOverlay(hover) {
  const ctx = setupCanvas(heatOv, heatW, heatH);
  ctx.clearRect(0, 0, heatW, heatH);
  // current frame marker
  {
    const x = xPix(state.frame);
    ctx.strokeStyle = cssVar('--accent');
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, MT); ctx.lineTo(x, heatH - MB);
    ctx.stroke();
    ctx.fillStyle = cssVar('--accent');
    ctx.beginPath();
    ctx.moveTo(x - 4, MT); ctx.lineTo(x + 4, MT); ctx.lineTo(x, MT + 6);
    ctx.closePath(); ctx.fill();
  }
  if (hover) {
    ctx.strokeStyle = cssVar('--ink-muted');
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(hover.x, MT); ctx.lineTo(hover.x, heatH - MB);
    ctx.moveTo(ML, hover.y); ctx.lineTo(heatW - MR, hover.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawColorbar() {
  const ctx = setupCanvas(cbar, 46, heatH);
  ctx.clearRect(0, 0, 46, heatH);
  const top = MT, bot = heatH - MB;
  for (let y = top; y < bot; y++) {
    const t = 1 - (y - top) / (bot - top);
    const li = Math.round(t * 255) * 3;
    ctx.fillStyle = `rgb(${LUT[li]},${LUT[li + 1]},${LUT[li + 2]})`;
    ctx.fillRect(2, y, 12, 1);
  }
  ctx.strokeStyle = cssVar('--border');
  ctx.strokeRect(2, top, 12, bot - top);
  ctx.font = '10px system-ui';
  ctx.fillStyle = cssVar('--ink-muted');
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (let v = 0; v >= -60; v -= 15) {
    const y = top + (bot - top) * (-v / 60);
    ctx.fillText(String(v), 17, y);
  }
}

// ---------- detail panel ----------
function drawDetail() {
  const p = state.frame;
  document.getElementById('detailTitle').textContent = `フレーム詳細 — frame ${p}`;

  // waveform
  {
    const boxW = waveCv.parentElement.clientWidth - 2;
    const h = 170;
    const ctx = setupCanvas(waveCv, boxW, h);
    ctx.clearRect(0, 0, boxW, h);
    ctx.strokeStyle = cssVar('--grid');
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(boxW, h / 2); ctx.stroke();
    const fr = audio.subarray(p * STRIDE, p * STRIDE + WIN);
    ctx.strokeStyle = cssVar('--accent');
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < WIN; i++) {
      const x = i / (WIN - 1) * boxW;
      const y = h / 2 - fr[i] * (h / 2 - 6);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  // spectrum stems (log2 x, dB y)
  {
    const boxW = specCv.parentElement.clientWidth - 2;
    const h = 210, mlt = 30, mbt = 18;
    const ctx = setupCanvas(specCv, boxW, h);
    ctx.clearRect(0, 0, boxW, h);
    const log2max = Math.log2(state.hmax);
    const sx = hh => mlt + (boxW - mlt - 6) * Math.log2(hh) / log2max;
    const sy = v => 4 + (h - 4 - mbt) * (-v / 60);
    ctx.font = '9px system-ui';
    ctx.fillStyle = cssVar('--ink-muted');
    ctx.strokeStyle = cssVar('--grid');
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let v = 0; v >= -60; v -= 20) {
      ctx.fillText(String(v), mlt - 3, sy(v));
      ctx.beginPath(); ctx.moveTo(mlt, sy(v)); ctx.lineTo(boxW - 6, sy(v)); ctx.stroke();
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let hh = 1; hh <= state.hmax; hh *= 4)
      ctx.fillText(String(hh), sx(hh), h - mbt + 3);
    ctx.strokeStyle = cssVar('--accent');
    ctx.lineWidth = Math.max(1, (boxW - mlt) / state.hmax / 1.2);
    for (let hh = 1; hh <= Math.min(state.hmax, NH - 1); hh++) {
      const v = db(p, hh);
      if (v <= -60) continue;
      ctx.beginPath();
      ctx.moveTo(sx(hh), sy(-60));
      ctx.lineTo(sx(hh), sy(v));
      ctx.stroke();
    }
  }
}

// ---------- audio playback ----------
let actx = null, cycle = null, allSrc = null;
let playingCycle = false, playingAll = false;
const btnCycle = document.getElementById('playCycle');
const btnAll = document.getElementById('playAll');
const XFADE = 0.015;

function ensureCtx() {
  if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
  if (actx.state === 'suspended') actx.resume();
  return actx;
}
// start (or crossfade-switch to) the looping single-cycle buffer of frame p
function startCycle(p) {
  const ctx = ensureCtx();
  const now = ctx.currentTime;
  const buf = ctx.createBuffer(1, WIN, AUDIO_SR);
  buf.copyToChannel(new Float32Array(audio.subarray(p * STRIDE, p * STRIDE + WIN)), 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.loop = true;
  const g = ctx.createGain();
  g.gain.setValueAtTime(0, now);
  g.gain.linearRampToValueAtTime(0.25, now + XFADE);
  src.connect(g).connect(ctx.destination);
  src.start();
  if (cycle) {
    const old = cycle;
    old.g.gain.cancelScheduledValues(now);
    old.g.gain.setValueAtTime(old.g.gain.value, now);
    old.g.gain.linearRampToValueAtTime(0, now + XFADE);
    old.src.stop(now + XFADE + 0.01);
  }
  cycle = { src, g };
  playingCycle = true;
  btnCycle.classList.add('active');
  btnCycle.textContent = '■ 停止';
}
function stopCycle() {
  if (cycle) {
    const ctx = ensureCtx();
    const now = ctx.currentTime;
    cycle.g.gain.cancelScheduledValues(now);
    cycle.g.gain.setValueAtTime(cycle.g.gain.value, now);
    cycle.g.gain.linearRampToValueAtTime(0, now + XFADE);
    cycle.src.stop(now + XFADE + 0.01);
    cycle = null;
  }
  playingCycle = false;
  btnCycle.classList.remove('active');
  btnCycle.textContent = '▶ このフレームをループ再生（43 Hz）';
}
btnCycle.addEventListener('click', () => {
  if (playingCycle) stopCycle(); else startCycle(state.frame);
});
function stopAll() {
  if (allSrc) { try { allSrc.stop(); } catch (e) {} allSrc = null; }
  playingAll = false;
  btnAll.classList.remove('active');
  btnAll.textContent = '▶ 全フレームを連続再生（5.97 s）';
}
btnAll.addEventListener('click', () => {
  if (playingAll) { stopAll(); return; }
  const ctx = ensureCtx();
  const buf = ctx.createBuffer(1, NSAMP, AUDIO_SR);
  buf.copyToChannel(new Float32Array(audio), 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const g = ctx.createGain();
  g.gain.value = 0.8;
  src.connect(g).connect(ctx.destination);
  src.onended = stopAll;
  src.start();
  allSrc = src;
  playingAll = true;
  btnAll.classList.add('active');
  btnAll.textContent = '■ 停止';
});

// ---------- tooltip ----------
function showTip(ev, html) {
  tooltip.innerHTML = html;
  tooltip.classList.remove('hidden');
  const pad = 14;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  const r = tooltip.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 4) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 4) y = ev.clientY - r.height - pad;
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}
function hideTip() { tooltip.classList.add('hidden'); }

// ---------- heatmap interaction: hover tooltip + drag scrub ----------
const heatWrap = document.getElementById('heatWrap');
let dragging = false;
let dragStartedAudio = false; // sound was started by this drag → stop on release

function setFrame(f) {
  if (f === state.frame) return;
  state.frame = f;
  drawDetail();
  if (playingCycle) startCycle(f); // crossfade to the new frame's cycle
}

heatWrap.addEventListener('pointerdown', ev => {
  const r = heat.getBoundingClientRect();
  const x = ev.clientX - r.left;
  if (x < ML || x > heatW - MR) return;
  dragging = true;
  heatWrap.setPointerCapture(ev.pointerId);
  if (!playingCycle) {
    dragStartedAudio = true;
    startCycle(pix2frame(x));
  }
  setFrame(pix2frame(x));
  drawOverlay(null);
  ev.preventDefault();
});

heatWrap.addEventListener('pointermove', ev => {
  const r = heat.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  if (dragging) {
    const cx = Math.max(ML, Math.min(heatW - MR, x));
    setFrame(pix2frame(cx));
    drawOverlay(null);
    const f = state.frame;
    showTip(ev, `frame <b>${f}</b>`);
    return;
  }
  if (x < ML || x > heatW - MR || y < MT || y > heatH - MB) {
    hideTip(); drawOverlay(null); return;
  }
  const f = pix2frame(x);
  const hh = Math.max(1, Math.min(state.hmax, Math.round(pix2harm(y))));
  const v = db(f, hh);
  showTip(ev,
    `frame <b>${f}</b> / h <b>${hh}</b> (${Math.round(hh * FUND)} Hz)<br>` +
    `${v <= -60 ? '≤ −60' : v.toFixed(1)} dB`);
  drawOverlay({ x, y });
});

function endDrag() {
  if (!dragging) return;
  dragging = false;
  if (dragStartedAudio) {
    stopCycle();
    dragStartedAudio = false;
  }
  save({ frame: state.frame });
  drawOverlay(null);
}
heatWrap.addEventListener('pointerup', endDrag);
heatWrap.addEventListener('pointercancel', endDrag);
heatWrap.addEventListener('mouseleave', () => { if (!dragging) { hideTip(); drawOverlay(null); } });

// ---------- spectrum interaction ----------
specCv.addEventListener('mousemove', ev => {
  const r = specCv.getBoundingClientRect();
  const boxW = r.width, mlt = 30;
  const x = ev.clientX - r.left;
  if (x < mlt) { hideTip(); return; }
  const hh = Math.max(1, Math.min(state.hmax,
    Math.round(Math.pow(2, (x - mlt) / (boxW - mlt - 6) * Math.log2(state.hmax)))));
  const v = db(state.frame, hh);
  showTip(ev, `h <b>${hh}</b> (${Math.round(hh * FUND)} Hz)<br>` +
    `${v <= -60 ? '≤ −60' : v.toFixed(1)} dB`);
});
specCv.addEventListener('mouseleave', hideTip);

// ---------- menu ----------
const menu = document.getElementById('menu');
const menuBtn = document.getElementById('menuBtn');
menuBtn.addEventListener('click', ev => {
  ev.stopPropagation();
  menu.classList.toggle('hidden');
});
document.addEventListener('click', ev => {
  if (!menu.classList.contains('hidden') && !menu.contains(ev.target)) menu.classList.add('hidden');
});

const optDark = document.getElementById('optDark');
const optAnn = document.getElementById('optAnn');
const optHmax = document.getElementById('optHmax');
optDark.checked = state.dark;
optAnn.checked = state.ann;
optHmax.value = String(state.hmax);

optDark.addEventListener('change', () => {
  save({ dark: optDark.checked });
  document.documentElement.classList.toggle('dark', state.dark);
  buildLUT();
  redrawAll();
});
optAnn.addEventListener('change', () => {
  save({ ann: optAnn.checked });
  drawHeat();
});
optHmax.addEventListener('change', () => {
  save({ hmax: parseInt(optHmax.value, 10) });
  redrawAll();
});

// ---------- layout & boot ----------
function redrawAll() {
  heatW = heatWrap.clientWidth;
  drawHeat();
  drawOverlay(null);
  drawColorbar();
  drawDetail();
}
window.addEventListener('resize', () => {
  if (heatWrap.clientWidth !== heatW) redrawAll();
});
document.documentElement.classList.toggle('dark', state.dark);
buildLUT();
redrawAll();
