// Animated tactics board ("prancheta") for a drill built on a real moment of the match.
// Mode 1: what happened (real defenders). Mode 2: the drill — same attack, defenders moving as the coach corrects.
// Mode 3: both, with arrows from the real position to the corrected one.
const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let D = null, mode = 1, demo = true, playing = true, speed = 1, t = 0, last = performance.now();

async function load() {
  const url = `/api/drill?match=${q.get('match')}&frame=${q.get('frame')}&team=${encodeURIComponent(q.get('team'))}&player=${q.get('player')}&tag=${q.get('tag')}`;
  D = await (await fetch(url)).json();
  if (D.error) { $('sub').textContent = D.error; return; }
  const dr = D.drill || {}, ld = D.load || {};
  $('title').textContent = dr.name || 'Drill';
  const focus = D.defenders.find((d) => d.focus);
  $('sub').innerHTML = `Built on a real moment of the match: <b>${esc(focus ? focus.name : '')}</b>, ${esc(D.team)}. Own goal on the left. Attack moves exactly as it did; the green defenders move as the drill teaches.`;
  $('chips').innerHTML = [D.format, ld.day, ld.sets, ld.work && `work ${ld.work} / rest ${ld.rest}`, ld.space, ld.rpe && `RPE ${ld.rpe}`].filter(Boolean).map((x) => `<span>${esc(x)}</span>`).join('');
  $('lgk').style.background = D.kit;
  $('side').innerHTML = `<h3>Objective</h3><p>${esc(dr.objective)}</p><h3>Set-up</h3><p>${esc(dr.setup)}</p>
    <h3>Rules</h3><p>${esc(dr.rules)}</p><h3>Coaching points</h3><ul>${(dr.coaching || []).map((c) => `<li>${esc(c)}</li>`).join('')}</ul>
    <h3>Progression</h3><p>${esc(dr.progression)}</p><h3>Check next match</h3><p>${esc(dr.kpi)}</p>
    <h3>Danger in this situation</h3><div id="dng"></div>
    <h3>Players in the drill</h3><p>${D.defenders.map((d) => `${d.n} ${esc(d.name)}${d.focus ? ' ★' : ''}`).join(' · ')}</p>
    <p class="sub">vs ${D.attackers.map((a) => `${a.n} ${esc(a.name)}`).join(' · ')}</p>`;
  t = D.frames[0].t;
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- geometry
function view() {
  const cv = $('cv'), [x0, y0, x1, y1] = D.area;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  let w = Math.max(x1 - x0 + 10, 30), h = Math.max(y1 - y0 + 10, 22);
  const ar = cv.width / cv.height;
  if (w / h > ar) h = w / ar; else w = h * ar;
  const s = cv.width / w;
  return { s, X: (x) => (x - (cx - w / 2)) * s, Y: (y) => ((cy + h / 2) - y) * s };
}
function frameAt(tt) {
  const F = D.frames; let k = 0;
  while (k < F.length - 2 && F[k + 1].t <= tt) k++;
  const a = F[k], b = F[k + 1] || a, u = b.t > a.t ? Math.min(1, Math.max(0, (tt - a.t) / (b.t - a.t))) : 0;
  const L = (p, r) => (p && r ? [p[0] + (r[0] - p[0]) * u, p[1] + (r[1] - p[1]) * u, p[2] != null && r[2] != null ? p[2] + (((r[2] - p[2] + 540) % 360) - 180) * u : p[2]] : p || r);
  const mix = (A, B) => A.map((p, i) => L(p, B[i]));
  return { A: mix(a.A, b.A), D: mix(a.D, b.D), S: mix(a.S, b.S), G: mix(a.G, b.G), ball: L(a.ball, b.ball), k, u };
}

// ---------------------------------------------------------------- drawing
function pitch(g, V) {
  const cv = $('cv');
  for (let i = 0; i < 14; i++) { g.fillStyle = i % 2 ? '#2e7d32' : '#307f34'; g.fillRect((i * cv.width) / 14, 0, cv.width / 14 + 1, cv.height); }
  g.strokeStyle = 'rgba(255,255,255,0.85)'; g.lineWidth = 2.5;
  const R = (x0, y0, x1, y1) => g.strokeRect(V.X(x0), V.Y(y1), V.X(x1) - V.X(x0), V.Y(y0) - V.Y(y1));
  R(-52.5, -34, 52.5, 34);
  g.beginPath(); g.moveTo(V.X(0), V.Y(34)); g.lineTo(V.X(0), V.Y(-34)); g.stroke();
  g.beginPath(); g.arc(V.X(0), V.Y(0), 9.15 * V.s, 0, Math.PI * 2); g.stroke();
  for (const sg of [-1, 1]) {
    const gx = sg * 52.5;
    R(Math.min(gx, gx - sg * 16.5), -20.16, Math.max(gx, gx - sg * 16.5), 20.16);
    R(Math.min(gx, gx - sg * 5.5), -9.16, Math.max(gx, gx - sg * 5.5), 9.16);
    g.fillStyle = '#fff'; g.fillRect(Math.min(V.X(gx), V.X(gx + sg * 1.6)), V.Y(3.66), 1.6 * V.s, 7.32 * V.s);
    g.beginPath(); g.arc(V.X(gx - sg * 11), V.Y(0), 3, 0, Math.PI * 2); g.fill();
  }
}
function cone(g, x, y) { g.fillStyle = '#fb923c'; g.beginPath(); g.moveTo(x, y - 9); g.lineTo(x - 7, y + 6); g.lineTo(x + 7, y + 6); g.closePath(); g.fill(); g.strokeStyle = '#7c2d12'; g.lineWidth = 1; g.stroke(); }
function props(g, V) {
  const [x0, y0, x1, y1] = D.area;
  g.setLineDash([10, 7]); g.strokeStyle = '#fde047'; g.lineWidth = 2.5;
  g.strokeRect(V.X(x0), V.Y(y1), V.X(x1) - V.X(x0), V.Y(y0) - V.Y(y1)); g.setLineDash([]);
  for (const [x, y] of [[x0, y0], [x0, y1], [x1, y0], [x1, y1], [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1]]) cone(g, V.X(x), V.Y(y));
  const gate = (x, y, w) => { g.strokeStyle = '#fff'; g.lineWidth = 5; g.beginPath(); g.moveTo(V.X(x), V.Y(y - w / 2)); g.lineTo(V.X(x), V.Y(y + w / 2)); g.stroke(); cone(g, V.X(x), V.Y(y - w / 2)); cone(g, V.X(x), V.Y(y + w / 2)); };
  const tag = D.tag, cy = (y0 + y1) / 2;
  if (tag === 'press_ball') { gate(x0, cy, 5); gate(x1, cy, 5); }
  if (tag === 'tuck_in') { for (const dy of [-8, 0, 8]) gate(x0, cy + dy, 3); }
  if (tag === 'line_drop' || tag === 'line_step') {           // the 12 m band between the lines
    const f = frameAt(t), xs = (mode === 1 ? f.D : f.S).filter(Boolean).map((p) => p[0]).sort((a, b) => a - b);
    if (xs.length >= 4) {
      const back = (xs[0] + xs[1] + xs[2] + xs[3]) / 4;
      g.fillStyle = 'rgba(253,224,71,0.13)'; g.fillRect(V.X(back), V.Y(y1), 12 * V.s, V.Y(y0) - V.Y(y1));
      g.strokeStyle = '#fde047'; g.lineWidth = 2; g.beginPath(); g.moveTo(V.X(back), V.Y(y1)); g.lineTo(V.X(back), V.Y(y0)); g.stroke();
      g.fillStyle = '#fde047'; g.font = 'bold 15px Inter, sans-serif'; g.fillText('back line · max 12 m to midfield', V.X(back) + 6, V.Y(y1) + 18);
    }
  }
  if (tag === 'scan') { g.strokeStyle = 'rgba(255,255,255,0.5)'; g.setLineDash([4, 6]); g.strokeRect(V.X(cy - 7 + (x0 + x1) / 2 - cy), V.Y(cy + 7), 14 * V.s, 14 * V.s); g.setLineDash([]); }
}
function disc(g, x, y, fill, txt, ring, alpha = 1, heading = null) {
  g.globalAlpha = alpha;
  g.beginPath(); g.arc(x, y, 22, 0, Math.PI * 2); g.fillStyle = fill; g.fill();
  g.lineWidth = ring ? 4 : 1.5; g.strokeStyle = ring || '#111'; g.stroke();
  if (heading != null) { // body orientation: small wedge in front of the counter
    const a = -(heading * Math.PI) / 180; g.beginPath(); g.moveTo(x + 34 * Math.cos(a), y + 34 * Math.sin(a));
    g.lineTo(x + 21 * Math.cos(a + 0.45), y + 21 * Math.sin(a + 0.45)); g.lineTo(x + 21 * Math.cos(a - 0.45), y + 21 * Math.sin(a - 0.45)); g.closePath(); g.fillStyle = fill; g.fill(); g.stroke();
  }
  g.fillStyle = luminance(fill) > 0.6 ? '#111' : '#fff'; g.font = 'bold 19px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(txt, x, y + 1); g.globalAlpha = 1;
}
function luminance(hex) { const h = hex.replace('#', ''); if (h.length < 6) return 0.5; const r = parseInt(h.slice(0, 2), 16), gg = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16); return (0.299 * r + 0.587 * gg + 0.114 * b) / 255; }
function trail(g, V, pick, idx, col, tt) {
  g.strokeStyle = col; g.lineWidth = 3; g.setLineDash([6, 6]); g.beginPath(); let first = true;
  for (const f of D.frames) { if (f.t > tt || f.t < tt - 2) continue; const p = pick(f)[idx]; if (!p) continue; if (first) { g.moveTo(V.X(p[0]), V.Y(p[1])); first = false; } else g.lineTo(V.X(p[0]), V.Y(p[1])); }
  g.stroke(); g.setLineDash([]);
}
function arrow(g, x0, y0, x1, y1, col) {
  const a = Math.atan2(y1 - y0, x1 - x0), d = Math.hypot(x1 - x0, y1 - y0); if (d < 8) return;
  g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 3; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1 - 10 * Math.cos(a), y1 - 10 * Math.sin(a)); g.stroke();
  g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - 14 * Math.cos(a - 0.4), y1 - 14 * Math.sin(a - 0.4)); g.lineTo(x1 - 14 * Math.cos(a + 0.4), y1 - 14 * Math.sin(a + 0.4)); g.closePath(); g.fill();
}

function draw() {
  const cv = $('cv'), g = cv.getContext('2d'), V = view(), f = frameAt(t);
  g.clearRect(0, 0, cv.width, cv.height); pitch(g, V); props(g, V);
  // attackers' runs (last 2 s)
  D.attackers.forEach((a, i) => trail(g, V, (fr) => fr.A, i, 'rgba(248,113,113,0.8)', t));
  const showReal = mode !== 2, showSim = mode !== 1;
  D.defenders.forEach((d, i) => {
    if (showReal) trail(g, V, (fr) => fr.D, i, mode === 3 ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.7)', t);
    if (showSim) trail(g, V, (fr) => fr.S, i, 'rgba(74,222,128,0.9)', t);
  });
  if (mode === 3) D.defenders.forEach((d, i) => { const p = f.D[i], s = f.S[i]; if (p && s) arrow(g, V.X(p[0]), V.Y(p[1]), V.X(s[0]), V.Y(s[1]), '#4ade80'); });
  f.G.forEach((p) => p && disc(g, V.X(p[0]), V.Y(p[1]), '#111827', 'GK'));
  D.defenders.forEach((d, i) => {
    const p = f.D[i], s = f.S[i], ring = d.focus ? '#facc15' : null;
    if (showReal && p) disc(g, V.X(p[0]), V.Y(p[1]), D.kit, d.n, ring, mode === 3 ? 0.45 : 1);
    if (showSim && s) disc(g, V.X(s[0]), V.Y(s[1]), '#4ade80', d.n, ring, 1, s[2]);
  });
  D.attackers.forEach((a, i) => { const p = f.A[i]; if (p) disc(g, V.X(p[0]), V.Y(p[1]), '#dc2626', a.n, a.carrier ? '#fff' : null); });
  if (f.ball) { g.beginPath(); g.arc(V.X(f.ball[0]), V.Y(f.ball[1]), 10, 0, Math.PI * 2); g.fillStyle = '#fff'; g.fill(); g.strokeStyle = '#000'; g.lineWidth = 1.5; g.stroke(); }
  // header on the board
  g.fillStyle = 'rgba(15,23,42,0.78)'; g.fillRect(12, 12, 470, 44); g.fillStyle = '#fff'; g.font = 'bold 24px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'middle';
  g.fillText(mode === 1 ? '1 · WHAT HAPPENED' : mode === 2 ? '2 · THE DRILL (corrected)' : '3 · REAL vs CORRECTED', 26, 35);
  // caption + danger
  const pts = (D.drill.coaching || []);
  const cap = t < -2 ? `<b>Set-up.</b> ${esc(D.drill.setup)}` : t < 0 ? `<b>Watch:</b> ${esc(pts[0] || '')}` : `<b>Key moment.</b> ${esc(pts[1] || pts[0] || '')} ${pts[2] ? ' · ' + esc(pts[2]) : ''}`;
  $('cap').innerHTML = (mode === 1 ? '<b>What happened.</b> ' : '') + cap;
  const k = f.k, dr = D.danger.real[k] ?? 0, ds = D.danger.sim[k] ?? 0, mx = Math.max(...D.danger.real, ...D.danger.sim, 1e-6);
  $('dng').innerHTML = `<div class="sub">real ${dr.toFixed(2)}</div><div class="meter"><i style="width:${(100 * dr) / mx}%;background:#f87171"></i></div>
    <div class="sub" style="margin-top:4px">drill (corrected) ${ds.toFixed(2)} · ${dr > 0 ? Math.round(100 * (1 - ds / dr)) : 0}% less</div><div class="meter"><i style="width:${(100 * ds) / mx}%;background:#4ade80"></i></div>`;
  $('clock').textContent = `${t >= 0 ? '+' : ''}${t.toFixed(1)} s`;
  const F = D.frames; $('tl').value = Math.round((1000 * (t - F[0].t)) / (F[F.length - 1].t - F[0].t));
}

function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  const F = D.frames, t0 = F[0].t, t1 = F[F.length - 1].t;
  if (playing) {
    t += dt * speed;
    if (t > t1 + 0.8) { t = t0; if (demo) { mode = mode === 1 ? 2 : 1; setModeButtons(); } }
  }
  draw(); requestAnimationFrame(loop);
}
function setModeButtons() { ['m1', 'm2', 'm3'].forEach((id, i) => $(id).classList.toggle('on', mode === i + 1)); $('demo').classList.toggle('on', demo); }
$('m1').onclick = () => { mode = 1; demo = false; setModeButtons(); };
$('m2').onclick = () => { mode = 2; demo = false; setModeButtons(); };
$('m3').onclick = () => { mode = 3; demo = false; setModeButtons(); };
$('demo').onclick = () => { demo = !demo; if (demo) { mode = 1; t = D.frames[0].t; } setModeButtons(); };
$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? '⏸' : '▶'; };
$('slow').onclick = () => { speed = speed === 1 ? 0.5 : 1; $('slow').textContent = speed === 1 ? '0.5×' : '1×'; };
$('tl').oninput = () => { const F = D.frames; t = F[0].t + ((F[F.length - 1].t - F[0].t) * $('tl').value) / 1000; };
setModeButtons();
load();
