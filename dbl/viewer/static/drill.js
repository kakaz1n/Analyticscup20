// Drill mode inside the 3D viewer (tactical view): the real moment of the match, only the players of the drill,
// the drill area / cones / gates on the grass, and the defence played twice — as it happened, then as the drill
// teaches (green, the better-defence replay). No separate board: it is the match itself, top-down.
import * as THREE from 'three';

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export function initDrill(ctx) {
  const { scene, S } = ctx;
  let group = null;

  function clearProps() { if (group) { scene.remove(group); group = null; } }
  function buildProps(d) {
    clearProps(); group = new THREE.Group();
    const s = d.s, [x0n, y0n, x1n, y1n] = d.area;
    const X = (x) => x * s, Y = (y) => y * s;
    const ax0 = Math.min(X(x0n), X(x1n)), ax1 = Math.max(X(x0n), X(x1n)), ay0 = Math.min(Y(y0n), Y(y1n)), ay1 = Math.max(Y(y0n), Y(y1n));
    // dashed area line
    const dash = new THREE.MeshBasicMaterial({ color: '#fde047', transparent: true, opacity: 0.95, depthWrite: false });
    const seg = (xa, ya, xb, yb) => {
      const len = Math.hypot(xb - xa, yb - ya), n = Math.floor(len / 1.6);
      for (let i = 0; i < n; i += 2) {
        const u0 = i / n, u1 = Math.min(1, (i + 1) / n);
        const m = new THREE.Mesh(new THREE.PlaneGeometry(len * (u1 - u0), 0.22), dash);
        m.position.set(xa + (xb - xa) * (u0 + u1) / 2, ya + (yb - ya) * (u0 + u1) / 2, 0.045);
        m.rotation.z = Math.atan2(yb - ya, xb - xa); m.renderOrder = 4; group.add(m);
      }
    };
    seg(ax0, ay0, ax1, ay0); seg(ax1, ay0, ax1, ay1); seg(ax1, ay1, ax0, ay1); seg(ax0, ay1, ax0, ay0);
    // cones
    const coneGeo = new THREE.ConeGeometry(0.35, 0.6, 12), coneMat = new THREE.MeshStandardMaterial({ color: '#fb923c', roughness: 0.6 });
    const cone = (x, y) => { const c = new THREE.Mesh(coneGeo, coneMat); c.rotation.x = Math.PI / 2; c.position.set(x, y, 0.3); c.castShadow = true; group.add(c); };
    const mx = (ax0 + ax1) / 2, my = (ay0 + ay1) / 2;
    for (const [x, y] of [[ax0, ay0], [ax1, ay0], [ax0, ay1], [ax1, ay1], [mx, ay0], [mx, ay1]]) cone(x, y);
    // mini gates
    const post = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.4 });
    const gate = (x, y, w) => {
      for (const dy of [-w / 2, w / 2]) { const p = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1, 8), post); p.rotation.x = Math.PI / 2; p.position.set(x, y + dy, 0.5); group.add(p); }
      const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, w, 8), post); bar.position.set(x, y, 1); group.add(bar);
    };
    const own = s > 0 ? ax0 : ax1, far = s > 0 ? ax1 : ax0;     // own goal side of the area
    if (d.tag === 'press_ball') { gate(own, my, 5); gate(far, my, 5); }
    if (d.tag === 'tuck_in') { for (const dy of [-8, 0, 8]) gate(far, my + dy, 3); }
    if (d.tag === 'line_drop' || d.tag === 'line_step') {            // 12 m band in front of the back line
      const band = new THREE.Mesh(new THREE.PlaneGeometry(12, ay1 - ay0), new THREE.MeshBasicMaterial({ color: '#fde047', transparent: true, opacity: 0.12, depthWrite: false }));
      band.position.set(own + (s > 0 ? 1 : -1) * (Math.abs(far - own) * 0.25 + 6), my, 0.035); group.add(band); d._band = band;
    }
    scene.add(group);
    d.view = { cx: mx, cy: my, h: Math.max(ax1 - ax0, (ay1 - ay0) * 1.6) * 1.05 + 12 };
  }

  async function open(qs) {
    const p = new URLSearchParams(qs);
    ctx.loading(true, 'Preparing the drill from the real moment…');
    try {
      const d = await (await fetch(`/api/drill?${qs}`)).json();
      if (d.error) throw new Error(d.error);
      if (Number(p.get('match')) !== S.match) { ctx.loading(false); return; }
      exit(true);
      const rp = await (await fetch(`/api/replay?match=${S.match}&frame=${d.start}&seconds=5&team=${encodeURIComponent(d.team)}`)).json();
      if (rp.error) throw new Error(rp.error);
      d.ids = new Set([...d.defender_ids, ...d.attacker_ids, ...d.gk_ids]);
      d.mode = 1; d.demo = true; d.prevCam = S.cam; d.prevOv = { ...S.ov }; d.prevToasts = S.toasts;
      S.drill = d; S.sim = null; S.edit = null; S.replay = rp; S.replayHide = false;
      for (const k of Object.keys(S.ov)) S.ov[k] = false;
      S.toasts = false;
      buildProps(d);
      ctx.setCam('drill'); ctx.select(d.defender_ids.find((id, i) => d.defenders[i] && d.defenders[i].focus) ?? d.defender_ids[0]);
      S.t = d.start - 5; if (!S.playing) ctx.togglePlay();
      document.querySelector('#tabs button[data-tab=live]').click();
    } catch (e) { alert(`Drill failed: ${e.message}`); }
    ctx.loading(false);
  }
  function exit(silent) {
    const d = S.drill; if (!d) return;
    S.drill = null; S.replay = null; S.replayHide = false; clearProps();
    Object.assign(S.ov, d.prevOv || {}); S.toasts = d.prevToasts;
    if (!silent) ctx.setCam(d.prevCam || 'broadcast');
  }
  function setMode(m) {
    const d = S.drill; if (!d) return;
    d.mode = m; d.demo = false; S.replayHide = m === 2; S.t = d.start - 5; if (!S.playing) ctx.togglePlay();
  }
  function demo() { const d = S.drill; if (!d) return; d.demo = !d.demo; if (d.demo) { d.mode = 1; S.replayHide = false; S.t = d.start - 5; } }

  // called every frame: loop the 5 s and alternate "what happened" → "the drill"
  function tick() {
    const d = S.drill; if (!d) return;
    if (S.t > d.end + 10 || S.t < d.start - 30) {
      S.t = d.start - 5;
      if (d.demo) { d.mode = d.mode === 1 ? 2 : 1; S.replayHide = d.mode === 2; }
    }
  }
  // drill filter for the renderer
  const hideOther = (pid) => !!(S.drill && !S.drill.ids.has(pid));
  const ghostsOn = () => !S.drill || S.drill.mode !== 1;
  const ghostIds = () => (S.drill ? new Set(S.drill.defender_ids) : null);

  function card() {
    const d = S.drill; if (!d) return '';
    const dr = d.drill || {}, ld = d.load || {};
    const t = (S.t - d.frame) / 10;
    const pts = dr.coaching || [];
    const cap = t < -1.5 ? `<b>Set-up.</b> ${esc(dr.setup)}` : t < 0.3 ? `<b>Watch:</b> ${esc(pts[0] || '')}` : `<b>Key moment.</b> ${esc(pts[1] || pts[0] || '')}${pts[2] ? ' · ' + esc(pts[2]) : ''}`;
    const R = S.replay, k = R ? Math.max(0, Math.min(R.frames.length - 1, Math.round((S.t - R.start) / 2))) : 0;
    const dr0 = R ? R.frames[k].danger_real : 0, ds0 = R ? R.frames[k].danger_sim : 0;
    const mx = R ? Math.max(...R.frames.map((f) => Math.max(f.danger_real, f.danger_sim)), 1e-6) : 1;
    const btn = (m, label) => `<button onclick="window.__drill.setMode(${m})" style="${d.mode === m && !d.demo ? 'background:#f59e0b;color:#111' : ''}">${label}</button>`;
    const focus = d.defenders.find((x) => x.focus);
    return `<div class="card" style="border-color:#fde047"><h4>🏋️ ${esc(dr.name)} <span class="muted" style="font-weight:400">${esc(d.format)}</span></h4>
      <div class="muted" style="font-size:11.5px">Real moment of the match: ${esc(focus ? focus.name : '')} (${esc(d.team)}). Only the drill's players are shown; the attack moves exactly as it did.</div>
      <div style="margin:6px 0;font-size:13px"><b style="color:${d.mode === 2 ? '#4ade80' : '#fbbf24'}">${d.mode === 1 ? '1 · WHAT HAPPENED' : d.mode === 2 ? '2 · THE DRILL — corrected defence (green)' : '3 · COMPARE'}</b>${d.demo ? ' <span class="muted">(demo: 1 → 2)</span>' : ''}</div>
      <div style="display:flex;gap:5px;flex-wrap:wrap">${btn(1, '1 · What happened')}${btn(2, '2 · The drill')}${btn(3, '3 · Compare')}<button onclick="window.__drill.demo()" style="${d.demo ? 'background:#f59e0b;color:#111' : ''}">▶ Demo</button></div>
      <div style="margin-top:8px;padding:7px 9px;background:rgba(253,224,71,0.1);border-radius:7px;font-size:12.5px">${cap}</div>
      <div class="row" style="margin-top:6px"><span>Danger now — real</span><span>${dr0.toFixed(2)}</span></div><div style="height:6px;background:rgba(255,255,255,.08);border-radius:3px"><i style="display:block;height:100%;width:${(100 * dr0) / mx}%;background:#f87171;border-radius:3px"></i></div>
      <div class="row"><span>— with the drill's behaviour</span><span style="color:#4ade80">${ds0.toFixed(2)}</span></div><div style="height:6px;background:rgba(255,255,255,.08);border-radius:3px"><i style="display:block;height:100%;width:${(100 * ds0) / mx}%;background:#4ade80;border-radius:3px"></i></div>
      <div class="muted" style="font-size:11.5px;margin-top:6px">${esc([ld.day, ld.sets, ld.work && `work ${ld.work} / rest ${ld.rest}`, ld.space, ld.rpe && `RPE ${ld.rpe}`].filter(Boolean).join(' · '))}</div>
      <details style="margin-top:4px"><summary style="cursor:pointer;font-size:12.5px">Objective, rules, coaching points</summary><div style="font-size:12px;line-height:1.45">
        <b>Objective</b> ${esc(dr.objective)}<br><b>Rules</b> ${esc(dr.rules)}<br><b>Coaching points</b><ul style="margin:2px 0 2px 16px;padding:0">${pts.map((c) => `<li>${esc(c)}</li>`).join('')}</ul>
        <b>Progression</b> ${esc(dr.progression)}<br><b>Check next match</b> ${esc(dr.kpi)}</div></details>
      <div style="display:flex;gap:6px;margin-top:6px"><button onclick="window.__drill.exit()">✕ Exit drill</button></div></div>`;
  }

  // ---------------------------------------------------------------- board-style counters on the grass
  // (the skeletons are small seen from above: each drill player also gets a big numbered disc, like a tactics board)
  const discGeo = new THREE.CircleGeometry(1.15, 32), ringGeo = new THREE.RingGeometry(1.15, 1.45, 32);
  const tex = new Map();
  function numberMat(n, fill, ink) {
    const key = `${n}|${fill}|${ink}`; if (tex.has(key)) return tex.get(key);
    const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d');
    g.fillStyle = fill; g.beginPath(); g.arc(64, 64, 62, 0, Math.PI * 2); g.fill();
    g.lineWidth = 6; g.strokeStyle = 'rgba(0,0,0,0.6)'; g.stroke();
    g.fillStyle = ink; g.font = 'bold 64px Inter, system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(n), 64, 68);
    const tx = new THREE.CanvasTexture(c); tx.colorSpace = THREE.SRGBColorSpace;
    const m = new THREE.MeshBasicMaterial({ map: tx, transparent: true, depthWrite: false }); tex.set(key, m); return m;
  }
  const pool = []; let used = 0;
  function counter(x, y, n, fill, ink, ring) {
    let o = pool[used];
    if (!o) {
      o = { disc: new THREE.Mesh(discGeo), ring: new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: '#facc15', transparent: true, depthWrite: false })) };
      o.disc.renderOrder = 8; o.ring.renderOrder = 9; scene.add(o.disc); scene.add(o.ring); pool.push(o);
    }
    used++;
    o.disc.material = numberMat(n, fill, ink); o.disc.position.set(x, y, 0.07); o.disc.visible = true;
    o.ring.position.set(x, y, 0.075); o.ring.visible = !!ring;
  }
  function drawCounters(tr, simPos) {
    used = 0;
    const d = S.drill;
    if (d && tr) {
      const showReal = d.mode !== 2, showSim = d.mode !== 1;
      d.attackers.forEach((a, i) => { const p = tr.pos.get(d.attacker_ids[i]); if (p) counter(p[0], p[1], a.n, '#dc2626', '#ffffff', a.carrier); });
      d.defenders.forEach((x, i) => {
        const id = d.defender_ids[i], p = tr.pos.get(id), s = simPos && simPos.get(id);
        if (showReal && p) counter(p[0], p[1], x.n, d.kit, '#111111', x.focus);
        if (showSim && s) counter(s[0], s[1], x.n, '#4ade80', '#052e16', x.focus);
      });
    }
    for (let k = used; k < pool.length; k++) { pool[k].disc.visible = false; pool[k].ring.visible = false; }
  }

  const api = { open, exit, setMode, demo, tick, card, hideOther, ghostsOn, ghostIds, drawCounters };
  window.__drill = api;
  return api;
}
