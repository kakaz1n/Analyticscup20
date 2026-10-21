// "Teach a different action": corrected defenders drawn as their OWN skeleton (moved, turned, lowered), with the
// field of view they should have had; plus an editor to drag / rotate defenders and get the danger re-scored live
// by the server (/api/whatif, posture-aware model in dbl/coach.py).
import * as THREE from 'three';

const GREEN = new THREE.Color('#4ade80');
const deg = (a) => (a * Math.PI) / 180;
const wrapDeg = (a) => ((((a + 180) % 360) + 360) % 360) - 180;

export function initCoach(ctx) {
  const { scene, camera, renderer, S, overlays } = ctx;
  const mat = () => new THREE.MeshStandardMaterial({ color: GREEN, emissive: '#14532d', transparent: true, opacity: 0.62, depthWrite: false, roughness: 0.6 });
  const MAX = 14;
  const boneG = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.055, 0.055, 1, 7), mat(), MAX * 30);
  const torsoG = new THREE.InstancedMesh(new THREE.CapsuleGeometry(0.5, 1.0, 6, 14), mat(), MAX);
  const headG = new THREE.InstancedMesh(new THREE.SphereGeometry(0.115, 14, 10), mat(), MAX);
  for (const m of [boneG, torsoG, headG]) { m.frustumCulled = false; m.renderOrder = 6; m.count = 0; scene.add(m); }
  // field-of-view cones of the corrected heads
  const cones = [];
  for (let k = 0; k < 6; k++) {
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(25 * 3), 3));
    const idx = []; for (let i = 1; i < 24; i++) idx.push(0, i, i + 1); g.setIndex(idx);
    const c = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
    c.frustumCulled = false; c.visible = false; scene.add(c); cones.push(c);
  }
  const handles = new THREE.InstancedMesh(new THREE.RingGeometry(0.9, 1.25, 28), new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide }), 14);
  handles.frustumCulled = false; handles.count = 0; handles.renderOrder = 7; scene.add(handles);
  // reach disc of the player being edited
  const reach = new THREE.Mesh(new THREE.RingGeometry(0.97, 1.0, 64), new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide }));
  reach.visible = false; reach.position.z = 0.03; scene.add(reach);

  const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _v = new THREE.Vector3(), _s = new THREE.Vector3(), _up = new THREE.Vector3(0, 1, 0);

  function cone(k, x, y, z, hDeg) {
    const c = cones[k]; if (!c) return;
    const pos = c.geometry.attributes.position.array, R = 9, half = Math.PI / 3, h = deg(hDeg);
    pos[0] = x; pos[1] = y; pos[2] = z;
    for (let i = 0; i < 24; i++) {
      const a = h - half + (2 * half * i) / 23;
      pos[(i + 1) * 3] = x + R * Math.cos(a); pos[(i + 1) * 3 + 1] = y + R * Math.sin(a); pos[(i + 1) * 3 + 2] = 0.04;
    }
    c.geometry.attributes.position.needsUpdate = true; c.geometry.computeBoundingSphere(); c.visible = true;
  }

  // ---------------------------------------------------------------- what to draw
  function specs() {
    const r = S.sim; if (!r) return [];
    if (S.edit && S.edit.actual) {
      return [...S.edit.edits.entries()].map(([pid, e]) => {
        const a = S.edit.actual.get(pid) || {};
        return { pid, from: [a.x, a.y], to: [e.x, e.y], dT: wrapDeg((e.torso ?? a.torso ?? 0) - (a.torso ?? 0)),
          head: e.head ?? a.head, showCone: true, load: !!(e.loaded && a.loaded === false), known: a.pose_known };
      });
    }
    return (r.moves || []).map((m) => ({
      pid: m.player_id, from: m.from, to: m.to,
      dT: m.torso_to != null ? wrapDeg(m.torso_to - m.torso_from) : 0,
      head: m.head_to, showCone: m.head_to != null && (Math.abs(wrapDeg(m.head_to - m.head_from)) >= 20 || Math.abs(wrapDeg(m.torso_to - m.torso_from)) >= 30),
      load: !!(m.loaded_to && m.loaded_from === false), known: m.pose_known,
    }));
  }

  // ---------------------------------------------------------------- better-defence replay (up to 5 s)
  function replayActive() {
    const R = S.replay; return !!(R && S.t >= R.start - 1 && S.t <= R.end + 20);
  }
  function replayAt(t) {
    const R = S.replay, F = R.frames, k = Math.max(0, Math.min(F.length - 1, (t - R.start) / (R.frames[1] ? (F[1].frame - F[0].frame) : 1)));
    const a = F[Math.floor(k)], b = F[Math.min(F.length - 1, Math.floor(k) + 1)], u = k - Math.floor(k);
    const lerpA = (x, y) => x + wrapDeg(y - x) * u;
    return { a, b, u, defenders: a.defenders.map((d, i) => {
      const e = b.defenders[i] || d;
      return { player_id: d.player_id, x: d.x + (e.x - d.x) * u, y: d.y + (e.y - d.y) * u, torso: lerpA(d.torso, e.torso),
        head: lerpA(d.head, e.head), loaded: u < 0.5 ? d.loaded : e.loaded };
    }), danger_real: a.danger_real + (b.danger_real - a.danger_real) * u, danger_sim: a.danger_sim + (b.danger_sim - a.danger_sim) * u,
    free_real: a.free_real, free_sim: a.free_sim };
  }
  function replaySpecs() {
    if (S.drill && S.drill.mode === 1) return [];                 // drill mode 1: only what happened
    const keep = S.drill ? new Set(S.drill.defender_ids) : null;
    const now = replayAt(S.t), tr = ctx.track();
    return now.defenders.filter((d) => !keep || keep.has(d.player_id)).map((d) => {
      const real = tr && tr.pos.get(d.player_id);
      return { pid: d.player_id, from: real ? [real[0], real[1]] : [d.x, d.y], to: [d.x, d.y], absTorso: d.torso, head: d.head,
        showCone: d.player_id === S.sel, load: d.loaded, known: true, replay: true };
    });
  }
  const hiding = (pid) => !!(S.replay && S.replayHide && replayActive() && S.replay.frames[0].defenders.some((d) => d.player_id === pid)
    && (!S.drill || S.drill.defender_ids.includes(pid)));

  // ---------------------------------------------------------------- gait: real running clips of each player
  let gait = null, gaitMatch = null;
  async function loadGait() {
    if (gaitMatch === S.match) return; gaitMatch = S.match; gait = null;
    try { gait = await (await fetch(`/api/gait?match=${S.match}`)).json(); } catch (e) { gait = null; }
  }
  // cumulative simulated distance per defender (feet advance with the distance actually covered)
  function prepReplay(R) {
    if (R._cum) return;
    R._cum = new Map();
    R.frames[0].defenders.forEach((d0, i) => {
      const c = [0];
      for (let k = 1; k < R.frames.length; k++) {
        const a = R.frames[k - 1].defenders[i], b = R.frames[k].defenders[i];
        c.push(c[k - 1] + Math.hypot(b.x - a.x, b.y - a.y));
      }
      R._cum.set(d0.player_id, c);
    });
  }
  function gaitPose(pid, t, d, nJ) {
    const lib = gait && (gait[pid] || gait[Object.keys(gait)[0]]);
    const R = S.replay; if (!lib || !lib.length || !R) return null;
    prepReplay(R);
    const step = R.frames[1] ? R.frames[1].frame - R.frames[0].frame : 2;
    const kf = Math.max(0, Math.min(R.frames.length - 1, (t - R.start) / step));
    const c = R._cum.get(pid); if (!c) return null;
    const k0 = Math.floor(kf), u = kf - k0, k1 = Math.min(c.length - 1, k0 + 1);
    const dist = c[k0] + (c[k1] - c[k0]) * u;
    const speed = (c[k1] - c[k0]) / (step / 10) || (k0 > 0 ? (c[k0] - c[k0 - 1]) / (step / 10) : 0);
    let clip = lib.find((g) => speed >= g.vmin && speed < g.vmax) || lib.reduce((a, g) => (Math.abs(g.speed - speed) < Math.abs(a.speed - speed) ? g : a));
    const F = clip.frames.length, dur = F / clip.fps;
    const phase = clip.speed < 0.8 ? ((t / 10) / dur) : (dist / Math.max(clip.speed * dur, 0.5));
    const fr = clip.frames[Math.floor(((phase % 1) + 1) % 1 * F) % F];
    const h = deg(d.absTorso), cs = Math.cos(h), sn = Math.sin(h), zk = d.load ? 0.93 : 1;
    const q = new Float32Array(nJ * 3);
    for (let j = 0; j < nJ; j++) {
      const x = fr[j][0] / 100, y = fr[j][1] / 100, z = fr[j][2] / 100;
      q[j * 3] = d.to[0] + cs * x - sn * y; q[j * 3 + 1] = d.to[1] + sn * x + cs * y; q[j * 3 + 2] = z * zk;
    }
    return q;
  }

  function draw(sk, show, ph) {
    let bi = 0, ti = 0, hi = 0, ci = 0;
    const capsules = [];
    const rep = replayActive();
    if (rep) { show = true; ph = 1; }
    const list = rep ? replaySpecs() : (show ? specs() : []);
    const { BONES, TORSO, BONE_R, J, torsoMatrix, headPose } = ctx.skel();
    if (rep) loadGait();
    for (const sp of list) {
      // replay: the player's own running clip at the simulated speed and heading (feet match the ground speed)
      const gq = rep ? gaitPose(sp.pid, S.t, sp, ctx.skel().J ? Object.keys(ctx.skel().J).length : 23) : null;
      const p = gq || sk.get(sp.pid);
      if (gq) { sp.dT = 0; sp.to = [sp.to[0], sp.to[1]]; }
      if (!p || !sp.known || J.midHip == null) { capsules.push({ from: sp.from, to: sp.to, capsule: true }); continue; }
      capsules.push({ from: sp.from, to: sp.to, capsule: false });
      const cx = p[J.midHip * 3], cy = p[J.midHip * 3 + 1];
      if (sp.absTorso != null && !gq) {   // turn his real (animated) body to the simulated heading
        const g = (n) => [p[J[n] * 3], p[J[n] * 3 + 1]];
        const ls = g('lShoulder'), rs = g('rShoulder'), lh = g('lHip'), rh = g('rHip');
        const lx = rs[0] - ls[0] + rh[0] - lh[0], ly = rs[1] - ls[1] + rh[1] - lh[1];
        sp.dT = Number.isFinite(lx) ? wrapDeg(sp.absTorso - (Math.atan2(lx, -ly) * 180) / Math.PI) : 0;
      }
      const th = deg(sp.dT) * ph, c = Math.cos(th), s = Math.sin(th);
      const px = cx + (sp.to[0] - cx) * ph, py = cy + (sp.to[1] - cy) * ph, zk = gq ? 1 : 1 - (sp.load ? 0.08 * ph : 0);
      const q = new Float32Array(p.length);
      for (let k = 0; k < p.length; k += 3) {
        const dx = p[k] - cx, dy = p[k + 1] - cy;
        q[k] = px + c * dx - s * dy; q[k + 1] = py + s * dx + c * dy; q[k + 2] = p[k + 2] * zk;
      }
      BONES.forEach(([a, b], k) => {
        if (TORSO.has(k)) return;
        const ax = q[a * 3], ay = q[a * 3 + 1], az = q[a * 3 + 2], bx = q[b * 3], by = q[b * 3 + 1], bz = q[b * 3 + 2];
        if (!(Number.isFinite(ax) && Number.isFinite(bx))) return;
        _v.set(bx - ax, by - ay, bz - az); const len = _v.length(); if (len < 1e-3 || len > 1.2) return;
        _q.setFromUnitVectors(_up, _v.clone().divideScalar(len));
        const r = BONE_R[k] || 1;
        _m.compose(_s.set((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2), _q, new THREE.Vector3(r, len, r));
        boneG.setMatrixAt(bi++, _m);
      });
      if (torsoMatrix(q, _m)) torsoG.setMatrixAt(ti++, _m);
      const h = headPose(q);
      if (h) {
        _m.compose(_s.set(h.x, h.y, h.z), _q.identity(), new THREE.Vector3(1, 1, 1)); headG.setMatrixAt(hi++, _m);
        if (sp.showCone && sp.head != null && ph > 0.6 && ci < cones.length) cone(ci++, h.x, h.y, h.z, sp.head);
      }
    }
    for (let k = ci; k < cones.length; k++) cones[k].visible = false;
    boneG.count = bi; torsoG.count = ti; headG.count = hi;
    for (const m of [boneG, torsoG, headG]) m.instanceMatrix.needsUpdate = true;
    overlays.updateSim(capsules, show, ph);
    // reach disc of the edited player
    const sel = S.edit && S.edit.sel != null && (S.sim.reach || []).find((r) => r.player_id === S.edit.sel);
    if (show && sel) { reach.position.set(sel.center[0], sel.center[1], 0.03); reach.scale.set(sel.radius, sel.radius, 1); reach.visible = true; }
    else reach.visible = false;
    // grab handles under every editable defender
    let hn = 0;
    if (S.edit && S.edit.actual && show) {
      const pulse = 1 + 0.12 * Math.sin(performance.now() / 180);
      for (const [pid, a] of S.edit.actual) {
        const e = S.edit.edits.get(pid) || a; const k = pid === S.edit.sel ? 1.35 * pulse : 1;
        _m.compose(_s.set(e.x, e.y, 0.06), _q.identity(), new THREE.Vector3(k, k, 1)); handles.setMatrixAt(hn++, _m);
      }
    }
    handles.count = hn; handles.instanceMatrix.needsUpdate = true;
  }

  // ---------------------------------------------------------------- editor
  async function whatif() {
    if (!S.edit) return;
    if (S.edit.busy) { S.edit.pending = true; return; }
    S.edit.busy = true;
    try {
      const edits = [...S.edit.edits.entries()].map(([player_id, e]) => ({ player_id, ...e }));
      const r = await (await fetch('/api/whatif', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ match: S.match, frame: S.sim.frame, focus: S.sim.focus_id, team: S.sim.defending_team, edits }) })).json();
      if (!r.error && S.edit) {
        S.edit.res = r;
        if (S.edit.edits.size) banner(true, `<b>✎ Your version:</b> danger <b style="color:${r.danger_reduction > 0 ? '#86efac' : '#fca5a5'}">${r.danger_reduction > 0 ? '−' : '+'}${Math.abs(Math.round(r.danger_reduction * 100))}%</b> vs what happened (auto coach −${Math.round(S.sim.danger_reduction * 100)}%) · drag / Q E / Z C / L · Esc to exit`);
        for (const pl of r.players) { const e = S.edit.edits.get(pl.player_id); if (e) Object.assign(e, { x: pl.x, y: pl.y, torso: pl.torso, head: pl.head }); }
      }
    } finally {
      if (S.edit) { S.edit.busy = false; if (S.edit.pending) { S.edit.pending = false; whatif(); } }
    }
  }

  async function enter(fromAuto = false) {
    if (!S.sim) return;
    if (S.edit) exit();
    S.edit = { edits: new Map(), actual: null, sel: null, res: null, busy: false, pending: false };
    if (S.playing) ctx.togglePlay();
    S.t = S.sim.frame;
    const base = await (await fetch('/api/whatif', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ match: S.match, frame: S.sim.frame, focus: S.sim.focus_id, team: S.sim.defending_team, edits: [] }) })).json();
    if (base.error) { S.edit = null; alert(base.error); return; }
    S.edit.actual = new Map(base.players.map((p) => [p.player_id, p]));
    S.edit.base = base;
    // start from WHAT HAPPENED: every change the analyst makes then shows up as a green player of its own
    if (fromAuto) for (const m of S.sim.moves || []) {
      S.edit.edits.set(m.player_id, { x: m.to[0], y: m.to[1], torso: m.torso_to, head: m.head_to, loaded: m.loaded_to });
    }
    S.edit.sel = S.sim.key_fix ? S.sim.key_fix.player_id : null;
    if (S.edit.sel != null) ctx.select(S.edit.sel);
    S.edit.prevCam = S.cam; ctx.setCam('tactical');            // top-down: easy to grab and drag players
    banner(true);
    whatif();
  }
  function exit() {
    if (S.edit && S.edit.prevCam) ctx.setCam(S.edit.prevCam);
    S.edit = null; reach.visible = false; handles.count = 0; banner(false); renderer.domElement.style.cursor = '';
  }
  // on-screen instructions while editing
  function banner(on, msg) {
    let b = document.getElementById('editBanner');
    if (!b) {
      b = document.createElement('div'); b.id = 'editBanner';
      b.style.cssText = 'position:fixed;left:50%;top:12px;transform:translateX(-50%);z-index:40;background:#14532d;color:#ecfdf5;border:1px solid #4ade80;border-radius:10px;padding:8px 14px;font:13px Inter,system-ui,sans-serif;box-shadow:0 6px 20px rgba(0,0,0,.4);text-align:center;pointer-events:none';
      document.body.appendChild(b);
    }
    b.style.display = on ? 'block' : 'none';
    if (on) b.innerHTML = msg || '<b>✎ EDIT MODE</b> — click and <b>drag</b> a defender (green handles) inside his reach circle · <b>Q/E</b> turn body · <b>Z/C</b> turn head · <b>L</b> stance · <b>Esc</b> exit';
  }
  function reset() {
    if (!S.edit) return;
    S.edit.edits.clear(); whatif();
  }
  function touch(pid) {
    if (!S.edit.edits.has(pid)) {
      const a = S.edit.actual.get(pid); if (!a) return null;
      S.edit.edits.set(pid, { x: a.x, y: a.y, torso: a.torso, head: a.head, loaded: a.loaded });
    }
    return S.edit.edits.get(pid);
  }

  // drag on the pitch (capture phase: wins over the orbit controls)
  const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  let dragging = null;
  const toScreen = (x, y) => { const v = new THREE.Vector3(x, y, 0.9).project(camera); return [(v.x + 1) / 2 * innerWidth, (1 - v.y) / 2 * innerHeight]; };
  const groundAt = (ev) => {
    mouse.set((ev.clientX / innerWidth) * 2 - 1, -(ev.clientY / innerHeight) * 2 + 1); ray.setFromCamera(mouse, camera);
    return ray.ray.intersectPlane(plane, new THREE.Vector3());
  };
  const stage = renderer.domElement.parentElement;
  stage.addEventListener('pointerdown', (ev) => {
    if (!S.edit || !S.edit.actual || Math.abs(S.t - S.sim.frame) > 25) return;
    let best = null, bd = 60;
    for (const [pid, a] of S.edit.actual) {
      const e = S.edit.edits.get(pid) || a;
      const [sx, sy] = toScreen(e.x, e.y); const d = Math.hypot(sx - ev.clientX, sy - ev.clientY);
      if (d < bd) { bd = d; best = pid; }
    }
    if (best == null) return;
    ev.stopPropagation(); ev.preventDefault();
    S.edit.sel = best; touch(best); dragging = best; ctx.select(best);
  }, true);
  addEventListener('pointermove', (ev) => {
    if (S.edit && S.edit.actual && dragging == null && ev.target === renderer.domElement) {
      let near = false;
      for (const [pid, a] of S.edit.actual) { const e = S.edit.edits.get(pid) || a; const [sx, sy] = toScreen(e.x, e.y); if (Math.hypot(sx - ev.clientX, sy - ev.clientY) < 60) { near = true; break; } }
      renderer.domElement.style.cursor = near ? 'grab' : '';
    }
    if (dragging == null || !S.edit) return;
    renderer.domElement.style.cursor = 'grabbing';
    const hit = groundAt(ev); if (!hit) return;
    const e = S.edit.edits.get(dragging), r = (S.sim.reach || []).find((z) => z.player_id === dragging);
    let x = hit.x, y = hit.y;
    if (r) { const dx = x - r.center[0], dy = y - r.center[1], d = Math.hypot(dx, dy); if (d > r.radius) { x = r.center[0] + dx * r.radius / d; y = r.center[1] + dy * r.radius / d; } }
    e.x = +x.toFixed(2); e.y = +y.toFixed(2); whatif();
  });
  addEventListener('pointerup', () => { dragging = null; });
  addEventListener('keydown', (ev) => {
    if (!S.edit || S.edit.sel == null || ev.target.tagName === 'INPUT') return;
    const e = touch(S.edit.sel); if (!e) return;
    const k = ev.key.toLowerCase();
    const clampHead = () => { const o = wrapDeg(e.head - e.torso); if (Math.abs(o) > 75) e.head = wrapDeg(e.torso + Math.sign(o) * 75); };
    if (k === 'q' || k === 'e') { const d = k === 'q' ? 15 : -15; e.torso = wrapDeg(e.torso + d); e.head = wrapDeg(e.head + d); }
    else if (k === 'z' || k === 'c') { e.head = wrapDeg(e.head + (k === 'z' ? 15 : -15)); clampHead(); }
    else if (k === 'l') e.loaded = !e.loaded;
    else if (k === 'r') { S.edit.edits.delete(S.edit.sel); }
    else if (k === 'escape') { exit(); return; }
    else return;
    ev.preventDefault(); whatif();
  });

  function turn(what, d) {
    if (!S.edit || S.edit.sel == null) return;
    const e = touch(S.edit.sel); if (!e) return;
    if (what === 'body') { e.torso = wrapDeg(e.torso + d); e.head = wrapDeg(e.head + d); }
    else if (what === 'head') { e.head = wrapDeg(e.head + d); const o = wrapDeg(e.head - e.torso); if (Math.abs(o) > 75) e.head = wrapDeg(e.torso + Math.sign(o) * 75); }
    else if (what === 'stance') e.loaded = !e.loaded;
    else if (what === 'undo') S.edit.edits.delete(S.edit.sel);
    whatif();
  }

  // ---------------------------------------------------------------- panel
  function editCard() {
    const E = S.edit; if (!E) return '';
    const r = E.res, auto = S.sim.danger_reduction;
    if (!r) return '<div class="card" style="border-color:#4ade80"><h4>✎ Your version</h4><span class="muted">Scoring…</span></div>';
    const b = r.before, a = r.after, red = r.danger_reduction;
    const col = red > 0 ? '#4ade80' : '#f87171';
    const row = (k, n, d = 1) => (b[k] == null ? '' : `<div class="row"><span>${n}</span><span>${(+b[k]).toFixed(d)} → <b>${(+a[k]).toFixed(d)}</b></span></div>`);
    const p = r.players.find((x) => x.player_id === E.sel);
    const yes = (v) => (v ? '<span class="pill good">yes</span>' : '<span class="pill bad">no</span>');
    const who = p ? `<div style="margin-top:6px"><b>${p.defender}</b>${p.pose_known ? '' : ' <span class="muted">(no pose at this instant: position only)</span>'}
      <div class="row"><span>Sees the ball</span><span>${yes(p.sees_ball)}</span></div>
      <div class="row"><span>Sees his man (${p.man || '–'})</span><span>${yes(p.sees_man)}</span></div>
      <div class="row"><span>Body vs own goal</span><span>${p.shape}${p.reaction_s != null ? ` · reacts ${p.reaction_s.toFixed(2)} s` : ''}</span></div>
      <div class="row"><span>Stance</span><span>${p.loaded == null ? '–' : (p.loaded ? 'loaded (low)' : 'upright')}</span></div></div>` : '<div class="muted" style="margin-top:6px">Click a defender to edit him.</div>';
    return `<div class="card" style="border-color:#4ade80"><h4>✎ Your version <span class="muted" style="font-weight:400">vs what happened</span></h4>
      <div class="row"><span>Danger</span><span><b style="color:${col}">${red > 0 ? '−' : '+'}${Math.abs(Math.round(red * 100))}%</b> <span class="muted">(auto coach −${Math.round(auto * 100)}%)</span></span></div>
      ${row('free_attackers', 'Free attackers near goal', 0)}${row('focus_marker_dist', 'Nearest defender to the focus attacker (m)')}
      ${row('carrier_pressure_dist', 'Pressure on the ball (m)')}${row('seeing_ball_and_man', 'Defenders seeing ball + man', 0)}
      ${who}
      ${r.warnings && r.warnings.length ? `<div class="muted" style="font-size:11px;color:#fbbf24">${r.warnings.join(' · ')}</div>` : ''}
      <div class="muted" style="font-size:11px;margin-top:5px"><b>Drag</b> a defender (within his green reach circle) · <b>Q/E</b> turn body · <b>Z/C</b> turn head · <b>L</b> stance · <b>R</b> undo player · <b>Esc</b> exit</div>
      ${E.sel != null ? `<div style="display:flex;gap:5px;margin-top:6px;flex-wrap:wrap"><button onclick="window.__coach.turn('body',15)">↺ body</button><button onclick="window.__coach.turn('body',-15)">↻ body</button><button onclick="window.__coach.turn('head',15)">↺ head</button><button onclick="window.__coach.turn('head',-15)">↻ head</button><button onclick="window.__coach.turn('stance')">stance</button><button onclick="window.__coach.turn('undo')">undo</button></div>` : ''}
      <div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap"><button onclick="window.__coach.reset()">Back to what happened</button><button onclick="window.__coach.enter(true)">Load the auto coach</button><button onclick="window.__coach.exit()">Done</button></div></div>`;
  }

  // screen position of a defender in the editor (used by automated tests / tours)
  const screenOf = (pid) => { const e = (S.edit && (S.edit.edits.get(pid) || (S.edit.actual && S.edit.actual.get(pid)))); return e ? toScreen(e.x, e.y) : null; };
  function replayCard() {
    const R = S.replay; if (!R) return '';
    const F = R.frames, W = 300, H = 70, mx = Math.max(...F.map((f) => Math.max(f.danger_real, f.danger_sim))) * 1.08 || 1;
    const X = (f) => ((f - R.start) / Math.max(1, R.end - R.start)) * W, Y = (v) => H - (v / mx) * H;
    const line = (k, col) => `<polyline fill="none" stroke="${col}" stroke-width="2.2" points="${F.map((f) => `${X(f.frame).toFixed(1)},${Y(f[k]).toFixed(1)}`).join(' ')}"/>`;
    const inWin = S.t >= R.start && S.t <= R.end;
    const cur = inWin ? replayAt(S.t) : null;
    const cx = Math.max(0, Math.min(W, X(S.t)));
    const pct = (x) => `${x >= 0 ? '−' : '+'}${Math.abs(Math.round(x * 100))}%`;
    const ev = (R.events || []).map((e) => `<div class="muted" style="font-size:11.5px">⚠ <a href="#" onclick="window.__viewer.t=${e.frame - 10};return false" style="color:#fbbf24">${e.defender}</a> could reach the ball here — from then on the real play would have changed.</div>`).join('');
    const diffs = (R.differences || []).slice(0, 4).map((d) => `${d.defender}: on ${d.ends_on} (${d.ends_dist_m} m${d.goal_side ? ', goal-side' : ''})`).join(' · ');
    return `<div class="card" style="border-color:#22c55e"><h4>🎬 Better-defence replay${R.goal ? ` · goal ${R.goal.goal_number} (${R.goal.scorer})` : ''} <span class="muted" style="font-weight:400">${R.seconds} s, attack unchanged · ${R.decisions.startsWith('clairvoyant') ? 'perfect anticipation' : `reads the play ${R.limits.reaction_s} s late`}</span></h4>
      <svg viewBox="0 0 ${W} ${H + 4}" style="width:100%;height:78px;background:rgba(255,255,255,0.03);border-radius:6px">${line('danger_real', '#f87171')}${line('danger_sim', '#4ade80')}
        <line x1="${cx}" x2="${cx}" y1="0" y2="${H}" stroke="#fbbf24" stroke-width="1.5"/></svg>
      <div class="muted" style="font-size:11px"><span style="color:#f87171">■</span> real defence · <span style="color:#4ade80">■</span> simulated defence (modelled danger)</div>
      ${cur ? `<div class="row"><span>Now</span><span>${cur.danger_real.toFixed(2)} → <b style="color:#4ade80">${cur.danger_sim.toFixed(2)}</b> (${pct(1 - cur.danger_sim / Math.max(cur.danger_real, 1e-9))}) · free ${cur.free_real} → ${cur.free_sim}</span></div>` : ''}
      <div class="row"><span>Danger, average / at the end</span><span><b style="color:#4ade80">${pct(R.mean_reduction)}</b> / <b style="color:#4ade80">${pct(R.end_reduction)}</b></span></div>
      ${R.focus_attacker && R.end_real.focus_marker_dist != null ? `<div class="row"><span>Nearest defender to ${R.focus_attacker} at the end</span><span>${R.end_real.focus_marker_dist} → <b>${R.end_sim.focus_marker_dist}</b> m</span></div>` : ''}
      <div class="row"><span>Free attackers near goal at the end</span><span>${R.end_real.free_attackers} → <b>${R.end_sim.free_attackers}</b></span></div>
      <div class="muted" style="font-size:11.5px;margin-top:4px">${diffs}</div>${ev}
      <div class="muted" style="font-size:10.5px;margin-top:4px">Green bodies = the simulated defenders (their real running animation, moved and turned); arrows = gap to where they really were. Physics: ${R.limits.accel_note || R.limits.speed_mps + ' m/s'}, ${R.limits.accel_mps2} m/s², body ${R.limits.body_turn_dps}°/s, look-ahead ${R.limits.lookahead_s} s. ${R.structure ? 'Structure: ' + R.structure + '.' : ''}</div>
      <div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap"><button onclick="window.__coach.playReplay()">▶ Play it</button>
      <button onclick="window.__viewer.replayHide=!window.__viewer.replayHide">${S.replayHide ? 'Show real defenders' : 'Hide real defenders'}</button>
      <button onclick='window.__runReplay(${JSON.stringify({ ...(R.goal ? { goal: R.goal.goal_number } : { frame: R.start }), mode: R.decisions.startsWith("clairvoyant") ? "realistic" : "clairvoyant" })})'>${R.decisions.startsWith('clairvoyant') ? 'Realistic reading of the play' : 'Perfect anticipation (upper bound)'}</button>
      <button onclick="window.__viewer.replay=null">Clear</button></div></div>`;
  }
  function playReplay() { if (!S.replay) return; S.t = S.replay.start - 10; if (!S.playing) ctx.togglePlay(); }

  // simulated positions for the drill counters (independent of "hide real defenders")
  function simPositions(t) {
    if (!S.replay || t < S.replay.start - 1 || t > S.replay.end + 20) return null;
    return new Map(replayAt(t).defenders.map((d) => [d.player_id, [d.x, d.y]]));
  }
  // simulated defender positions at time t (used to run the live analysis on the simulated defence)
  function replayPositions(t) {
    if (!S.replay || !S.replayHide || t < S.replay.start - 1 || t > S.replay.end + 20) return null;
    return new Map(replayAt(t).defenders.map((d) => [d.player_id, [d.x, d.y]]));
  }
  const api = { simPositions, turn, draw, enter, exit, reset, editCard, screenOf, replayCard, playReplay, hiding, replayActive, replayPositions };
  window.__coach = api;
  return api;
}
