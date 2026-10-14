// Full-match 3D viewer: SkillCorner body pose (12.5 fps, interpolated) + tracking (10 fps).
// Coordinates: SkillCorner metres, x along the pitch, y across, z up.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { analyse, pitchControl } from './analysis.js';
import { Overlays } from './overlays.js';
import { renderLive, renderReport, selectedCard, addMsg, md } from './panel.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const CHUNK = 300;              // tracking frames per data window (30 s)
const NAN16 = -32768;
const KIND_COLORS = { run: '#f59e0b', pass: '#38bdf8', duel: '#f472b6', recovery: '#a3e635' };
const params = new URLSearchParams(location.search);

// ------------------------------------------------------------------ state
const S = {
  ana: null, lastPoss: '', lastPossT: -1e9, ov: { shape: true, marking: true, lanes: false, space: false },
  insights: null, series: null, cards: {}, baseline: {}, dangerSpace: NaN, chatHist: [],
  sessionId: Math.random().toString(36).slice(2) + Date.now().toString(36),
  meta: null, match: null, t: 0, playing: false, speed: 1, cam: params.get('cam') || 'broadcast',
  sel: params.get('focus') ? Number(params.get('focus')) : null,
  chunks: new Map(), pending: new Map(), labels: true, ghosts: true, cone: true, toasts: true,
  kinds: new Set(['run', 'pass', 'duel', 'recovery']), keyOnly: true, lastToastF: -1,
  playerInfo: new Map(), teamOf: new Map(), colorOf: new Map(),
};
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ three
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
$('stage').appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0d1726');
scene.fog = new THREE.Fog('#0d1726', 140, 260);
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 600);
camera.position.set(0, -70, 30);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;
controls.addEventListener('start', () => { if (S.cam !== 'free') setCam('free'); });

scene.add(new THREE.HemisphereLight(0xe8f1ff, 0x2f4a2a, 1.15));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-30, -45, 70);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -65, right: 65, top: 45, bottom: -45, near: 10, far: 200 });
scene.add(sun);

function layoutSide() {
  const c = $('cams').getBoundingClientRect();
  $('side').style.top = `${Math.round(c.bottom + 8)}px`;
}
function resize() {
  layoutSide();
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  drawEvents();
}
addEventListener('resize', resize);

// ------------------------------------------------------------------ pitch
let pitchGroup = null;
function buildPitch(L, W) {
  if (pitchGroup) scene.remove(pitchGroup);
  pitchGroup = new THREE.Group();
  const M = 6, px = 20; // margin (m), pixels per metre
  const cw = Math.round((L + 2 * M) * px), ch = Math.round((W + 2 * M) * px);
  const c = document.createElement('canvas'); c.width = cw; c.height = ch;
  const g = c.getContext('2d');
  g.fillStyle = '#2f7a3d'; g.fillRect(0, 0, cw, ch);
  const stripes = 18;
  for (let i = 0; i < stripes; i++) {
    g.fillStyle = i % 2 ? '#358744' : '#2f7a3d';
    g.fillRect((M + (L / stripes) * i) * px, M * px, (L / stripes) * px, W * px);
  }
  const X = (x) => (x + L / 2 + M) * px, Y = (y) => (W / 2 - y + M) * px;
  g.strokeStyle = 'rgba(255,255,255,0.92)'; g.lineWidth = 0.12 * px;
  const rect = (x0, y0, x1, y1) => g.strokeRect(X(x0), Y(y1), (x1 - x0) * px, (y1 - y0) * px);
  rect(-L / 2, -W / 2, L / 2, W / 2);
  g.beginPath(); g.moveTo(X(0), Y(-W / 2)); g.lineTo(X(0), Y(W / 2)); g.stroke();
  g.beginPath(); g.arc(X(0), Y(0), 9.15 * px, 0, 2 * Math.PI); g.stroke();
  for (const s of [-1, 1]) {
    const gx = s * L / 2;
    rect(Math.min(gx, gx - s * 16.5), -20.16, Math.max(gx, gx - s * 16.5), 20.16);
    rect(Math.min(gx, gx - s * 5.5), -9.16, Math.max(gx, gx - s * 5.5), 9.16);
    g.beginPath(); g.arc(X(gx - s * 11), Y(0), 0.25 * px, 0, 2 * Math.PI); g.fillStyle = 'white'; g.fill();
    g.save(); g.beginPath(); g.rect(s < 0 ? X(gx + 16.5) : 0, 0, s < 0 ? cw : X(gx - 16.5), ch); g.clip();
    g.beginPath(); g.arc(X(gx - s * 11), Y(0), 9.15 * px, 0, 2 * Math.PI); g.stroke(); g.restore();
  }
  g.beginPath(); g.arc(X(0), Y(0), 0.25 * px, 0, 2 * Math.PI); g.fill();
  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  tex.colorSpace = THREE.SRGBColorSpace;
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(L + 2 * M, W + 2 * M), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }));
  plane.receiveShadow = true;
  pitchGroup.add(plane);
  const surround = new THREE.Mesh(new THREE.PlaneGeometry(L + 120, W + 100), new THREE.MeshStandardMaterial({ color: '#1d4a28', roughness: 1 }));
  surround.position.z = -0.02; pitchGroup.add(surround);
  const postMat = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.4 });
  for (const s of [-1, 1]) {
    const gx = s * L / 2;
    for (const gy of [-3.66, 3.66]) {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 2.44, 10), postMat);
      p.rotation.x = Math.PI / 2; p.position.set(gx, gy, 1.22); p.castShadow = true; pitchGroup.add(p);
    }
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 7.32, 10), postMat);
    bar.position.set(gx, 0, 2.44); bar.castShadow = true; pitchGroup.add(bar);
    const net = new THREE.Mesh(new THREE.BoxGeometry(2, 7.32, 2.44), new THREE.MeshStandardMaterial({ color: '#ffffff', transparent: true, opacity: 0.12, depthWrite: false }));
    net.position.set(gx + s * 1, 0, 1.22); pitchGroup.add(net);
  }
  scene.add(pitchGroup);
}

// ------------------------------------------------------------------ actors
const MAXP = 40;
let BONES = [], JOINTS = [], J = {}, TORSO = new Set();
const boneMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.055, 0.055, 1, 7), new THREE.MeshStandardMaterial({ roughness: 0.55 }), MAXP * 30);
const headMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.115, 14, 10), new THREE.MeshStandardMaterial({ roughness: 0.5 }), MAXP);
const jointMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.06, 8, 6), new THREE.MeshStandardMaterial({ roughness: 0.5 }), MAXP * 8);
const ghostMesh = new THREE.InstancedMesh(new THREE.CapsuleGeometry(0.27, 1.15, 4, 10), new THREE.MeshStandardMaterial({ transparent: true, opacity: 0.38, roughness: 0.8, depthWrite: false }), MAXP);
for (const m of [boneMesh, headMesh, jointMesh]) { m.castShadow = true; m.frustumCulled = false; scene.add(m); }
ghostMesh.frustumCulled = false; scene.add(ghostMesh);
const ball = new THREE.Mesh(new THREE.SphereGeometry(0.13, 18, 14), new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.35 }));
ball.castShadow = true; scene.add(ball);
const ballShadow = new THREE.Mesh(new THREE.CircleGeometry(0.16, 20), new THREE.MeshBasicMaterial({ color: '#000000', transparent: true, opacity: 0.35 }));
ballShadow.position.z = 0.01; scene.add(ballShadow);

const coneMat = new THREE.MeshBasicMaterial({ color: '#f59e0b', transparent: true, opacity: 0.28, side: THREE.DoubleSide, depthWrite: false });
const coneGeo = new THREE.BufferGeometry();
const CONE_N = 24;
coneGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((CONE_N + 1) * 3), 3));
const idx = []; for (let i = 1; i < CONE_N; i++) idx.push(0, i, i + 1);
coneGeo.setIndex(idx);
const cone = new THREE.Mesh(coneGeo, coneMat); cone.frustumCulled = false; scene.add(cone);
const ring = new THREE.Mesh(new THREE.RingGeometry(0.55, 0.75, 32), new THREE.MeshBasicMaterial({ color: '#f59e0b', transparent: true, opacity: 0.9, depthWrite: false }));
ring.position.z = 0.02; scene.add(ring);

const labelSprites = new Map();
function label(pid) {
  if (labelSprites.has(pid)) return labelSprites.get(pid);
  const info = S.playerInfo.get(pid) || {};
  const team = S.teamOf.get(pid);
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = team ? team.color : '#888'; g.beginPath(); g.arc(32, 32, 29, 0, 2 * Math.PI); g.fill();
  g.lineWidth = 4; g.strokeStyle = 'rgba(0,0,0,0.55)'; g.stroke();
  g.fillStyle = team ? team.number_color : '#fff'; g.font = 'bold 30px system-ui, sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(info.number ?? '?', 32, 34);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set(0.75, 0.75, 1); sp.renderOrder = 10; scene.add(sp);
  labelSprites.set(pid, sp);
  return sp;
}

// ------------------------------------------------------------------ data
async function loadMeta(id) {
  const r = await fetch(`/api/meta?match=${id}`);
  const m = await r.json();
  S.meta = m; S.match = id; S.chunks.clear(); S.pending.clear();
  JOINTS = m.joints; BONES = m.bones; J = Object.fromEntries(JOINTS.map((n, i) => [n, i]));
  TORSO = new Set(BONES.map((b, i) => [b, i]).filter(([b]) => {
    const n = [JOINTS[b[0]], JOINTS[b[1]]];
    return n.every((x) => /Shoulder|Hip|neck/.test(x));
  }).map(([, i]) => i));
  S.playerInfo = new Map(m.players.map((p) => [p.id, p]));
  const teams = { [m.teams.home.id]: m.teams.home, [m.teams.away.id]: m.teams.away };
  S.teamOf = new Map(m.players.map((p) => [p.id, teams[p.team]]));
  S.colorOf = new Map(m.players.map((p) => {
    const t = teams[p.team];
    const base = new THREE.Color(t.color);
    if (p.gk) base.lerp(new THREE.Color('#111111'), 0.55);
    return [p.id, base];
  }));
  for (const sp of labelSprites.values()) scene.remove(sp);
  labelSprites.clear();
  $('title').textContent = `${m.title}`;
  buildPitch(m.pitch[0], m.pitch[1]);
  const p1 = m.periods[0], p2 = m.periods[m.periods.length - 1];
  $('slider').min = p1.start; $('slider').max = p2.end;
  const f0 = Number(params.get('f')) || p1.start;
  S.t = Math.max(p1.start, f0 - (params.get('f') ? 30 : 0));
  buildPlayerList();
  drawEvents();
  loadAnalysis(id);
}

function decodePose(buf) {
  const dv = new DataView(buf);
  const n = dv.getInt32(0, true), nj = dv.getInt32(4, true);
  const frames = new Int32Array(buf, 8, n), pids = new Int32Array(buf, 8 + 4 * n, n);
  const joints = new Int16Array(buf, 8 + 8 * n, n * nj * 3);
  const byFrame = new Map();
  for (let i = 0; i < n; i++) {
    let m = byFrame.get(frames[i]); if (!m) { m = new Map(); byFrame.set(frames[i], m); }
    m.set(pids[i], i);
  }
  return { n, nj, joints, byFrame };
}

function decodeTrack(js) {
  const players = new Map();
  for (const [pid, [f, x, y]] of Object.entries(js.players)) players.set(Number(pid), { f: Int32Array.from(f), x: Float32Array.from(x), y: Float32Array.from(y) });
  const [bf, bx, by, bz, poss] = js.ball;
  return { players, ball: { f: Int32Array.from(bf), x: Float32Array.from(bx), y: Float32Array.from(by), z: Float32Array.from(bz), poss } };
}

function ensure(ci) {
  if (ci < 0 || S.chunks.has(ci) || S.pending.has(ci)) return;
  const a = ci * CHUNK, b = a + CHUNK + 4, m = S.match;
  const p = Promise.all([
    fetch(`/api/pose?match=${m}&from=${a}&to=${b}`).then((r) => r.arrayBuffer()),
    fetch(`/api/track?match=${m}&from=${a}&to=${b}`).then((r) => r.json()),
  ]).then(([pb, tj]) => {
    if (m !== S.match) return;
    S.chunks.set(ci, { pose: decodePose(pb), track: decodeTrack(tj) });
  }).finally(() => S.pending.delete(ci));
  S.pending.set(ci, p);
}

function manageChunks() {
  const ci = Math.floor(S.t / CHUNK);
  ensure(ci); ensure(ci + 1);
  if ((S.t % CHUNK) < 40) ensure(ci - 1);
  for (const k of [...S.chunks.keys()]) if (k < ci - 2 || k > ci + 2) S.chunks.delete(k);
  return S.chunks.has(ci);
}

function bsearch(arr, v) { let lo = 0, hi = arr.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (arr[m] <= v) lo = m; else hi = m - 1; } return lo; }

function trackAt(t) {
  const c = S.chunks.get(Math.floor(t / CHUNK)); if (!c) return null;
  const pos = new Map();
  for (const [pid, d] of c.track.players) {
    if (!d.f.length || t < d.f[0] - 1 || t > d.f[d.f.length - 1] + 1) continue;
    const i = bsearch(d.f, t), j = Math.min(i + 1, d.f.length - 1);
    const w = d.f[j] > d.f[i] ? Math.min(Math.max((t - d.f[i]) / (d.f[j] - d.f[i]), 0), 1) : 0;
    pos.set(pid, [d.x[i] + (d.x[j] - d.x[i]) * w, d.y[i] + (d.y[j] - d.y[i]) * w]);
  }
  const b = c.track.ball; let bp = null, poss = '';
  if (b.f.length) {
    const i = bsearch(b.f, t), j = Math.min(i + 1, b.f.length - 1);
    const w = b.f[j] > b.f[i] ? Math.min(Math.max((t - b.f[i]) / (b.f[j] - b.f[i]), 0), 1) : 0;
    if (b.x[i] > -900 && b.x[j] > -900) bp = [b.x[i] + (b.x[j] - b.x[i]) * w, b.y[i] + (b.y[j] - b.y[i]) * w, b.z[i] + (b.z[j] - b.z[i]) * w];
    poss = b.poss[i];
  }
  return { pos, ball: bp, poss };
}

function poseRow(c, f25, pid) {
  const m = c && c.pose.byFrame.get(f25); if (!m) return -1;
  const r = m.get(pid); return r === undefined ? -1 : r;
}

// skeleton for every player at time t (tracking clock), interpolated between 12.5 fps samples
function skeletonsAt(t) {
  const out = new Map();
  const ci = Math.floor(t / CHUNK), c = S.chunks.get(ci), c2 = S.chunks.get(ci + 1);
  if (!c) return out;
  const f25 = t * 2.5, f0 = Math.floor(f25 / 2) * 2, f1 = f0 + 2, w = (f25 - f0) / 2;
  const nj = c.pose.nj;
  const pids = new Set();
  for (const f of [f0 - 2, f0, f1, f1 + 2]) { const m = (c.pose.byFrame.get(f) || c2?.pose.byFrame.get(f)); if (m) for (const p of m.keys()) pids.add(p); }
  for (const pid of pids) {
    const src = (f) => { let r = poseRow(c, f, pid); if (r >= 0) return [c, r]; r = poseRow(c2, f, pid); return r >= 0 ? [c2, r] : null; };
    let A = src(f0), B = src(f1), wa = w;
    if (!A && !B) { A = src(f0 - 2); B = src(f1 + 2); wa = 0.5; if (!A && !B) continue; }
    if (!A) { A = B; wa = 0; } if (!B) { B = A; wa = 0; }
    const pts = new Float32Array(nj * 3); let ok = 0;
    for (let k = 0; k < nj * 3; k++) {
      const a = A[0].pose.joints[A[1] * nj * 3 + k], b = B[0].pose.joints[B[1] * nj * 3 + k];
      if (a === NAN16 && b === NAN16) { pts[k] = NaN; continue; }
      const va = a === NAN16 ? b : a, vb = b === NAN16 ? a : b;
      pts[k] = (va + (vb - va) * wa) / 100; ok++;
    }
    if (ok < nj) continue;
    // stand the skeleton on the grass (pose z is relative to the body, not the pitch)
    let minFoot = Infinity;
    for (const n of ['lAnkle', 'rAnkle', 'lHeel', 'rHeel', 'lBigToe', 'rBigToe']) { const z = pts[J[n] * 3 + 2]; if (Number.isFinite(z)) minFoot = Math.min(minFoot, z); }
    if (Number.isFinite(minFoot)) { const off = Math.min(Math.max(minFoot - 0.03, -0.35), 0.35); for (let k = 2; k < pts.length; k += 3) pts[k] -= off; }
    out.set(pid, pts);
  }
  return out;
}

function headPose(p) {
  const le = J.lEar * 3, re = J.rEar * 3, no = J.nose * 3;
  if (![le, re, no].every((i) => Number.isFinite(p[i]) && Number.isFinite(p[i + 1]))) return null;
  const dx = p[re] - p[le], dy = p[re + 1] - p[le + 1];
  const perp = Math.atan2(dx, -dy);
  const mx = (p[le] + p[re]) / 2, my = (p[le + 1] + p[re + 1]) / 2, mz = (p[le + 2] + p[re + 2]) / 2;
  const nose = Math.atan2(p[no + 1] - my, p[no] - mx);
  const h = Math.atan2(Math.sin(perp) + Math.sin(nose), Math.cos(perp) + Math.cos(nose));
  return { x: mx, y: my, z: mz, h };
}

// ------------------------------------------------------------------ draw
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _v = new THREE.Vector3(), _s = new THREE.Vector3(), _up = new THREE.Vector3(0, 1, 0);
const SEL = new THREE.Color('#f59e0b');
let lastSk = new Map(), lastTr = null;

function draw() {
  const sk = skeletonsAt(S.t), tr = trackAt(S.t);
  lastSk = sk; lastTr = tr;
  runAnalysis(tr);
  let bi = 0, hi = 0, ji = 0, gi = 0;
  const shown = new Set();
  for (const sp of labelSprites.values()) sp.visible = false;
  for (const [pid, p] of sk) {
    if (tr && !tr.pos.has(pid) && !S.playerInfo.has(pid)) continue;
    shown.add(pid);
    if (pid === S.sel && S.cam === 'eyes') { updateCone(headPose(p)); ring.visible = false; continue; } // don't render your own head
    const col = pid === S.sel ? SEL : (S.colorOf.get(pid) || new THREE.Color('#aaa'));
    BONES.forEach(([a, b], k) => {
      const ax = p[a * 3], ay = p[a * 3 + 1], az = p[a * 3 + 2], bx = p[b * 3], by = p[b * 3 + 1], bz = p[b * 3 + 2];
      if (!(Number.isFinite(ax) && Number.isFinite(bx))) return;
      _v.set(bx - ax, by - ay, bz - az); const len = _v.length(); if (len < 1e-3 || len > 1.2) return;
      _q.setFromUnitVectors(_up, _v.clone().divideScalar(len));
      const r = TORSO.has(k) ? 2.1 : 1;
      _m.compose(_s.set((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2), _q, new THREE.Vector3(r, len, r));
      boneMesh.setMatrixAt(bi, _m); boneMesh.setColorAt(bi, col); bi++;
    });
    const h = headPose(p);
    if (h) {
      _m.compose(_s.set(h.x, h.y, h.z), _q.identity(), new THREE.Vector3(1, 1, 1));
      headMesh.setMatrixAt(hi, _m); headMesh.setColorAt(hi, col); hi++;
    }
    for (const n of ['lWrist', 'rWrist', 'lBigToe', 'rBigToe', 'lKnee', 'rKnee']) {
      const i = J[n] * 3; if (!Number.isFinite(p[i])) continue;
      _m.compose(_s.set(p[i], p[i + 1], p[i + 2]), _q.identity(), new THREE.Vector3(1, 1, 1));
      jointMesh.setMatrixAt(ji, _m); jointMesh.setColorAt(ji, col); ji++;
    }
    if (S.labels) { const sp = label(pid); const z = h ? h.z + 0.55 : 2.3; sp.position.set(h ? h.x : p[0], h ? h.y : p[1], z); sp.visible = true; }
    if (pid === S.sel) {
      ring.visible = true; ring.position.set(p[J.midHip * 3], p[J.midHip * 3 + 1], 0.02);
      updateCone(h);
    }
  }
  if (tr) {
    for (const [pid, [x, y]] of tr.pos) {
      if (shown.has(pid)) continue;
      if (S.ghosts) {
        const col = pid === S.sel ? SEL : (S.colorOf.get(pid) || new THREE.Color('#aaa'));
        _q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
        _m.compose(_s.set(x, y, 0.85), _q, new THREE.Vector3(1, 1, 1));
        ghostMesh.setMatrixAt(gi, _m); ghostMesh.setColorAt(gi, col); gi++;
      }
      if (S.labels) { const sp = label(pid); sp.position.set(x, y, 2.05); sp.visible = true; }
      if (pid === S.sel) { ring.visible = true; ring.position.set(x, y, 0.02); updateCone(null); }
    }
    if (tr.ball) {
      const z = Math.max(tr.ball[2] || 0, 0.13);
      ball.position.set(tr.ball[0], tr.ball[1], z); ball.visible = true;
      ballShadow.position.set(tr.ball[0], tr.ball[1], 0.012); ballShadow.visible = true;
    } else { ball.visible = false; ballShadow.visible = false; }
  }
  if (S.sel == null || (!shown.has(S.sel) && !(tr && tr.pos.has(S.sel)))) { ring.visible = false; cone.visible = false; }
  boneMesh.count = bi; headMesh.count = hi; jointMesh.count = ji; ghostMesh.count = gi;
  for (const m of [boneMesh, headMesh, jointMesh, ghostMesh]) { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; }
  const nTrack = tr ? tr.pos.size : 0;
  $('cov').textContent = `${shown.size}/${nTrack || shown.size}`;
  updatePoseDots(shown);
}

function updateCone(h) {
  if (!S.cone || !h) { cone.visible = false; return; }
  const pos = coneGeo.attributes.position.array, R = 9, half = Math.PI / 3;
  pos[0] = h.x; pos[1] = h.y; pos[2] = h.z;
  for (let i = 0; i < CONE_N; i++) {
    const a = h.h - half + (2 * half * i) / (CONE_N - 1);
    pos[(i + 1) * 3] = h.x + R * Math.cos(a); pos[(i + 1) * 3 + 1] = h.y + R * Math.sin(a); pos[(i + 1) * 3 + 2] = 0.03;
  }
  coneGeo.attributes.position.needsUpdate = true; coneGeo.computeBoundingSphere();
  cone.visible = true;
}

// ------------------------------------------------------------------ cameras
const camTarget = new THREE.Vector3(), camPos = new THREE.Vector3();
function setCam(c) {
  S.cam = c;
  document.querySelectorAll('#cams [data-cam]').forEach((b) => b.classList.toggle('on', b.dataset.cam === c));
  camera.fov = c === 'eyes' ? 75 : 42; camera.updateProjectionMatrix();
}
function selectedPoint() {
  if (S.sel == null) return null;
  const p = lastSk.get(S.sel);
  if (p) { const h = headPose(p); return { x: p[J.midHip * 3], y: p[J.midHip * 3 + 1], h }; }
  const q = lastTr && lastTr.pos.get(S.sel);
  return q ? { x: q[0], y: q[1], h: null } : null;
}
function updateCamera(dt) {
  if (!S.meta) return;
  const [L, W] = S.meta.pitch;
  const b = (lastTr && lastTr.ball) || [0, 0, 0];
  const k = 1 - Math.pow(0.0015, dt);
  if (S.cam === 'free') { controls.enabled = true; controls.update(); return; }
  controls.enabled = false;
  if (S.cam === 'broadcast') {
    camTarget.lerp(_v.set(b[0] * 0.92, b[1] * 0.35, 0), k);
    camPos.lerp(_v.set(b[0] * 0.8, -W / 2 - 22, 17), k);
  } else if (S.cam === 'tactical') {
    camTarget.lerp(_v.set(0, 0, 0), k); camPos.lerp(_v.set(0, -8, 112), k);
  } else if (S.cam === 'goal') {
    const s = b[0] >= 0 ? 1 : -1;
    camTarget.lerp(_v.set(s * (L / 2 - 22), b[1] * 0.5, 0), k);
    camPos.lerp(_v.set(s * (L / 2 + 16), b[1] * 0.2, 13), k);
  } else if (S.cam === 'follow' || S.cam === 'eyes') {
    const sp = selectedPoint();
    if (!sp) { camTarget.lerp(_v.set(b[0], b[1], 0), k); camPos.lerp(_v.set(b[0] - 18, b[1] - 18, 14), k); }
    else if (S.cam === 'follow') {
      const toBall = Math.atan2(b[1] - sp.y, b[0] - sp.x);
      camTarget.lerp(_v.set((sp.x + b[0]) / 2, (sp.y + b[1]) / 2, 0.8), k);
      camPos.lerp(_v.set(sp.x - 9 * Math.cos(toBall), sp.y - 9 * Math.sin(toBall), 5.5), k);
    } else {
      const h = sp.h;
      if (h) {
        camPos.set(h.x - 0.12 * Math.cos(h.h), h.y - 0.12 * Math.sin(h.h), h.z + 0.08);
        camTarget.set(h.x + 10 * Math.cos(h.h), h.y + 10 * Math.sin(h.h), 0.6);
      } else { camPos.lerp(_v.set(sp.x, sp.y - 0.5, 1.75), k); camTarget.lerp(_v.set(b[0], b[1], 0.5), k); }
    }
  }
  camera.position.copy(camPos); camera.lookAt(camTarget); controls.target.copy(camTarget);
}

// ------------------------------------------------------------------ UI
function clock(f) {
  const ps = S.meta.periods; let p = ps[0];
  for (const q of ps) if (f >= q.start) p = q;
  const base = p.period === 2 ? 45 * 60 : 0, cap = p.period === 2 ? 90 * 60 : 45 * 60;
  const sec = Math.max(0, (f - p.start) / 10) + base;
  const fmt = (x) => `${String(Math.floor(x / 60)).padStart(2, '0')}:${String(Math.floor(x % 60)).padStart(2, '0')}`;
  const txt = sec <= cap ? fmt(sec) : `${cap / 60}+${fmt(sec - cap)}`;
  return [txt, p.period === 1 ? '1st half' : '2nd half'];
}

function buildPlayerList() {
  const el = $('players'); el.innerHTML = '';
  for (const side of ['home', 'away']) {
    const t = S.meta.teams[side];
    const h = document.createElement('h3'); h.textContent = t.name; el.appendChild(h);
    S.meta.players.filter((p) => p.team === t.id).sort((a, b) => (a.number ?? 99) - (b.number ?? 99)).forEach((p) => {
      const d = document.createElement('div'); d.className = 'pl'; d.dataset.pid = p.id;
      d.innerHTML = `<span class="num" style="background:${t.color};color:${t.number_color}">${p.number ?? ''}</span><span>${p.name}</span><span class="muted" style="font-size:11px">${p.role ?? ''}</span><span class="pose-dot"></span>`;
      d.onclick = () => select(S.sel === p.id ? null : p.id);
      el.appendChild(d);
    });
  }
  markSel();
}
function markSel() { document.querySelectorAll('.pl').forEach((d) => d.classList.toggle('sel', Number(d.dataset.pid) === S.sel)); }
let dotsT = 0;
function updatePoseDots(shown) {
  const now = performance.now(); if (now - dotsT < 400) return; dotsT = now;
  document.querySelectorAll('.pl').forEach((d) => d.querySelector('.pose-dot').classList.toggle('on', shown.has(Number(d.dataset.pid))));
}
function select(pid) { S.sel = pid; markSel(); if (pid != null && (S.cam === 'broadcast' || S.cam === 'tactical')) {} }

function visibleEvents() { return (S.meta?.events || []).filter((e) => S.kinds.has(e.kind) && (!S.keyOnly || e.key)); }
function drawEvents() {
  const cv = $('evcanvas'); if (!S.meta) return;
  const w = cv.clientWidth, h = cv.clientHeight; cv.width = w * devicePixelRatio; cv.height = h * devicePixelRatio;
  const g = cv.getContext('2d'); g.scale(devicePixelRatio, devicePixelRatio); g.clearRect(0, 0, w, h);
  const a = Number($('slider').min), b = Number($('slider').max);
  const rows = { run: 0, pass: 1, duel: 2, recovery: 3 };
  g.fillStyle = 'rgba(255,255,255,0.05)'; g.fillRect(0, 0, w, h);
  const p2 = S.meta.periods[1]; if (p2) { const x = ((p2.start - a) / (b - a)) * w; g.fillStyle = 'rgba(255,255,255,0.35)'; g.fillRect(x, 0, 1, h); }
  for (const e of visibleEvents()) {
    const x = ((e.f - a) / (b - a)) * w; g.fillStyle = KIND_COLORS[e.kind];
    g.globalAlpha = e.key ? 1 : 0.55; g.fillRect(x - 1, 2 + rows[e.kind] * 5.5, 2, 5);
  }
  g.globalAlpha = 1;
  for (const gb of S.goals || []) {
    const x = ((gb.goal.frame - a) / (b - a)) * w;
    g.fillStyle = '#ffffff'; g.beginPath(); g.moveTo(x, h - 1); g.lineTo(x - 5, h - 9); g.lineTo(x + 5, h - 9); g.closePath(); g.fill();
    g.fillRect(x - 0.5, 0, 1, h);
  }
  const xt = ((S.t - a) / (b - a)) * w; g.fillStyle = '#ffffff'; g.fillRect(xt - 1, 0, 2, h);
}
function goalAtX(clientX) {
  const cv = $('evcanvas'), r = cv.getBoundingClientRect();
  const a = Number($('slider').min), b = Number($('slider').max);
  const f = a + ((clientX - r.left) / r.width) * (b - a), tol = (b - a) / r.width * 6;
  return (S.goals || []).find((gb) => Math.abs(gb.goal.frame - f) < tol);
}
function eventAtX(clientX) {
  const cv = $('evcanvas'), r = cv.getBoundingClientRect();
  const a = Number($('slider').min), b = Number($('slider').max);
  const f = a + ((clientX - r.left) / r.width) * (b - a), tol = (b - a) / r.width * 4;
  let best = null, bd = Infinity;
  for (const e of visibleEvents()) { const d = Math.abs(e.f - f); if (d < tol && d < bd) { best = e; bd = d; } }
  return best;
}
$('evcanvas').addEventListener('mousemove', (ev) => {
  const gb = goalAtX(ev.clientX);
  if (gb) {
    const tip = $('tip'); tip.style.display = 'block'; tip.style.left = `${ev.clientX + 10}px`; tip.style.top = `${ev.clientY - 40}px`;
    tip.textContent = `⚽ ${gb.goal.time} ${gb.goal.scorer} (${gb.goal.team}) — ${gb.goal.score_after}. Click to watch the build-up.`;
    return;
  }
  const e = eventAtX(ev.clientX), tip = $('tip');
  if (!e) { tip.style.display = 'none'; return; }
  tip.style.display = 'block'; tip.style.left = `${ev.clientX + 10}px`; tip.style.top = `${ev.clientY - 40}px`;
  tip.textContent = `${clock(e.f)[0]} · ${e.label}`;
});
$('evcanvas').addEventListener('mouseleave', () => { $('tip').style.display = 'none'; });
$('evcanvas').addEventListener('click', (ev) => {
  const gb = goalAtX(ev.clientX);
  if (gb) {
    const f0 = gb.sequence && gb.sequence.length ? Math.max(gb.sequence[0].frame, gb.goal.frame - 150) : gb.goal.frame - 120;
    S.t = f0; S.lastToastF = -1; if (gb.responsibility?.[0]) {
      const p = S.meta.players.find((x) => x.name === gb.responsibility[0].defender); if (p) select(p.id);
    }
    if (!S.playing) togglePlay();
    return;
  }
  const e = eventAtX(ev.clientX); if (!e) return;
  S.t = e.f - 30; if (e.defender) select(e.defender); S.lastToastF = -1;
});
$('slider').addEventListener('input', (ev) => { S.t = Number(ev.target.value); S.lastToastF = -1; });
$('play').onclick = () => togglePlay();
$('speed').onchange = (ev) => { S.speed = Number(ev.target.value); };
document.querySelectorAll('#cams [data-cam]').forEach((b) => (b.onclick = () => setCam(b.dataset.cam)));
const toggle = (id, key) => { $(id).onclick = () => { S[key] = !S[key]; $(id).classList.toggle('on', S[key]); }; };
toggle('tLabels', 'labels'); toggle('tGhosts', 'ghosts'); toggle('tCone', 'cone'); toggle('tToasts', 'toasts');
document.querySelectorAll('.chip[data-k]').forEach((c) => (c.onclick = () => {
  const k = c.dataset.k; S.kinds.has(k) ? S.kinds.delete(k) : S.kinds.add(k); c.classList.toggle('off', !S.kinds.has(k)); drawEvents();
}));
$('keyOnly').classList.toggle('on', S.keyOnly);
$('keyOnly').onclick = () => { S.keyOnly = !S.keyOnly; $('keyOnly').classList.toggle('off', !S.keyOnly); drawEvents(); };
function togglePlay() { S.playing = !S.playing; $('play').textContent = S.playing ? '⏸ Pause' : '▶ Play'; }
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'ArrowRight') S.t += 50; else if (e.key === 'ArrowLeft') S.t -= 50;
  else if (e.key === '.') S.t += 1; else if (e.key === ',') S.t -= 1;
  else if ('123456'.includes(e.key)) setCam(['broadcast', 'tactical', 'goal', 'follow', 'eyes', 'free'][Number(e.key) - 1]);
});

// click a player in the scene to select him
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
renderer.domElement.addEventListener('dblclick', (ev) => {
  mouse.set((ev.clientX / innerWidth) * 2 - 1, -(ev.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), new THREE.Vector3());
  if (!hit || !lastTr) return;
  let best = null, bd = 1.8;
  for (const [pid, [x, y]] of lastTr.pos) { const d = Math.hypot(x - hit.x, y - hit.y); if (d < bd) { bd = d; best = pid; } }
  select(best);
});

let toastTimer = 0;
function maybeToast() {
  if (!S.toasts || !S.playing) return;
  for (const e of visibleEvents()) {
    if (e.f > S.lastToastF && e.f <= S.t && S.t - e.f < 15) {
      S.lastToastF = e.f;
      const el = $('toast'); el.innerHTML = `<b style="color:${KIND_COLORS[e.kind]}">${clock(e.f)[0]}</b> · ${e.label}`;
      el.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
    }
  }
}

// ------------------------------------------------------------------ loop
let last = performance.now(), sliderT = 0;
function loop(now) {
  const dt = Math.min((now - last) / 1000, 0.1); last = now;
  if (S.meta) {
    const ready = manageChunks();
    $('loading').style.display = ready ? 'none' : 'block';
    if (ready) {
      if (S.playing) {
        S.t += dt * 10 * S.speed;
        const ps = S.meta.periods;
        if (ps[1] && S.t > ps[0].end && S.t < ps[1].start) S.t = ps[1].start;
        if (S.t >= ps[ps.length - 1].end) { S.t = ps[ps.length - 1].end; togglePlay(); }
      }
      draw(); maybeToast(); panelTick(now);
    }
    if (now - sliderT > 200) { $('slider').value = Math.round(S.t); sliderT = now; drawEvents(); drawChart(); }
    const [c, half] = clock(S.t); $('clock').textContent = c; $('half').textContent = half;
    updateCamera(dt);
  }
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ defensive analysis
const overlays = new Overlays(scene);
function periodAt(t) { let p = 1; for (const q of S.meta.periods) if (t >= q.start) p = q.period; return p; }
function runAnalysis(tr) {
  if (!tr) { overlays.hide(); S.ana = null; return; }
  if (tr.poss) { S.lastPoss = tr.poss; S.lastPossT = S.t; }
  if (S.t - S.lastPossT > 80 || S.t < S.lastPossT - 5) S.lastPoss = tr.poss || '';
  const prevT = trackAt(S.t - 3);
  const prev = prevT ? { pos: prevT.pos, dt: 0.3 } : null;
  S.ana = analyse(S.meta, tr, prev, periodAt(S.t), S.lastPoss, S.playerInfo);
  overlays.update(S.ana, S.ov, S.meta.pitch[1], performance.now() / 1000);
  overlays.updateSim(S.sim, S.sim && Math.abs(S.t - S.sim.frame) < 25, performance.now() / 1000);
  const now = performance.now();
  if (S.ov.space && S.ana && S.ana.inPlay && (!runAnalysis.pcT || now - runAnalysis.pcT > 250)) {
    runAnalysis.pcT = now;
    S.dangerSpace = overlays.updateSpace(pitchControl(S.ana, S.meta.pitch[0], S.meta.pitch[1]), S.ana, S.meta.pitch[0], S.meta.pitch[1]);
  }
  if (!S.ov.space) S.dangerSpace = NaN;
}

async function loadAnalysis(id) {
  $('tab-report').innerHTML = '<span class="muted">Computing match report…</span>';
  const goalsP = fetch(`/api/goals?match=${id}`).then((r) => r.json()).catch(() => []);
  const [ins, ser, cards] = await Promise.all([
    fetch(`/api/insights?match=${id}`).then((r) => r.json()).catch(() => null),
    fetch(`/api/series?match=${id}`).then((r) => r.json()).catch(() => null),
    fetch(`/api/players?match=${id}`).then((r) => r.json()).catch(() => ({})),
  ]);
  if (id !== S.match) return;
  S.insights = ins; S.series = ser; S.cards = cards || {};
  S.baseline = {};
  for (const r of (ins?.shape_by_phase || [])) {
    if (!['medium_block', 'low_block', 'high_block'].includes(r.phase_oop)) continue;
    const b = (S.baseline[r.defending_team] ||= { n: 0, line_height: 0, ball_zone_area: 0, line_gap: 0 });
    const w = r.share_of_time || 0; b.n += w;
    for (const k of ['line_height', 'ball_zone_area', 'line_gap']) b[k] += (r[k] || 0) * w;
  }
  for (const b of Object.values(S.baseline)) for (const k of ['line_height', 'ball_zone_area', 'line_gap']) b[k] /= (b.n || 1);
  S.goals = await goalsP;
  if (id !== S.match) return;
  renderReport($('tab-report'), ins, S.meta.events, (f) => clock(f)[0], (f, d, play) => {
    S.t = f - 30; if (d) select(d); S.lastToastF = -1; if (play && !S.playing) togglePlay();
  }, S.goals, (q) => { document.querySelector('#tabs button[data-tab=ask]').click(); ask(q); });
  drawChart();
}

const teamNameById = (id) => (S.meta.teams.home.id === id ? S.meta.teams.home.name : S.meta.teams.away.name);

function selectedLive() {
  if (S.sel == null || !lastTr) return null;
  const p = S.playerInfo.get(S.sel); if (!p) return null;
  const pos = lastTr.pos.get(S.sel); if (!pos) return null;
  const out = { pose: false };
  const prevT = trackAt(S.t - 3);
  const pv = prevT && prevT.pos.get(S.sel);
  if (pv) out.speed = Math.hypot(pos[0] - pv[0], pos[1] - pv[1]) / 0.3;
  const sDef = ((side) => { const homeDefLeft = (S.meta.home_side[periodAt(S.t) - 1] || 'left_to_right') === 'left_to_right'; return (p.team === S.meta.teams.home.id ? homeDefLeft : !homeDefLeft) ? 1 : -1; })();
  // nearest opponent
  let man = null, bd = Infinity;
  for (const [pid, [x, y]] of lastTr.pos) {
    const q = S.playerInfo.get(pid); if (!q || q.team === p.team || q.gk) continue;
    const d = Math.hypot(x - pos[0], y - pos[1]); if (d < bd) { bd = d; man = { pid, x, y, ...q, dist: d, goalSide: pos[0] * sDef < x * sDef }; }
  }
  out.man = man;
  const sk = lastSk.get(S.sel);
  if (sk) {
    out.pose = true;
    const h = headPose(sk);
    const ang = (a) => Math.abs(((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI) * 180 / Math.PI;
    if (h) {
      const b = lastTr.ball;
      out.sees = {
        ball: b ? ang(h.h - Math.atan2(b[1] - h.y, b[0] - h.x)) <= 60 : null,
        man: man ? ang(h.h - Math.atan2(man.y - h.y, man.x - h.x)) <= 60 : null,
      };
    }
    const ls = J.lShoulder * 3, rs = J.rShoulder * 3, lh = J.lHip * 3, rh = J.rHip * 3;
    const hd = (l, r) => Math.atan2(sk[r] - sk[l], -(sk[r + 1] - sk[l + 1]));
    const tor = Math.atan2(Math.sin(hd(ls, rs)) + Math.sin(hd(lh, rh)), Math.cos(hd(ls, rs)) + Math.cos(hd(lh, rh)));
    const goalDir = sDef > 0 ? Math.PI : 0;
    const tg = ang(tor - goalDir);
    out.shape = tg < 45 ? 'turned' : tg <= 135 ? 'open' : 'square';
  }
  return out;
}

let panelT = 0;
function panelTick(now) {
  if (now - panelT < 250 || !$('tab-live').classList.contains('on')) return; panelT = now;
  const p = S.sel != null ? S.playerInfo.get(S.sel) : null;
  const card = simCard() + (p ? selectedCard(p, S.teamOf.get(S.sel)?.name || '', selectedLive(), S.cards[S.sel]) : '');
  const m = S.ana;
  renderLive($('tab-live'), m, { teamName: teamNameById, sel: S.sel, selCard: card,
    baseline: m && m.inPlay ? S.baseline[teamNameById(m.defTeam)] : null, dangerSpace: S.dangerSpace });
}

function drawChart() {
  const cv = $('chart'); if (!S.meta || !cv) return;
  const w = cv.clientWidth, h = cv.clientHeight; cv.width = w * devicePixelRatio; cv.height = h * devicePixelRatio;
  const g = cv.getContext('2d'); g.scale(devicePixelRatio, devicePixelRatio); g.clearRect(0, 0, w, h);
  const a = Number($('slider').min), b = Number($('slider').max);
  g.fillStyle = 'rgba(255,255,255,0.03)'; g.fillRect(0, 0, w, h);
  if (!S.series) return;
  const ymax = 60;
  for (const side of ['home', 'away']) {
    const t = S.meta.teams[side], d = S.series[t.name]; if (!d) continue;
    g.strokeStyle = t.color; g.lineWidth = 1.6; g.globalAlpha = 0.9; g.beginPath();
    let started = false, lastF = -1e9;
    d.f.forEach((f, i) => {
      const x = ((f - a) / (b - a)) * w, y = h - 3 - Math.min(d.line_height[i], ymax) / ymax * (h - 6);
      if (!started || f - lastF > 600) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
      lastF = f;
    });
    g.stroke();
  }
  g.globalAlpha = 1; g.fillStyle = 'rgba(255,255,255,0.55)'; g.font = '10px system-ui';
  g.fillText('back-line height, m from own goal (1-min avg)', 6, 11);
  const xt = ((S.t - a) / (b - a)) * w; g.fillStyle = '#ffffff'; g.fillRect(xt - 1, 0, 2, h);
}

// overlay toggles
for (const [id, k] of [['oShape', 'shape'], ['oMarking', 'marking'], ['oLanes', 'lanes'], ['oSpace', 'space']]) {
  $(id).classList.toggle('on', S.ov[k]);
  $(id).onclick = () => { S.ov[k] = !S.ov[k]; $(id).classList.toggle('on', S.ov[k]); };
}
// side tabs
document.querySelectorAll('#tabs button').forEach((b) => (b.onclick = () => {
  document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('on', x === b));
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('on', x.id === `tab-${b.dataset.tab}`));
  const ask = b.dataset.tab === 'ask';
  $('quick').style.display = ask ? 'flex' : 'none'; $('askrow').style.display = ask ? 'flex' : 'none';
  if (ask && !$('chat').children.length) addMsg($('chat'), 'bot', md('Ask about **what you are watching**: I receive the match, the current time, the selected player and the live defensive state, and answer from computed data (shape, body pose, Dynamic Events). I **remember the conversation**, so follow-ups work (*"and the second goal?"*, *"how could it be avoided?"*). Try *Explain this moment*.'));
}));
async function runSim(params, fromChat = false) {
  const qs = new URLSearchParams({ match: S.match, ...params }).toString();
  $('loading').textContent = 'Simulating better defending…'; $('loading').style.display = 'block';
  try {
    const r = await (await fetch(`/api/simulate?${qs}`)).json();
    if (r.error) throw new Error(r.error);
    S.sim = r; S.t = r.frame; S.lastToastF = -1; if (S.playing) togglePlay();
    if (r.key_fix) select(r.key_fix.player_id);
    if (!fromChat) document.querySelector('#tabs button[data-tab=live]').click();
  } catch (e) { alert(`Simulation failed: ${e.message}`); }
  $('loading').textContent = 'Loading…'; $('loading').style.display = 'none';
}
window.__runSim = runSim;
function simCard() {
  const r = S.sim; if (!r) return '';
  const b = r.before, a = r.after;
  const row = (k, n, d = 1) => (b[k] == null ? '' : `<div class="row"><span>${n}</span><span>${(+b[k]).toFixed(d)} → <b>${(+a[k]).toFixed(d)}</b></span></div>`);
  return `<div class="card" style="border-color:#22c55e"><h4>🧪 Simulation ${r.goal ? `· goal ${r.goal.goal_number} (${r.goal.scorer})` : ''} <span class="muted" style="font-weight:400">${r.moment || clock(r.frame)[0]}</span></h4>
    <div class="row"><span>Danger</span><span><b style="color:#4ade80">−${Math.round(r.danger_reduction * 100)}%</b></span></div>
    ${row('free_attackers', 'Free attackers near goal', 0)}${row('focus_marker_dist', `Nearest defender to ${r.focus_attacker || 'scorer'} (m)`)}
    ${row('carrier_pressure_dist', 'Pressure on the ball (m)')}${row('lane_open', 'Open shot lane', 2)}${row('attack_space_m2', 'Attack space near goal (m²)', 0)}
    ${r.key_fix ? `<div style="margin-top:6px"><b>Key fix:</b> ${r.key_fix.defender} — ${r.key_fix.why} (${r.key_fix.move_m} m)</div>` : ''}
    <div class="muted" style="font-size:11.5px;margin-top:4px">${r.moves.slice(1, 6).map((m) => `${m.defender}: ${m.why}`).join(' · ')}</div>
    <div class="muted" style="font-size:10.5px;margin-top:4px">Green = corrected positions reachable in ${r.reaction_window_s}s (≤${r.max_move_m} m); attackers fixed.</div>
    <div style="display:flex;gap:6px;margin-top:6px"><button onclick="window.__viewer.t=window.__viewer.sim.frame-40;">↺ Replay into it</button><button onclick="window.__viewer.sim=null">Clear</button></div></div>`;
}
function goFocus(fc) {
  if (fc.sim) { runSim(fc.sim, true); return; }
  S.t = fc.frame; S.lastToastF = -1;
  if (fc.select) select(fc.select);
  if (fc.cam) setCam(fc.cam);
  if (fc.play && !S.playing) togglePlay();
}
function secToFrame(sec) {
  const cm = S.meta.clockmap || []; let best = null, bd = Infinity;
  for (const [s, p, f] of cm) { const d = Math.abs(s - sec) + (p === 2 && sec < 45 * 60 ? 1e6 : 0); if (d < bd) { bd = d; best = f; } }
  return best;
}
function linkTimes(html) {
  return html.replace(/(^|[^\d:>])(\d{1,3}):(\d{2})(?![\d:])/g, (m, pre, mm, ss) =>
    `${pre}<a href="#" class="tlink" data-sec="${Number(mm) * 60 + Number(ss)}" style="color:#fbbf24">${mm}:${ss}</a>`);
}
function wireLinks(el, focus) {
  el.querySelectorAll('.tlink').forEach((a) => (a.onclick = (e) => {
    e.preventDefault(); const f = secToFrame(Number(a.dataset.sec)); if (f != null) { S.t = f - 30; S.lastToastF = -1; if (!S.playing) togglePlay(); }
  }));
  const b = el.querySelector('.gofocus'); if (b && focus) b.onclick = () => goFocus(focus);
}
async function ask(q) {
  if (!q.trim()) return;
  if (S.playing) togglePlay();
  addMsg($('chat'), 'user', md(q));
  const wait = addMsg($('chat'), 'bot', '<span class="muted">Thinking with the data…</span>');
  const m = S.ana;
  const live = m && m.inPlay ? {
    defending: teamNameById(m.defTeam), line_height: +m.lineHeight.toFixed(1), length: +m.length.toFixed(1), width: +m.width.toFixed(1),
    ball_zone_area: Number.isFinite(m.ballZoneArea) ? Math.round(m.ballZoneArea) : null, line_gap: +m.lineGap.toFixed(1),
    behind_ball: m.behindBall, press_dist: Number.isFinite(m.pressDist) ? +m.pressDist.toFixed(1) : null,
    free_attackers: m.freeAttackers, open_forward_lanes: m.openLanesAhead ?? null, alerts: m.alerts.map((a) => a.txt),
    selected_live: selectedLive(),
  } : null;
  try {
    const r = await fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, match: S.match, frame: Math.round(S.t), clock: clock(S.t)[0], selected: S.sel, live, history: S.chatHist, session: S.sessionId }) });
    const a = await r.json();
    if (a.error) throw new Error(a.error);
    const calls = (a.calls || []).map((c) => `${c.tool}(${JSON.stringify(c.args)})`).join('<br>');
    const rows = a.data && (a.data.rows || a.data.findings);
    let tbl = '';
    if (rows && rows.length) {
      const cols = Object.keys(rows[0]).slice(0, 7);
      const fmt = (v) => (typeof v === 'number' ? (Math.abs(v) < 1 && v !== 0 ? v.toFixed(2) : String(+v.toFixed(2))) : String(v ?? ''));
      tbl = `<table style="width:100%;font-size:11.5px;border-collapse:collapse;margin-top:6px"><tr>${cols.map((c) => `<th style="text-align:left;color:#9aa7b8;font-weight:500;padding:2px">${c}</th>`).join('')}</tr>` +
        rows.slice(0, 10).map((r) => `<tr>${cols.map((c) => `<td style="padding:2px;border-top:1px solid rgba(255,255,255,.08)">${fmt(r[c])}</td>`).join('')}</tr>`).join('') + '</table>';
    }
    const go = a.focus ? `<div style="margin-top:6px"><button class="gofocus">▶ Go to ${a.focus.label || 'moment'}</button></div>` : '';
    wait.innerHTML = linkTimes(md(a.text)) + tbl + go + `<details><summary>${a.backend === 'gemini' ? 'Gemini' : 'offline'} · ${(a.calls || []).length} tool call(s)</summary>${calls || '—'}</details>`;
    S.chatHist.push({ role: 'user', text: q }, { role: 'model', text: a.text });
    wireLinks(wait, a.focus);
    if (a.focus) goFocus(a.focus);
  } catch (e) { wait.innerHTML = `<span style="color:#f87171">Error: ${e.message}</span>`; }
}
$('askgo').onclick = () => { ask($('askq').value); $('askq').value = ''; };
$('askq').addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') { ask($('askq').value); $('askq').value = ''; } });
document.querySelectorAll('#quick button[data-q]').forEach((b) => (b.onclick = () => ask(b.dataset.q)));
$('newchat').onclick = async () => {
  try { await fetch('/api/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: S.sessionId }) }); } catch (e) { /* ignore */ }
  S.sessionId = Math.random().toString(36).slice(2) + Date.now().toString(36); S.chatHist = [];
  $('chat').innerHTML = ''; addMsg($('chat'), 'bot', md('New conversation. I remember everything we discuss here — ask follow-ups like *"and the second goal?"*, *"why him?"*, *"how could it be avoided?"*.'));
};

// ------------------------------------------------------------------ boot
(async () => {
  resize();
  const ms = await (await fetch('/api/matches')).json();
  const sel = $('match');
  ms.forEach((m) => { const o = document.createElement('option'); o.value = m.id; o.textContent = m.title; sel.appendChild(o); });
  const want = Number(params.get('match')) || (ms[0] && ms[0].id);
  sel.value = want;
  sel.onchange = async () => { S.sel = null; S.sim = null; $('newchat').click(); await loadMeta(Number(sel.value)); };
  if (want) await loadMeta(want);
  setCam(S.cam);
  if (params.get('play') === '1') togglePlay();
  window.__viewer = S; // for debugging / automated screenshots
  requestAnimationFrame(loop);
})();
