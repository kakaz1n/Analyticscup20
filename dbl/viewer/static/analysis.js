// Live defensive analysis from tracking at the current instant (pure functions, no three.js).
// All metrics are computed in the DEFENDING team's frame: own goal at x = -L/2 ("goal-ward" = -x).

export const TH = {
  freeRadius: 5,        // m: an attacker with no defender this close is "free"
  dangerZone: 35,       // m from goal: free attackers only count inside this distance
  pressRadius: 5,       // m: defenders "on the ball"
  laneBlock: 1.6,       // m: a defender this close to a pass lane blocks it
  lineGapAlert: 15,     // m between back line and midfield line
  ballZoneAlert: 260,   // m² hull of the 5 nearest defenders to the ball
  noPressAlert: 7,      // m nearest defender to the ball carrier
};

function hullArea(pts) {
  if (pts.length < 3) return NaN;
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (const q of [...p].reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  const h = lo.slice(0, -1).concat(up.slice(0, -1));
  let a = 0; for (let i = 0; i < h.length; i++) { const j = (i + 1) % h.length; a += h[i][0] * h[j][1] - h[j][0] * h[i][1]; }
  return { area: Math.abs(a) / 2, hull: h };
}
export { hullArea };

export function defendsLeft(meta, teamId, period) {
  const side = meta.home_side[(period || 1) - 1] || 'left_to_right';
  const homeDefLeft = side === 'left_to_right';
  return teamId === meta.teams.home.id ? homeDefLeft : !homeDefLeft;
}

function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  let t = l2 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0; t = Math.max(0, Math.min(1, t));
  return [Math.hypot(px - ax - t * dx, py - ay - t * dy), t];
}

/**
 * @param tr   {pos: Map(pid->[x,y]), ball:[x,y,z]|null, poss:'home team'|'away team'|''}
 * @param prev same structure ~0.3 s earlier (for velocities)
 * Returns metrics + geometry in RAW pitch coords (for drawing) and normalised numbers.
 */
export function analyse(meta, tr, prev, period, possHint, info) {
  if (!tr || !tr.pos.size) return null;
  const [L] = meta.pitch;
  const possTeam = possHint === 'home team' ? meta.teams.home.id : possHint === 'away team' ? meta.teams.away.id : null;
  if (possTeam == null) return { inPlay: false };
  const defTeam = possTeam === meta.teams.home.id ? meta.teams.away.id : meta.teams.home.id;
  const s = defendsLeft(meta, defTeam, period) ? 1 : -1;  // raw -> normalised multiplier
  const gx = -L / 2;
  const D = [], A = [], Dgk = [];
  for (const [pid, [x, y]] of tr.pos) {
    const p = info.get(pid); if (!p) continue;
    const o = { pid, x, y, nx: x * s, ny: y * s, gk: p.gk };
    const pv = prev && prev.pos.get(pid);
    o.vx = pv ? (x - pv[0]) / prev.dt : 0; o.vy = pv ? (y - pv[1]) / prev.dt : 0;
    if (p.team === defTeam) { (p.gk ? Dgk : D).push(o); } else if (!p.gk) A.push(o); else A.gkObj = o;
  }
  if (D.length < 6) return { inPlay: false };
  const ball = tr.ball ? { x: tr.ball[0], y: tr.ball[1], nx: tr.ball[0] * s, ny: tr.ball[1] * s } : null;
  const byDepth = [...D].sort((a, b) => a.nx - b.nx);
  const back = byDepth.slice(0, 4), mid = byDepth.slice(4, 7);
  const mean = (arr, k) => arr.reduce((t, o) => t + o[k], 0) / arr.length;
  const allDef = [...D, ...Dgk].sort((a, b) => a.nx - b.nx);
  const m = {
    inPlay: true, defTeam, possTeam, s,
    lineHeight: mean(back, 'nx') - gx,
    blockHeight: mean(D, 'nx') - gx,
    length: byDepth[byDepth.length - 1].nx - byDepth[0].nx,
    width: Math.max(...D.map((o) => o.ny)) - Math.min(...D.map((o) => o.ny)),
    lineGap: mid.length ? mean(mid, 'nx') - mean(back, 'nx') : NaN,
    offsideX: allDef.length >= 2 ? allDef[1].nx : NaN, // normalised x of 2nd-last defender
  };
  const hull = hullArea(D.map((o) => [o.x, o.y]));
  m.area = hull.area; m.hull = hull.hull;
  m.backLine = [...back].sort((a, b) => a.ny - b.ny).map((o) => [o.x, o.y]);
  m.midLine = [...mid].sort((a, b) => a.ny - b.ny).map((o) => [o.x, o.y]);
  if (ball) {
    const near = [...D].sort((a, b) => Math.hypot(a.x - ball.x, a.y - ball.y) - Math.hypot(b.x - ball.x, b.y - ball.y));
    const bz = hullArea(near.slice(0, 5).map((o) => [o.x, o.y]));
    m.ballZoneArea = bz.area; m.ballZoneHull = bz.hull;
    m.pressDist = Math.hypot(near[0].x - ball.x, near[0].y - ball.y);
    m.pressN = near.filter((o) => Math.hypot(o.x - ball.x, o.y - ball.y) <= TH.pressRadius).length;
    m.presser = near[0].pid;
    m.behindBall = D.filter((o) => o.nx < ball.nx).length;
    // ball carrier = nearest attacker to the ball (if within 2.5 m)
    const carrier = [...A].sort((a, b) => Math.hypot(a.x - ball.x, a.y - ball.y) - Math.hypot(b.x - ball.x, b.y - ball.y))[0];
    if (carrier && Math.hypot(carrier.x - ball.x, carrier.y - ball.y) < 2.5) {
      m.carrier = carrier.pid;
      // passing lanes from the carrier to each team-mate, blocked if a defender is near the lane
      m.lanes = A.filter((o) => o !== carrier).map((o) => {
        let blocker = null, best = Infinity;
        for (const d of D.concat(Dgk)) {
          const [dist, t] = segDist(d.x, d.y, carrier.x, carrier.y, o.x, o.y);
          if (t > 0.08 && t < 0.97 && dist < best) { best = dist; blocker = d.pid; }
        }
        const ahead = (o.nx - carrier.nx) < -2; // toward the defending goal
        return { to: o.pid, from: [carrier.x, carrier.y], end: [o.x, o.y], open: best > TH.laneBlock, blocker, ahead,
          len: Math.hypot(o.x - carrier.x, o.y - carrier.y) };
      }).filter((l) => l.len < 45);
      m.openLanesAhead = m.lanes.filter((l) => l.open && l.ahead).length;
    }
  }
  // marking: nearest defender to each attacker; free attackers near goal
  m.marks = A.map((a) => {
    let best = null, bd = Infinity;
    for (const d of D) { const dd = Math.hypot(a.x - d.x, a.y - d.y); if (dd < bd) { bd = dd; best = d; } }
    const goalSide = best ? best.nx < a.nx : false;
    const inZone = a.nx - gx <= TH.dangerZone;
    return { att: a.pid, def: best && best.pid, dist: bd, goalSide, free: bd > TH.freeRadius && inZone, inZone,
      a: [a.x, a.y], d: best ? [best.x, best.y] : null };
  });
  m.freeAttackers = m.marks.filter((k) => k.free).length;
  m.attackersInZone = m.marks.filter((k) => k.inZone).length;
  // alerts, analyst wording
  const al = [];
  if (m.lineGap > TH.lineGapAlert) al.push({ lvl: 'red', txt: `Gap between lines ${m.lineGap.toFixed(0)} m` });
  if (m.freeAttackers >= 1) al.push({ lvl: m.freeAttackers >= 2 ? 'red' : 'amber', txt: `${m.freeAttackers} free attacker(s) within ${TH.dangerZone} m of goal` });
  if (m.ballZoneArea > TH.ballZoneAlert) al.push({ lvl: 'amber', txt: `Not compact around the ball (${m.ballZoneArea.toFixed(0)} m²)` });
  if (m.carrier && m.pressDist > TH.noPressAlert) al.push({ lvl: 'amber', txt: `No pressure on the ball (${m.pressDist.toFixed(1)} m)` });
  if (m.openLanesAhead >= 3) al.push({ lvl: 'amber', txt: `${m.openLanesAhead} open forward passing lanes` });
  if (ball && m.behindBall <= 5) al.push({ lvl: 'red', txt: `Only ${m.behindBall} outfielders behind the ball` });
  m.alerts = al;
  m.D = D; m.A = A; m.Dgk = Dgk; m.ball = ball;
  return m;
}

/**
 * Space control (simplified Spearman 2018 / Fernández & Bornn 2018 time-to-intercept model).
 * Returns Float32Array of P(attacking team controls cell) on a grid over the pitch.
 */
export function pitchControl(m, L, W, nx = 70, ny = 46) {
  const out = new Float32Array(nx * ny);
  const react = 0.7, vmax = 7.0, k = Math.PI / Math.sqrt(3) / 0.45;
  const players = [...m.D, ...m.Dgk].map((o) => ({ ...o, att: false })).concat(m.A.map((o) => ({ ...o, att: true })));
  for (const p of players) { p.rx = p.x + p.vx * react; p.ry = p.y + p.vy * react; }
  for (let j = 0; j < ny; j++) {
    const y = -W / 2 + (j + 0.5) * W / ny;
    for (let i = 0; i < nx; i++) {
      const x = -L / 2 + (i + 0.5) * L / nx;
      let ta = Infinity, td = Infinity;
      for (const p of players) {
        const t = react + Math.hypot(x - p.rx, y - p.ry) / vmax;
        if (p.att) { if (t < ta) ta = t; } else if (t < td) td = t;
      }
      out[j * nx + i] = 1 / (1 + Math.exp(-k * (td - ta)));
    }
  }
  return { grid: out, nx, ny };
}
