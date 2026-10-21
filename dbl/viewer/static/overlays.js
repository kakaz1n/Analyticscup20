// Analyst overlays drawn on the grass: block hull, back/midfield lines, offside line, ball-zone hull,
// marking links + free attackers, passing lanes, pressure ring, space control heatmap.
import * as THREE from 'three';

const COL = {
  hull: new THREE.Color('#60a5fa'), back: new THREE.Color('#ffffff'), mid: new THREE.Color('#cbd5e1'),
  offside: new THREE.Color('#e879f9'), mark: new THREE.Color('#22d3ee'), markLoose: new THREE.Color('#f43f5e'),
  open: new THREE.Color('#22c55e'), blocked: new THREE.Color('#64748b'), free: new THREE.Color('#ef4444'),
  ballZone: new THREE.Color('#8b5cf6'), under: new THREE.Color('#0b1220'),
};

export class Overlays {
  constructor(scene) {
    this.scene = scene;
    const flat = (opacity) => new THREE.MeshBasicMaterial({ transparent: true, opacity, depthWrite: false });
    this.seg = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), flat(0.9), 600);
    this.seg.frustumCulled = false; this.seg.renderOrder = 3; scene.add(this.seg);
    this.rings = new THREE.InstancedMesh(new THREE.RingGeometry(0.75, 1.05, 32), flat(0.95), 30);
    this.rings.frustumCulled = false; this.rings.renderOrder = 4; scene.add(this.rings);
    this.hullMesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: COL.hull, transparent: true, opacity: 0.13, depthWrite: false, side: THREE.DoubleSide }));
    this.hullMesh.position.z = 0.025; this.hullMesh.renderOrder = 2; scene.add(this.hullMesh);
    this.bzMesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: COL.ballZone, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
    this.bzMesh.position.z = 0.03; this.bzMesh.renderOrder = 2; scene.add(this.bzMesh);
    this.space = null; this.spaceTex = null;
    this._m = new THREE.Matrix4(); this._q = new THREE.Quaternion(); this._z = new THREE.Vector3(0, 0, 1);
  }

  // ---- counterfactual simulation: green ghosts at corrected positions + arrows from the real ones
  // specs: [{from, to, capsule}] in raw coords; ph = animation phase 0..1 (real -> corrected)
  updateSim(specs, show, ph) {
    if (!this.simGhost) {
      const mat = new THREE.MeshStandardMaterial({ color: '#22c55e', transparent: true, opacity: 0.55, depthWrite: false, emissive: '#14532d' });
      this.simGhost = new THREE.InstancedMesh(new THREE.CapsuleGeometry(0.3, 1.15, 4, 10), mat, 14);
      this.simGhost.frustumCulled = false; this.scene.add(this.simGhost);
      this.simSeg = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: '#4ade80', transparent: true, opacity: 0.95, depthWrite: false }), 90);
      this.simSeg.frustumCulled = false; this.simSeg.renderOrder = 5; this.scene.add(this.simSeg);
      this.simRing = new THREE.InstancedMesh(new THREE.RingGeometry(0.55, 0.8, 28), new THREE.MeshBasicMaterial({ color: '#4ade80', transparent: true, opacity: 0.9, depthWrite: false }), 14);
      this.simRing.frustumCulled = false; this.scene.add(this.simRing);
    }
    if (!show || !specs || !specs.length) { this.simGhost.count = 0; this.simSeg.count = 0; this.simRing.count = 0; return; }
    let g = 0, k = 0, r = 0;
    const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
    for (const m of specs.slice(0, 14)) {
      if (!m.from || !m.to || !Number.isFinite(m.from[0]) || !Number.isFinite(m.to[0])) continue;
      if (m.capsule) {
        const x = m.from[0] + (m.to[0] - m.from[0]) * ph, y = m.from[1] + (m.to[1] - m.from[1]) * ph;
        this._m.compose(new THREE.Vector3(x, y, 0.88), q, new THREE.Vector3(1, 1, 1));
        this.simGhost.setMatrixAt(g++, this._m);
      }
      this._m.compose(new THREE.Vector3(m.to[0], m.to[1], 0.05), this._q.identity(), new THREE.Vector3(1, 1, 1));
      this.simRing.setMatrixAt(r++, this._m);
      // dashed arrow from real to corrected position
      const dx = m.to[0] - m.from[0], dy = m.to[1] - m.from[1], len = Math.hypot(dx, dy);
      if (len < 0.5) continue;
      const ux = dx / len, uy = dy / len;
      for (let s = 0; s < len - 0.4 && k < 88; s += 1.0) {
        const e = Math.min(s + 0.6, len - 0.4);
        const a = [m.from[0] + ux * s, m.from[1] + uy * s], b = [m.from[0] + ux * e, m.from[1] + uy * e];
        this._q.setFromAxisAngle(this._z, Math.atan2(uy, ux));
        this._m.compose(new THREE.Vector3((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.06), this._q, new THREE.Vector3(e - s, 0.16, 0.01));
        this.simSeg.setMatrixAt(k++, this._m);
      }
    }
    this.simGhost.count = g; this.simRing.count = r; this.simSeg.count = k;
    for (const mm of [this.simGhost, this.simSeg, this.simRing]) mm.instanceMatrix.needsUpdate = true;
  }

  _segment(i, a, b, width, color, z = 0.04) {
    const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
    if (len < 0.05) return i;
    this._q.setFromAxisAngle(this._z, Math.atan2(dy, dx));
    this._m.compose(new THREE.Vector3((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z), this._q, new THREE.Vector3(len, width, 0.01));
    this.seg.setMatrixAt(i, this._m); this.seg.setColorAt(i, color);
    return i + 1;
  }

  _dashed(i, a, b, width, color, dash = 0.8, gap = 0.6) {
    const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len;
    for (let s = 0; s < len && i < 590; s += dash + gap) {
      const e = Math.min(s + dash, len);
      i = this._segment(i, [a[0] + ux * s, a[1] + uy * s], [a[0] + ux * e, a[1] + uy * e], width, color);
    }
    return i;
  }

  _poly(mesh, pts) {
    if (!pts || pts.length < 3) { mesh.visible = false; return; }
    const shape = new THREE.Shape(pts.map(([x, y]) => new THREE.Vector2(x, y)));
    mesh.geometry.dispose(); mesh.geometry = new THREE.ShapeGeometry(shape); mesh.visible = true;
  }

  hide() {
    this.seg.count = 0; this.rings.count = 0; this.hullMesh.visible = false; this.bzMesh.visible = false;
    if (this.space) this.space.visible = false;
  }

  // team colours (kit jersey colours from the match sheet): lines and block of each team
  setTeams(meta) {
    this.teamCol = new Map();
    for (const side of ['home', 'away']) {
      const tm = meta.teams[side];
      this.teamCol.set(tm.id, new THREE.Color(tm.color || (side === 'home' ? '#60a5fa' : '#f87171')));
    }
  }

  update(m, opt, W, t) {
    if (!m || !m.inPlay) { this.hide(); return; }
    let i = 0, r = 0;
    // --- SHAPE: team hull, back line, midfield line, offside line, ball-zone hull
    if (opt.shape) {
      this._poly(this.hullMesh, m.hull);
      this.hullMesh.material.color.copy((this.teamCol && this.teamCol.get(m.defTeam)) || COL.hull);
      this._poly(this.bzMesh, m.ballZoneHull);
      // lines of BOTH teams, each in its own kit colour (defending team thicker); dark underlay keeps
      // light kits (white / yellow) readable on the grass
      for (const tl of m.teamLines || []) {
        const c = (this.teamCol && this.teamCol.get(tl.team)) || (tl.team === m.defTeam ? COL.back : COL.mid);
        const w = tl.team === m.defTeam ? 0.24 : 0.15;
        tl.lines.forEach((ln, li) => {
          const ww = li === 0 ? w : w * 0.75;
          for (let k = 0; k + 1 < ln.length && i < 390; k++) {
            if (Math.hypot(ln[k + 1][0] - ln[k][0], ln[k + 1][1] - ln[k][1]) > 32) continue; // not one line
            i = this._segment(i, ln[k], ln[k + 1], ww + 0.12, COL.under, 0.036);
            i = this._segment(i, ln[k], ln[k + 1], ww, c, 0.04);
          }
        });
      }
      if (Number.isFinite(m.offsideX)) {
        const x = m.offsideX * m.s; // back to raw coords
        i = this._dashed(i, [x, -W / 2], [x, W / 2], 0.14, COL.offside, 1.2, 0.8);
      }
    } else { this.hullMesh.visible = false; this.bzMesh.visible = false; }
    // --- MARKING: nearest defender per attacker, free attackers ringed red
    if (opt.marking) {
      for (const k of m.marks) {
        if (k.d && k.dist < 12) i = this._segment(i, k.a, k.d, 0.07, k.goalSide ? COL.mark : COL.markLoose, 0.05);
        if (k.free && r < 28) {
          const pulse = 1 + 0.15 * Math.sin(t * 6);
          this._m.compose(new THREE.Vector3(k.a[0], k.a[1], 0.06), this._q.identity(), new THREE.Vector3(pulse, pulse, 1));
          this.rings.setMatrixAt(r, this._m); this.rings.setColorAt(r, COL.free); r++;
        }
      }
    }
    // --- PRESSURE / LANES: ring on the carrier (green = pressed, red = free), passing lanes
    if (opt.lanes && m.lanes) {
      for (const l of m.lanes) {
        if (l.open) i = this._segment(i, l.from, l.end, l.ahead ? 0.13 : 0.08, COL.open, 0.045);
        else i = this._dashed(i, l.from, l.end, 0.07, COL.blocked);
      }
    }
    if ((opt.lanes || opt.marking) && m.carrier != null && m.ball && r < 29) {
      const c = m.pressDist <= 2.5 ? new THREE.Color('#22c55e') : m.pressDist <= 5 ? new THREE.Color('#f59e0b') : new THREE.Color('#ef4444');
      this._m.compose(new THREE.Vector3(m.ball.x, m.ball.y, 0.07), this._q.identity(), new THREE.Vector3(1.6, 1.6, 1));
      this.rings.setMatrixAt(r, this._m); this.rings.setColorAt(r, c); r++;
    }
    this.seg.count = i; this.rings.count = r;
    this.seg.instanceMatrix.needsUpdate = true; if (this.seg.instanceColor) this.seg.instanceColor.needsUpdate = true;
    this.rings.instanceMatrix.needsUpdate = true; if (this.rings.instanceColor) this.rings.instanceColor.needsUpdate = true;
    if (this.space) this.space.visible = !!opt.space;
  }

  // P(attacking team controls) grid -> red (attack) / blue (defence) heatmap; space BEHIND the
  // defensive line that the attack controls is drawn strongest (the dangerous space).
  updateSpace(pc, m, L, W) {
    const { grid, nx, ny } = pc;
    if (!this.space) {
      this.spaceData = new Uint8Array(nx * ny * 4);
      this.spaceTex = new THREE.DataTexture(this.spaceData, nx, ny, THREE.RGBAFormat);
      this.spaceTex.magFilter = THREE.LinearFilter; this.spaceTex.minFilter = THREE.LinearFilter;
      this.space = new THREE.Mesh(new THREE.PlaneGeometry(L, W), new THREE.MeshBasicMaterial({ map: this.spaceTex, transparent: true, depthWrite: false }));
      this.space.position.z = 0.02; this.space.renderOrder = 1; this.scene.add(this.space);
    }
    const d = this.spaceData;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const p = grid[j * nx + i], k = (j * nx + i) * 4;
      const x = (-L / 2 + (i + 0.5) * L / nx) * m.s; // normalised x of the cell
      const behind = Number.isFinite(m.offsideX) && x < m.offsideX;
      if (p >= 0.5) { d[k] = 239; d[k + 1] = 68; d[k + 2] = 68; d[k + 3] = Math.round((p - 0.5) * 2 * (behind ? 200 : 95)); }
      else { d[k] = 59; d[k + 1] = 130; d[k + 2] = 246; d[k + 3] = Math.round((0.5 - p) * 2 * 70); }
    }
    this.spaceTex.needsUpdate = true;
    // area of dangerous space: attack-controlled cells behind the defensive line
    let cells = 0; for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const x = (-L / 2 + (i + 0.5) * L / nx) * m.s;
      if (grid[j * nx + i] > 0.5 && Number.isFinite(m.offsideX) && x < m.offsideX && x > -L / 2) cells++;
    }
    return cells * (L / nx) * (W / ny);
  }
}
