"""Posture-aware counterfactual coaching: "what should he (or they) have done instead?"

Extends the positional simulation (``dbl.simulate``) with the body: every defender has a torso heading,
a head heading and a stance taken from SkillCorner body pose at that frame. Posture changes danger through
three effects, each calibrated on the two matches by the framework's own evidence (``calibration()``):

* SEE    — an attacker is only fully covered if the defender sees **ball and man** (both within ±60° of
           his head). Otherwise coverage is scaled by the measured effect of dual vision on
           "receiver left unpressured" (14% vs 32% → ×0.82).
* SHAPE  — body shape to the own goal (turned / open / square) sets the reaction lag to runs, measured at
           run starts (square ≈1.07 s, open ≈0.49 s, turned ≈0.10 s). While he reacts, a runner gains
           ``speed × lag`` metres, and the defender reaches space later.
* STANCE — a loaded stance (lower / more flexed than his own norm at that speed) makes pressure on the
           ball carrier more effective, scaled by the measured regain effect (19% vs 14% → ×0.75 when not
           loaded).

Positions stay limited to what each defender could reach reacting 1.5 s earlier (≤ 9 m). Heads may turn
up to 75° from the torso. Attackers are held fixed. Search: alternating coordinate descent, per defender
position then posture.

Two ways to use it:
* ``simulate(..., posture=True)`` / ``coach_player()``: automatic coach for the team or one defender.
* ``whatif(match, frame, edits)``: score any configuration an analyst drags/rotates in the 3D viewer.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import BUILD_DIR, MATCHES
from .simulate import Scene, REACT_S, V_MAX, GRID_R, GRID_A, _value

FOV = np.radians(60)
NECK = np.radians(75)
TORSO_GRID = np.radians(np.arange(-180, 180, 22.5))
HEAD_OFF = np.radians([-60, -30, 0, 30, 60])


# ------------------------------------------------------------------ calibration from the framework's evidence
@lru_cache(maxsize=1)
def calibration() -> dict:
    cal = {"lag": {"open": 0.49, "square": 1.07, "turned": 0.10}, "k_vision": 0.82, "k_unloaded": 0.75,
           "source": {}}
    try:
        ev = {e["id"]: e for e in json.loads((BUILD_DIR / "evidence.json").read_text(encoding="utf-8"))}
        e = ev.get("dual_vision_free")
        if e:  # P(receiver pressured): seen = 1 - rate_exposed; not seen = 1 - (rate_exposed - adj_diff)
            seen, unseen = 1 - e["rate_exposed"], 1 - (e["rate_exposed"] - e["adj_diff"])
            cal["k_vision"] = round(float(np.clip(unseen / seen, 0.5, 1.0)), 3)
            cal["source"]["k_vision"] = f"dual vision → receiver unpressured {e['rate_exposed']:.0%} vs " \
                                        f"{e['rate_exposed'] - e['adj_diff']:.0%} (adjusted)"
        e = ev.get("loaded_regain")
        if e:
            other = e["rate_exposed"] - e["adj_diff"]
            cal["k_unloaded"] = round(float(np.clip(other / e["rate_exposed"], 0.5, 1.0)), 3)
            cal["source"]["k_unloaded"] = f"loaded stance → regain {e['rate_exposed']:.0%} vs {other:.0%} (adjusted)"
    except Exception:  # noqa: BLE001
        pass
    runs = [pd.read_parquet(p, columns=["body_shape", "reaction_lag_s"])
            for m in MATCHES if (p := BUILD_DIR / str(m) / "runs.parquet").exists()]
    if runs:
        r = pd.concat(runs).dropna()
        lag = r.groupby("body_shape").reaction_lag_s.mean()
        for k in ("open", "square", "turned"):
            if k in lag:
                cal["lag"][k] = round(float(lag[k]), 2)
        cal["source"]["lag"] = "mean reaction lag to forward runs by body shape at run start " + \
            ", ".join(f"{k} {cal['lag'][k]:.2f} s" for k in ("turned", "open", "square"))
        # 'turned' (facing the own goal) shows a tiny lag because those defenders are usually already
        # retreating when the run starts: a confound, not a benefit of turning your back. Never reward it.
        if cal["lag"]["turned"] < cal["lag"]["open"]:
            cal["source"]["lag"] += f" (turned capped at open: measured {cal['lag']['turned']:.2f} s is confounded by retreating defenders)"
            cal["lag"]["turned"] = cal["lag"]["open"]
    return cal


# ------------------------------------------------------------------ posture at a frame
@lru_cache(maxsize=2)
def _pose(match_id: int) -> pd.DataFrame:
    p = BUILD_DIR / str(match_id) / "pose_features.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, columns=["frame25", "player_id", "torso_deg", "head_deg", "knee_deg", "hip_h",
                                       "head_err", "torso_err", "frame10"]).sort_values("frame25", kind="stable") \
        .reset_index(drop=True)


@lru_cache(maxsize=2)
def _norms(match_id: int) -> dict:
    """Speed-adjusted posture norms (as in MatchContext.posture_model) for the stance call."""
    pf = _pose(match_id)
    if pf.empty:
        return {}
    xy = pd.read_parquet(BUILD_DIR / str(match_id) / "players_xy.parquet", columns=["frame", "player_id", "speed"])
    p = pf[["frame10", "player_id", "hip_h", "knee_deg"]].dropna().copy()
    p["frame"] = p.frame10.round().astype(int)
    p = p.merge(xy, on=["frame", "player_id"], how="inner")
    bins = np.arange(0, 10.5, 0.5)
    p["sb"] = np.clip(np.digitize(p.speed, bins) - 1, 0, len(bins) - 2)
    ch = p.groupby("sb").hip_h.median().reindex(range(len(bins) - 1)).ffill().bfill().to_numpy()
    ck = p.groupby("sb").knee_deg.median().reindex(range(len(bins) - 1)).ffill().bfill().to_numpy()
    p["rh"], p["rk"] = p.hip_h - ch[p.sb], p.knee_deg - ck[p.sb]
    return {"bins": bins, "ch": ch, "ck": ck, "oh": p.groupby("player_id").rh.median().to_dict(),
            "ok": p.groupby("player_id").rk.median().to_dict()}


def posture_at(match_id: int, frame: int, player_ids, speeds) -> dict:
    """player_id -> {torso, head (raw deg), loaded (bool)} from the nearest pose frame (±0.2 s)."""
    pf = _pose(match_id)
    if pf.empty:
        return {}
    f25 = frame * 2.5
    f = pf.frame25.to_numpy()
    lo, hi = np.searchsorted(f, f25 - 5, "left"), np.searchsorted(f, f25 + 5, "right")
    w = pf.iloc[lo:hi]
    w = w[w.player_id.isin(list(player_ids))]
    w = w[(w.torso_err.fillna(99) < 18) & w.torso_deg.notna() & w.head_deg.notna()]
    if w.empty:
        return {}
    w = w.assign(dt=(w.frame25 - f25).abs()).sort_values("dt").drop_duplicates("player_id")
    n = _norms(match_id)
    out = {}
    for r in w.itertuples():
        loaded = None
        sp = speeds.get(r.player_id, np.nan)
        if n and np.isfinite(sp) and np.isfinite(r.hip_h) and np.isfinite(r.knee_deg):
            sb = int(np.clip(np.digitize(sp, n["bins"]) - 1, 0, len(n["bins"]) - 2))
            dh = n["ch"][sb] + n["oh"].get(r.player_id, 0) - r.hip_h
            dk = n["ck"][sb] + n["ok"].get(r.player_id, 0) - r.knee_deg
            loaded = bool(dh >= 0.03 or dk >= 5)
        out[int(r.player_id)] = {"torso": float(r.torso_deg), "head": float(r.head_deg), "loaded": loaded}
    return out


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def shape_of(torso_n: float) -> str:
    """Body shape vs own goal (own goal direction = pi in the normalised frame)."""
    a = abs(np.degrees(_wrap(torso_n - np.pi)))
    return "turned" if a < 45 else ("open" if a <= 135 else "square")


# ------------------------------------------------------------------ the scene
class PostureScene(Scene):
    def __init__(self, match_id: int, frame: int, defending_team: str | None = None,
                 focus_attacker: int | None = None, react_s: float = REACT_S):
        super().__init__(match_id, frame, defending_team, focus_attacker, react_s)
        self.cal = calibration()
        self.match_id = match_id
        D = self.D
        speeds = dict(zip(D.player_id, np.hypot(D.vx, D.vy)))
        pos = posture_at(match_id, self.frame, D.player_id.tolist(), speeds)
        n = len(D)
        rot = 0.0 if self.s > 0 else np.pi                       # raw -> normalised heading
        self.known = np.array([pid in pos for pid in D.player_id])
        th = np.zeros(n); hh = np.zeros(n); ld = np.ones(n, bool); ldk = np.zeros(n, bool)
        for i, pid in enumerate(D.player_id):
            if pid in pos:
                th[i] = _wrap(np.radians(pos[pid]["torso"]) + rot)
                hh[i] = _wrap(np.radians(pos[pid]["head"]) + rot)
                if pos[pid]["loaded"] is not None:
                    ld[i], ldk[i] = pos[pid]["loaded"], True
            elif self.ball is not None:                         # unknown: face the ball (neutral, no penalty)
                th[i] = hh[i] = np.arctan2(self.ball[1] - D.y.iloc[i], self.ball[0] - D.x.iloc[i])
        self.th0, self.hh0, self.ld0, self.ld_known = th, hh, ld, ldk
        self.run_speed = np.maximum(0.0, -self.A.vx.to_numpy())  # goal-ward speed of each attacker
        self.lag_ref = self.cal["lag"]["open"]

    # posture -> model effects
    def lags(self, th):
        lag = np.array([self.cal["lag"][shape_of(t)] for t in th])
        return np.where(self.known, lag, self.lag_ref)

    def _in_view(self, hh, px, py, dx, dy):
        ang = np.arctan2(py[:, None] - dy[None], px[:, None] - dx[None])
        return np.abs(_wrap(ang - hh[None])) <= FOV

    def score(self, dx, dy, th=None, hh=None, ld=None, parts=False):
        th = self.th0 if th is None else th
        hh = self.hh0 if hh is None else hh
        ld = self.ld0 if ld is None else ld
        ax, ay = self.A.x.to_numpy(), self.A.y.to_numpy()
        lag = self.lags(th)
        cov, sees_ball = self._coverage(dx, dy, th, hh, ld, lag)
        mark = float((self.w * (1 - cov)).sum())
        return self._rest(dx, dy, th, hh, lag, sees_ball, mark, parts)

    def coverage(self, dx=None, dy=None, th=None, hh=None, ld=None):
        """Per-attacker coverage in [0, 1] (1 = tightly marked goal-side by a defender who sees ball and man)."""
        dx = self.D.x.to_numpy() if dx is None else dx
        dy = self.D.y.to_numpy() if dy is None else dy
        th = self.th0 if th is None else th
        hh = self.hh0 if hh is None else hh
        ld = self.ld0 if ld is None else ld
        return self._coverage(dx, dy, th, hh, ld, self.lags(th))[0]

    def _coverage(self, dx, dy, th, hh, ld, lag):
        ax, ay = self.A.x.to_numpy(), self.A.y.to_numpy()
        d = np.hypot(ax[:, None] - dx[None], ay[:, None] - dy[None])
        d_eff = np.maximum(0.0, d + self.run_speed[:, None] * (lag - self.lag_ref)[None])
        goal_side = dx[None] < ax[:, None] + 0.5
        sees_man = self._in_view(hh, ax, ay, dx, dy)
        if self.ball is not None:
            sees_ball = self._in_view(hh, np.array([self.ball[0]]), np.array([self.ball[1]]), dx, dy)[0]
        else:
            sees_ball = np.ones(len(dx), bool)
        both = sees_man & sees_ball[None]
        vis = np.where(self.known[None] & ~both, self.cal["k_vision"], 1.0)
        stance = np.ones_like(d)
        if self.carrier is not None:
            stance[self.carrier] = np.where(self.known & ~ld, self.cal["k_unloaded"], 1.0)
        cov = (np.exp(-(d_eff / 2.5) ** 2) * np.where(goal_side, 1.0, 0.6) * vis * stance).max(1) \
            if len(dx) else np.zeros(len(ax))
        return cov, sees_ball

    def _rest(self, dx, dy, th, hh, lag, sees_ball, mark, parts):
        lane = 0.0
        if self.ball is not None:
            gx = -self.L / 2
            bx, by = self.ball
            vx, vy = gx - bx, -by
            l2 = vx * vx + vy * vy + 1e-9
            t = np.clip(((dx - bx) * vx + (dy - by) * vy) / l2, 0, 1)
            dist = np.hypot(bx + t * vx - dx, by + t * vy - dy)
            block = np.max(np.exp(-(dist / 1.0) ** 2) * (t > 0.02) * np.where(sees_ball, 1.0, 0.85)) if len(dx) else 0.0
            lane = float(_value(bx, by, self.L) * (1 - block))
        dd = np.hypot(self.gxx[..., None] - dx, self.gyy[..., None] - dy)
        td = (0.7 + np.maximum(-0.3, lag - self.lag_ref) + dd / 7.0).min(-1) if len(dx) else np.inf
        td = np.minimum(td, self.t_gk)
        p_att = 1 / (1 + np.exp(-(np.pi / np.sqrt(3) / 0.45) * (td - self.ta)))
        space = float((p_att * self.gval).sum() / self.gval.sum())
        move = float(((dx - self.D.x.to_numpy()) ** 2 + (dy - self.D.y.to_numpy()) ** 2).sum())
        turn = float((np.degrees(_wrap(th - self.th0)) ** 2 + np.degrees(_wrap(hh - self.hh0)) ** 2).sum())
        total = 1.0 * mark + 1.2 * lane + 2.0 * space + getattr(self, 'reg', 1.0) * (0.002 * move + 2e-6 * turn)
        if parts:
            area = float((p_att > 0.5).sum() * (31 / 23) * (52 / 29))
            return {"total": total, "marking_danger": mark, "lane_open": lane, "space_share": space,
                    "attack_space_m2": area}
        return total

    def score_variants(self, i, dx, dy, th, hh, ld, vx, vy, vth, vhh, vld):
        """Vectorised score of K variants of defender i (others fixed): returns K totals, identical to score()."""
        vx, vy = np.asarray(vx, float), np.asarray(vy, float)
        vth, vhh, vld = np.asarray(vth, float), np.asarray(vhh, float), np.asarray(vld, bool)
        K = len(vx)
        ax, ay = self.A.x.to_numpy(), self.A.y.to_numpy()
        n = len(dx)
        oth = np.array([j for j in range(n) if j != i], int)
        lag_all = self.lags(th)
        # --- others' coverage (Na) and lane block
        if len(oth):
            do = np.hypot(ax[:, None] - dx[oth][None], ay[:, None] - dy[oth][None])
            deo = np.maximum(0.0, do + self.run_speed[:, None] * (lag_all[oth] - self.lag_ref)[None])
            gso = dx[oth][None] < ax[:, None] + 0.5
            smo = self._in_view(hh[oth], ax, ay, dx[oth], dy[oth])
            sbo = (self._in_view(hh[oth], np.array([self.ball[0]]), np.array([self.ball[1]]), dx[oth], dy[oth])[0]
                   if self.ball is not None else np.ones(len(oth), bool))
            viso = np.where(self.known[oth][None] & ~(smo & sbo[None]), self.cal["k_vision"], 1.0)
            sto = np.ones_like(do)
            if self.carrier is not None:
                sto[self.carrier] = np.where(self.known[oth] & ~ld[oth], self.cal["k_unloaded"], 1.0)
            cov_o = (np.exp(-(deo / 2.5) ** 2) * np.where(gso, 1.0, 0.6) * viso * sto).max(1)
        else:
            cov_o = np.zeros(len(ax)); sbo = np.ones(0, bool)
        # --- candidate i (Na x K)
        known_i = bool(self.known[i])
        lag_i = np.array([self.cal["lag"][shape_of(t)] for t in vth]) if known_i else np.full(K, self.lag_ref)
        d = np.hypot(ax[:, None] - vx[None], ay[:, None] - vy[None])
        de = np.maximum(0.0, d + self.run_speed[:, None] * (lag_i - self.lag_ref)[None])
        gs = vx[None] < ax[:, None] + 0.5
        sm = np.abs(_wrap(np.arctan2(ay[:, None] - vy[None], ax[:, None] - vx[None]) - vhh[None])) <= FOV
        sb = (np.abs(_wrap(np.arctan2(self.ball[1] - vy, self.ball[0] - vx) - vhh)) <= FOV) if self.ball is not None \
            else np.ones(K, bool)
        vis = np.where(known_i & ~(sm & sb[None]), self.cal["k_vision"], 1.0)
        st = np.ones_like(d)
        if self.carrier is not None:
            st[self.carrier] = np.where(known_i & ~vld, self.cal["k_unloaded"], 1.0)
        cov_i = np.exp(-(de / 2.5) ** 2) * np.where(gs, 1.0, 0.6) * vis * st
        cov = np.maximum(cov_o[:, None], cov_i)
        mark = (self.w[:, None] * (1 - cov)).sum(0)
        lane = np.zeros(K)
        if self.ball is not None:
            gx = -self.L / 2
            bx, by = self.ball
            lx, ly = gx - bx, -by
            l2 = lx * lx + ly * ly + 1e-9
            def block(px, py, seeb):
                tt = np.clip(((px - bx) * lx + (py - by) * ly) / l2, 0, 1)
                dist = np.hypot(bx + tt * lx - px, by + tt * ly - py)
                return np.exp(-(dist / 1.0) ** 2) * (tt > 0.02) * np.where(seeb, 1.0, 0.85)
            bo = block(dx[oth], dy[oth], sbo).max() if len(oth) else 0.0
            lane = _value(bx, by, self.L) * (1 - np.maximum(bo, block(vx, vy, sb)))
        # --- space: others' arrival grid min, candidate arrival per K
        if len(oth):
            ddo = np.hypot(self.gxx[..., None] - dx[oth], self.gyy[..., None] - dy[oth])
            tdo = (0.7 + np.maximum(-0.3, lag_all[oth] - self.lag_ref) + ddo / 7.0).min(-1)
        else:
            tdo = np.full(self.gxx.shape, np.inf)
        tdo = np.minimum(tdo, self.t_gk)
        ddi = np.hypot(self.gxx[..., None] - vx, self.gyy[..., None] - vy)
        tdi = 0.7 + np.maximum(-0.3, lag_i - self.lag_ref) + ddi / 7.0
        td = np.minimum(tdo[..., None], tdi)
        p_att = 1 / (1 + np.exp(-(np.pi / np.sqrt(3) / 0.45) * (td - self.ta[..., None])))
        space = (p_att * self.gval[..., None]).sum((0, 1)) / self.gval.sum()
        reg = getattr(self, "reg", 1.0)
        tot = 1.0 * mark + 1.2 * lane + 2.0 * space
        if reg:
            Dx, Dy = self.D.x.to_numpy(), self.D.y.to_numpy()
            mo = float(((dx[oth] - Dx[oth]) ** 2 + (dy[oth] - Dy[oth]) ** 2).sum())
            to = float((np.degrees(_wrap(th[oth] - self.th0[oth])) ** 2 + np.degrees(_wrap(hh[oth] - self.hh0[oth])) ** 2).sum())
            mi = (vx - Dx[i]) ** 2 + (vy - Dy[i]) ** 2
            ti = np.degrees(_wrap(vth - self.th0[i])) ** 2 + np.degrees(_wrap(vhh - self.hh0[i])) ** 2
            tot = tot + reg * (0.002 * (mo + mi) + 2e-6 * (to + ti))
        return tot

    def metrics(self, dx, dy, th=None, hh=None, ld=None) -> dict:
        p = self.score(dx, dy, th, hh, ld, parts=True)
        ax, ay = self.A.x.to_numpy(), self.A.y.to_numpy()
        d = np.hypot(ax[:, None] - dx[None], ay[:, None] - dy[None]).min(1) if len(dx) else np.full(len(ax), 99)
        zone = (ax + self.L / 2) <= 35
        p["free_attackers"] = int(((d > 5) & zone).sum())
        if self.focus is not None:
            p["focus_marker_dist"] = round(float(d[self.focus]), 1)
        if self.carrier is not None:
            p["carrier_pressure_dist"] = round(float(d[self.carrier]), 1)
        th = self.th0 if th is None else th
        hh = self.hh0 if hh is None else hh
        p["seeing_ball_and_man"] = int(sum(self.diag_one(i, dx, dy, th, hh)["sees_ball_and_man"]
                                           for i in range(len(dx)) if self.known[i]))
        p["square_defenders"] = int(sum(shape_of(t) == "square" for t, k in zip(th, self.known) if k))
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}

    def diag_one(self, i, dx, dy, th, hh) -> dict:
        """What defender i sees and how he stands (his 'man' = nearest attacker)."""
        ax, ay = self.A.x.to_numpy(), self.A.y.to_numpy()
        j = int(np.argmin(np.hypot(ax - dx[i], ay - dy[i]))) if len(ax) else None
        sees_ball = True
        if self.ball is not None:
            sees_ball = bool(abs(_wrap(np.arctan2(self.ball[1] - dy[i], self.ball[0] - dx[i]) - hh[i])) <= FOV)
        sees_man = bool(j is not None and abs(_wrap(np.arctan2(ay[j] - dy[i], ax[j] - dx[i]) - hh[i])) <= FOV)
        shp = shape_of(th[i])
        return {"man": self.A.name.iloc[j] if j is not None else None, "sees_ball": sees_ball, "sees_man": sees_man,
                "sees_ball_and_man": sees_ball and sees_man, "shape": shp,
                "reaction_s": self.cal["lag"][shp] if self.known[i] else None}

    def seen_with_ball(self, i, dx, dy, hh, radius=15.0) -> list[str]:
        """Attackers within `radius` that defender i sees together with the ball (nearest first)."""
        if self.ball is not None and abs(_wrap(np.arctan2(self.ball[1] - dy[i], self.ball[0] - dx[i]) - hh[i])) > FOV:
            return []
        ax, ay = self.A.x.to_numpy(), self.A.y.to_numpy()
        dist = np.hypot(ax - dx[i], ay - dy[i])
        ok = (dist <= radius) & (np.abs(_wrap(np.arctan2(ay - dy[i], ax - dx[i]) - hh[i])) <= FOV)
        return [self.A.name.iloc[j] for j in np.argsort(dist) if ok[j]]

    # ---------------------------------------------------------- search
    def optimise(self, sweeps: int = 3, which=None, positions=True, posture=True):
        """Coordinate descent, defender by defender; every candidate set is scored in one vectorised call."""
        dx, dy = self.D.x.to_numpy().copy(), self.D.y.to_numpy().copy()
        th, hh, ld = self.th0.copy(), self.hh0.copy(), self.ld0.copy()
        idx = range(len(dx)) if which is None else which
        gr, ga = getattr(self, "grid", (GRID_R, GRID_A))
        rr = np.linspace(0, self.R, gr)
        aa = np.linspace(0, 2 * np.pi, ga, endpoint=False)
        cand = np.array([(0.0, 0.0)] + [(r * np.cos(a), r * np.sin(a)) for r in rr[1:] for a in aa])
        best = self.score(dx, dy, th, hh, ld)
        for _ in range(sweeps):
            improved = False
            for i in idx:
                if positions:
                    P = np.vstack([cand + [self.cx[i], self.cy[i]], [[dx[i], dy[i]]]])
                    P = P[(np.abs(P[:, 0]) <= self.L / 2) & (np.abs(P[:, 1]) <= self.W / 2)]
                    if len(P):
                        K = len(P)
                        sc = self.score_variants(i, dx, dy, th, hh, ld, P[:, 0], P[:, 1], np.full(K, th[i]), np.full(K, hh[i]), np.full(K, ld[i]))
                        b = int(np.argmin(sc))
                        if sc[b] < best - 1e-6:
                            best, improved = float(sc[b]), True
                            dx[i], dy[i] = P[b]
                if posture and self.known[i]:
                    ax, ay = self.A.x.to_numpy(), self.A.y.to_numpy()
                    near = np.argsort(np.hypot(ax - dx[i], ay - dy[i]))[:2]
                    tg = [np.arctan2(ay[j] - dy[i], ax[j] - dx[i]) for j in near]
                    if self.ball is not None:
                        b_ = np.arctan2(self.ball[1] - dy[i], self.ball[0] - dx[i])
                        tg += [b_] + [_wrap(b_ + _wrap(a_ - b_) / 2) for a_ in tg]
                    combos = []
                    for tt in np.concatenate([getattr(self, 'torso_grid', TORSO_GRID), [self.th0[i]], tg]):
                        for hd in [_wrap(tt + o) for o in HEAD_OFF] + [g for g in tg if abs(_wrap(g - tt)) <= NECK]:
                            for l_ in ((ld[i], True) if self.ld_known[i] else (ld[i],)):
                                combos.append((tt, hd, float(l_)))
                    C = np.array(combos)
                    K = len(C)
                    sc = self.score_variants(i, dx, dy, th, hh, ld, np.full(K, dx[i]), np.full(K, dy[i]), C[:, 0], C[:, 1], C[:, 2] > 0.5)
                    b = int(np.argmin(sc))
                    if sc[b] < best - 1e-6:
                        best, improved = float(sc[b]), True
                        th[i], hh[i], ld[i] = C[b, 0], C[b, 1], bool(C[b, 2] > 0.5)
            if not improved:
                break
        return dx, dy, th, hh, ld


# ------------------------------------------------------------------ public API
def _raw_deg(a, s):
    return round(float(np.degrees(_wrap(a + (0 if s > 0 else np.pi)))), 1)


def _turn_words(da_deg: float) -> str:
    return f"{abs(da_deg):.0f}° {'left' if da_deg > 0 else 'right'}"


def _explain(sc, i, x0, y0, x1, y1, t0, h0, t1, h1, l0, l1) -> list[str]:
    why = []
    ax, ay = sc.A.x.to_numpy(), sc.A.y.to_numpy()
    dist = float(np.hypot(x1[i] - x0[i], y1[i] - y0[i]))
    if dist >= 0.5:
        j = int(np.argmin(np.hypot(ax - x1[i], ay - y1[i])))
        if np.hypot(ax[j] - x1[i], ay[j] - y1[i]) < 3:
            why.append(f"tighter on {sc.A.name.iloc[j]}" + (" (goal-side)" if x1[i] < ax[j] else ""))
        if x1[i] < x0[i] - 1:
            why.append(f"{x0[i] - x1[i]:.1f} m deeper")
        elif x1[i] > x0[i] + 1:
            why.append(f"{x1[i] - x0[i]:.1f} m higher")
        if abs(y1[i] - y0[i]) > 1:
            why.append(f"{abs(y1[i] - y0[i]):.1f} m {'inside' if abs(y1[i]) < abs(y0[i]) else 'wider'}")
    if sc.known[i]:
        a, b = sc.diag_one(i, x0, y0, t0, h0), sc.diag_one(i, x1, y1, t1, h1)
        dt_ = float(np.degrees(_wrap(t1[i] - t0[i])))
        if a["shape"] != b["shape"] and a["reaction_s"] != b["reaction_s"]:
            why.append(f"open the body: {a['shape']} → {b['shape']} (reacts {a['reaction_s']:.2f} → {b['reaction_s']:.2f} s)")
        elif abs(dt_) >= 30:
            why.append(f"turn the body {_turn_words(dt_)}")
        dh = float(np.degrees(_wrap(h1[i] - h0[i])))
        if abs(dh) >= 20:
            new = [n for n in sc.seen_with_ball(i, x1, y1, h1) if n not in sc.seen_with_ball(i, x0, y0, h0)]
            why.append(f"head {_turn_words(dh)}" + (f": ball and {', '.join(new[:2])} in view" if new else ""))
        if l1[i] and not l0[i]:
            why.append("get low on the ball carrier")
    return why


def coach(match_id: int, frame: int, defending_team: str | None = None, focus_attacker: int | None = None,
          only: list[int] | None = None, react_s: float = REACT_S) -> dict:
    """Automatic coach. ``only`` = defender ids to coach (default: the whole back unit)."""
    sc = PostureScene(match_id, frame, defending_team, focus_attacker, react_s)
    if len(sc.D) < 5:
        return {"error": "not enough defenders tracked at this frame"}
    x0, y0, t0, h0, l0 = sc.D.x.to_numpy(), sc.D.y.to_numpy(), sc.th0, sc.hh0, sc.ld0
    which = None
    if only:
        which = [int(i) for i in np.flatnonzero(sc.D.player_id.isin([int(o) for o in only]).to_numpy())]
        if not which:
            return {"error": "that player is not an outfield defender at this frame"}
    before = sc.metrics(x0, y0)
    # decomposition: posture only / position only / both
    _, _, tp, hp, lp = sc.optimise(which=which, positions=False, posture=True)
    xq, yq, *_ = sc.optimise(which=which, positions=True, posture=False)
    x1, y1, t1, h1, l1 = sc.optimise(which=which, positions=True, posture=True)
    # keep a posture correction only if it is worth teaching (≥ 0.5% of the danger on its own)
    for i in range(len(x1)):
        if not sc.known[i]:
            continue
        base = sc.score(x1, y1, t1, h1, l1)
        tt, hh_, ll = t1.copy(), h1.copy(), l1.copy()
        tt[i], hh_[i], ll[i] = t0[i], h0[i], l0[i]
        if sc.score(x1, y1, tt, hh_, ll) - base < 0.005 * before["total"]:
            t1, h1, l1 = tt, hh_, ll
    after = sc.metrics(x1, y1, t1, h1, l1)
    tot0 = before["total"] or 1e-9
    red = lambda s: round(float((tot0 - s) / tot0), 3)  # noqa: E731
    parts = {"posture_only": red(sc.score(x0, y0, tp, hp, lp)), "position_only": red(sc.score(xq, yq)),
             "both": red(after["total"])}
    moves = []
    for i, r in sc.D.iterrows():
        why = _explain(sc, i, x0, y0, x1, y1, t0, h0, t1, h1, l0, l1)
        if not why:
            continue
        xs, ys, ts, hs, ls = x0.copy(), y0.copy(), t0.copy(), h0.copy(), l0.copy()
        xs[i], ys[i], ts[i], hs[i], ls[i] = x1[i], y1[i], t1[i], h1[i], l1[i]
        gain = before["total"] - sc.score(xs, ys, ts, hs, ls)
        xs2, ys2 = x0.copy(), y0.copy(); xs2[i], ys2[i] = x1[i], y1[i]
        g_pos = before["total"] - sc.score(xs2, ys2)
        d0, d1 = sc.diag_one(i, x0, y0, t0, h0), sc.diag_one(i, x1, y1, t1, h1)
        moves.append({
            "player_id": int(r.player_id), "defender": r["name"],
            "move_m": round(float(np.hypot(x1[i] - x0[i], y1[i] - y0[i])), 1),
            "gain": round(float(gain), 3), "gain_position": round(float(g_pos), 3),
            "gain_posture": round(float(gain - g_pos), 3), "why": ", ".join(why), "pose_known": bool(sc.known[i]),
            "from": [round(float(x0[i] * sc.s), 2), round(float(y0[i] * sc.s), 2)],
            "to": [round(float(x1[i] * sc.s), 2), round(float(y1[i] * sc.s), 2)],
            "torso_from": _raw_deg(t0[i], sc.s), "torso_to": _raw_deg(t1[i], sc.s),
            "head_from": _raw_deg(h0[i], sc.s), "head_to": _raw_deg(h1[i], sc.s),
            "loaded_from": bool(l0[i]) if sc.ld_known[i] else None, "loaded_to": bool(l1[i]) if sc.ld_known[i] else None,
            "before": d0, "after": d1})
    moves.sort(key=lambda m: -m["gain"])
    meta = sc.meta
    return {
        "match_id": match_id, "frame": sc.frame, "defending_team": meta.team_name(sc.dteam),
        "mode": "posture+position", "coached": [sc.D.name.iloc[i] for i in which] if which else "all outfield defenders",
        "reaction_window_s": react_s, "max_move_m": round(sc.R, 1),
        "before": before, "after": after, "danger_reduction": parts["both"], "reduction_parts": parts,
        "moves": moves, "key_fix": moves[0] if moves else None,
        "focus_attacker": sc.A.name.iloc[sc.focus] if sc.focus is not None else None,
        "focus_id": int(sc.A.player_id.iloc[sc.focus]) if sc.focus is not None else None,
        "reach": [{"player_id": int(pid), "center": [round(float(cx * sc.s), 2), round(float(cy * sc.s), 2)],
                   "radius": round(sc.R, 1)} for pid, cx, cy in zip(sc.D.player_id, sc.cx, sc.cy)],
        "calibration": {k: v for k, v in sc.cal.items()},
        "note": "Positions reachable within the reaction window; posture effects calibrated on the framework's "
                "evidence (vision, body shape → reaction, stance → regains). Attackers held fixed. A safer "
                "alternative to teach from, not a guaranteed outcome.",
    }


def coach_player(match_id: int, frame: int, player_id: int, focus_attacker: int | None = None) -> dict:
    return coach(match_id, frame, None, focus_attacker, only=[player_id])


def whatif(match_id: int, frame: int, edits: list[dict], focus_attacker: int | None = None,
           defending_team: str | None = None) -> dict:
    """Score an analyst-edited configuration. edits: [{player_id, x, y, torso, head, loaded}] in RAW
    pitch coordinates/degrees (any field optional). Returns danger vs. what happened + per-defender read-outs."""
    sc = PostureScene(match_id, frame, defending_team, focus_attacker)
    x0, y0 = sc.D.x.to_numpy(), sc.D.y.to_numpy()
    x1, y1, t1, h1, l1 = x0.copy(), y0.copy(), sc.th0.copy(), sc.hh0.copy(), sc.ld0.copy()
    rot = 0.0 if sc.s > 0 else np.pi
    ids = list(sc.D.player_id)
    warn = []
    for e in edits or []:
        pid = int(e.get("player_id", -1))
        if pid not in ids:
            continue
        i = ids.index(pid)
        if e.get("x") is not None and e.get("y") is not None:
            nx, ny = float(e["x"]) * sc.s, float(e["y"]) * sc.s
            dc = np.hypot(nx - sc.cx[i], ny - sc.cy[i])
            if dc > sc.R:                                       # clamp to what he could reach
                nx, ny = sc.cx[i] + (nx - sc.cx[i]) * sc.R / dc, sc.cy[i] + (ny - sc.cy[i]) * sc.R / dc
                warn.append(f"{sc.D.name.iloc[i]}: limited to his {sc.R:.0f} m reach")
            x1[i], y1[i] = nx, ny
        if e.get("torso") is not None:
            t1[i] = _wrap(np.radians(float(e["torso"])) + rot)
        if e.get("head") is not None:
            h1[i] = _wrap(np.radians(float(e["head"])) + rot)
        if abs(_wrap(h1[i] - t1[i])) > NECK:                    # anatomical neck limit
            h1[i] = _wrap(t1[i] + np.sign(_wrap(h1[i] - t1[i])) * NECK)
        if e.get("loaded") is not None:
            l1[i] = bool(e["loaded"])
    before, after = sc.metrics(x0, y0), sc.metrics(x1, y1, t1, h1, l1)
    red = (before["total"] - after["total"]) / before["total"] if before["total"] else 0.0
    players = []
    for i, pid in enumerate(ids):
        d = sc.diag_one(i, x1, y1, t1, h1)
        players.append({"player_id": int(pid), "defender": sc.D.name.iloc[i], "pose_known": bool(sc.known[i]),
                        "x": round(float(x1[i] * sc.s), 2), "y": round(float(y1[i] * sc.s), 2),
                        "torso": _raw_deg(t1[i], sc.s), "head": _raw_deg(h1[i], sc.s),
                        "loaded": bool(l1[i]) if sc.ld_known[i] else None, **d})
    return {"match_id": match_id, "frame": sc.frame, "defending_team": sc.meta.team_name(sc.dteam),
            "before": before, "after": after, "danger_reduction": round(float(red), 3), "players": players,
            "warnings": warn,
            "reach": [{"player_id": int(pid), "center": [round(float(cx * sc.s), 2), round(float(cy * sc.s), 2)],
                       "radius": round(sc.R, 1)} for pid, cx, cy in zip(sc.D.player_id, sc.cx, sc.cy)]}
