"""Counterfactual defending: "where should the defenders have been to stop this?"

At a chosen frame we search better positions for the defending outfield players, each
constrained to where he could physically have been: a disc around his position T seconds
earlier with radius v·T (default T = 1.5 s, v = 6 m/s). Attackers and ball are kept fixed.

Objective (lower = safer), all in the defending team's normalised frame (own goal at x = -L/2):

* marking danger   Σ_attackers value(attacker) · (1 − coverage)
                   value = threat of the location (distance/angle to goal); the ball carrier and the
                   focus attacker (e.g. the scorer) weigh more; coverage = best defender's
                   exp(−(d/2.5 m)²), discounted ×0.6 if he is not goal-side
* shot/pass lane   value(ball) · (1 − block of the ball→goal lane by the nearest defender)
* space behind     attack-controlled, threat-weighted space within 32 m of goal
                   (time-to-intercept pitch control, Spearman 2018 simplified)
* movement         small penalty per m² moved, so the answer changes as little as possible

Search: coordinate descent over a polar grid inside each defender's reachable disc (3 sweeps).
Inspired by positional-optimisation work in the first Analytics Cup (Shah, 2026) but implemented
independently. It shows *better-defended alternatives*, not a guaranteed save.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .data.load import load_meta, load_phases, load_tracking

V_MAX = 6.0
REACT_S = 1.5
GRID_R, GRID_A = 6, 16


@lru_cache(maxsize=2)
def _data(match_id: int):
    meta = load_meta(match_id)
    fr, xy = load_tracking(match_id)
    return meta, fr.set_index("frame"), xy.set_index("frame").sort_index(), load_phases(match_id)


def _value(x, y, L):
    """Threat of a location for the attack (own goal of the defence at x=-L/2): 0..1."""
    gx = -L / 2
    d = np.hypot(x - gx, y)
    ang = np.abs(np.arctan2(y, np.maximum(x - gx, 0.5)))
    return np.exp(-d / 16.0) * (1 - 0.45 * ang / (np.pi / 2))


class Scene:
    def __init__(self, match_id: int, frame: int, defending_team: str | None = None,
                 focus_attacker: int | None = None, react_s: float = REACT_S):
        meta, fr, xy, ph = _data(match_id)
        self.meta, self.L, self.W = meta, *meta.pitch
        frame = int(frame)
        if frame not in xy.index:
            frame = int(xy.index[np.argmin(np.abs(xy.index.to_numpy() - frame))])
        self.frame = frame
        per = int(fr.loc[frame, "period"])
        if defending_team is None:
            p = ph[(ph.frame_start <= frame) & (ph.frame_end >= frame)]
            if len(p):
                poss = int(p.team_in_possession_id.iloc[0])
            else:
                g = fr.loc[frame, "poss_group"]
                poss = meta.home_id if g == "home team" else meta.away_id
            dteam = meta.away_id if poss == meta.home_id else meta.home_id
        else:
            dteam = meta.home_id if meta.team_name(meta.home_id) == defending_team else meta.away_id
        self.dteam, self.period = dteam, per
        self.s = 1.0 if meta.defends_left(dteam, per) else -1.0
        pl = meta.players.set_index("player_id")
        snap = xy.loc[[frame]].reset_index()
        snap["team"] = snap.player_id.map(pl.team_id)
        snap["gk"] = snap.player_id.map(pl.is_gk).fillna(False).astype(bool)
        snap["name"] = snap.player_id.map(pl.name)
        snap["x"], snap["y"] = snap.x * self.s, snap.y * self.s
        snap["vx"], snap["vy"] = snap.vx * self.s, snap.vy * self.s
        D = snap[(snap.team == dteam) & ~snap.gk].reset_index(drop=True)
        self.GK = snap[(snap.team == dteam) & snap.gk]
        A = snap[(snap.team != dteam) & ~snap.gk].reset_index(drop=True)
        self.D, self.A = D, A
        # reachable discs: position react_s earlier (if tracked) with radius V_MAX * react_s
        f0 = frame - int(react_s * 10)
        prev = xy.loc[[f0]].set_index("player_id") if f0 in xy.index else None
        cx, cy = D.x.to_numpy().copy(), D.y.to_numpy().copy()
        if prev is not None:
            for i, pid in enumerate(D.player_id):
                if pid in prev.index:
                    cx[i], cy[i] = prev.loc[pid, "x"] * self.s, prev.loc[pid, "y"] * self.s
        self.cx, self.cy, self.R = cx, cy, V_MAX * react_s
        b = fr.loc[frame]
        self.ball = np.array([float(b.ball_x) * self.s, float(b.ball_y) * self.s]) if pd.notna(b.ball_x) else None
        ax, ay = A.x.to_numpy(), A.y.to_numpy()
        w = _value(ax, ay, self.L)
        if self.ball is not None and len(A):
            carrier = int(np.argmin(np.hypot(ax - self.ball[0], ay - self.ball[1])))
            if np.hypot(ax[carrier] - self.ball[0], ay[carrier] - self.ball[1]) < 2.5:
                w[carrier] *= 1.6
            self.carrier = carrier
        else:
            self.carrier = None
        self.focus = None
        if focus_attacker is not None and focus_attacker in set(A.player_id):
            self.focus = int(np.flatnonzero(A.player_id.to_numpy() == focus_attacker)[0])
            w[self.focus] *= 2.0
        self.w = w
        # pitch-control grid in the danger zone (within 32 m of goal)
        gx = -self.L / 2
        xs = np.linspace(gx + 1, gx + 32, 24)
        ys = np.linspace(-26, 26, 30)
        self.gxx, self.gyy = np.meshgrid(xs, ys)
        self.gval = _value(self.gxx, self.gyy, self.L)
        self.ta = self._arrival(ax + A.vx.to_numpy() * 0.5, ay + A.vy.to_numpy() * 0.5)
        if len(self.GK):
            self.t_gk = self._arrival(self.GK.x.to_numpy(), self.GK.y.to_numpy())
        else:
            self.t_gk = np.full(self.gxx.shape, np.inf)

    def _arrival(self, px, py):
        if len(px) == 0:
            return np.full(self.gxx.shape, np.inf)
        d = np.hypot(self.gxx[..., None] - px, self.gyy[..., None] - py)
        return (0.7 + d / 7.0).min(-1)

    def score(self, dx: np.ndarray, dy: np.ndarray, parts: bool = False):
        ax, ay = self.A.x.to_numpy(), self.A.y.to_numpy()
        d = np.hypot(ax[:, None] - dx[None], ay[:, None] - dy[None])
        goal_side = dx[None] < ax[:, None] + 0.5
        cov = (np.exp(-(d / 2.5) ** 2) * np.where(goal_side, 1.0, 0.6)).max(1) if len(dx) else np.zeros(len(ax))
        mark = float((self.w * (1 - cov)).sum())
        lane = 0.0
        if self.ball is not None:
            gx = -self.L / 2
            bx, by = self.ball
            vx, vy = gx - bx, -by
            l2 = vx * vx + vy * vy + 1e-9
            t = np.clip(((dx - bx) * vx + (dy - by) * vy) / l2, 0, 1)
            dist = np.hypot(bx + t * vx - dx, by + t * vy - dy)
            block = np.max(np.exp(-(dist / 1.0) ** 2) * (t > 0.02)) if len(dx) else 0.0
            lane = float(_value(bx, by, self.L) * (1 - block))
        td = np.minimum(self._arrival(dx, dy), self.t_gk)
        p_att = 1 / (1 + np.exp(-(np.pi / np.sqrt(3) / 0.45) * (td - self.ta)))
        space = float((p_att * self.gval).sum() / self.gval.sum())
        move = float(((dx - self.D.x.to_numpy()) ** 2 + (dy - self.D.y.to_numpy()) ** 2).sum())
        total = 1.0 * mark + 1.2 * lane + 2.0 * space + 0.002 * move
        if parts:
            area = float((p_att > 0.5).sum() * (31 / 23) * (52 / 29))
            return {"total": total, "marking_danger": mark, "lane_open": lane, "space_share": space,
                    "attack_space_m2": area}
        return total

    def metrics(self, dx, dy) -> dict:
        p = self.score(dx, dy, parts=True)
        ax, ay = self.A.x.to_numpy(), self.A.y.to_numpy()
        d = np.hypot(ax[:, None] - dx[None], ay[:, None] - dy[None]).min(1) if len(dx) else np.full(len(ax), 99)
        zone = (ax + self.L / 2) <= 35
        p["free_attackers"] = int(((d > 5) & zone).sum())
        if self.focus is not None:
            p["focus_marker_dist"] = round(float(d[self.focus]), 1)
        if self.carrier is not None:
            p["carrier_pressure_dist"] = round(float(d[self.carrier]), 1)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}

    def optimise(self, sweeps: int = 3):
        dx, dy = self.D.x.to_numpy().copy(), self.D.y.to_numpy().copy()
        rr = np.linspace(0, self.R, GRID_R)
        aa = np.linspace(0, 2 * np.pi, GRID_A, endpoint=False)
        cand = np.array([(0.0, 0.0)] + [(r * np.cos(a), r * np.sin(a)) for r in rr[1:] for a in aa])
        best = self.score(dx, dy)
        for _ in range(sweeps):
            improved = False
            for i in range(len(dx)):
                ox, oy = dx[i], dy[i]
                bi = (ox, oy)
                for cxo, cyo in np.vstack([cand + [self.cx[i], self.cy[i]], [[ox, oy]]]):
                    if abs(cxo) > self.L / 2 or abs(cyo) > self.W / 2:
                        continue
                    dx[i], dy[i] = cxo, cyo
                    sc = self.score(dx, dy)
                    if sc < best - 1e-6:
                        best, bi, improved = sc, (cxo, cyo), True
                dx[i], dy[i] = bi
            if not improved:
                break
        return dx, dy


def simulate(match_id: int, frame: int, defending_team: str | None = None, focus_attacker: int | None = None,
             react_s: float = REACT_S, posture: bool = True, only: list[int] | None = None) -> dict:
    """Better reachable defending at a frame. posture=True (default) also corrects body orientation,
    head and stance (dbl.coach); posture=False is the position-only model. `only` = coach these defenders."""
    if posture or only:
        from .coach import coach
        return coach(match_id, frame, defending_team, focus_attacker, only=only, react_s=react_s)
    sc = Scene(match_id, frame, defending_team, focus_attacker, react_s)
    if len(sc.D) < 5:
        return {"error": "not enough defenders tracked at this frame"}
    x0, y0 = sc.D.x.to_numpy(), sc.D.y.to_numpy()
    before = sc.metrics(x0, y0)
    x1, y1 = sc.optimise()
    after = sc.metrics(x1, y1)
    moves = []
    for i, r in sc.D.iterrows():
        dist = float(np.hypot(x1[i] - x0[i], y1[i] - y0[i]))
        if dist < 0.5:
            continue
        # marginal value: only this defender moved
        xs, ys = x0.copy(), y0.copy()
        xs[i], ys[i] = x1[i], y1[i]
        gain = before["total"] - sc.score(xs, ys)
        ax, ay = sc.A.x.to_numpy(), sc.A.y.to_numpy()
        j = int(np.argmin(np.hypot(ax - x1[i], ay - y1[i])))
        toward = sc.A.name.iloc[j]
        why = []
        if np.hypot(ax[j] - x1[i], ay[j] - y1[i]) < 3:
            why.append(f"tighter on {toward}" + (" (goal-side)" if x1[i] < ax[j] else ""))
        if x1[i] < x0[i] - 1:
            why.append(f"{x0[i] - x1[i]:.1f} m deeper")
        elif x1[i] > x0[i] + 1:
            why.append(f"{x1[i] - x0[i]:.1f} m higher")
        if abs(y1[i] - y0[i]) > 1:
            why.append(f"{abs(y1[i] - y0[i]):.1f} m {'inside' if abs(y1[i]) < abs(y0[i]) else 'wider'}")
        moves.append({"player_id": int(r.player_id), "defender": r["name"], "move_m": round(dist, 1),
                      "gain": round(float(gain), 3), "why": ", ".join(why) or "covers space/lane",
                      # raw pitch coordinates for drawing
                      "from": [round(float(x0[i] * sc.s), 2), round(float(y0[i] * sc.s), 2)],
                      "to": [round(float(x1[i] * sc.s), 2), round(float(y1[i] * sc.s), 2)]})
    moves.sort(key=lambda m: -m["gain"])
    red = (before["total"] - after["total"]) / before["total"] if before["total"] else 0
    meta = sc.meta
    return {
        "match_id": match_id, "frame": sc.frame, "defending_team": meta.team_name(sc.dteam),
        "reaction_window_s": react_s, "max_move_m": round(sc.R, 1),
        "before": before, "after": after, "danger_reduction": round(float(red), 3),
        "moves": moves,
        "key_fix": moves[0] if moves else None,
        "focus_attacker": sc.A.name.iloc[sc.focus] if sc.focus is not None else None,
        "note": "Counterfactual positions each defender could have reached within the reaction window; attackers held "
                "fixed. It shows a safer configuration, not a guaranteed outcome.",
    }


def simulate_goal(match_id: int, goal_number: int, react_s: float = REACT_S, posture: bool = True,
                  only: list[int] | None = None) -> dict:
    """Simulate the conceding team's best reachable shape at the assist (or 1 s before the shot)."""
    from .goals import find_goals
    g = find_goals(match_id)[goal_number - 1]
    f = g["assist_frame"] or (g["frame"] - 10)
    res = simulate(match_id, f, g["conceding"], g.get("scorer_id"), react_s, posture, only)
    res["goal"] = {k: g[k] for k in ("goal_number", "time", "scorer", "assist", "team", "conceding", "frame")}
    res["moment"] = "assist pass" if g["assist_frame"] else "1 s before the shot"
    return res
