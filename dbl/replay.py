"""Better-defence replay: re-simulate the MOVEMENT of every defender for up to 5 s.

The attack (all attackers and the ball) replays exactly as it happened. The defending outfield players start where
they really were, with their real velocity, and from then on are steered every 0.2 s to reduce danger, under
physical limits:

* speed ≤ 7 m/s, acceleration ≤ 5 m/s² (so a direction change takes time, as on the pitch);
* body turns ≤ 180°/s, head ≤ 360°/s and ≤ 75° from the body; stance can change at any step.

At each step the controller picks, defender by defender (2 sweeps of coordinate descent), the velocity change and
the posture that minimise the posture-aware danger (``dbl.coach.PostureScene``: marking with vision / body shape /
stance effects, shot lane, attack-controlled space near goal) now and 0.6 s ahead along the real attack — a
well-organised defence reading the play, not a prediction of what these players would have done.

Because the attack is frozen to reality, the replay stops being "what would have happened" as soon as a defender
reaches the ball: those moments are reported as ``ball_reachable`` events (block / interception chances).
"""
from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np

from .coach import PostureScene, NECK, _wrap, shape_of, _raw_deg
from .simulate import _value

DT_FRAMES = 2            # 0.2 s per step
V_MAX = 7.0
A_MAX = 5.0
TURN_BODY = np.radians(180)   # per second
TURN_HEAD = np.radians(360)
LOOK = 3                 # look-ahead steps (0.6 s)
REACT_STEPS = 2          # decisions use what was visible 0.4 s earlier (≈ measured reaction of an open defender)
MAX_S = 5.0
SMOOTH = 0.005           # cost per (m/s)² of velocity change: players do not change direction for marginal gains


def _cands(v, dt, vmax=V_MAX, torso=None):
    """Velocity candidates reachable in one step (|Δv| ≤ a·dt). Speed is capped by the player's own top speed and
    by the running direction relative to his body: forwards 100 %, sideways 70 %, backwards (back-pedal) 50 %."""
    dv = A_MAX * dt
    out = [np.zeros(2)]
    for r in (0.5 * dv, dv):
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            out.append(np.array([r * np.cos(a), r * np.sin(a)]))
    c = v[None] + np.array(out)
    sp = np.hypot(c[:, 0], c[:, 1])
    cap = np.full(len(c), vmax)
    if torso is not None:
        rel = np.abs(_wrap(np.arctan2(c[:, 1], c[:, 0]) - torso))
        cap = vmax * np.where(rel <= np.radians(45), 1.0, np.where(rel <= np.radians(120), 0.7, 0.5))
    # the direction cap can never brake harder than the acceleration limit (no instant stops)
    cap = np.maximum(cap, min(float(np.hypot(*v)), vmax) - dv)
    k = sp > cap
    c[k] *= (cap[k] / sp[k])[:, None]
    return c


@lru_cache(maxsize=4)
def _top_speeds(match_id: int) -> dict:
    """Each player's own top speed in this match (99.5th percentile, 5.5–8.5 m/s)."""
    from .config import BUILD_DIR
    import pandas as pd
    xy = pd.read_parquet(BUILD_DIR / str(match_id) / "players_xy.parquet", columns=["player_id", "speed"])
    q = xy.groupby("player_id").speed.quantile(0.995)
    return {int(k): float(np.clip(v, 5.5, 8.5)) for k, v in q.items()}


class _Frame:
    """One scene + mapping between the simulated defender list and this scene's defender order."""
    def __init__(self, sc: PostureScene, ids: list[int]):
        self.sc = sc
        sc.reg = 0.0
        self.order = list(sc.D.player_id)
        self.map = [ids.index(p) if p in ids else -1 for p in self.order]
        self.ax, self.ay = sc.D.x.to_numpy(), sc.D.y.to_numpy()
        self.known_real = sc.known.copy()
        self.pos_of = {j: k for k, j in enumerate(self.map) if j >= 0}

    def variants(self, j, X, Y, TH, HH, LD, vx, vy, vth, vhh, vld):
        """K variants of simulated defender j scored at once (others as given)."""
        k = self.pos_of.get(j)
        if k is None:
            return np.zeros(len(vx))
        self.sc.known = np.ones(len(self.order), bool)
        try:
            return self.sc.score_variants(k, *self.arrays(X, Y, TH, HH, LD), vx, vy, vth, vhh, vld)
        finally:
            self.sc.known = self.known_real

    def arrays(self, X, Y, TH, HH, LD):
        n = len(self.order)
        x, y = self.ax.copy(), self.ay.copy()
        th, hh, ld = self.sc.th0.copy(), self.sc.hh0.copy(), self.sc.ld0.copy()
        for k, j in enumerate(self.map):
            if j >= 0:
                x[k], y[k], th[k], hh[k], ld[k] = X[j], Y[j], TH[j], HH[j], LD[j]
        return x, y, th, hh, ld

    def score_sim(self, X, Y, TH, HH, LD):
        self.sc.known = np.ones(len(self.order), bool)   # the simulated players' posture is chosen, so it counts
        try:
            return self.sc.score(*self.arrays(X, Y, TH, HH, LD))
        finally:
            self.sc.known = self.known_real

    def metrics_sim(self, X, Y, TH, HH, LD):
        self.sc.known = np.ones(len(self.order), bool)
        try:
            return self.sc.metrics(*self.arrays(X, Y, TH, HH, LD))
        finally:
            self.sc.known = self.known_real

    def metrics_real(self):
        return self.sc.metrics(self.ax, self.ay)


def _predicted(sc: PostureScene, ahead: float, ball_v) -> PostureScene:
    """The scene as a defender could PREDICT it `ahead` seconds later from what he sees now: attackers and ball
    continue at their current velocity (no knowledge of the real future)."""
    if ahead <= 0:
        return sc
    p = copy.copy(sc)
    A = sc.A.copy()
    ox, oy = A.x.to_numpy(), A.y.to_numpy()
    A["x"], A["y"] = ox + A.vx.to_numpy() * ahead, oy + A.vy.to_numpy() * ahead
    p.A = A
    nx, ny = A.x.to_numpy(), A.y.to_numpy()
    p.w = sc.w * _value(nx, ny, sc.L) / np.maximum(_value(ox, oy, sc.L), 1e-6)
    p.ta = sc._arrival(nx + A.vx.to_numpy() * 0.5, ny + A.vy.to_numpy() * 0.5)
    if sc.ball is not None and ball_v is not None:
        p.ball = sc.ball + ball_v * ahead
    return p


def _structure(i, px, py, X, Y, line, order, real_xy):
    """Football structure, as a penalty on defender i's candidate positions:
    * the back line stays a line (≤ 2.5 m from the others' median depth) and keeps its left-to-right order;
    * everybody stays within ~8 m of his real zone at that instant (roles are kept: a centre-back does not become a
      winger)."""
    pen = np.zeros(len(px))
    if i in line:
        others = [j for j in line if j != i]
        if others:
            med = float(np.median(X[others]))
            pen += 0.02 * np.maximum(0.0, np.abs(px - med) - 2.5) ** 2
        r = order.index(i)
        if r > 0:
            pen += 0.05 * np.maximum(0.0, Y[order[r - 1]] + 1.0 - py) ** 2
        if r < len(order) - 1:
            pen += 0.05 * np.maximum(0.0, py - (Y[order[r + 1]] - 1.0)) ** 2
    if real_xy is not None:
        pen += 0.003 * np.maximum(0.0, np.hypot(px - real_xy[0], py - real_xy[1]) - 8.0) ** 2
    return pen


def _posture_targets(sc: PostureScene, x, y):
    ax, ay = sc.A.x.to_numpy(), sc.A.y.to_numpy()
    near = np.argsort(np.hypot(ax - x, ay - y))[:2]
    tg = [np.arctan2(ay[j] - y, ax[j] - x) for j in near]
    if sc.ball is not None:
        b = np.arctan2(sc.ball[1] - y, sc.ball[0] - x)
        tg += [b] + [_wrap(b + _wrap(a - b) / 2) for a in tg]
    return tg


def _toward(a, target, maxd):
    d = _wrap(target - a)
    return _wrap(a + np.clip(d, -maxd, maxd))


@lru_cache(maxsize=16)
def replay(match_id: int, start: int, seconds: float = MAX_S, defending_team: str | None = None,
           focus_attacker: int | None = None, clairvoyant: bool = False) -> dict:
    """clairvoyant=False (default): each decision only uses what was visible REACT_STEPS earlier, extrapolated.
    clairvoyant=True: decisions see the real future of the attack (upper bound, not realistic)."""
    seconds = float(min(max(seconds, 1.0), MAX_S))
    n = int(round(seconds * 10 / DT_FRAMES))
    sc0 = PostureScene(match_id, start, defending_team, focus_attacker)
    team = sc0.meta.team_name(sc0.dteam)
    period = sc0.period
    ids = [int(p) for p in sc0.D.player_id]
    names = dict(zip(sc0.D.player_id, sc0.D.name))
    frames = []
    for k in range(n + 1):
        sc = PostureScene(match_id, sc0.frame + k * DT_FRAMES, team, focus_attacker)
        if sc.period != period:
            break
        frames.append(_Frame(sc, ids))
    n = len(frames) - 1
    dt = DT_FRAMES / 10
    X, Y = sc0.D.x.to_numpy().copy(), sc0.D.y.to_numpy().copy()
    V = np.stack([sc0.D.vx.to_numpy(), sc0.D.vy.to_numpy()], 1).astype(float)
    TH, HH, LD = sc0.th0.copy(), sc0.hh0.copy(), sc0.ld0.copy()
    s = sc0.s
    tops = _top_speeds(match_id)
    vmax = np.array([tops.get(p, V_MAX) for p in ids])
    grp = sc0.meta.players.set_index("player_id").position_group.to_dict()
    line = [i for i, p in enumerate(ids) if str(grp.get(p, "")) in ("Central Defender", "Full Back")]
    if len(line) < 3:
        line = list(np.argsort(X)[:4])
    line_order = sorted(line, key=lambda i: Y[i])
    out_frames, danger_real, danger_sim, events = [], [], [], []
    reached = set()

    def snapshot(fr, k):
        dr, ds = fr.metrics_real(), fr.metrics_sim(X, Y, TH, HH, LD)
        danger_real.append(dr["total"]); danger_sim.append(ds["total"])
        out_frames.append({
            "frame": int(fr.sc.frame), "danger_real": round(dr["total"], 3), "danger_sim": round(ds["total"], 3),
            "free_real": dr["free_attackers"], "free_sim": ds["free_attackers"],
            "defenders": [{"player_id": pid, "x": round(float(X[i] * s), 2), "y": round(float(Y[i] * s), 2),
                           "torso": _raw_deg(TH[i], s), "head": _raw_deg(HH[i], s), "loaded": bool(LD[i])}
                          for i, pid in enumerate(ids)]})
        ball = fr.sc.ball
        if ball is not None and k > 0:
            from .simulate import _data
            fz = _data(match_id)[1].loc[fr.sc.frame, "ball_z"]
            low = not np.isfinite(fz) or fz < 1.8          # a ball above head height cannot be played
            d = np.hypot(X - ball[0], Y - ball[1])
            for i in (np.flatnonzero(d < 1.0) if low else []):
                if ids[i] not in reached:
                    reached.add(ids[i])
                    events.append({"frame": int(fr.sc.frame), "player_id": ids[i], "defender": names[ids[i]],
                                   "type": "ball_reachable", "dist_m": round(float(d[i]), 2)})

    snapshot(frames[0], 0)
    ball_v = [None] + [((frames[k].sc.ball - frames[k - 1].sc.ball) / dt)
                       if frames[k].sc.ball is not None and frames[k - 1].sc.ball is not None else None
                       for k in range(1, n + 1)]
    for k in range(1, n + 1):
        if clairvoyant:
            fr, fl = frames[k], frames[min(k + LOOK, n)]
        else:                                   # decide from what was visible REACT_STEPS ago, extrapolated
            ko = max(0, k - REACT_STEPS)
            gap = (k - ko) * dt
            fr = _Frame(_predicted(frames[ko].sc, gap, ball_v[ko]), ids)
            fl = _Frame(_predicted(frames[ko].sc, gap + LOOK * dt, ball_v[ko]), ids)
        look = LOOK * dt if not clairvoyant else (min(k + LOOK, n) - k) * dt
        real = frames[k]
        rx = {j: (real.ax[kk], real.ay[kk]) for j, kk in real.pos_of.items()}
        Xp, Yp = X.copy(), Y.copy()
        Vn = V.copy()
        X, Y = Xp + Vn[:, 0] * dt, Yp + Vn[:, 1] * dt
        for _ in range(2):
            for i in range(len(ids)):
                c = _cands(V[i], dt, vmax[i], TH[i])
                px, py = Xp[i] + c[:, 0] * dt, Yp[i] + c[:, 1] * dt
                # stay on the pitch: leaving it costs (a player running at a line must brake within his
                # acceleration limit); a hard clip 3 m outside remains only as a last resort
                hl, hw = frames[k].sc.L / 2, frames[k].sc.W / 2
                stop = np.hypot(c[:, 0], c[:, 1]) / (2 * A_MAX)            # where he would stop if braking now
                sx, sy = px + c[:, 0] * stop, py + c[:, 1] * stop
                out_ = np.maximum(0, np.abs(sx) - hl - 0.5) + np.maximum(0, np.abs(sy) - hw - 0.5)
                px = np.clip(px, -hl - 3, hl + 3)
                py = np.clip(py, -hw - 3, hw + 3)
                K = len(c)
                rep_ = lambda v: np.full(K, v)  # noqa: E731
                cost = fr.variants(i, X, Y, TH, HH, LD, px, py, rep_(TH[i]), rep_(HH[i]), rep_(LD[i]))
                if look > 0:
                    Xl, Yl = X + Vn[:, 0] * look, Y + Vn[:, 1] * look
                    cost = cost + 0.6 * fl.variants(i, Xl, Yl, TH, HH, LD, px + c[:, 0] * look, py + c[:, 1] * look,
                                                    rep_(TH[i]), rep_(HH[i]), rep_(LD[i]))
                cost = cost + _structure(i, px, py, X, Y, line, line_order, rx.get(i)) \
                    + SMOOTH * ((c - V[i]) ** 2).sum(1) + 5.0 * out_
                b_ = int(np.argmin(cost))
                Vn[i] = c[b_]; X[i], Y[i] = px[b_], py[b_]
        V = Vn
        # posture within turn-rate limits; a player running fast faces where he runs (no sideways sprint)
        for i in range(len(ids)):
            sp = float(np.hypot(*V[i])); vdir = float(np.arctan2(V[i][1], V[i][0]))
            tg = _posture_targets(fr.sc, X[i], Y[i])
            tors = [_toward(TH[i], g, TURN_BODY * dt) for g in tg + [TH[i], vdir]]
            if sp > 4.0:
                tors = [t_ for t_ in tors if abs(_wrap(t_ - vdir)) <= np.radians(60)] or [_toward(TH[i], vdir, TURN_BODY * dt)]
            combos = []
            for t_new in tors:
                for hg in tg + [t_new]:
                    h_new = _toward(HH[i], hg, TURN_HEAD * dt)
                    o = _wrap(h_new - t_new)
                    if abs(o) > NECK:
                        h_new = _wrap(t_new + np.sign(o) * NECK)
                    for l_ in {bool(LD[i]), True}:
                        combos.append((t_new, h_new, l_))
            cb = np.array([(a_, b2, float(c2)) for a_, b2, c2 in combos])
            K = len(cb)
            cost = fr.variants(i, X, Y, TH, HH, LD, np.full(K, X[i]), np.full(K, Y[i]), cb[:, 0], cb[:, 1], cb[:, 2] > 0.5)
            b_ = int(np.argmin(cost))
            TH[i], HH[i], LD[i] = cb[b_, 0], cb[b_, 1], bool(cb[b_, 2] > 0.5)
        snapshot(frames[k], k)

    dr, ds = np.array(danger_real), np.array(danger_sim)
    red = 1 - ds / np.maximum(dr, 1e-9)
    end_real, end_sim = frames[-1].metrics_real(), frames[-1].metrics_sim(X, Y, TH, HH, LD)
    # who defended most differently
    first, last = out_frames[0]["defenders"], out_frames[-1]["defenders"]
    diffs = []
    for i, pid in enumerate(ids):
        real_end = frames[-1].sc.D[frames[-1].sc.D.player_id == pid]
        if real_end.empty:
            continue
        rx, ry = float(real_end.x.iloc[0]), float(real_end.y.iloc[0])
        gap = float(np.hypot(X[i] - rx, Y[i] - ry))
        path_sim = sum(np.hypot(out_frames[k]["defenders"][i]["x"] - out_frames[k - 1]["defenders"][i]["x"],
                                out_frames[k]["defenders"][i]["y"] - out_frames[k - 1]["defenders"][i]["y"])
                       for k in range(1, len(out_frames)))
        ax, ay = frames[-1].sc.A.x.to_numpy(), frames[-1].sc.A.y.to_numpy()
        j = int(np.argmin(np.hypot(ax - X[i], ay - Y[i])))
        diffs.append({"player_id": pid, "defender": names[pid], "end_gap_m": round(gap, 1),
                      "ran_m": round(float(path_sim), 1),
                      "ends_on": frames[-1].sc.A.name.iloc[j], "ends_dist_m": round(float(np.hypot(ax[j] - X[i], ay[j] - Y[i])), 1),
                      "goal_side": bool(X[i] < ax[j]), "shape_end": shape_of(TH[i])})
    diffs.sort(key=lambda d: -d["end_gap_m"])
    return {
        "match_id": match_id, "defending_team": team, "start": int(frames[0].sc.frame), "end": int(frames[-1].sc.frame),
        "seconds": round(n * dt, 1), "step_s": dt, "frames": out_frames,
        "mean_reduction": round(float(red.mean()), 3), "end_reduction": round(float(red[-1]), 3),
        "peak_real": round(float(dr.max()), 3), "peak_sim": round(float(ds.max()), 3),
        "end_real": end_real, "end_sim": end_sim,
        "focus_attacker": sc0.A.name.iloc[sc0.focus] if sc0.focus is not None else None,
        "events": events, "differences": diffs,
        "decisions": "clairvoyant (sees the real future of the attack)" if clairvoyant else
                     f"realistic: reads the play {REACT_STEPS * dt:.1f} s late and extrapolates it",
        "structure": "back line kept as a line in its left-right order; each player within ~8 m of his real zone",
        "limits": {"speed_mps": V_MAX, "accel_note": "own top speed per player; sideways 70 %, back-pedal 50 %", "reaction_s": 0.0 if clairvoyant else round(REACT_STEPS * dt, 1), "accel_mps2": A_MAX, "body_turn_dps": 180, "head_turn_dps": 360, "lookahead_s": round(LOOK * dt, 1)},
        "note": "Attack and ball replay exactly as they happened; only the defenders' movement and posture are "
                "re-simulated (physical limits above, 0.6 s look-ahead). Once a defender can reach the ball the real "
                "play would have changed — see events.",
    }


def replay_goal(match_id: int, goal_number: int, seconds: float = MAX_S, clairvoyant: bool = False) -> dict:
    """Re-simulate the conceding defence over the `seconds` before the shot."""
    from .goals import find_goals
    g = find_goals(match_id)[goal_number - 1]
    start = int(g["frame"] - round(seconds * 10))
    res = dict(replay(match_id, start, seconds, g["conceding"], g.get("scorer_id"), clairvoyant))
    res["goal"] = {k: g[k] for k in ("goal_number", "time", "scorer", "assist", "team", "conceding", "frame")}
    return res
