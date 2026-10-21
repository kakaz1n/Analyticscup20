"""Defensive error patterns mined from the counterfactual coach, and training plans built from them.

1. **Moments.** Every dangerous situation the defence faced: opponent line-breaking passes, passes in possessions
   that led to a shot, dangerous off-ball runs, runs that got in behind, duels lost. Within 1.5 s they are merged.
   A **control** sample of ordinary out-of-possession frames is added, so a pattern can be checked for being
   specific to danger rather than just "always there".
2. **Coach every moment** (``dbl.coach.PostureScene``, position + posture, faster grid). Each correction is
   decomposed per defender into its position and posture gain, and tagged with a coaching *error type*
   (``TAGS``) from the geometry of the correction.
3. **Patterns** = error types per team (and per player) ranked by *avoidable danger* (the sum of the modelled danger
   the correction removes). The output also gives:
   - how often they occur;
   - the **lift** vs the control sample;
   - where (third, channel) and in which phase they happen;
   - examples to watch.
4. **Training plan** = the top patterns of a team mapped to drills (``DRILLS``). Each drill comes with coaching
   points in the framework's language and the KPI to track in the next match.

Everything is derived from the model. It is a way to prioritise what to train, not proof that a drill fixes it.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import BUILD_DIR, MATCHES
from .coach import PostureScene, _wrap, shape_of
from .data.load import load_meta

GRID_FAST = (4, 12)
TORSO_FAST = np.radians(np.arange(-180, 180, 45))

# where each drill is played: examples are taken there when the team has one (third from the attack's view)
DRILL_ZONE = {"tight_marking": "attacking_third", "goal_side": None, "press_ball": None, "line_drop": None,
              "line_step": None, "tuck_in": None, "widen": None, "cover_space": "attacking_third", "scan": None,
              "open_body": None, "get_low": None}

TAGS = {
    "goal_side": ("Lost goal-side position", "SHAPE/REACT", "position"),
    "tight_marking": ("Too loose on his man", "REACT", "position"),
    "press_ball": ("No pressure on the ball carrier", "STANCE/REACT", "position"),
    "line_drop": ("Back line too high: space in behind", "SHAPE", "position"),
    "line_step": ("Back line too deep: gap to midfield", "SHAPE", "position"),
    "tuck_in": ("Too wide: central gap open", "SHAPE", "position"),
    "widen": ("Too narrow: wide attacker free", "SHAPE", "position"),
    "cover_space": ("Not covering the dangerous space / lane", "SHAPE", "position"),
    "scan": ("Ball-watching: man not in view", "SEE", "posture"),
    "open_body": ("Square body to the runner", "SHAPE", "posture"),
    "get_low": ("Upright when engaging the carrier", "STANCE", "posture"),
}


# ------------------------------------------------------------------ candidate moments
def candidates(match_id: int, control_per_team: int = 60, seed: int = 7) -> pd.DataFrame:
    rows = []
    b = BUILD_DIR / str(match_id)
    p = pd.read_parquet(b / "passes.parquet")
    for col, kind in (("line_break", "line-breaking pass"), ("lead_to_shot", "pass in a shot possession")):
        s = p[p[col].fillna(False).astype(bool)]
        rows += [dict(frame=int(r.frame), team=r.defending_team, kind=kind, focus=int(r.attacker_id) if pd.notna(r.attacker_id) else None,
                      third=r.third, channel=r.channel, phase=r.phase_oop, minute=int(r.minute)) for r in s.itertuples()]
    r_ = pd.read_parquet(b / "runs.parquet")
    for col, kind in (("dangerous", "dangerous run"), ("beaten_in_behind", "run in behind")):
        s = r_[r_[col].fillna(False).astype(bool)]
        rows += [dict(frame=int(r.frame), team=r.defending_team, kind=kind, focus=int(r.attacker_id) if pd.notna(r.attacker_id) else None,
                      third=r.third, channel=r.channel, phase=r.phase_oop, minute=int(r.minute)) for r in s.itertuples()]
    d = pd.read_parquet(b / "duels.parquet")
    s = d[d.beaten.fillna(False).astype(bool)]
    rows += [dict(frame=int(r.frame), team=r.defending_team, kind="duel lost", focus=int(r.attacker_id) if pd.notna(r.attacker_id) else None,
                  third=r.third, channel=r.channel, phase=r.phase_oop, minute=int(r.minute)) for r in s.itertuples()]
    c = pd.DataFrame(rows).sort_values("frame")
    # merge moments of the same team within 1.5 s (keep the first; collect kinds)
    keep, last = [], {}
    for r in c.itertuples():
        if r.team in last and r.frame - last[r.team]["frame"] <= 15:
            last[r.team]["kind"] = ", ".join(sorted(set(last[r.team]["kind"].split(", ")) | {r.kind}))
            continue
        last[r.team] = r._asdict(); keep.append(last[r.team])
    c = pd.DataFrame(keep).drop(columns=["Index"], errors="ignore")
    c["control"] = False
    # control: ordinary out-of-possession frames of each team, ≥ 3 s away from any dangerous moment
    meta = load_meta(match_id)
    ph = pd.read_parquet(b / "frames.parquet", columns=["frame", "poss_group"]).dropna()
    rng = np.random.default_rng(seed)
    ctl = []
    for side, team_id in (("home team", meta.home_id), ("away team", meta.away_id)):
        team_def = meta.team_name(meta.away_id if team_id == meta.home_id else meta.home_id)
        fr = ph[ph.poss_group == side].frame.to_numpy()
        busy = c[c.team == team_def].frame.to_numpy()
        fr = fr[[np.min(np.abs(busy - f)) > 30 if len(busy) else True for f in fr]] if len(fr) else fr
        for f in rng.choice(fr, size=min(control_per_team, len(fr)), replace=False) if len(fr) else []:
            ctl.append(dict(frame=int(f), team=team_def, kind="control", focus=None, third=None, channel=None,
                            phase=None, minute=int(f // 600), control=True))
    return pd.concat([c, pd.DataFrame(ctl)], ignore_index=True)


# ------------------------------------------------------------------ coach one moment, tag its corrections
def coach_moment(match_id: int, frame: int, team: str, focus: int | None):
    sc = PostureScene(match_id, frame, team, focus)
    if len(sc.D) < 6 or len(sc.A) < 3:
        return None, []
    sc.grid, sc.torso_grid = GRID_FAST, TORSO_FAST
    x0, y0, t0, h0, l0 = sc.D.x.to_numpy(), sc.D.y.to_numpy(), sc.th0, sc.hh0, sc.ld0
    before = sc.score(x0, y0)
    x1, y1, t1, h1, l1 = sc.optimise(sweeps=2)
    after = sc.score(x1, y1, t1, h1, l1)
    ax, ay = sc.A.x.to_numpy(), sc.A.y.to_numpy()
    back4 = set(np.argsort(x0)[:4])
    eps = 0.01 * before
    rows = []
    for i in range(len(x0)):
        xs, ys, ts, hs, ls = x0.copy(), y0.copy(), t0.copy(), h0.copy(), l0.copy()
        xs[i], ys[i] = x1[i], y1[i]
        g_pos = before - sc.score(xs, ys)
        ts[i], hs[i], ls[i] = t1[i], h1[i], l1[i]
        g_all = before - sc.score(xs, ys, ts, hs, ls)
        g_post = g_all - g_pos
        tags_pos, tags_post = [], []
        moved = np.hypot(x1[i] - x0[i], y1[i] - y0[i])
        if moved >= 1.0 and g_pos > eps:
            j = int(np.argmin(np.hypot(ax - x1[i], ay - y1[i])))
            d0, d1 = np.hypot(ax[j] - x0[i], ay[j] - y0[i]), np.hypot(ax[j] - x1[i], ay[j] - y1[i])
            gs0, gs1 = x0[i] < ax[j], x1[i] < ax[j]
            if j == sc.carrier and d0 - d1 > 1.5:
                tags_pos.append("press_ball")
            elif gs1 and not gs0 and d1 < 4:
                tags_pos.append("goal_side")
            elif d1 < 3 and d0 - d1 > 1.5:
                tags_pos.append("tight_marking")
            if i in back4 and x1[i] - x0[i] < -1.5:
                tags_pos.append("line_drop")
            elif i in back4 and x1[i] - x0[i] > 1.5:
                tags_pos.append("line_step")
            if abs(y1[i]) < abs(y0[i]) - 2:
                tags_pos.append("tuck_in")
            elif abs(y1[i]) > abs(y0[i]) + 2:
                tags_pos.append("widen")
            if not tags_pos:
                tags_pos.append("cover_space")
        if sc.known[i] and g_post > eps:
            if abs(np.degrees(_wrap(h1[i] - h0[i]))) >= 20 and \
                    len(sc.seen_with_ball(i, x1, y1, h1)) > len(sc.seen_with_ball(i, x1, y1, h0)):
                tags_post.append("scan")
            if shape_of(t0[i]) == "square" and shape_of(t1[i]) != "square":
                tags_post.append("open_body")
            if l1[i] and not l0[i]:
                tags_post.append("get_low")
        for tg in tags_pos:
            rows.append(dict(player_id=int(sc.D.player_id.iloc[i]), defender=sc.D.name.iloc[i], tag=tg,
                             gain=float(g_pos / len(tags_pos)), move_m=round(float(moved), 1)))
        for tg in tags_post:
            rows.append(dict(player_id=int(sc.D.player_id.iloc[i]), defender=sc.D.name.iloc[i], tag=tg,
                             gain=float(g_post / len(tags_post)), move_m=round(float(moved), 1)))
    return dict(before=float(before), after=float(after), reduction=float((before - after) / before) if before else 0.0), rows


def _work(args):
    match_id, rec = args
    try:
        mom, rows = coach_moment(match_id, rec["frame"], rec["team"], rec["focus"])
    except Exception as exc:  # noqa: BLE001
        return rec, None, [], str(exc)
    return rec, mom, rows, None


def build_patterns(match_id: int, workers: int = 2, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_m, out_c = BUILD_DIR / str(match_id) / "sim_moments.parquet", BUILD_DIR / str(match_id) / "sim_corrections.parquet"
    if out_m.exists() and out_c.exists() and not force:
        return pd.read_parquet(out_m), pd.read_parquet(out_c)
    cand = candidates(match_id)
    jobs = [(match_id, r) for r in cand.to_dict("records")]
    res = []
    if workers > 1:
        import multiprocessing as mp
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"  # Windows/macOS: spawn
        with mp.get_context(method).Pool(workers) as pool:
            for k, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
                res.append(r)
                if k % 50 == 0:
                    print(f"  {match_id}: {k}/{len(jobs)} moments coached", flush=True)
    else:
        res = [_work(j) for j in jobs]
    M, C = [], []
    for rec, mom, rows, err in res:
        if mom is None:
            continue
        mid = f"{match_id}_{rec['frame']}_{rec['team']}"
        M.append({**rec, **mom, "moment_id": mid, "match_id": match_id})
        C += [{**r, "moment_id": mid, "match_id": match_id, "frame": rec["frame"], "team": rec["team"],
               "kind": rec["kind"], "control": rec["control"], "third": rec["third"], "channel": rec["channel"],
               "phase": rec["phase"]} for r in rows]
    Mdf, Cdf = pd.DataFrame(M), pd.DataFrame(C)
    Mdf.to_parquet(out_m); Cdf.to_parquet(out_c)
    return Mdf, Cdf


@lru_cache(maxsize=1)
def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    M, C = [], []
    for m in MATCHES:
        a, b = BUILD_DIR / str(m) / "sim_moments.parquet", BUILD_DIR / str(m) / "sim_corrections.parquet"
        if a.exists() and b.exists():
            M.append(pd.read_parquet(a)); C.append(pd.read_parquet(b))
    if not M:
        return pd.DataFrame(), pd.DataFrame()
    # deterministic order (moments are coached in parallel, so file order varies between runs)
    M = pd.concat(M, ignore_index=True).sort_values(["match_id", "frame", "team"], kind="stable").reset_index(drop=True)
    C = pd.concat(C, ignore_index=True).sort_values(["match_id", "frame", "team", "player_id", "tag"], kind="stable").reset_index(drop=True)
    return M, C


@lru_cache(maxsize=1)
def _shot_after() -> dict:
    """moment_id -> did the attacking team shoot within 10 s after the moment (Dynamic Events)."""
    from .simulate import _data
    from .validate import _shots
    M, _ = _load()
    out = {}
    for m, g in M.groupby("match_id"):
        meta = _data(int(m))[0]
        sh = _shots(int(m))
        ids = {meta.team_name(meta.home_id): meta.home_id, meta.team_name(meta.away_id): meta.away_id}
        for r in g.itertuples():
            att = meta.away_id if ids.get(r.team) == meta.home_id else meta.home_id
            f = sh.frame.to_numpy()[sh.team_id.to_numpy() == att]
            out[r.moment_id] = bool(((f > r.frame) & (f <= r.frame + 100)).any())
    return out


@lru_cache(maxsize=1)
def tag_outcomes(n_boot: int = 400) -> dict:
    """Is each error type linked to REAL shots?  Over all dangerous moments of both matches: shot within 10 s
    when the coach flags the error vs when it does not, stratified by pitch third (Mantel-Haenszel weights),
    95% bootstrap CI resampling moments."""
    M, C = _load()
    if M.empty:
        return {}
    d = M[~M.control].copy()
    try:
        y = _shot_after()
    except Exception:  # noqa: BLE001  (raw events missing)
        return {}
    d["shot10"] = d.moment_id.map(y).astype(bool)
    flagged = C[~C.control].groupby("tag").moment_id.apply(set).to_dict()
    third = d.third.fillna("?").to_numpy()
    shot = d.shot10.to_numpy()
    strata = [third == t for t in np.unique(third)]

    def rd(hit, idx=None):
        num = den = 0.0
        for st in strata:
            sel = st if idx is None else st[idx]
            h, ys = (hit[sel], shot[sel]) if idx is None else (hit[idx][sel], shot[idx][sel])
            n1, n0 = h.sum(), (~h).sum()
            if n1 and n0:
                wgt = n1 * n0 / (n1 + n0)
                num += wgt * (ys[h].mean() - ys[~h].mean()); den += wgt
        return num / den if den else np.nan

    rng = np.random.default_rng(7)
    out = {}
    for tag, ids in flagged.items():
        hit = d.moment_id.isin(ids).to_numpy()
        if hit.sum() < 10:
            continue
        est = rd(hit)
        bs = [rd(hit, rng.integers(0, len(hit), len(hit))) for _ in range(n_boot)]
        ci = np.nanpercentile(bs, [2.5, 97.5])
        out[tag] = {"n": int(hit.sum()), "shot_with": round(float(shot[hit].mean()), 3),
                    "shot_without": round(float(shot[~hit].mean()), 3), "rd": round(float(est), 3),
                    "ci": [round(float(ci[0]), 3), round(float(ci[1]), 3)], "linked": bool(ci[0] > 0)}
    return out


@lru_cache(maxsize=4)
def _clock_map(match_id: int) -> dict:
    out = {}
    for tb in ("passes", "runs", "duels"):
        p = BUILD_DIR / str(match_id) / f"{tb}.parquet"
        if p.exists():
            d = pd.read_parquet(p, columns=["frame", "minute", "second"])
            out.update({int(f): f"{int(m):02d}:{int(s):02d}" for f, m, s in zip(d.frame, d.minute, d.second)})
    return out


def _top(s: pd.Series, k=3):
    v = s.dropna().value_counts(normalize=True).head(k)
    return [{"value": str(i), "share": round(float(x), 2)} for i, x in v.items()]


def error_patterns(team: str | None = None, min_moments: int = 3) -> list[dict]:
    """Error types per team ranked by avoidable danger in dangerous moments, with lift vs ordinary frames."""
    M, C = _load()
    if M.empty:
        return []
    out = []
    for tm, Mt in M.groupby("team"):
        if team and tm != team:
            continue
        dang, ctl = Mt[~Mt.control], Mt[Mt.control]
        Ct = C[C.team == tm]
        for tag, g in Ct.groupby("tag"):
            gd, gc = g[~g.control], g[g.control]
            n_d = gd.moment_id.nunique()
            if n_d < min_moments:
                continue
            rate_d = n_d / max(len(dang), 1)
            rate_c = gc.moment_id.nunique() / max(len(ctl), 1)
            # bootstrap CI of the lift (resampling moments)
            hit_d = dang.moment_id.isin(set(gd.moment_id)).to_numpy()
            hit_c = ctl.moment_id.isin(set(gc.moment_id)).to_numpy()
            rng = np.random.default_rng(11)
            lifts = []
            if len(hit_d) and len(hit_c):
                for _ in range(400):
                    a_ = hit_d[rng.integers(0, len(hit_d), len(hit_d))].mean()
                    c_ = hit_c[rng.integers(0, len(hit_c), len(hit_c))].mean()
                    if c_ > 0:
                        lifts.append(a_ / c_)
            lift_ci = [round(float(np.percentile(lifts, 2.5)), 2), round(float(np.percentile(lifts, 97.5)), 2)] if lifts else None
            per = gd.groupby("player_id").agg(defender=("defender", "first"), n=("moment_id", "nunique"), gain=("gain", "sum"))
            per = per.sort_values("gain", ascending=False)
            # one example per focus player (his most avoidable moment), so the clips match the names shown
            zone = DRILL_ZONE.get(tag)

            def _best(pid):
                g_ = gd[gd.player_id == pid]
                if zone and (g_.third == zone).any():
                    g_ = g_[g_.third == zone]
                return g_.sort_values("gain", ascending=False).head(1)
            ex = pd.concat([_best(pid) for pid in per.index[:4]])
            label, principle, kind = TAGS[tag]
            oc = tag_outcomes().get(tag)
            out.append({
                "outcome": oc, "linked": bool(oc and oc["linked"]),
                "team": tm, "tag": tag, "error": label, "principle": principle, "kind": kind,
                "moments": int(n_d), "dangerous_moments": int(len(dang)), "rate": round(rate_d, 3),
                "rate_control": round(rate_c, 3), "lift": round(rate_d / rate_c, 2) if rate_c > 0 else None, "lift_ci": lift_ci,
                "robust": bool(lift_ci and lift_ci[0] > 1.0),
                "avoidable_danger": round(float(gd.gain.sum()), 2),
                # the part of the avoidable danger that is specific to dangerous moments (above the ordinary rate)
                "excess_avoidable": round(float(gd.gain.sum()) * max(0.0, 1 - rate_c / rate_d), 2) if rate_d > 0 else 0.0,
                "avoidable_share": round(float(gd.gain.sum() / max(dang.before.sum(), 1e-9)), 3),
                "players": [{"player_id": int(pid), "defender": r.defender, "moments": int(r.n), "avoidable": round(float(r.gain), 2)}
                            for pid, r in per.head(4).iterrows()],
                "where": {"third": _top(gd.third), "channel": _top(gd.channel), "phase": _top(gd.phase), "situation": _top(gd.kind)},
                "examples": [{"match_id": int(r.match_id), "frame": int(r.frame), "time": _clock_map(int(r.match_id)).get(int(r.frame)),
                          "defender": r.defender,
                              "player_id": int(r.player_id), "kind": r.kind, "gain": round(float(r.gain), 3)} for r in ex.itertuples()],
            })
    out.sort(key=lambda d: (d["team"], -d["excess_avoidable"], -d["avoidable_danger"]))
    return out


# ------------------------------------------------------------------ drills
DRILLS = {
    "goal_side": {
        "name": "Recovery race — get back goal-side",
        "objective": "When the ball is played past you, recover on the inside line between your man and the goal before engaging.",
        "setup": "Half-pitch, 2 attackers vs 2 defenders + GK. Coach serves through-balls from the halfway line; defenders start 2 m ahead (on the wrong side) of their attacker.",
        "rules": "Defenders score a point by being goal-side of their man when he receives; attackers score normally. 6 reps × 4 sets.",
        "coaching": ["REACT: first step backwards on the pass, not after it", "SHAPE: open hips to the goal, curve run to the inside line", "SEE: check the runner over the shoulder while recovering"],
        "progression": "Delay the defender's start by 0.5 s; add a third attacker arriving late.",
        "kpi": "goal_side rate of the receiver's marker at line-breaking passes (passes table)",
    },
    "tight_marking": {
        "name": "Touch-tight in the box",
        "objective": "Reduce distance to your man as the ball approaches the danger zone (inside 25 m).",
        "setup": "Box + D: 4 attackers vs 4 defenders, 2 wide servers crossing/cutting back, GK.",
        "rules": "Marker must be within an arm's length (≈1.5 m) at the moment of the cross or the attack gets a bonus point.",
        "coaching": ["REACT: tighten when the server's head goes down", "SEE: ball and man in one view (side-on)", "SHAPE: goal-side, between man and goal centre"],
        "progression": "Add blind-side runs and late arrivals at the penalty spot.",
        "kpi": "marking_distance at passes into the box; receiver_free rate",
    },
    "press_ball": {
        "name": "First defender: close down and stop forward play",
        "objective": "Nearest player engages the carrier quickly and on a curved run that blocks the forward pass.",
        "setup": "3v3 in a 25 × 20 m channel with two end gates; triggers called by the coach (bad touch, back to goal, pass to full-back).",
        "rules": "Defending team wins a point for every regain or backward pass forced within 5 s.",
        "coaching": ["REACT: go on the pass, arrive as the ball arrives", "STANCE: decelerate, get low, side-on in the last 2 m", "SHAPE: curve to show him away from goal"],
        "progression": "Second defender must cover within 6 m (2v2 pressure-cover).",
        "kpi": "pressure on the ball (carrier_pressure_dist), PPDA, regains after engagements",
    },
    "line_drop": {
        "name": "Back-four line control — drop on no-pressure",
        "objective": "When the ball carrier is unpressed and facing forward, the line drops together to protect the space in behind.",
        "setup": "8v6 (back four + 2 midfielders vs 8) between halfway and the box; coach calls 'pressure' / 'no pressure'.",
        "rules": "Through-balls behind the line are worth double; offside counts as a defensive point only if the line moved together.",
        "coaching": ["SEE: read the carrier (head up, no pressure = drop)", "SHAPE: half-turned bodies, ready to run back", "REACT: line moves together, 1 m per 1 m of opponent progression"],
        "progression": "Remove the coach's call; players communicate the trigger.",
        "kpi": "runs in behind conceded (beaten_in_behind), last-line breaks conceded",
    },
    "line_step": {
        "name": "Back-four line control — step up to squeeze",
        "objective": "When the ball goes backwards or sideways under pressure, the line steps up to keep the block compact.",
        "setup": "Same 8v6 as the drop drill; the attacking team must recycle the ball back every 3 passes.",
        "rules": "Defence scores when the distance between back line and midfield stays below 12 m at the moment of a forward pass.",
        "coaching": ["SEE: trigger = backward pass / pressured receiver", "REACT: step up together, never alone", "SHAPE: compact distance to midfield ≤ 12 m"],
        "progression": "Live 10v10 with offside; video review of line height on the 3D viewer.",
        "kpi": "space between lines (line_gap), back-line height by phase",
    },
    "tuck_in": {
        "name": "Close the middle — narrow block",
        "objective": "Defenders far from the ball tuck inside to close central gaps; the ball-side full-back engages.",
        "setup": "6v6 + 2 neutral wide players on a 50 × 40 m pitch with 3 small central gates at each end.",
        "rules": "Goals through the central gate count 3, wide goals 1.",
        "coaching": ["SHAPE: weak-side defenders inside the posts", "SEE: check the central runner, not the ball only", "REACT: shift with every pass across"],
        "progression": "Remove the neutrals: switches of play must be stopped by the shift.",
        "kpi": "ball-zone compactness; line-breaking passes through the middle",
    },
    "widen": {
        "name": "Shift & balance against wide overloads",
        "objective": "The block shifts so that the wide attacker on the ball side is engaged without leaving the far post free.",
        "setup": "Back four + 1 vs 4 attackers + 2 full-backs overlapping; coach switches play.",
        "rules": "Attack scores double from a cross with the wide man unmarked.",
        "coaching": ["SHAPE: ball-side full-back out, centre-backs slide", "SEE: far-side full-back tucks but checks his man", "REACT: shift on the ball's travel"],
        "progression": "Add an underlapping midfielder.",
        "kpi": "free attackers in wide channels inside 35 m",
    },
    "cover_space": {
        "name": "Cover & balance (3v2 + 1)",
        "objective": "The second and third defenders protect the space and the shot lane behind the pressing defender.",
        "setup": "3 defenders vs 2 attackers + 1 joker in a 30 × 30 m area with a goal and GK.",
        "rules": "Points for blocked shots and intercepted passes into the space behind.",
        "coaching": ["SHAPE: cover at 45° behind the presser, balance across", "SEE: scan for the joker's run", "REACT: rotate when the presser is beaten"],
        "progression": "Joker becomes a third attacker (3v3).",
        "kpi": "attack-controlled space near goal (simulation space_share); open shot lanes",
    },
    "scan": {
        "name": "Scan before the pass — ball and man",
        "objective": "Keep ball and man in one view; check the shoulder before every opponent pass.",
        "setup": "4v4 + 3 neutrals rondo; behind each defender a coach holds a coloured bib for 1 s at random.",
        "rules": "A defender who calls the colour before the next pass earns a point; receivers free in front of an unscanned marker give the attack a point.",
        "coaching": ["SEE: shoulder check in the 3 s before each pass", "SHAPE: side-on so the turn of the head is short", "REACT: adjust distance after each check"],
        "progression": "Transfer into 6v6 with blind-side runs; review with the 3D 'Player's eyes' camera.",
        "kpi": "dual vision at passes, shoulder checks per pass, blind-side runs conceded",
    },
    "open_body": {
        "name": "Open your hips to the runner",
        "objective": "Stand side-on/open to your own goal when a runner can go in behind, so the first step is backwards.",
        "setup": "1v1 + server: attacker makes a timed run in behind from a stationary start; defender starts square.",
        "rules": "Defender must touch the cone at the edge of the box before the attacker; progress to starting open.",
        "coaching": ["SHAPE: hips at 45–135° to the own goal, not facing the ball square", "REACT: drop step on the server's touch", "SEE: runner on the shoulder"],
        "progression": "Two runners, one checking short, one in behind.",
        "kpi": "square body at run starts, reaction lag to runs, beaten in behind",
    },
    "get_low": {
        "name": "Jockey stance in 1v1",
        "objective": "Engage the carrier with a lowered centre of gravity and short steps, to react to the change of direction.",
        "setup": "1v1 channel 10 × 20 m to a small goal; 6 reps each.",
        "rules": "A defender who is beaten while upright loses double; a forced backward touch is a point.",
        "coaching": ["STANCE: knees flexed, hips below normal running height", "SHAPE: side-on, show to the touchline", "REACT: stay on the toes, don't dive in"],
        "progression": "2v1 with a supporting runner.",
        "kpi": "loaded stance at engagements, regains, duels lost",
    },
}


def training_plan(team: str, top: int = 4) -> dict:
    pats = [p for p in error_patterns(team, min_moments=10) if p["avoidable_danger"] > 0]
    if not pats:
        return {"team": team, "error": "no simulated moments for this team; run `python -m dbl patterns` first"}
    # rank by the avoidable danger that is SPECIFIC to dangerous moments (lift > 1); errors that are just as frequent
    # in ordinary defending are a model/style baseline, not this team's problem
    # priority: errors that are specific to dangerous moments (lift CI > 1) AND linked to real shots come first
    pats.sort(key=lambda p: (-(int(p["robust"]) + int(p["linked"])), -p["excess_avoidable"], -p["avoidable_danger"]))
    sel = [p for p in pats if p["excess_avoidable"] > 0][:top] or pats[:top]
    blocks = []
    for k, p in enumerate(sel, 1):
        d = DRILLS[p["tag"]]
        blocks.append({
            "priority": k, "pattern": p["error"], "tag": p["tag"], "principle": p["principle"],
            "evidence": f"{p['moments']} of {p['dangerous_moments']} dangerous moments ({p['rate']:.0%}); "
                        + (f"{p['lift']:.1f}× more often than in ordinary defending"
                           + (f" (95% CI {p['lift_ci'][0]:.1f}–{p['lift_ci'][1]:.1f})" if p.get("lift_ci") else "") + "; " if p.get("lift") else "")
                        + f"{p['avoidable_share']:.0%} of the modelled danger in those moments is avoidable through this correction"
                        + (f"; real outcome (both matches): shot within 10 s {p['outcome']['shot_with']:.0%} when flagged vs "
                           f"{p['outcome']['shot_without']:.0%} ({p['outcome']['rd'] * 100:+.0f} pp by third, 95% CI "
                           f"{p['outcome']['ci'][0] * 100:+.0f} to {p['outcome']['ci'][1] * 100:+.0f})" if p.get("outcome") else ""),
            "robust": p["robust"], "linked": p["linked"], "outcome": p.get("outcome"),
            "moments": p["moments"], "dangerous_moments": p["dangerous_moments"], "rate": p["rate"],
            "lift": p.get("lift"), "lift_ci": p.get("lift_ci"), "players": p["players"], "where": p["where"], "examples": p["examples"][:3],
            "drill": d, "minutes": 20 if k <= 2 else 15,
        })
    M, _ = _load()
    n = int((~M[M.team == team].control).sum()) if not M.empty else 0
    return {"team": team, "dangerous_moments_coached": n, "session": blocks,
            "session_outline": ["Warm-up with the scanning rondo (10')"] + [f"{b['drill']['name']} ({b['minutes']}')" for b in blocks] +
                               ["Game 9v9 with the session's conditions as scoring bonuses (20')"],
            "note": "Patterns come from the counterfactual coach (position + posture) applied to every dangerous moment; "
                    "lift compares with ordinary out-of-possession frames. Drills are coaching suggestions linked to the "
                    "error type; track the KPI in the next match to verify."}


def write_training_doc(path=None) -> str:
    from .config import ROOT
    M, _ = _load()
    L = ["# Training plans from the simulations", "",
         "_Generated by `python -m dbl patterns`. Every dangerous moment the defence faced was re-simulated by the "
         "posture-aware coach; corrections were tagged with an error type and ranked by the danger they would have "
         "removed. Lift = how much more often the error appears in dangerous moments than in ordinary defending. Real "
         "outcome = how often the attack actually shot within 10 s when the coach flags this error, against dangerous "
         "moments without it (same pitch third, both matches pooled). Priorities: errors that are both specific to "
         "danger (lift CI above 1) and linked to real shots come first._", ""]
    for team in sorted(M.team.unique()) if not M.empty else []:
        tp = training_plan(team)
        if "error" in tp:
            continue
        L += [f"## {team} ({tp['dangerous_moments_coached']} dangerous moments simulated)", "",
              "| # | Error pattern | Principle | How often (dangerous moments) | Specific to danger (lift, 95% CI) | "
              "Linked to real shots (shot ≤ 10 s: flagged vs not; Δ by third, 95% CI) | Players to focus |",
              "|---|---|---|---|---|---|---|"]
        pmap = {p["tag"]: p for p in error_patterns(team, min_moments=10)}
        for b in tp["session"]:
            p_ = pmap[b["tag"]]
            lift = (f"{p_['lift']:.1f}× ({p_['lift_ci'][0]:.1f}–{p_['lift_ci'][1]:.1f})" if p_.get("lift_ci") else "–") \
                + (" ✔" if p_["robust"] else "")
            oc = p_.get("outcome")
            link = (f"{oc['shot_with']:.0%} vs {oc['shot_without']:.0%}; {oc['rd'] * 100:+.0f} pp "
                    f"({oc['ci'][0] * 100:+.0f} to {oc['ci'][1] * 100:+.0f})" + (" ✔" if oc["linked"] else "")) if oc else "–"
            L.append(f"| {b['priority']} | **{b['pattern']}** | {b['principle']} | {p_['moments']} of {p_['dangerous_moments']} "
                     f"({p_['rate']:.0%}) | {lift} | {link} | "
                     + ", ".join(f"{p['defender']} ({p['moments']})" for p in b["players"][:3]) + " |")
        L += ["", "✔ = 95% CI excludes no effect.", "", "**Session**: " + " → ".join(tp["session_outline"]), ""]
        for b in tp["session"]:
            d = b["drill"]
            L += [f"### {b['priority']}. {d['name']} — fixes *{b['pattern'].lower()}*", "",
                  f"- **Objective:** {d['objective']}", f"- **Setup:** {d['setup']}", f"- **Rules:** {d['rules']}",
                  "- **Coaching points:** " + "; ".join(d["coaching"]), f"- **Progression:** {d['progression']}",
                  f"- **Track next match:** {d['kpi']}",
                  "- **Watch in the 3D viewer:** " + ", ".join(f"{e.get('time') or 'frame ' + str(e['frame'])} ({e['defender']}, {e['kind']})" for e in b["examples"]), ""]
    L += ["_Model-derived priorities, not proof that a drill fixes the error: verify with the KPI in the next match._", ""]
    path = path or ROOT / "docs" / "TRAINING.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return str(path)
