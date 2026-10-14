"""Team-level defensive shape from tracking (what analysts read off the tactical cam).

For every in-play frame (sampled at 2 Hz) the team OUT of possession (from
SkillCorner Phases of Play) is measured in its own normalised frame (own goal
at x = -L/2):

* line_height   distance of the back line (deepest 4 outfielders) from own goal
* block_height  distance of the outfield centroid from own goal
* length/width  depth and breadth of the outfield block
* area          convex hull of the 10 outfielders (team compactness)
* ball_zone_area convex hull of the 5 defenders nearest the ball -- the
                compactness that relates to defensive success (Vienna study, 2023)
* line_gap      distance between back line and the next 3 players (space between lines)
* behind_ball   outfielders goal-side of the ball
* free_attackers attackers within 35 m of goal with no defender within 5 m
* press_n       defenders within 5 m of the ball;  press_dist nearest defender to ball
* offside_line  distance from goal of the second-last defender (GK included)

Plus match-level defensive KPIs from Dynamic Events (PPDA*, counter-pressing,
regains, line breaks conceded) and the SEE-SHAPE-STANCE-REACT body metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from .config import BUILD_DIR
from .data.load import load_events, load_meta, load_phases, load_tracking

STEP = 5  # every 5th tracking frame -> 2 Hz


def _hull_area(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return np.nan
    try:
        return float(ConvexHull(xy).volume)
    except Exception:  # noqa: BLE001  (collinear)
        return np.nan


def build_shape(match_id: int) -> pd.DataFrame:
    meta = load_meta(match_id)
    fr, xy = load_tracking(match_id)
    ph = load_phases(match_id)
    L = meta.pitch[0]
    pl = meta.players.set_index("player_id")
    team_of, gk = pl.team_id.to_dict(), pl.is_gk.to_dict()
    ids = (meta.home_id, meta.away_id)

    # frame -> phase (team in possession + phase type of the team out of possession)
    rows = []
    for r in ph.itertuples():
        f = np.arange(int(r.frame_start), int(r.frame_end) + 1)
        f = f[f % STEP == 0]
        rows.append(pd.DataFrame({"frame": f, "in_poss": int(r.team_in_possession_id),
                                  "phase_oop": r.team_out_of_possession_phase_type,
                                  "phase_ip": r.team_in_possession_phase_type, "period": int(r.period)}))
    fp = pd.concat(rows).drop_duplicates("frame").set_index("frame")
    xy = xy[xy.frame.isin(fp.index)]
    balls = fr.set_index("frame")[["ball_x", "ball_y"]]
    out = []
    for f, g in xy.groupby("frame"):
        p = fp.loc[f]
        dteam = ids[1] if p.in_poss == ids[0] else ids[0]
        s = 1.0 if meta.defends_left(dteam, int(p.period)) else -1.0
        tid = g.player_id.map(team_of).to_numpy()
        isgk = g.player_id.map(gk).fillna(False).to_numpy(bool)
        X, Y = g.x.to_numpy() * s, g.y.to_numpy() * s
        dm, am = (tid == dteam), (tid != dteam) & ~pd.isna(tid)
        D = np.c_[X[dm & ~isgk], Y[dm & ~isgk]]
        A = np.c_[X[am & ~isgk], Y[am & ~isgk]]
        Dall = np.c_[X[dm], Y[dm]]
        if len(D) < 6:
            continue
        bx, by = (balls.ball_x.get(f, np.nan) * s, balls.ball_y.get(f, np.nan) * s)
        gx = -L / 2
        xs = np.sort(D[:, 0])
        back = xs[:4]
        mid = xs[4:7]
        rec = {
            "frame": int(f), "period": int(p.period), "defending_team": meta.team_name(dteam),
            "phase_oop": p.phase_oop, "phase_ip": p.phase_ip,
            "line_height": float(back.mean() - gx), "block_height": float(D[:, 0].mean() - gx),
            "length": float(xs[-1] - xs[0]), "width": float(D[:, 1].max() - D[:, 1].min()),
            "area": _hull_area(D), "line_gap": float(mid.mean() - back.mean()) if len(mid) else np.nan,
            "offside_line": float(np.sort(Dall[:, 0])[1] - gx) if len(Dall) >= 2 else np.nan,
        }
        if np.isfinite(bx):
            d2b = np.hypot(D[:, 0] - bx, D[:, 1] - by)
            near = D[np.argsort(d2b)[:5]]
            rec.update(ball_x=float(bx - gx), ball_zone_area=_hull_area(near),
                       behind_ball=int((D[:, 0] < bx).sum()), press_dist=float(d2b.min()),
                       press_n=int((d2b <= 5).sum()))
        if len(A):
            dd = np.hypot(A[:, None, 0] - D[None, :, 0], A[:, None, 1] - D[None, :, 1]).min(1)
            danger = (A[:, 0] - gx) <= 35
            rec["free_attackers"] = int(((dd > 5) & danger).sum())
            rec["attackers_in_zone"] = int(danger.sum())
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_parquet(BUILD_DIR / str(match_id) / "shape.parquet", index=False)
    return df


def load_shape(match_id: int) -> pd.DataFrame:
    p = BUILD_DIR / str(match_id) / "shape.parquet"
    return pd.read_parquet(p) if p.exists() else build_shape(match_id)


SHAPE_COLS = ["line_height", "block_height", "length", "width", "area", "ball_zone_area", "line_gap",
              "behind_ball", "free_attackers", "press_n", "press_dist", "offside_line"]
ORGANISED = ["high_block", "medium_block", "low_block"]


def shape_by_phase(match_id: int) -> pd.DataFrame:
    s = load_shape(match_id)
    g = s.groupby(["defending_team", "phase_oop"])
    out = g[SHAPE_COLS].median().round(1)
    out["share_of_time"] = (g.size() / s.groupby("defending_team").size()).round(3)
    return out.reset_index()


def event_kpis(match_id: int) -> pd.DataFrame:
    """StatsBomb-style defensive KPIs rebuilt from SkillCorner Dynamic Events (no StatsBomb data used)."""
    ev = load_events(match_id)
    meta = load_meta(match_id)
    rows = []
    for t in (meta.home_id, meta.away_id):
        opp = meta.away_id if t == meta.home_id else meta.home_id
        eng = ev[(ev.event_type == "on_ball_engagement") & (ev.team_id == t)]
        opp_pp = ev[(ev.event_type == "player_possession") & (ev.team_id == opp)]
        build_zone = ("defensive_third", "middle_third")  # opponent's own 2/3 (attacking perspective)
        opp_passes = opp_pp[(opp_pp.end_type == "pass") & opp_pp.third_start.isin(build_zone)]
        eng_high = eng[eng.third_start.isin(build_zone)]
        lb = opp_pp[(opp_pp.end_type == "pass")]
        lb_any = lb.first_line_break.fillna(False).astype(bool) | lb.second_last_line_break.fillna(False).astype(bool) | \
            lb.last_line_break.fillna(False).astype(bool)
        actions = eng_high[eng_high.end_type.isin(["direct_regain", "indirect_regain", "direct_disruption",
                                                    "indirect_disruption", "foul_committed"])]
        rows.append({
            "team": meta.team_name(t),
            # classic PPDA uses tackles/interceptions/challenges/fouls -> closest SkillCorner analogue:
            # engagements that ended in a regain, a disruption or a foul
            "ppda": round(len(opp_passes) / max(len(actions), 1), 1),
            "passes_per_pressure": round(len(opp_passes) / max(len(eng_high), 1), 2),
            "engagements": len(eng),
            "pressures_high": len(eng_high),
            "counter_presses": int((eng.event_subtype == "counter_press").sum()),
            "pressing_chains": int(eng.pressing_chain.fillna(False).astype(bool).sum()),
            "chain_regains": int((eng.pressing_chain_end_type == "regain").sum()),
            "regains": int(eng.end_type.isin(["direct_regain", "indirect_regain"]).sum()),
            "opp_passes": int(len(lb)),
            "line_breaks_conceded": int(lb_any.sum()),
            "last_line_breaks_conceded": int(lb.last_line_break.fillna(False).astype(bool).sum()),
            "line_break_rate": round(float(lb_any.mean()), 3) if len(lb) else np.nan,
            "opp_shots": int((opp_pp.end_type == "shot").sum()),
        })
    return pd.DataFrame(rows)


def insights(match_id: int, store=None) -> dict:
    """Deterministic defensive report: numbers + plain-language bullets (no LLM)."""
    meta = load_meta(match_id)
    sp = shape_by_phase(match_id)
    sh = load_shape(match_id)
    kp = event_kpis(match_id).set_index("team")
    teams = [meta.team_name(meta.home_id), meta.team_name(meta.away_id)]
    org = sh[sh.phase_oop.isin(ORGANISED)]
    overall = org.groupby("defending_team")[SHAPE_COLS].median().round(1)
    mix = (sh[sh.phase_oop.isin(ORGANISED)].groupby(["defending_team", "phase_oop"]).size()
           .groupby(level=0).apply(lambda x: (x / x.sum()).round(3)).droplevel(0).unstack(fill_value=0))
    body = None
    if store is not None:
        tp = store.team_profile().set_index("team")
        body = tp.loc[[t for t in teams if t in tp.index]]
    bullets: dict[str, list[str]] = {}
    for t in teams:
        o = teams[1] if t == teams[0] else teams[0]
        b = []
        if t in mix.index:
            m = mix.loc[t]
            main = m.idxmax()
            b.append(f"Out of possession (organised phases) mostly in a **{main.replace('_', ' ')}** "
                     f"({m[main]:.0%}); high {m.get('high_block', 0):.0%} · medium {m.get('medium_block', 0):.0%} · low {m.get('low_block', 0):.0%}.")
        if t in overall.index and o in overall.index:
            a, c = overall.loc[t], overall.loc[o]
            b.append(f"Back line **{a.line_height:.0f} m** from goal (opponent {c.line_height:.0f} m); block length "
                     f"{a.length:.0f} m × width {a.width:.0f} m, area {a.area:.0f} m².")
            b.append(f"Ball-zone compactness (5 nearest defenders): **{a.ball_zone_area:.0f} m²** "
                     f"(opponent {c.ball_zone_area:.0f} m²) — the compactness that predicts winning the ball back.")
            b.append(f"Space between lines: **{a.line_gap:.1f} m** median; {a.behind_ball:.0f} outfielders goal-side of the ball.")
            fa = org[org.defending_team == t].free_attackers
            b.append(f"An attacker was free (no defender within 5 m) inside 35 m of goal in **{(fa > 0).mean():.0%}** of organised-defence time.")
        if t in kp.index:
            k = kp.loc[t]
            I = lambda c: int(k[c])  # noqa: E731
            b.append(f"PPDA **{k.ppda:.1f}** (opponent passes in their own two-thirds per regain/disruption/foul there; "
                     f"lower = more aggressive) · {k.passes_per_pressure:.2f} passes per pressing engagement; "
                     f"{I('counter_presses')} counter-presses, {I('chain_regains')} pressing chains ending in a regain, {I('regains')} regains.")
            b.append(f"Line-breaking passes conceded: **{I('line_breaks_conceded')}** of {I('opp_passes')} opponent passes "
                     f"({k.line_break_rate:.0%}); {I('last_line_breaks_conceded')} broke the last line; opponent shots {I('opp_shots')}.")
        if body is not None and t in body.index:
            r = body.loc[t]
            b.append(f"Body language: marker saw ball **and** man on {r.dual_vision:.0%} of passes, "
                     f"{r.checks_per_pass:.2f} shoulder checks per pass; {r.blind_side_runs:.0%} of forward runs started on a "
                     f"defender's blind side; square body at {r.square_at_run:.0%} of runs; beaten in behind {r.beaten_in_behind:.0%}.")
        bullets[t] = b
    return {
        "teams": teams,
        "bullets": bullets,
        "shape_by_phase": sp.to_dict(orient="records"),
        "kpis": kp.reset_index().to_dict(orient="records"),
        "body": body.reset_index().to_dict(orient="records") if body is not None else [],
    }


def series(match_id: int, every_s: int = 10) -> dict:
    """Coarse time series for the viewer's chart: per team, median line height per `every_s` seconds."""
    s = load_shape(match_id)
    s = s.assign(bucket=(s.frame // (every_s * 10)) * every_s * 10)
    g = s.groupby(["defending_team", "bucket"])[["line_height", "ball_zone_area", "free_attackers"]].median().reset_index()
    # 1-minute rolling mean so the chart shows the trend, not frame noise
    g[["line_height", "ball_zone_area", "free_attackers"]] = (
        g.groupby("defending_team")[["line_height", "ball_zone_area", "free_attackers"]]
        .transform(lambda x: x.rolling(6, center=True, min_periods=2).mean()))
    return {t: {"f": d.bucket.astype(int).tolist(), "line_height": d.line_height.round(1).tolist(),
                "ball_zone_area": d.ball_zone_area.round(0).tolist(), "free": d.free_attackers.round(1).tolist()}
            for t, d in g.groupby("defending_team")}


def state_at(match_id: int, frame: int) -> dict:
    """Shape metrics of the defending team at (the nearest sampled frame to) `frame`."""
    s = load_shape(match_id)
    i = (s.frame - frame).abs().idxmin()
    r = s.loc[i]
    if abs(int(r.frame) - frame) > 25:
        return {"frame": frame, "note": "ball out of play / no phase at this time"}
    return {k: (round(float(v), 1) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
            for k, v in r.items() if pd.notna(v)}
