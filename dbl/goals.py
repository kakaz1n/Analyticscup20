"""Goal-concession analysis: how did each goal happen and which defenders carry responsibility?

Goals are found from the score change in SkillCorner Dynamic Events (robust to goals
not flagged `lead_to_goal`). For each goal we rebuild the attacking sequence and audit the
defence in the preceding window with the framework (SEE / SHAPE / STANCE / REACT) plus team shape.

"Guilt" is reported as **responsibility indicators** with explicit reasons and points, never as a
verdict: tracking + pose show positions and orientation, not instructions given by the coach.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .data.load import load_events, load_meta
from .shape import state_at

WINDOW_S = 15


@lru_cache(maxsize=4)
def find_goals(match_id: int) -> list[dict]:
    """Goals = possessions SkillCorner marks with `game_interruption_after == goal_for` or `lead_to_goal`
    (moved forward to the team's shot if the flag sits on the assist). Falls back to the score change
    in the event stream if a goal is missing."""
    meta = load_meta(match_id)
    ev = load_events(match_id).sort_values("frame_start")
    home, away = meta.team_name(meta.home_id), meta.team_name(meta.away_id)
    pp = ev[ev.event_type == "player_possession"]
    flag = pp[(pp.game_interruption_after == "goal_for") | ((pp.end_type == "shot") & (pp.lead_to_goal == True))]  # noqa: E712
    rows = []
    for _, r in flag.iterrows():
        if r.end_type != "shot":
            nxt = pp[(pp.team_shortname == r.team_shortname) & (pp.end_type == "shot")
                     & (pp.frame_start >= r.frame_start) & (pp.frame_start <= r.frame_start + 60)]
            if len(nxt):
                r = nxt.iloc[0]
        if rows and abs(rows[-1].frame_start - r.frame_start) < 100 and rows[-1].team_shortname == r.team_shortname:
            continue
        rows.append(r)
    expected = int(meta.raw["home_team_score"]) + int(meta.raw["away_team_score"])
    if len(rows) < expected:  # fallback: score change
        for team in (home, away):
            sc = np.where(ev.team_shortname == team, ev.team_score, ev.opponent_team_score)
            for f_change in ev.frame_start[np.r_[False, np.diff(sc) > 0]]:
                if any(abs(x.frame_start - f_change) < 400 and x.team_shortname == team for x in rows):
                    continue
                c = pp[(pp.team_shortname == team) & (pp.frame_start <= f_change) & (pp.frame_start >= f_change - 250)]
                sh = c[c.end_type == "shot"]
                if len(sh) or len(c):
                    rows.append((sh if len(sh) else c).iloc[-1])
    rows.sort(key=lambda r: r.frame_start)
    goals, score = [], {home: 0, away: 0}
    for i, g in enumerate(rows, 1):
        team = g.team_shortname
        score[team] += 1
        prev = pp[(pp.team_shortname == team) & (pp.frame_start < g.frame_start) & (pp.frame_start >= g.frame_start - 150)]
        assist = prev[(prev.end_type == "pass") & (prev.player_targeted_name == g.player_name)]
        goals.append({
            "goal_number": i, "team": team, "conceding": away if team == home else home,
            "frame": int(g.frame_start), "period": int(g.period),
            "time": f"{int(g.minute_start):02d}:{int(g.second_start):02d}",
            "scorer": g.player_name, "scorer_id": int(g.player_id) if pd.notna(g.player_id) else None,
            "assist": assist.iloc[-1].player_name if len(assist) else None,
            "assist_frame": int(assist.iloc[-1].frame_end) if len(assist) else None,
            "score_after": f"{home} {score[home]}-{score[away]} {away}",
            "phase": g.get("team_in_possession_phase_type"),
        })
    return goals


def _sequence(ev: pd.DataFrame, team: str, f_goal: int) -> list[dict]:
    """The scoring team's possession chain leading to the goal (back to the last opponent possession)."""
    pp = ev[(ev.event_type == "player_possession") & (ev.frame_start <= f_goal)].sort_values("frame_start")
    chain = []
    for _, r in pp[::-1].iterrows():
        if r.team_shortname != team or f_goal - r.frame_start > 30 * 10:
            break
        lb = [n for n, c in (("1st line", "first_line_break"), ("2nd-last line", "second_last_line_break"),
                             ("last line", "last_line_break")) if r.get(c) is True]
        chain.append({"time": f"{int(r.minute_start):02d}:{int(r.second_start):02d}", "frame": int(r.frame_start),
                      "player": r.player_name, "action": r.end_type, "to": r.get("player_targeted_name"),
                      "start": r.get("start_type"), "line_break": ", ".join(lb) or None,
                      "pressure_on_ball": r.get("overall_pressure_start")})
    return chain[::-1]


def goal_breakdown(match_id: int, goal_number: int, store=None) -> dict:
    goals = find_goals(match_id)
    if not 1 <= goal_number <= len(goals):
        return {"error": f"goal_number must be 1..{len(goals)}", "goals": goals}
    g = goals[goal_number - 1]
    ev = load_events(match_id)
    f1, f0 = g["frame"], g["frame"] - WINDOW_S * 10
    seq = _sequence(ev, g["team"], f1)
    if seq:
        f0 = min(f0, seq[0]["frame"] - 20)
    if store is None:
        from .query.tools import store as _s
        store = _s()
    T = store.tables
    sel = lambda t: T[t][(T[t].match_id == match_id) & (T[t].frame >= f0) & (T[t].frame <= f1 + 5)  # noqa: E731
                         & (T[t].defending_team == g["conceding"])]
    passes, runs, duels, recs = sel("passes"), sel("runs"), sel("duels"), sel("recoveries")

    blame: dict[str, dict] = {}

    def add(name, pts, why, t):
        if not isinstance(name, str):
            return
        b = blame.setdefault(name, {"defender": name, "points": 0, "reasons": []})
        b["points"] += pts
        b["reasons"].append(f"{t}: {why} (+{pts})")

    key_players = {g["scorer"], g["assist"]}
    for r in passes.itertuples():
        w = 2 if r.attacker in key_players else 1
        if r.line_break and pd.notna(r.defender):
            add(r.defender, w, f"line-breaking pass {r.passer} → {r.attacker} while he was the receiver's marker", r.time)
        if isinstance(r.vision, str) and r.vision not in ("ball_and_man",) and r.attacker in key_players:
            add(r.defender, 2, f"could not see ball and man ({r.vision}) when {r.attacker} received", r.time)
        if r.attacker in key_players and pd.notna(r.marking_distance) and r.marking_distance > 5:
            add(r.defender, 1, f"{r.marking_distance:.1f} m from {r.attacker} at the pass", r.time)
        if r.attacker in key_players and r.shoulder_checks == 0:
            add(r.defender, 1, "no shoulder check in the 3 s before the pass", r.time)
    for r in runs.itertuples():
        w = 2 if r.attacker in key_players else 1
        if r.beaten_in_behind is True:
            add(r.defender, 3 * w // 1, f"{r.attacker} got in behind him ({str(r.subtype).replace('_', ' ')})", r.time)
        if r.blind_side is True:
            add(r.defender, w, f"run by {r.attacker} started on his blind side", r.time)
        if r.body_shape == "square":
            add(r.defender, w, "square/flat body when the run started", r.time)
        if pd.notna(r.reaction_lag_s) and r.reaction_lag_s >= 0.8:
            add(r.defender, w, f"reacted {r.reaction_lag_s:.1f} s after the runner", r.time)
    for r in duels.itertuples():
        if r.beaten:
            add(r.defender, 2, f"beaten in the duel by {r.attacker}", r.time)
    for r in recs.itertuples():
        if not r.recovered:
            add(r.defender, 1, "did not recover goal-side within 3 s", getattr(r, "time", ""))

    # at the shot: who was nearest to the scorer and was he close?
    shot_marker = None
    try:
        from .data.load import load_tracking
        meta = load_meta(match_id)
        _, xy = load_tracking(match_id)
        dteam = meta.home_id if meta.team_name(meta.home_id) == g["conceding"] else meta.away_id
        sgn = 1.0 if meta.defends_left(dteam, g["period"]) else -1.0
        pl = meta.players.set_index("player_id")
        snap = xy[xy.frame == f1].copy()
        snap["x"], snap["y"] = snap.x * sgn, snap.y * sgn
        snap["team"] = snap.player_id.map(pl.team_id)
        snap["gk"] = snap.player_id.map(pl.is_gk).fillna(False).astype(bool)
        if g["scorer_id"] in set(snap.player_id):
            sc = snap[snap.player_id == g["scorer_id"]].iloc[0]
            d = snap[(snap.team == dteam) & ~snap.gk]
            dist = np.hypot(d.x - sc.x, d.y - sc.y).to_numpy()
            j = int(np.argmin(dist))
            nd = d.iloc[j]
            gx = -meta.pitch[0] / 2
            between = int(((d.x < sc.x) & (np.abs(d.y - sc.y * (d.x - gx) / (sc.x - gx + 1e-9)) < 2)).sum())
            shot_marker = {"defender": pl.name.get(int(nd.player_id)), "distance_m": round(float(dist[j]), 1),
                           "goal_side": bool(nd.x < sc.x), "defenders_in_shot_line": between,
                           "shot_distance_m": round(float(np.hypot(sc.x - gx, sc.y)), 1)}
            if shot_marker["distance_m"] > 2.5:
                add(shot_marker["defender"], 2, f"nearest defender to the scorer at the shot but {shot_marker['distance_m']} m away",
                    g["time"])
    except Exception as exc:  # noqa: BLE001
        shot_marker = {"note": f"tracking unavailable: {exc}"}

    ranking = sorted(blame.values(), key=lambda b: -b["points"])
    shape_assist = state_at(match_id, g["assist_frame"]) if g.get("assist_frame") else None
    return {
        "goal": g,
        "sequence": seq,
        "shape_at_assist": shape_assist,
        "shape_at_shot": state_at(match_id, f1),
        "at_shot": shot_marker,
        "responsibility": ranking[:6],
        "method": "Points from framework moments in the attacking sequence (weights x2 when the scorer/assister is "
                  "involved): beaten in behind 3, vision failure 2, beaten in duel 2, unmarked scorer at shot 2, "
                  "line-breaking pass while marking 1, >5 m from receiver 1, no shoulder check 1, blind side 1, "
                  "square body 1, late reaction 1, no recovery 1. Indicators, not a verdict.",
        "window_frames": [int(f0), int(f1)],
    }
