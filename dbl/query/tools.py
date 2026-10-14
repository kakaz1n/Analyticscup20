"""Typed, side-effect-free tools over the MomentStore.

These are the ONLY things the LLM can call. Each returns JSON-serialisable data
computed by pandas, so every number in an answer is traceable to a table.
"""
import json
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ..framework import FINDINGS, METRICS, OUTCOMES, PRINCIPLES
from ..store import MomentStore

MAX_ROWS = 40


@lru_cache(maxsize=1)
def store() -> MomentStore:
    return MomentStore()


def _records(df: pd.DataFrame, limit: int = MAX_ROWS) -> list[dict]:
    df = df.head(limit).copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    return json.loads(df.to_json(orient="records", default_handler=str))


def _parse(obj: Any) -> Any:
    if isinstance(obj, str) and obj.strip().startswith(("{", "[")):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return obj
    return obj


def describe_framework() -> dict:
    """Return the SEE-SHAPE-STANCE-REACT framework: principles, every metric (definition, coaching cue,
    table it lives in) and every outcome. Call this first when unsure which column answers a question."""
    return {"principles": PRINCIPLES, "metrics": METRICS, "outcomes": OUTCOMES,
            "tables": {k: v[:80] for k, v in store().schema().items()},
            "teams": store().teams}


def list_columns(table: str) -> dict:
    """List the columns of a moment table ('duels', 'passes', 'runs' or 'recoveries') with example values."""
    df = store().tables[table]
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        out[c] = {"dtype": str(df[c].dtype),
                  "examples": [str(v) for v in (s.unique()[:6] if not pd.api.types.is_numeric_dtype(s) else s.head(3))]}
    return {"table": table, "rows": len(df), "columns": out}


def find_moments(table: str, where_json: str = "{}", query: str = "", columns: list[str] | None = None,
                 sort_by: str = "", ascending: bool = False, limit: int = 15) -> dict:
    """Filter individual defensive moments.

    Args:
        table: 'duels' (pressing engagements), 'passes' (opponent passes, receiver's marker),
            'runs' (opponent off-ball runs, nearest/assigned defender) or 'recoveries' (after being beaten).
        where_json: JSON object of exact filters, e.g. '{"defending_team": "Sydney FC", "vision": "ball_only",
            "minute": {"min": 60}}'. Lists mean "any of". Text matching is case-insensitive.
        query: optional pandas query string for anything else, e.g. "distance_m < 3 and body_shape == 'square'".
        columns: columns to return (default: a sensible subset).
        sort_by: column to sort by. ascending: sort direction. limit: max rows (<= 40).
    Returns: total count matching and the first rows (with match_id, minute, event_id to open the clip).
    """
    s = store()
    df = s.filter(table, _parse(where_json) or {}, query or None)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    default = {
        "duels": ["match_id", "time", "event_id", "defending_team", "defender", "attacker", "subtype", "phase_oop",
                  "distance_m", "body_shape", "body_angle", "shows", "loaded_stance", "regain", "carrier_outcome"],
        "passes": ["match_id", "time", "event_id", "defending_team", "defender", "passer", "attacker", "third",
                   "vision", "shoulder_checks", "marking_distance", "receiver_pressure", "line_break", "lead_to_shot"],
        "runs": ["match_id", "time", "event_id", "defending_team", "defender", "attacker", "subtype", "blind_side",
                 "body_shape", "reaction_lag_s", "beaten_in_behind", "dangerous", "received"],
        "recoveries": ["match_id", "time", "event_id", "defending_team", "defender", "attacker", "source",
                       "turned_within_1s", "recovered", "time_to_goal_side_s"],
    }[table]
    cols = [c for c in (columns or default) if c in df.columns]
    return {"table": table, "matching": int(len(df)), "rows": _records(df[cols], min(limit, MAX_ROWS))}


def aggregate(table: str, metrics: list[str], group_by: list[str] | None = None, where_json: str = "{}",
              query: str = "", min_n: int = 5, sort_by: str = "", limit: int = 25) -> dict:
    """Average metrics/outcomes (booleans become rates) over moments, optionally grouped.

    Example: aggregate('runs', ['beaten_in_behind', 'dangerous'], ['defending_team', 'body_shape'],
    where_json='{"forward_run": true}'). Always report the 'n' column with any rate.
    """
    s = store()
    df = s.aggregate(table, metrics, group_by, _parse(where_json) or {}, query or None, min_n)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    return {"table": table, "rows": _records(df, limit)}


def player_profiles(team: str = "", sort_by: str = "", ascending: bool = False, min_moments: int = 10,
                    limit: int = 20) -> dict:
    """One row per defender with SEE/SHAPE/STANCE/REACT summary metrics (rates in 0-1, counts as n).
    Filter by team name and sort by any column (e.g. 'blind_side', 'dual_vision', 'regain')."""
    df = store().player_profile(min_moments)
    if team:
        df = df[df.team.str.lower() == team.lower()]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    return {"rows": _records(df, limit)}


def team_profiles() -> dict:
    """One row per team (as defending side): side-on rate, loaded stance, regains, dual vision, shoulder checks,
    receivers left free, blind-side runs conceded, square body at runs, beaten in behind, reaction lag."""
    return {"rows": _records(store().team_profile())}


def framework_evidence(finding_id: str = "") -> dict:
    """The pre-registered comparisons linking each body-pose cue to an outcome, with stratified
    risk differences and 95% bootstrap CIs. finding_id optional (e.g. 'open_body_runs')."""
    df = store().all_evidence()
    if finding_id:
        df = df[df.id == finding_id]
    return {"findings": _records(df, 50), "caveat": "Two matches only: associations, not causal effects."}


def open_moment(table: str, event_id: str, match_id: int) -> dict:
    """Full detail for one moment (all computed pose/tracking features and outcomes)."""
    d = store().moment(table, event_id, match_id)
    return {k: (round(v, 3) if isinstance(v, float) else (v if isinstance(v, (int, str, bool)) else str(v)))
            for k, v in d.items()}


# ------------------------------------------------------------- team shape / match tools
def list_matches() -> dict:
    """The matches available, with ids, titles and the two teams of each."""
    from ..config import MATCHES
    from ..data.load import load_meta
    out = []
    for m in MATCHES:
        meta = load_meta(m)
        out.append({"match_id": m, "title": meta.title, "home": meta.team_name(meta.home_id),
                    "away": meta.team_name(meta.away_id)})
    return {"matches": out}


def match_insights(match_id: int) -> dict:
    """Deterministic defensive report for one match, per team: block type mix (high/medium/low), back-line height,
    block length/width/area, ball-zone compactness (5 defenders nearest the ball), space between lines, free attackers
    near goal, PPDA, counter-pressing, regains, line-breaking passes conceded, and SEE-SHAPE-STANCE-REACT body metrics."""
    from ..shape import insights
    ins = insights(match_id, store())
    return {"bullets": ins["bullets"], "kpis": ins["kpis"], "body": ins["body"]}


def shape_by_phase(match_id: int, team: str = "") -> dict:
    """Median defensive-shape metrics of each team per out-of-possession phase (high_block, medium_block, low_block,
    defending_transition, defending_quick_break, defending_direct, defending_set_play, chaotic), with share of time.
    Distances in metres from own goal; areas in m^2."""
    from ..shape import shape_by_phase as sbp
    df = sbp(match_id)
    if team:
        df = df[df.defending_team.str.lower() == team.lower()]
    return {"rows": _records(df, 40)}


def defensive_state(match_id: int, frame: int) -> dict:
    """Snapshot of the team out of possession at a tracking frame (10 fps; minute*600 approx. + period offset):
    phase, back-line height, block length/width/area, ball-zone compactness, line gap, players behind ball,
    nearest defender to ball, defenders within 5 m of ball, free attackers within 35 m of goal, offside line."""
    from ..shape import state_at
    return state_at(match_id, int(frame))


def moments_window(match_id: int, frame_from: int, frame_to: int) -> dict:
    """All framework moments (duels, passes, runs, recoveries) between two tracking frames of a match,
    with their body-pose features and outcomes. Use to explain what happened in a passage of play."""
    s = store()
    out = {}
    for t in ("duels", "passes", "runs", "recoveries"):
        df = s.tables[t]
        df = df[(df.match_id == match_id) & (df.frame >= frame_from) & (df.frame <= frame_to)]
        keep = [c for c in ("time", "event_id", "defending_team", "defender", "attacker", "subtype", "vision",
                            "shoulder_checks", "body_shape", "body_angle", "shows", "loaded_stance", "blind_side",
                            "reaction_lag_s", "receiver_pressure", "line_break", "beaten_in_behind", "dangerous",
                            "regain", "carrier_outcome", "recovered", "time_to_goal_side_s") if c in df]
        out[t] = _records(df[keep], 15)
    return out


def goals(match_id: int) -> dict:
    """All goals of a match in order: goal_number, scoring team, conceding team, time, tracking frame, scorer,
    assister, score after the goal. Use this FIRST for any question about a goal ('first goal', 'the 3rd goal')."""
    from ..goals import find_goals
    return {"goals": find_goals(match_id)}


def goal_breakdown(match_id: int, goal_number: int) -> dict:
    """How a goal happened and which defenders carry responsibility. Returns the attacking sequence (passes, line
    breaks, pressure on the ball), the conceding team's shape at the assist and at the shot, the situation at the
    shot (nearest defender to the scorer, distance, goal-side, defenders in the shot line) and a ranked list of
    defenders with responsibility points and the concrete reasons (vision, marking distance, beaten in behind,
    blind side, square body, late reaction, duels lost, no recovery). Points are indicators, not a verdict."""
    from ..goals import goal_breakdown as gb
    return gb(int(match_id), int(goal_number), store())


def simulate_goal_prevention(match_id: int, goal_number: int) -> dict:
    """Counterfactual: the safest positions the conceding defenders could physically have reached (within 1.5 s at
    6 m/s) at the assist pass of a goal, with attackers fixed. Returns before/after danger metrics (free attackers,
    distance of the nearest defender to the scorer and to the ball carrier, open shot lane, attack-controlled space near
    goal), % danger reduction, and the moves ranked by impact ('key_fix' = the single most valuable correction)."""
    from ..simulate import simulate_goal
    return simulate_goal(int(match_id), int(goal_number))


def simulate_defence(match_id: int, frame: int) -> dict:
    """Same counterfactual optimisation at any tracking frame (the team out of possession is optimised)."""
    from ..simulate import simulate
    return simulate(int(match_id), int(frame))


def _json_safe(o):
    """NaN/inf -> None, numpy -> Python: Gemini rejects tool results that are not strict JSON."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if o is pd.NaT:
        return None
    return o


def _safe(fn):
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        return _json_safe(fn(*a, **kw))
    return wrapper


TOOLS = [describe_framework, list_columns, find_moments, aggregate, player_profiles, team_profiles,
         framework_evidence, open_moment, list_matches, match_insights, shape_by_phase, defensive_state,
         moments_window, goals, goal_breakdown, simulate_goal_prevention, simulate_defence]
TOOLS = [_safe(f) for f in TOOLS]
TOOL_MAP = {f.__name__: f for f in TOOLS}
