"""Integration tests: run only when the data has been built (`python -m dbl all`)."""
import numpy as np
import pytest

from dbl.config import BUILD_DIR, MATCHES

built = all((BUILD_DIR / str(m) / "runs.parquet").exists() for m in MATCHES)
pytestmark = pytest.mark.skipif(not built, reason="data not built")


@pytest.fixture(scope="module")
def store():
    from dbl.store import MomentStore
    return MomentStore()


def test_tables_have_pose_coverage(store):
    assert store.tables["duels"].body_shape.notna().mean() > 0.8
    assert store.tables["runs"].blind_side.notna().mean() > 0.6


def test_torso_matches_running_direction():
    """Body pose sanity check: a player sprinting faces where he runs."""
    import pandas as pd
    from dbl.geometry import angdiff
    m = next(iter(MATCHES))
    p = pd.read_parquet(BUILD_DIR / str(m) / "pose_features.parquet", columns=["frame10", "player_id", "torso_deg"])
    xy = pd.read_parquet(BUILD_DIR / str(m) / "players_xy.parquet", columns=["frame", "player_id", "vx", "vy", "speed"])
    p["frame"] = p.frame10.round().astype(int)
    j = p.merge(xy[xy.speed > 5], on=["frame", "player_id"])
    d = np.abs(angdiff(j.torso_deg, np.degrees(np.arctan2(j.vy, j.vx))))
    assert np.nanmedian(d) < 20


def test_filter_and_aggregate(store):
    team = store.teams[0]
    df = store.filter("runs", {"defending_team": team.lower(), "forward_run": True})
    assert len(df) and (df.defending_team == team).all()
    agg = store.aggregate("runs", ["beaten_in_behind"], ["body_shape"], {"forward_run": True})
    assert set(agg.columns) >= {"body_shape", "beaten_in_behind", "n"}


def test_offline_agent_answers(store):
    from dbl.query.agent import ask_offline
    a = ask_offline("who checks their shoulder most at " + store.teams[0])
    assert a.data and a.data.get("rows")


def test_team_shape_is_plausible():
    from dbl.shape import load_shape, insights
    m = next(iter(MATCHES))
    s = load_shape(m)
    org = s[s.phase_oop.isin(["high_block", "medium_block", "low_block"])]
    assert 15 < org.line_height.median() < 60          # back line 15-60 m from own goal
    assert 15 < org.length.median() < 45               # block depth
    assert (org.area > 0).mean() > 0.95
    # a high block should sit higher than a low block
    med = org.groupby("phase_oop").line_height.median()
    assert med["high_block"] > med["low_block"]
    ins = insights(m)
    assert len(ins["bullets"]) == 2


def test_viewer_api_contract():
    import struct
    from dbl.viewer.server import available_matches, match_meta, pose_window, track_window
    ms = available_matches()
    if not ms:
        pytest.skip("full-match skeletons not built")
    meta = match_meta(ms[0])
    assert {"teams", "players", "periods", "bones", "events", "home_side"} <= set(meta)
    f = meta["periods"][0]["start"] + 3000
    n, nj = struct.unpack("<ii", pose_window(ms[0], f, f + 100)[:8])
    assert nj == len(meta["joints"]) and n > 0
    assert len(track_window(ms[0], f, f + 100)["players"]) >= 20


def test_goals_match_the_match_sheet():
    from dbl.data.load import load_meta
    from dbl.goals import find_goals, goal_breakdown
    for m in MATCHES:
        meta = load_meta(m)
        gs = find_goals(m)
        assert len(gs) == meta.raw["home_team_score"] + meta.raw["away_team_score"]
        scorers = {p["short_name"] for p in meta.raw["players"] if p.get("goal")}
        assert {g["scorer"] for g in gs} <= scorers
    b = goal_breakdown(next(iter(MATCHES)), 1)
    assert b["sequence"] and b["sequence"][-1]["action"] == "shot"
    assert b["responsibility"] and all(r["reasons"] for r in b["responsibility"])


def test_simulation_reduces_danger_within_reach():
    from dbl.simulate import simulate_goal
    r = simulate_goal(next(iter(MATCHES)), 1)
    assert r["after"]["total"] <= r["before"]["total"]
    assert all(m["move_m"] <= r["max_move_m"] + 9.5 for m in r["moves"])  # reach disc is centred 1.5 s earlier
    assert r["key_fix"] is None or r["key_fix"]["gain"] >= max(m["gain"] for m in r["moves"]) - 1e-9
