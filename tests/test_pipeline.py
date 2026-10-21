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


def test_posture_coach_and_whatif():
    from dbl.coach import calibration, coach, whatif
    from dbl.goals import find_goals
    m = next(iter(MATCHES))
    cal = calibration()
    assert cal["lag"]["square"] > cal["lag"]["open"] >= 0 and 0.5 <= cal["k_vision"] <= 1 and 0.5 <= cal["k_unloaded"] <= 1
    g = find_goals(m)[0]
    r = coach(m, g["assist_frame"], g["conceding"], g["scorer_id"])
    p = r["reduction_parts"]
    assert r["after"]["total"] <= r["before"]["total"] + 1e-9
    assert p["both"] >= max(p["position_only"], p["posture_only"]) - 0.02  # joint search at least as good
    assert all(-180 <= mv["torso_to"] <= 180 for mv in r["moves"] if mv["pose_known"])
    # what-if: nothing edited = what happened; a far drag is clamped to the reach disc
    base = whatif(m, g["assist_frame"], [], g["scorer_id"], g["conceding"])
    assert abs(base["danger_reduction"]) < 1e-9
    pid = r["moves"][0]["player_id"]
    w = whatif(m, g["assist_frame"], [{"player_id": pid, "x": 0, "y": 0}], g["scorer_id"], g["conceding"])
    reach = next(z for z in w["reach"] if z["player_id"] == pid)
    me = next(z for z in w["players"] if z["player_id"] == pid)
    assert ((me["x"] - reach["center"][0]) ** 2 + (me["y"] - reach["center"][1]) ** 2) ** 0.5 <= reach["radius"] + 0.05


def test_better_defence_replay_respects_physics():
    import numpy as np
    from dbl.replay import replay_goal, V_MAX, A_MAX
    r = replay_goal(next(iter(MATCHES)), 1, seconds=2.0)
    assert r["end"] - r["start"] <= 20 and len(r["frames"]) >= 2
    f0, f1 = r["frames"][0], r["frames"][-1]
    assert all(a["player_id"] == b["player_id"] for a, b in zip(f0["defenders"], f1["defenders"]))
    dt = r["step_s"]
    for k in range(1, len(r["frames"])):
        for a, b in zip(r["frames"][k - 1]["defenders"], r["frames"][k]["defenders"]):
            assert np.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt <= 8.7
    # acceleration limit holds (speed change per step <= A_MAX, small tolerance for clipping at the pitch edge)
    P = np.array([[[d["x"], d["y"]] for d in f["defenders"]] for f in r["frames"]])
    sp = np.linalg.norm(np.diff(P, axis=0), axis=2) / dt
    assert np.all(np.abs(np.diff(sp, axis=0)) / dt <= A_MAX * 1.15)
    assert r["frames"][0]["danger_sim"] == r["frames"][0]["danger_real"]   # starts from what happened
    assert np.mean([f["danger_sim"] for f in r["frames"]]) <= np.mean([f["danger_real"] for f in r["frames"]]) + 1e-6


def test_error_patterns_and_training_plan():
    from dbl.patterns import TAGS, DRILLS, error_patterns, training_plan, coach_moment, _load
    assert set(TAGS) == set(DRILLS)
    from dbl.goals import find_goals
    m = next(iter(MATCHES))
    g = find_goals(m)[0]
    mom, rows = coach_moment(m, g["assist_frame"], g["conceding"], g["scorer_id"])
    assert mom["after"] <= mom["before"] + 1e-9 and all(r["tag"] in TAGS and r["gain"] > 0 for r in rows)
    M, _ = _load()
    if M.empty:   # patterns not built in this environment
        return
    team = M.team.iloc[0]
    pats = error_patterns(team)
    assert pats and all(p["moments"] <= p["dangerous_moments"] for p in pats)
    tp = training_plan(team)
    assert tp["session"] and all(b["drill"]["name"] and b["drill"]["kpi"] for b in tp["session"])


def test_drill_board():
    from dbl.patterns import training_plan, _load
    from dbl.session import drill_board, FORMAT
    M, _ = _load()
    if M.empty:
        return
    team = M[~M.control].team.iloc[0]
    b = training_plan(team)["session"][0]
    e = b["examples"][0]
    d = drill_board(int(e["match_id"]), int(e["frame"]), team, int(e["player_id"]), b["tag"])
    nd, na = FORMAT[b["tag"]]
    assert len(d["defenders"]) == nd and len(d["attackers"]) <= na and any(x["focus"] for x in d["defenders"])
    assert len(d["frames"]) >= 20 and all(len(f["S"]) == nd for f in d["frames"])


def test_validation_statistics():
    from dbl.validate import auc, _logit_fit, _pred
    y = np.array([0, 0, 1, 1], bool)
    assert auc([0.1, 0.2, 0.8, 0.9], y) == 1.0 and auc([0.9, 0.8, 0.2, 0.1], y) == 0.0
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000); yy = rng.random(2000) < 1 / (1 + np.exp(-(0.5 + 2 * x)))
    w = _logit_fit(x[:, None], yy.astype(float))
    assert abs(w[1] - 2) < 0.4 and 0 < _pred(w, np.array([[0.0]]))[0] < 1


def test_coverage_is_part_of_the_score():
    from dbl.coach import PostureScene
    from dbl.goals import find_goals
    m = next(iter(MATCHES))
    g = find_goals(m)[0]
    sc = PostureScene(m, g["assist_frame"], g["conceding"], g["scorer_id"])
    cov = sc.coverage()
    assert cov.shape == (len(sc.A),) and np.all((cov >= 0) & (cov <= 1))
    parts = sc.score(sc.D.x.to_numpy(), sc.D.y.to_numpy(), parts=True)
    assert abs(parts["marking_danger"] - float((sc.w * (1 - cov)).sum())) < 1e-9
