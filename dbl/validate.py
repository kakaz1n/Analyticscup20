"""Does the simulation's danger score mean anything?  Validation against what really happened.

Every reduction the coach and the replay report ("−56 % danger") is a change in the same score
(``PostureScene.score``: marking + 1.2·lane + 2·space).  If that score did not track real chances, the
simulations would be optimising noise.  This module tests it on the two matches:

* **Sample.**  Every in-possession second of both matches (one frame per second, ball in play), with the
  defending team taken from Phases of Play.
* **Outcome.**  The team in possession takes a shot within the next 10 s (Dynamic Events: player possession
  ending in a shot).
* **Baseline.**  Ball location alone (the same zone value the score uses).  The question is whether the
  *defensive configuration* adds information beyond where the ball is.
* **Out-of-match test.**  Fit on one match, score the other (both directions); AUC with a 95 % bootstrap CI that
  resamples whole possessions.
* **Calibration.**  A logistic map from danger to P(shot within 10 s), fitted on both matches and saved to
  ``data/build/danger_calibration.json``.  The replay and the reports use it to express a reduction as a change
  in shot probability.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import BUILD_DIR, MATCHES, RAW_DIR

HORIZON_S = 10
STEP = 10  # tracking frames are 10 Hz -> one sample per second


def _shots(match_id: int) -> pd.DataFrame:
    e = pd.read_csv(RAW_DIR / str(match_id) / f"{match_id}_dynamic_events.csv", low_memory=False)
    s = e[(e.event_type == "player_possession") & (e.end_type == "shot")]
    return s[["frame_end", "team_id"]].astype(int).rename(columns={"frame_end": "frame"})


def _sample_frames(match_id: int) -> list[tuple[int, int, int]]:
    from .simulate import _data
    _, fr, xy, ph = _data(match_id)
    have = set(xy.index.unique())
    out = []
    for k, p in enumerate(ph.itertuples()):
        for f in range(int(p.frame_start), int(p.frame_end) + 1, STEP):
            if f in have and pd.notna(fr.loc[f, "ball_x"]):
                out.append((f, int(p.team_in_possession_id), k))
    return out


def _one(args):
    match_id, f, poss, k = args
    from .coach import PostureScene
    try:
        sc = PostureScene(match_id, f)
    except Exception:  # noqa: BLE001
        return None
    if sc.ball is None or len(sc.D) < 5 or len(sc.A) < 5:
        return None
    from .simulate import _value
    p = sc.score(sc.D.x.to_numpy(), sc.D.y.to_numpy(), parts=True)
    return dict(match_id=match_id, frame=f, poss_team=poss, phase=k, danger=p["total"],
                marking=p["marking_danger"], lane=p["lane_open"], space=p["space_share"],
                ball_value=float(_value(sc.ball[0], sc.ball[1], sc.L)), ball_x=float(sc.ball[0]))


def build_samples(match_id: int, workers: int = 2, force: bool = False) -> pd.DataFrame:
    out = BUILD_DIR / str(match_id) / "danger_samples.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    jobs = [(match_id, f, t, k) for f, t, k in _sample_frames(match_id)]
    if workers > 1:
        import multiprocessing as mp
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(workers) as pool:
            rows = list(pool.imap(_one, jobs, chunksize=32))
    else:
        rows = [_one(j) for j in jobs]
    df = pd.DataFrame([r for r in rows if r])
    sh = _shots(match_id)
    lab = np.zeros(len(df), bool)
    for t, g in sh.groupby("team_id"):
        fs = np.sort(g.frame.to_numpy())
        m = df.poss_team.to_numpy() == t
        f = df.frame.to_numpy()[m]
        j = np.searchsorted(fs, f, side="right")          # first shot strictly after f
        nxt = np.where(j < len(fs), fs[np.minimum(j, len(fs) - 1)], 10 ** 9)
        lab[m] = (nxt - f) <= HORIZON_S * 10
    df["shot10"] = lab
    df.to_parquet(out)
    return df


# ------------------------------------------------------------------ statistics
def auc(score, y) -> float:
    score, y = np.asarray(score, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _logit_fit(X, y, l2=1e-3, iters=50):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):  # Newton / IRLS
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def _pred(w, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-X @ w))


def _boot_auc(df, col, n=400, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(["match_id", "phase"])]
    vals = []
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(auc(s[col], s.shot10))
    return np.nanpercentile(vals, [2.5, 97.5]).round(3).tolist()


FEATS = {"ball": ["ball_value"], "ball+danger": ["ball_value", "danger"], "danger": ["danger"]}


def validate(workers: int = 2, force: bool = False) -> dict:
    df = pd.concat([build_samples(m, workers, force) for m in MATCHES], ignore_index=True)
    res = {"n_seconds": int(len(df)), "n_positive": int(df.shot10.sum()), "horizon_s": HORIZON_S,
           "in_sample_auc": {}, "out_of_match_auc": {}}
    for c in ("ball_value", "danger", "marking", "lane", "space"):
        res["in_sample_auc"][c] = dict(auc=round(auc(df[c], df.shot10), 3), ci=_boot_auc(df, c))
    # out-of-match: fit on one, test on the other (pooled predictions)
    for name, cols in FEATS.items():
        pred = np.zeros(len(df))
        for m in MATCHES:
            tr, te = df.match_id != m, df.match_id == m
            mu, sd = df.loc[tr, cols].mean(), df.loc[tr, cols].std() + 1e-9
            w = _logit_fit(((df.loc[tr, cols] - mu) / sd).to_numpy(), df.loc[tr, "shot10"].to_numpy(float))
            pred[te.to_numpy()] = _pred(w, ((df.loc[te, cols] - mu) / sd).to_numpy())
        df["_p"] = pred
        res["out_of_match_auc"][name] = dict(auc=round(auc(pred, df.shot10), 3), ci=_boot_auc(df, "_p"))
    # same ball zone, different defence: within ball-value quintiles, is higher danger more shots?
    df["bq"] = pd.qcut(df.ball_value, 5, labels=False, duplicates="drop")
    strat = []
    for q, g in df.groupby("bq"):
        hi = g.danger > g.danger.median()
        strat.append(dict(ball_quintile=int(q) + 1, n=int(len(g)),
                          shot_rate_low_danger=round(float(g.shot10[~hi].mean()), 3),
                          shot_rate_high_danger=round(float(g.shot10[hi].mean()), 3)))
    res["within_ball_zone"] = strat
    # calibration danger -> P(shot in 10 s), all data
    w = _logit_fit(df[["danger"]].to_numpy(), df.shot10.to_numpy(float))
    res["calibration"] = {"intercept": float(w[0]), "slope": float(w[1]),
                          "note": "P(shot within 10 s) = 1/(1+exp(-(intercept + slope·danger)))"}
    df["dq"] = pd.qcut(df.danger, 10, labels=False, duplicates="drop")
    res["reliability"] = [dict(decile=int(q) + 1, danger=round(float(g.danger.mean()), 3),
                               observed=round(float(g.shot10.mean()), 3),
                               predicted=round(float(_pred(w, g[["danger"]].to_numpy()).mean()), 3))
                          for q, g in df.groupby("dq")]
    (BUILD_DIR / "danger_calibration.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    _cal.cache_clear()
    return res


@lru_cache(maxsize=1)
def _cal():
    p = BUILD_DIR / "danger_calibration.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["calibration"]


def p_shot(danger) -> float | None:
    """Calibrated probability that the attack shoots within 10 s, given the danger score (None if not built)."""
    c = _cal()
    if c is None or danger is None:
        return None
    return float(1 / (1 + np.exp(-(c["intercept"] + c["slope"] * float(danger)))))


# ------------------------------------------------------------------ attacker-level test of the marking term
def _options(match_id: int) -> pd.DataFrame:
    e = pd.read_csv(RAW_DIR / str(match_id) / f"{match_id}_dynamic_events.csv", low_memory=False)
    pp = e[e.event_type == "player_possession"].set_index("event_id")
    po = e[(e.event_type == "passing_option") & (e.passing_option_at_pass_moment == True)].copy()  # noqa: E712
    po["pass_frame"] = po.associated_player_possession_event_id.map(pp.frame_end)
    po = po.dropna(subset=["pass_frame"])
    po["pass_frame"] = po.pass_frame.astype(int)
    return po[["associated_player_possession_event_id", "pass_frame", "player_id", "targeted", "received",
               "xpass_completion", "xthreat", "dangerous"]].rename(columns={"associated_player_possession_event_id": "pid_event"})


def _cov_one(args):
    match_id, frame, ids = args
    from .coach import PostureScene
    try:
        sc = PostureScene(match_id, frame)
    except Exception:  # noqa: BLE001
        return []
    cov = sc.coverage()
    known = sc.known.copy()
    sc.known[:] = False                                   # same scene, posture switched off
    cov_pos = sc.coverage()
    sc.known = known
    idx = {int(p): j for j, p in enumerate(sc.A.player_id)}
    return [dict(match_id=match_id, pass_frame=frame, player_id=p, cov=float(cov[idx[p]]),
                 cov_pos=float(cov_pos[idx[p]]), pose_known=int(known.sum())) for p in ids if p in idx]


def build_option_cover(match_id: int, workers: int = 2, force: bool = False) -> pd.DataFrame:
    out = BUILD_DIR / str(match_id) / "option_cover.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    po = _options(match_id)
    jobs = [(match_id, int(f), [int(p) for p in g.player_id]) for f, g in po.groupby("pass_frame")]
    if workers > 1:
        import multiprocessing as mp
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(workers) as pool:
            rows = [r for rs in pool.imap(_cov_one, jobs, chunksize=16) for r in rs]
    else:
        rows = [r for j in jobs for r in _cov_one(j)]
    df = po.merge(pd.DataFrame(rows), on=["pass_frame", "player_id"])
    df.to_parquet(out)
    return df


def _within_auc(df, col, n_boot=400, seed=0):
    """P(the targeted option has a higher `col` than an untargeted one at the same pass), with group bootstrap CI."""
    pairs = []
    for _, g in df.groupby(["match_id", "pid_event"]):
        t, u = g[g.targeted][col].to_numpy(), g[~g.targeted][col].to_numpy()
        if len(t) == 1 and len(u):
            pairs.append(float(np.mean((t[0] > u) + 0.5 * (t[0] == u))))
    pairs = np.array(pairs)
    rng = np.random.default_rng(seed)
    bs = [pairs[rng.integers(0, len(pairs), len(pairs))].mean() for _ in range(n_boot)]
    return dict(auc=round(float(pairs.mean()), 3), ci=np.percentile(bs, [2.5, 97.5]).round(3).tolist(), n=len(pairs))


def validate_marking(workers: int = 2, force: bool = False) -> dict:
    df = pd.concat([build_option_cover(m, workers, force) for m in MATCHES], ignore_index=True)
    df["free"], df["free_pos"] = 1 - df["cov"], 1 - df["cov_pos"]
    df["targeted"], df["received"] = df.targeted.astype(bool), df.received.astype(bool)
    res = {"n_options": int(len(df)), "n_passes": int(df.groupby(["match_id", "pid_event"]).ngroups)}
    # 1) who gets the ball: within the same pass, is the targeted option freer?
    res["targeted_within_pass"] = {c: _within_auc(df, c) for c in ("free", "free_pos", "xpass_completion", "xthreat")}
    # 2) does the pass arrive: completion of targeted passes, out-of-match logistic
    t = df[df.targeted].reset_index(drop=True)
    comp = {}
    for name, cols in {"xpass (SkillCorner)": ["xpass_completion"], "free (model)": ["free"],
                       "xpass + free": ["xpass_completion", "free"], "xpass + free_pos": ["xpass_completion", "free_pos"]}.items():
        pred = np.zeros(len(t))
        for m in MATCHES:
            tr, te = (t.match_id != m).to_numpy(), (t.match_id == m).to_numpy()
            X = t[cols]
            mu, sd = X[tr].mean(), X[tr].std() + 1e-9
            w = _logit_fit(((X[tr] - mu) / sd).to_numpy(), t.received[tr].to_numpy(float))
            pred[te] = _pred(w, ((X[te] - mu) / sd).to_numpy())
        comp[name] = round(auc(pred, t.received), 3)
    res["completion_out_of_match_auc"] = comp
    res["n_targeted"] = int(len(t))
    # 3) completion by model coverage tercile (descriptive)
    t["tercile"] = pd.qcut(t["cov"], 3, labels=["free", "half", "tight"])
    res["completion_by_cover"] = {str(k): round(float(g.received.mean()), 3) for k, g in t.groupby("tercile", observed=True)}
    res["corr_free_vs_xpass"] = round(float(df[["free", "xpass_completion"]].corr(method="spearman").iloc[0, 1]), 3)
    return res


# ------------------------------------------------------------------ is the simulated movement physically plausible?
def validate_replay() -> dict:
    """Run the better-defence replay on every goal and compare the simulated defenders' kinematics with the real
    defenders over the same 5 s (same players): speed, acceleration and distance run."""
    from .goals import find_goals
    from .replay import replay_goal
    from .simulate import _data
    sim_v, sim_a, real_v, real_a, ran_s, ran_r, goals = [], [], [], [], [], [], []
    for m in MATCHES:
        _, _, xy, _ = _data(m)
        for g in find_goals(m):
            r = replay_goal(m, g["goal_number"])
            dt = r["step_s"]
            F = r["frames"]
            ids = [d["player_id"] for d in F[0]["defenders"]]
            P = np.array([[[d["x"], d["y"]] for d in f["defenders"]] for f in F])        # T x N x 2 (normalised)
            v = np.linalg.norm(np.diff(P, axis=0), axis=2) / dt
            a = np.abs(np.diff(v, axis=0)) / dt
            sim_v.append(v.ravel()); sim_a.append(a.ravel()); ran_s.append(v.sum(0) * dt)
            fr = [f["frame"] for f in F]
            R = xy[xy.index.isin(fr) & xy.player_id.isin(ids)].reset_index()
            Rp = R.pivot_table(index="frame", columns="player_id", values=["x", "y"]).reindex(fr)
            if Rp.isna().any().any():
                Rp = Rp.interpolate(limit_direction="both")
            rx, ry = Rp["x"][ids].to_numpy(), Rp["y"][ids].to_numpy()
            rv = np.hypot(np.diff(rx, axis=0), np.diff(ry, axis=0)) / dt
            ra = np.abs(np.diff(rv, axis=0)) / dt
            real_v.append(rv.ravel()); real_a.append(ra.ravel()); ran_r.append(rv.sum(0) * dt)
            goals.append(dict(match_id=m, goal=g["goal_number"], conceding=g["conceding"],
                              end_reduction=r["end_reduction"], mean_reduction=r["mean_reduction"],
                              ran_sim_m=round(float(np.mean(v.sum(0) * dt)), 1), ran_real_m=round(float(np.mean(rv.sum(0) * dt)), 1)))
    sv, sa, rv, ra = map(np.concatenate, (sim_v, sim_a, real_v, real_a))
    q = lambda x: {f"p{p}": round(float(np.percentile(x, p)), 2) for p in (50, 90, 99)} | {"max": round(float(x.max()), 2)}
    # sensitivity of the headline numbers to the one free control weight (velocity-change cost)
    from . import replay as R
    keep, sens = R.SMOOTH, []
    try:
        for w in (0.0, 0.002, R.SMOOTH, 0.01):
            R.SMOOTH = w
            R.replay.cache_clear()
            row = {"smooth": w}
            for m in MATCHES:
                for g in find_goals(m):
                    r = replay_goal(m, g["goal_number"])
                    row[f"{g['conceding']} conceding, goal {g['goal_number']} ({m})"] = [r["end_reduction"], r["mean_reduction"]]
            sens.append(row)
    finally:
        R.SMOOTH = keep
        R.replay.cache_clear()
    return {"speed_mps": {"simulated": q(sv), "real": q(rv)}, "accel_mps2": {"simulated": q(sa), "real": q(ra)},
            "sensitivity": sens,
            "distance_5s_m": {"simulated_mean": round(float(np.mean(np.concatenate(ran_s))), 1),
                              "real_mean": round(float(np.mean(np.concatenate(ran_r))), 1)},
            "goals": goals,
            "note": "Kinematics from 0.2 s steps (finite differences); real = the same defenders in the same 5 s."}
