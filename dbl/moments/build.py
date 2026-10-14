"""Turn SkillCorner Dynamic Events into defensive *moments* enriched with body pose.

Four moment families, one per coaching question:

* ``duels``      -- on-ball engagements: was the presser side-on, low, showing the
                    carrier the right way? (SHAPE + STANCE)
* ``passes``     -- every opponent pass: did the receiver's marker see ball AND man,
                    and did he check his shoulder beforehand? (SEE)
* ``runs``       -- every opponent off-ball run: did the nearest defender see the
                    runner, was his body open, how fast did he react? (SEE + SHAPE + REACT)
* ``recoveries`` -- after a defender is beaten: did he turn and get back goal-side? (REACT)

All geometry is in the defender-normalised frame (own goal at x = -L/2).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..config import BUILD_DIR, TH, TRACKING_FPS
from ..geometry import angdiff, bearing
from .context import MatchContext

GOAL_WARD = 180.0  # in the normalised frame the own goal is in the -x direction


def _bool(v) -> bool:
    return bool(v) if pd.notna(v) else False


def _phase_cols(r: pd.Series) -> dict:
    return {
        "match_id": int(r.match_id),
        "period": int(r.period),
        "frame": int(r.frame_start),
        "minute": int(r.minute_start),
        "second": int(r.second_start),
        "phase_oop": r.get("team_out_of_possession_phase_type"),
        "phase_ip": r.get("team_in_possession_phase_type"),
        "third": r.get("third_start"),
        "channel": r.get("channel_start"),
        "game_state": r.get("game_state"),
    }


def _shoulder_checks(pw, ball: pd.DataFrame, dx: float, dy: float) -> tuple[int, float]:
    """Count head turns away from the ball (> check_away_deg for >= check_min_frames).
    Returns (n_checks, share_of_window_looking_at_ball)."""
    if pw is None or pw.n < 5 or ball.empty:
        return 0, np.nan
    bf = ball.index.to_numpy(float)
    bx = np.interp(pw.frame10_series, bf, ball.ball_x.to_numpy(float))
    by = np.interp(pw.frame10_series, bf, ball.ball_y.to_numpy(float))
    # defender position approximated as constant over the window (head turns are fast)
    away = np.abs(angdiff(pw.head_series, bearing(dx, dy, bx, by)))
    valid = ~np.isnan(away)
    if valid.sum() < 5:
        return 0, np.nan
    flag = (away > TH.check_away_deg) & valid
    n, run = 0, 0
    for f in flag:
        run = run + 1 if f else 0
        if run == TH.check_min_frames:
            n += 1
    return n, float((away[valid] <= TH.fov_half_deg).mean())


# ---------------------------------------------------------------------- duels
def build_duels(ctx: MatchContext) -> pd.DataFrame:
    ev = ctx.events
    rows = []
    poss = ev[ev.event_type == "player_possession"].set_index("event_id")
    for _, r in ev[ev.event_type == "on_ball_engagement"].iterrows():
        dfd, car, team, per, f0 = int(r.player_id), r.player_in_possession_id, int(r.team_id), int(r.period), int(r.frame_start)
        if pd.isna(car):
            continue
        car = int(car)
        tr_d = ctx.track(dfd, f0 - 10, f0, team, per)
        tr_c = ctx.track(car, f0 - 10, f0, team, per)
        if f0 not in tr_d.index or f0 not in tr_c.index:
            continue
        d, c = tr_d.loc[f0], tr_c.loc[f0]
        dist = float(np.hypot(c.x - d.x, c.y - d.y))
        phi = float(bearing(d.x, d.y, c.x, c.y))
        w = TH.engage_window_s * TRACKING_FPS
        pw = ctx.pose_window(dfd, f0 - w, f0 + w / 2, team, per)
        out = {
            **_phase_cols(r), "event_id": r.event_id, "moment": "duel",
            "subtype": r.event_subtype, "defending_team": ctx.meta.team_name(team),
            "attacking_team": ctx.meta.team_name(ctx.meta.away_id if team == ctx.meta.home_id else ctx.meta.home_id),
            "defender_id": dfd, "defender": ctx.name_of.get(dfd), "defender_role": ctx.role_of.get(dfd),
            "attacker_id": car, "attacker": ctx.name_of.get(car),
            "x": float(d.x), "y": float(d.y), "carrier_x": float(c.x), "carrier_y": float(c.y),
            "distance_m": dist, "def_speed": float(d.speed),
            "approach_decel": float(tr_d.speed.max() - d.speed),
            "closing_speed": float(-np.diff(np.hypot(tr_c.x.reindex(tr_d.index) - tr_d.x,
                                                      tr_c.y.reindex(tr_d.index) - tr_d.y).dropna()).mean() * TRACKING_FPS)
            if len(tr_d) > 2 else np.nan,
            "goal_side": _bool(r.get("goal_side_start")),
            "pose_frames": 0 if pw is None else pw.n,
        }
        if pw is not None and not np.isnan(pw.torso):
            rel = float(angdiff(pw.torso, phi))
            out["body_angle"] = abs(rel)
            out["body_shape"] = ("square" if abs(rel) < TH.side_on_min_deg
                                 else "side_on" if abs(rel) <= TH.side_on_max_deg else "turned")
            # the side the chest opens toward is the side the carrier is shown
            s = np.sign(rel)
            lat = s * np.cos(np.radians(phi))
            if abs(lat) > 0.3 and abs(rel) >= TH.side_on_min_deg:
                out["shows"] = "outside" if np.sign(lat) == np.sign(c.y) else "inside"
            else:
                out["shows"] = "none"
            out["knee_deg"] = pw.knee
            dh, dk = ctx.crouch(dfd, float(d.speed), pw.hip_h, pw.knee)
            out["hip_drop_m"], out["knee_extra_deg"] = dh, dk
            out["stagger_m"] = pw.stagger
            if not (np.isnan(dh) and np.isnan(dk)):
                out["loaded_stance"] = bool((dh >= TH.ready_hip_drop_m) if not np.isnan(dh) else False) or \
                    bool((dk >= TH.ready_knee_extra_deg) if not np.isnan(dk) else False)
            out["eyes_on_ball"] = bool(abs(angdiff(pw.head, phi)) <= TH.fov_half_deg) if not np.isnan(pw.head) else np.nan
        et = r.get("end_type")
        out.update({
            "beaten": _bool(r.get("beaten_by_possession")) or _bool(r.get("beaten_by_movement")),
            "beaten_by_possession": _bool(r.get("beaten_by_possession")),
            "beaten_by_movement": _bool(r.get("beaten_by_movement")),
            "regain": et in ("direct_regain", "indirect_regain"),
            "disruption": et in ("direct_disruption", "indirect_disruption"),
            "foul": et == "foul_committed",
            "danger_stopped": _bool(r.get("stop_possession_danger")) or _bool(r.get("reduce_possession_danger")),
            "forced_backward": _bool(r.get("force_backward")),
            "pressing_chain_end": r.get("pressing_chain_end_type"),
            "frame_end": int(r.frame_end),
        })
        # what the pressed carrier managed to do with the ball
        pid = r.get("associated_player_possession_event_id")
        if pd.notna(pid) and pid in poss.index:
            c = poss.loc[pid]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[0]
            end = c.get("end_type")
            fwd = c.get("pass_direction") == "forward"
            lb = _bool(c.get("first_line_break")) or _bool(c.get("second_last_line_break")) or _bool(c.get("last_line_break"))
            out["carrier_outcome"] = ("lost_ball" if end == "possession_loss" or out["regain"]
                                      else "shot" if end == "shot"
                                      else "line_break_pass" if end == "pass" and lb
                                      else "forward_pass" if end == "pass" and fwd
                                      else "safe_pass" if end == "pass" else str(end))
            out["carrier_progressed"] = out["carrier_outcome"] in ("shot", "line_break_pass", "forward_pass")
            out["carrier_pressure"] = c.get("overall_pressure_end")
        rows.append(out)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- passes
def build_passes(ctx: MatchContext) -> pd.DataFrame:
    ev = ctx.events
    pp = ev[ev.event_type == "player_possession"].sort_values("frame_start")
    passes = pp[(pp.end_type == "pass") & pp.player_targeted_id.notna()]
    rows = []
    for _, r in passes.iterrows():
        per, fp = int(r.period), int(r.frame_end)
        att_team = int(r.team_id)
        team = ctx.meta.away_id if att_team == ctx.meta.home_id else ctx.meta.home_id
        recv = int(r.player_targeted_id)
        snap = ctx.players_at(fp, team, per)
        if snap.empty or recv not in set(snap.player_id):
            continue
        bx, by = ctx.ball_at(fp, team, per)
        marks = ctx.assign_markers(snap, TH.marker_max_dist_m)
        marker = marks.get(recv)
        rv = snap[snap.player_id == recv].iloc[0]
        # next possession by the receiver (reception quality)
        nxt = pp[(pp.frame_start >= fp) & (pp.frame_start <= fp + 40) & (pp.player_id == recv)].head(1)
        out = {
            **_phase_cols(r), "frame": fp, "event_id": r.event_id, "moment": "pass",
            "defending_team": ctx.meta.team_name(team), "attacking_team": ctx.meta.team_name(att_team),
            "passer": r.player_name, "attacker_id": recv, "attacker": ctx.name_of.get(recv),
            "receiver_x": float(rv.x), "receiver_y": float(rv.y),
            "pass_success": r.get("pass_outcome") == "successful",
            "line_break": _bool(r.get("first_line_break")) or _bool(r.get("second_last_line_break")) or _bool(r.get("last_line_break")),
            "last_line_break": _bool(r.get("last_line_break")),
            "lead_to_shot": _bool(r.get("lead_to_shot")),
            "pass_xthreat": r.get("player_targeted_xthreat"),
            "dangerous": _bool(r.get("player_targeted_dangerous")),
            "receiver_pressure": nxt.overall_pressure_start.iloc[0] if len(nxt) else None,
            "receiver_pressure_id": nxt.overall_pressure_start_id.iloc[0] if len(nxt) else np.nan,
            "receiver_epv_gain": nxt.possession_epv_delta_for.iloc[0] if len(nxt) else np.nan,
            "marker_id": marker,
        }
        # team-level awareness: share of outfield defenders with pose who have the ball in view
        seen = []
        for _, d in snap[snap.defending & ~snap.is_gk].iterrows():
            pw = ctx.pose_window(int(d.player_id), fp - 2, fp + 1, team, per)
            if pw is not None and not np.isnan(pw.head):
                seen.append(abs(angdiff(pw.head, bearing(d.x, d.y, bx, by))) <= TH.fov_half_deg)
        out["team_ball_awareness"] = float(np.mean(seen)) if seen else np.nan
        out["team_pose_n"] = len(seen)
        if marker is not None:
            m = snap[snap.player_id == marker].iloc[0]
            out.update({"defender_id": marker, "defender": ctx.name_of.get(marker), "defender_role": ctx.role_of.get(marker),
                        "x": float(m.x), "y": float(m.y),
                        "marking_distance": float(np.hypot(rv.x - m.x, rv.y - m.y)),
                        "goal_side": bool(m.x < rv.x)})
            pw = ctx.pose_window(marker, fp - 2, fp + 1, team, per)
            if pw is not None and not np.isnan(pw.head):
                phb, phm = bearing(m.x, m.y, bx, by), bearing(m.x, m.y, rv.x, rv.y)
                ab, am = abs(float(angdiff(pw.head, phb))), abs(float(angdiff(pw.head, phm)))
                out.update({
                    "pose_frames": pw.n,
                    "head_to_ball_deg": ab, "head_to_man_deg": am,
                    "ball_man_split_deg": abs(float(angdiff(phb, phm))),
                    "ball_in_view": ab <= TH.fov_half_deg, "man_in_view": am <= TH.fov_half_deg,
                })
                out["vision"] = ("ball_and_man" if out["ball_in_view"] and out["man_in_view"]
                                 else "ball_only" if out["ball_in_view"]
                                 else "man_only" if out["man_in_view"] else "neither")
                # shoulder checks in the look-back window
                f_start = fp - TH.scan_window_s * TRACKING_FPS
                pwl = ctx.pose_window(marker, f_start, fp, team, per)
                ball = ctx.ball_series(int(f_start), fp, team, per)
                n, share = _shoulder_checks(pwl, ball, m.x, m.y)
                out["shoulder_checks"] = n
                out["ball_watch_share"] = share
        rows.append(out)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- runs
def build_runs(ctx: MatchContext) -> pd.DataFrame:
    ev = ctx.events
    rows = []
    for _, r in ev[ev.event_type == "off_ball_run"].iterrows():
        per, f0, runner = int(r.period), int(r.frame_start), int(r.player_id)
        att_team = int(r.team_id)
        team = ctx.meta.away_id if att_team == ctx.meta.home_id else ctx.meta.home_id
        snap = ctx.players_at(f0, team, per)
        if snap.empty or runner not in set(snap.player_id):
            continue
        rn = snap[snap.player_id == runner].iloc[0]
        defs = snap[snap.defending & ~snap.is_gk]
        if defs.empty:
            continue
        dd = np.hypot(defs.x - rn.x, defs.y - rn.y)
        marks = ctx.assign_markers(snap, TH.marker_max_dist_m)
        dfd = marks.get(runner, int(defs.player_id.iloc[int(np.argmin(dd))]))
        d = snap[snap.player_id == dfd].iloc[0]
        bx, by = ctx.ball_at(f0, team, per)
        out = {
            **_phase_cols(r), "event_id": r.event_id, "moment": "run", "subtype": r.event_subtype,
            "defending_team": ctx.meta.team_name(team), "attacking_team": ctx.meta.team_name(att_team),
            "attacker_id": runner, "attacker": ctx.name_of.get(runner),
            "defender_id": dfd, "defender": ctx.name_of.get(dfd), "defender_role": ctx.role_of.get(dfd),
            "x": float(d.x), "y": float(d.y), "runner_x": float(rn.x), "runner_y": float(rn.y),
            "runner_distance": float(np.hypot(rn.x - d.x, rn.y - d.y)),
            "runner_goal_side_of_defender": bool(rn.x < d.x),
            "targeted": _bool(r.get("targeted")), "received": _bool(r.get("received")),
            "dangerous": _bool(r.get("dangerous")), "lead_to_shot": _bool(r.get("lead_to_shot")),
            "separation_gain": r.get("separation_gain"),
            "break_line": _bool(r.get("break_defensive_line")),
        }
        pw = ctx.pose_window(dfd, f0 - 3, f0 + 1, team, per)
        if pw is not None and not np.isnan(pw.head):
            hr = abs(float(angdiff(pw.head, bearing(d.x, d.y, rn.x, rn.y))))
            hb = abs(float(angdiff(pw.head, bearing(d.x, d.y, bx, by))))
            out.update({"pose_frames": pw.n, "head_to_runner_deg": hr, "head_to_ball_deg": hb,
                        "runner_in_view": hr <= TH.fov_half_deg,
                        "blind_side": (hr > TH.fov_half_deg) and (hb <= TH.fov_half_deg)})
        if pw is not None and not np.isnan(pw.torso):
            # 0 = facing own goal (turned), 90 = side-on / open, 180 = facing play
            tg = abs(float(angdiff(pw.torso, GOAL_WARD)))
            out["torso_to_goal_deg"] = tg
            out["body_shape"] = "turned" if tg < TH.open_body_min_deg else "open" if tg <= 135 else "square"
        # did the runner get goal-side of his defender by the end of the run?
        f1 = int(r.frame_end)
        e_snap = ctx.players_at(f1, team, per)
        if not e_snap.empty and {runner, dfd} <= set(e_snap.player_id):
            rx = float(e_snap.loc[e_snap.player_id == runner, "x"].iloc[0])
            dx = float(e_snap.loc[e_snap.player_id == dfd, "x"].iloc[0])
            out["beaten_in_behind"] = bool(rn.x >= d.x and rx < dx)
        out["frame_end"] = f1
        # reaction: first frame the defender moves goal-ward fast, relative to the runner
        horizon = int(TH.react_max_s * TRACKING_FPS)
        td = ctx.track(dfd, f0, f0 + horizon, team, per)
        trn = ctx.track(runner, f0, f0 + horizon, team, per)
        if len(td) and len(trn):
            dgo = td.index[(-td.vx) >= TH.react_speed_mps]
            rgo = trn.index[(-trn.vx) >= TH.react_speed_mps]
            if len(rgo):
                t_r = (rgo[0] - f0) / TRACKING_FPS
                t_d = (dgo[0] - f0) / TRACKING_FPS if len(dgo) else TH.react_max_s
                out["reaction_lag_s"] = max(t_d - t_r, 0.0) if len(dgo) else TH.react_max_s - t_r
                out["reacted"] = bool(len(dgo))
        rows.append(out)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ recoveries
def build_recoveries(ctx: MatchContext, duels: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """After being beaten (dribbled in a duel, or a runner got in behind):
    did the defender turn quickly and get back goal-side of the threat?"""
    cands = []
    if len(duels):
        for _, r in duels[duels.beaten].iterrows():
            cands.append((r, "duel", "ball"))
    if len(runs) and "beaten_in_behind" in runs:
        for _, r in runs[runs.beaten_in_behind.fillna(False).astype(bool)].iterrows():
            cands.append((r, "run", "runner"))
    rows = []
    teams = {ctx.meta.team_name(ctx.meta.home_id): ctx.meta.home_id, ctx.meta.team_name(ctx.meta.away_id): ctx.meta.away_id}
    horizon = int(TH.recover_window_s * TRACKING_FPS)
    for r, source, threat in cands:
        team, per, f1 = teams[r.defending_team], int(r.period), int(r.frame_end)
        dfd = int(r.defender_id)
        td = ctx.track(dfd, f1, f1 + horizon, team, per)
        if td.empty:
            continue
        if threat == "ball":
            tx = ctx.ball_series(f1, f1 + horizon, team, per).reindex(td.index).ball_x
        else:
            tx = ctx.track(int(r.attacker_id), f1, f1 + horizon, team, per).x.reindex(td.index)
        gs = (td.x < tx).fillna(False)
        t_rec = (gs[gs].index[0] - f1) / TRACKING_FPS if gs.any() else np.nan
        pw = ctx.pose_window(dfd, f1, f1 + TRACKING_FPS, team, per)
        out = {k: r.get(k) for k in ("match_id", "period", "minute", "second", "event_id", "defending_team",
                                     "attacking_team", "defender_id", "defender", "defender_role",
                                     "attacker_id", "attacker", "phase_oop", "third", "channel", "subtype")}
        out.update({
            "moment": "recovery", "source": source, "frame": f1,
            "x": float(td.x.iloc[0]), "y": float(td.y.iloc[0]),
            "recovered": bool(gs.any()), "time_to_goal_side_s": t_rec,
            "max_recovery_speed": float(td.speed.max()),
            "turned_within_1s": bool(abs(angdiff(pw.torso, GOAL_WARD)) < 60)
            if pw is not None and not np.isnan(pw.torso) else np.nan,
            "pose_frames": 0 if pw is None else pw.n,
        })
        rows.append(out)
    return pd.DataFrame(rows)


def build_match(match_id: int, verbose: bool = True) -> dict[str, pd.DataFrame]:
    ctx = MatchContext(match_id)
    if ctx.pose.empty and verbose:
        print(f"  !! no pose features for {match_id}: moments will lack body-pose columns", file=sys.stderr)
    out = {}
    out["duels"] = build_duels(ctx)
    out["passes"] = build_passes(ctx)
    out["runs"] = build_runs(ctx)
    out["recoveries"] = build_recoveries(ctx, out["duels"], out["runs"])
    out["players"] = ctx.meta.players.assign(match_id=match_id)
    dest = BUILD_DIR / str(match_id)
    dest.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(dest / f"{k}.parquet", index=False)
        if verbose:
            print(f"  {k}: {len(v)} rows", file=sys.stderr)
    return out
