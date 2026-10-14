"""MatchContext: everything needed to evaluate a defensive moment, in a
*defender-normalised* frame where the defending team always defends the goal
at x = -L/2 (so "goal-ward" is always -x)."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..config import BUILD_DIR
from ..data.load import MatchMeta, load_events, load_meta, load_phases, load_tracking
from ..geometry import circmean


@dataclass
class PoseWindow:
    n: int
    torso: float
    head: float
    knee: float
    hip_h: float
    stagger: float
    stance_w: float
    head_series: np.ndarray = field(repr=False)   # per 25 fps frame
    frame10_series: np.ndarray = field(repr=False)


class MatchContext:
    def __init__(self, match_id: int, pose: pd.DataFrame | None = None):
        self.match_id = match_id
        self.meta: MatchMeta = load_meta(match_id)
        self.events = load_events(match_id)
        self.phases = load_phases(match_id)
        self.frames, self.xy = load_tracking(match_id)
        self.frames = self.frames.set_index("frame")
        self.L, self.W = self.meta.pitch
        if pose is None:
            p = BUILD_DIR / str(match_id) / "pose_features.parquet"
            pose = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        self.pose = pose
        self._xy_by_frame = self.xy.set_index("frame").sort_index()
        self._pose_by_player: dict[int, pd.DataFrame] = {}
        if not pose.empty:
            for pid, g in pose.sort_values("frame10").groupby("player_id"):
                self._pose_by_player[int(pid)] = g.reset_index(drop=True)
        pl = self.meta.players.set_index("player_id")
        self.team_of = pl["team_id"].to_dict()
        self.is_gk = pl["is_gk"].to_dict()
        self.name_of = pl["name"].to_dict()
        self.role_of = pl["role"].to_dict()

    # ------------------------------------------------------------------ frames
    def flip(self, team_id: int, period: int) -> float:
        """-1 if coordinates must be rotated 180 deg so `team_id` defends x = -L/2."""
        return 1.0 if self.meta.defends_left(team_id, period) else -1.0

    def players_at(self, frame: int, defending_team: int, period: int) -> pd.DataFrame:
        """All tracked players at `frame`, normalised for `defending_team`."""
        try:
            df = self._xy_by_frame.loc[[frame]].reset_index()
        except KeyError:
            return pd.DataFrame()
        s = self.flip(defending_team, period)
        df = df.assign(x=df.x * s, y=df.y * s, vx=df.vx * s, vy=df.vy * s)
        df["team_id"] = df.player_id.map(self.team_of)
        df["is_gk"] = df.player_id.map(self.is_gk).fillna(False).astype(bool)
        df["defending"] = df.team_id == defending_team
        return df

    def ball_at(self, frame: int, defending_team: int, period: int) -> tuple[float, float]:
        s = self.flip(defending_team, period)
        try:
            r = self.frames.loc[frame]
            return float(r.ball_x) * s, float(r.ball_y) * s
        except (KeyError, TypeError):
            return float("nan"), float("nan")

    def ball_series(self, f0: int, f1: int, defending_team: int, period: int) -> pd.DataFrame:
        s = self.flip(defending_team, period)
        b = self.frames.loc[f0:f1, ["ball_x", "ball_y"]].astype(float) * s
        return b.interpolate(limit_direction="both")

    def track(self, player_id: int, f0: int, f1: int, defending_team: int, period: int) -> pd.DataFrame:
        s = self.flip(defending_team, period)
        g = self.xy[(self.xy.player_id == player_id) & (self.xy.frame >= f0) & (self.xy.frame <= f1)]
        return g.assign(x=g.x * s, y=g.y * s, vx=g.vx * s, vy=g.vy * s).set_index("frame")

    # -------------------------------------------------------------------- pose
    def pose_window(self, player_id: int, f0: float, f1: float, defending_team: int, period: int) -> PoseWindow | None:
        g = self._pose_by_player.get(int(player_id))
        if g is None:
            return None
        fr = g["frame10"].to_numpy()
        i0, i1 = np.searchsorted(fr, f0, "left"), np.searchsorted(fr, f1, "right")
        if i1 <= i0:
            return None
        w = g.iloc[i0:i1]
        rot = 0.0 if self.flip(defending_team, period) > 0 else 180.0
        head = ((w.head_deg.to_numpy(float) + rot + 180) % 360) - 180
        torso = ((w.torso_deg.to_numpy(float) + rot + 180) % 360) - 180
        return PoseWindow(
            n=len(w), torso=circmean(torso), head=circmean(head),
            knee=float(np.nanmedian(w.knee_deg)) if w.knee_deg.notna().any() else np.nan,
            hip_h=float(np.nanmedian(w.hip_h)) if w.hip_h.notna().any() else np.nan,
            stagger=float(np.nanmedian(w.stagger)) if w.stagger.notna().any() else np.nan,
            stance_w=float(np.nanmedian(w.stance_w)) if w.stance_w.notna().any() else np.nan,
            head_series=head, frame10_series=w.frame10.to_numpy(float),
        )

    @cached_property
    def posture_model(self) -> dict:
        """Speed-adjusted posture norms.

        Knee flexion and hip height are dominated by gait: a sprinting player is
        'low' by construction. So 'readiness' is measured as a *residual*: how much
        lower (hip) / more flexed (knee) a player is than (a) the match-wide norm at
        that speed, (b) shifted by the player's own offset (height, running style).
        """
        if self.pose.empty:
            return {}
        p = self.pose[["frame10", "player_id", "hip_h", "knee_deg"]].copy()
        p["frame"] = p.frame10.round().astype(int)
        p = p.merge(self.xy[["frame", "player_id", "speed"]], on=["frame", "player_id"], how="inner")
        bins = np.arange(0, 10.5, 0.5)
        p["sb"] = np.clip(np.digitize(p.speed, bins) - 1, 0, len(bins) - 2)
        curve_h = p.groupby("sb").hip_h.median().reindex(range(len(bins) - 1)).ffill().bfill().to_numpy()
        curve_k = p.groupby("sb").knee_deg.median().reindex(range(len(bins) - 1)).ffill().bfill().to_numpy()
        p["res_h"] = p.hip_h - curve_h[p.sb]
        p["res_k"] = p.knee_deg - curve_k[p.sb]
        return {"bins": bins, "curve_h": curve_h, "curve_k": curve_k,
                "off_h": p.groupby("player_id").res_h.median().to_dict(),
                "off_k": p.groupby("player_id").res_k.median().to_dict()}

    def crouch(self, player_id: int, speed: float, hip_h: float, knee: float) -> tuple[float, float]:
        """(hip drop m, extra knee flexion deg) vs the player's own norm at this speed.
        Positive = lower / more flexed than usual = 'loaded' stance."""
        m = self.posture_model
        if not m or np.isnan(speed):
            return np.nan, np.nan
        sb = int(np.clip(np.digitize(speed, m["bins"]) - 1, 0, len(m["bins"]) - 2))
        dh = (m["curve_h"][sb] + m["off_h"].get(player_id, 0.0)) - hip_h
        dk = (m["curve_k"][sb] + m["off_k"].get(player_id, 0.0)) - knee
        return float(dh), float(dk)

    # -------------------------------------------------------------- marking
    def assign_markers(self, snap: pd.DataFrame, max_dist: float) -> dict[int, int]:
        """Defender -> attacker assignment (Hungarian) on distance to the attacker's
        goal-side point (2 m toward the defended goal). Returns attacker_id -> defender_id."""
        d = snap[snap.defending & ~snap.is_gk]
        a = snap[~snap.defending & ~snap.is_gk]
        if d.empty or a.empty:
            return {}
        gx, gy = -self.L / 2, 0.0
        ax, ay = a.x.to_numpy(), a.y.to_numpy()
        vx, vy = gx - ax, gy - ay
        nrm = np.hypot(vx, vy) + 1e-9
        px, py = ax + 2 * vx / nrm, ay + 2 * vy / nrm
        cost = np.hypot(d.x.to_numpy()[:, None] - px[None], d.y.to_numpy()[:, None] - py[None])
        r, c = linear_sum_assignment(cost)
        out = {}
        for i, j in zip(r, c):
            if cost[i, j] <= max_dist:
                out[int(a.player_id.iloc[j])] = int(d.player_id.iloc[i])
        return out
