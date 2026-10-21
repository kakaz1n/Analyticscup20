"""Parse SkillCorner match sheet, tracking, dynamic events and phases."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..config import BUILD_DIR, TRACKING_FPS
from .download import match_files

try:  # orjson is ~3x faster on the 90 MB tracking files, but optional
    import orjson as _json

    _loads = _json.loads
except ImportError:  # pragma: no cover
    _loads = json.loads


@dataclass
class MatchMeta:
    match_id: int
    raw: dict = field(repr=False)

    @property
    def home_id(self) -> int:
        return self.raw["home_team"]["id"]

    @property
    def away_id(self) -> int:
        return self.raw["away_team"]["id"]

    def team_name(self, team_id: int) -> str:
        for k in ("home_team", "away_team"):
            if self.raw[k]["id"] == team_id:
                return self.raw[k]["short_name"]
        return str(team_id)

    @property
    def title(self) -> str:
        r = self.raw
        return (f"{r['home_team']['short_name']} {r['home_team_score']}-{r['away_team_score']} "
                f"{r['away_team']['short_name']}")

    @property
    def pitch(self) -> tuple[float, float]:
        return float(self.raw["pitch_length"]), float(self.raw["pitch_width"])

    def defends_left(self, team_id: int, period: int) -> bool:
        """True if `team_id` defends the goal at x = -L/2 in `period`."""
        home_side = self.raw["home_team_side"][int(period) - 1]  # direction home ATTACKS
        home_defends_left = home_side == "left_to_right"
        return home_defends_left if team_id == self.home_id else not home_defends_left

    @cached_property
    def players(self) -> pd.DataFrame:
        rows = []
        for p in self.raw["players"]:
            role = p.get("player_role") or {}
            rows.append({
                "player_id": p["id"],
                "name": p["short_name"],
                "team_id": p["team_id"],
                "team": self.team_name(p["team_id"]),
                "number": p.get("number"),
                "role": role.get("acronym"),
                "position_group": role.get("position_group"),
                "is_gk": role.get("acronym") == "GK",
                "minutes": ((p.get("playing_time") or {}).get("total") or {}).get("minutes_played"),
            })
        return pd.DataFrame(rows)


def load_meta(match_id: int) -> MatchMeta:
    return MatchMeta(match_id, json.loads(match_files(match_id)["match"].read_text(encoding="utf-8")))


def load_events(match_id: int) -> pd.DataFrame:
    return pd.read_csv(match_files(match_id)["events"], low_memory=False)


def load_phases(match_id: int) -> pd.DataFrame:
    return pd.read_csv(match_files(match_id)["phases"])


def _parse_tracking(match_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, players = [], {"frame": [], "player_id": [], "x": [], "y": [], "detected": []}
    with open(match_files(match_id)["tracking"], "rb") as fh:
        for line in fh:
            r = _loads(line)
            if r.get("period") is None:
                continue
            b = r["ball_data"] or {}
            poss = r.get("possession") or {}
            frames.append((r["frame"], r["period"], b.get("x"), b.get("y"), b.get("z"),
                           poss.get("group"), poss.get("player_id")))
            for p in r["player_data"]:
                players["frame"].append(r["frame"])
                players["player_id"].append(p["player_id"])
                players["x"].append(p["x"])
                players["y"].append(p["y"])
                players["detected"].append(bool(p.get("is_detected")))
    fr = pd.DataFrame(frames, columns=["frame", "period", "ball_x", "ball_y", "ball_z",
                                       "poss_group", "poss_player"])
    pl = pd.DataFrame(players)
    pl = pl.merge(fr[["frame", "period"]], on="frame", how="left")
    return fr, pl


def add_kinematics(pl: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Savitzky-Golay smoothed velocity per player (SkillCorner recommends smoothing)."""
    pl = pl.sort_values(["player_id", "frame"]).reset_index(drop=True)
    vx = np.full(len(pl), np.nan, dtype="float32")
    vy = np.full(len(pl), np.nan, dtype="float32")
    dt = 1.0 / TRACKING_FPS
    # split into contiguous runs (same player, consecutive frames)
    brk = (pl["player_id"].ne(pl["player_id"].shift()) | pl["frame"].diff().ne(1)).cumsum().to_numpy()
    x, y = pl["x"].to_numpy(float), pl["y"].to_numpy(float)
    starts = np.flatnonzero(np.r_[True, brk[1:] != brk[:-1]])
    ends = np.r_[starts[1:], len(pl)]
    for s, e in zip(starts, ends):
        n = e - s
        if n >= window:
            vx[s:e] = savgol_filter(x[s:e], window, 2, deriv=1, delta=dt)
            vy[s:e] = savgol_filter(y[s:e], window, 2, deriv=1, delta=dt)
        elif n >= 2:
            vx[s:e] = np.gradient(x[s:e], dt)
            vy[s:e] = np.gradient(y[s:e], dt)
    pl["vx"], pl["vy"] = vx, vy
    pl["speed"] = np.hypot(vx, vy).astype("float32")
    return pl


def load_tracking(match_id: int, cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (frames, players) at 10 fps; cached as parquet after the first parse."""
    out = BUILD_DIR / str(match_id)
    f_fr, f_pl = out / "frames.parquet", out / "players_xy.parquet"
    if cache and f_fr.exists() and f_pl.exists():
        return pd.read_parquet(f_fr), pd.read_parquet(f_pl)
    fr, pl = _parse_tracking(match_id)
    pl = add_kinematics(pl)
    for c in ("x", "y"):
        pl[c] = pl[c].astype("float32")
    if cache:
        out.mkdir(parents=True, exist_ok=True)
        fr.to_parquet(f_fr, index=False)
        pl.to_parquet(f_pl, index=False)
    return fr, pl
