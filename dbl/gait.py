"""Gait library: short running/walking clips of each player, so simulated defenders move with real legs.

From the full-match skeletons (12.5 fps) each player's frames are grouped into speed bands. For each band the
smoothest continuous stretch of ≥ 1.28 s is kept, *in place* (pelvis at the origin) and rotated so that the body
faces +x. The viewer plays the clip of the band matching the simulated speed, advancing it by the distance the
simulated player covers (so feet and ground speed agree), and rotates it to the simulated body heading.

Output: ``data/build/<match>/gait.json`` — {player_id: [{vmin, vmax, speed, fps, frames: int16 cm [F][23][3]}]}.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import BUILD_DIR
from .pose.skeletons import JOINTS

BANDS = [(0.0, 0.8), (0.8, 2.2), (2.2, 3.8), (3.8, 5.5), (5.5, 10.0)]
MIN_F = 16          # 1.28 s at 12.5 fps
FPS = 12.5


def _cols(j):
    return [f"{j}_x", f"{j}_y", f"{j}_z"]


def build_gait(match_id: int, force: bool = False) -> dict:
    out = BUILD_DIR / str(match_id) / "gait.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    files = sorted((BUILD_DIR / str(match_id) / "full").glob("pose_*.parquet"))
    if not files:
        return {}
    cols = ["frame25", "player_id"] + [c for j in JOINTS for c in _cols(j)]
    lib: dict[str, list] = {}
    best: dict[tuple, tuple] = {}
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        for pid, g in df.groupby("player_id"):
            g = g.sort_values("frame25")
            fr = g.frame25.to_numpy()
            J = np.stack([g[_cols(j)].to_numpy(np.float32) for j in JOINTS], 1) / 100.0   # F x 23 x 3, metres
            J[np.abs(J) > 300] = np.nan
            hip = J[:, JOINTS.index("midHip"), :2]
            ls, rs = J[:, JOINTS.index("lShoulder"), :2], J[:, JOINTS.index("rShoulder"), :2]
            lh, rh = J[:, JOINTS.index("lHip"), :2], J[:, JOINTS.index("rHip"), :2]
            lat = (rs - ls) + (rh - lh)
            head = np.arctan2(lat[:, 0], -lat[:, 1])                    # facing = perpendicular, as in the viewer
            ok = np.isfinite(J).all((1, 2)) & np.isfinite(head)
            dtf = np.diff(fr, prepend=fr[0] - 2)
            sp = np.r_[0, np.hypot(*np.diff(hip, axis=0).T)] / np.maximum(dtf / 25.0, 1e-3)
            sp = pd.Series(sp).rolling(3, center=True, min_periods=1).median().to_numpy()
            # continuous runs (no gaps, all joints present)
            brk = np.flatnonzero((dtf != 2) | ~ok)
            starts = np.r_[0, brk]
            ends = np.r_[brk, len(fr)]
            for a, b in zip(starts, ends):
                a = a + (0 if ok[a] and (a == 0 or dtf[a] == 2) else 1)
                if b - a < MIN_F:
                    continue
                for bi, (lo, hi) in enumerate(BANDS):
                    inb = (sp[a:b] >= lo) & (sp[a:b] < hi)
                    # longest stretch inside the band
                    run, s0 = 0, a
                    for k, v in enumerate(inb):
                        if v:
                            run += 1
                            if run >= MIN_F:
                                st = a + k - MIN_F + 1
                                seg = slice(st, st + MIN_F)
                                turn = np.abs(np.diff(np.unwrap(head[seg]))).sum()
                                score = turn + np.std(sp[seg]) * 0.5
                                key = (int(pid), bi)
                                if key not in best or score < best[key][0]:
                                    best[key] = (score, J[seg].copy(), head[seg].copy(), hip[seg].copy(), float(sp[seg].mean()))
                        else:
                            run = 0
    for (pid, bi), (_, Jc, hd, hp, spd) in best.items():
        # in place and facing +x: subtract the pelvis, rotate each frame by -heading (smoothed)
        h = np.unwrap(hd)
        h = pd.Series(h).rolling(5, center=True, min_periods=1).mean().to_numpy()
        C = Jc.copy()
        C[..., 0] -= hp[:, None, 0]; C[..., 1] -= hp[:, None, 1]
        c, s = np.cos(-h)[:, None], np.sin(-h)[:, None]
        x, y = C[..., 0].copy(), C[..., 1].copy()
        C[..., 0], C[..., 1] = c * x - s * y, s * x + c * y
        lo, hi = BANDS[bi]
        lib.setdefault(str(pid), []).append({"vmin": lo, "vmax": hi, "speed": round(spd, 2), "fps": FPS,
                                             "frames": np.round(C * 100).astype(int).tolist()})
    for v in lib.values():
        v.sort(key=lambda c: c["vmin"])
    out.write_text(json.dumps(lib), encoding="utf-8")
    return lib


@lru_cache(maxsize=4)
def gait_json(match_id: int) -> bytes:
    return json.dumps(build_gait(match_id)).encode()
