"""Keep the raw 3D skeletons around every defensive moment, for the 3D viewer.

The full pose files (3.3 GB/match) are too big to ship, and the feature table
only keeps angles. Here we stream the raw file once more and keep the 29 joints
(rounded to 1 cm) for:

* every player at each moment's key frame, and
* every player within ``RADIUS_M`` of the focal defender, at 6.25 fps for
  ``CLIP_S`` seconds either side of the key frame (the animation).

Output: ``data/build/<match>/skeletons.parquet`` -- one row per (pose frame, player).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BUILD_DIR, POSE_FPS, TRACKING_FPS
from .extract import iter_records

# 23 of the 29 joints (fingers and small toes dropped: not drawn, ~20% smaller files)
JOINTS = ("nose", "neck", "lEye", "rEye", "lEar", "rEar", "lShoulder", "rShoulder", "lElbow", "rElbow",
          "lWrist", "rWrist", "midHip", "lHip", "rHip", "lKnee",
          "rKnee", "lAnkle", "rAnkle", "lHeel", "rHeel", "lBigToe", "rBigToe")
BONES = (("nose", "neck"), ("lEye", "rEye"), ("lEar", "lEye"), ("rEar", "rEye"), ("nose", "lEye"), ("nose", "rEye"),
         ("neck", "lShoulder"), ("neck", "rShoulder"), ("lShoulder", "rShoulder"),
         ("lShoulder", "lElbow"), ("lElbow", "lWrist"), ("rShoulder", "rElbow"), ("rElbow", "rWrist"),
         ("neck", "midHip"), ("lShoulder", "lHip"), ("rShoulder", "rHip"), ("lHip", "rHip"),
         ("lHip", "lKnee"), ("lKnee", "lAnkle"), ("rHip", "rKnee"), ("rKnee", "rAnkle"),
         ("lAnkle", "lHeel"), ("lAnkle", "lBigToe"), ("lHeel", "lBigToe"),
         ("rAnkle", "rHeel"), ("rAnkle", "rBigToe"), ("rHeel", "rBigToe"))
RADIUS_M = 15.0
CLIP_S = 1.2
CLIP_STEP = 4  # pose frames (25 fps / 4 = 6.25 fps)
TABLES = ("duels", "passes", "runs", "recoveries")


def plan(match_id: int) -> dict[int, list[int] | None]:
    """pose frame -> player ids to keep (None = everybody)."""
    xy = pd.read_parquet(BUILD_DIR / str(match_id) / "players_xy.parquet", columns=["frame", "player_id", "x", "y"])
    xy = xy.set_index("frame").sort_index()
    keep: dict[int, set | None] = {}
    half = int(CLIP_S * POSE_FPS)
    for t in TABLES:
        p = BUILD_DIR / str(match_id) / f"{t}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["frame", "defender_id"])
        for f10, dfd in zip(df.frame.astype(int), df.defender_id):
            k = int(round(f10 * POSE_FPS / TRACKING_FPS))
            keep[k] = None  # everyone at the key frame
            try:
                snap = xy.loc[[f10]]
            except KeyError:
                continue
            d = snap[snap.player_id == dfd]
            if d.empty:
                continue
            near = set(snap.player_id[np.hypot(snap.x - d.x.iloc[0], snap.y - d.y.iloc[0]) <= RADIUS_M].astype(int))
            for f in range(k - half, k + half + 1, CLIP_STEP):
                if f == k:
                    continue
                cur = keep.get(f, set())
                if cur is not None:
                    keep[f] = cur | near
    return {f: (None if v is None else sorted(v)) for f, v in keep.items()}


def save_plan(match_id: int, path: Path) -> Path:
    path.write_text(json.dumps({str(k): v for k, v in plan(match_id).items()}), encoding="utf-8")
    return path


def extract(records, keep: dict[int, list[int] | None]) -> pd.DataFrame:
    keep = {int(k): (None if v is None else set(v)) for k, v in keep.items()}
    cols = defaultdict(list)
    for rec in records:
        f = rec.get("frame")
        if f not in keep or rec.get("period") is None:
            continue
        who = keep[f]
        for p in rec.get("player_data") or []:
            j = p.get("joints")
            if not j or (who is not None and p["player_id"] not in who):
                continue
            cols["frame25"].append(f)
            cols["period"].append(rec["period"])
            cols["player_id"].append(p["player_id"])
            for n in JOINTS:
                xyz = (j.get(n) or {}).get("xyz") or (np.nan, np.nan, np.nan)
                cols[n + "_x"].append(xyz[0]); cols[n + "_y"].append(xyz[1]); cols[n + "_z"].append(xyz[2])
        bd = rec.get("ball_data") or {}
        cols["frame25"].append(f); cols["period"].append(rec["period"]); cols["player_id"].append(-1)
        for n in JOINTS:  # ball stored as player -1 with its xyz in "nose"
            v = (bd.get("x"), bd.get("y"), bd.get("z")) if n == "nose" else (np.nan,) * 3
            cols[n + "_x"].append(v[0] if v[0] is not None else np.nan)
            cols[n + "_y"].append(v[1] if v[1] is not None else np.nan)
            cols[n + "_z"].append(v[2] if v[2] is not None else np.nan)
    df = pd.DataFrame(cols)
    return to_compact(df.drop_duplicates(["frame25", "player_id"]))


def to_compact(df: pd.DataFrame) -> pd.DataFrame:
    """Joints as int16 centimetres (NaN -> -32768): 1 cm is far below pose error, ~3x smaller files."""
    df = df.copy()
    df["frame25"] = df.frame25.astype("int64")
    df["period"] = df.period.astype("int8")
    df["player_id"] = df.player_id.astype("int64")
    for c in df.columns[3:]:
        v = df[c].astype("float64") * 100
        df[c] = v.round().fillna(-32768).clip(-32768, 32767).astype("int16")
    return df


def joints_m(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse of `to_compact`: joint columns back to metres (float, NaN restored)."""
    out = df.copy()
    for c in out.columns[3:]:
        v = out[c].astype("float32")
        out[c] = v.where(v != -32768) / 100
    return out


def iter_byte_range(path: Path, start: int, end: int):
    import json as _j
    with open(path, "rb") as fh:
        fh.seek(start)
        if start:
            fh.readline()
        while fh.tell() <= end:
            line = fh.readline()
            if not line:
                break
            yield _j.loads(line)


def build_skeletons(match_id: int, src: Path | str, force: bool = False) -> Path:
    out = BUILD_DIR / str(match_id)
    if list(out.glob("skeletons_p*.parquet")) and not force:
        return out
    print(f"  extracting 3D skeletons from {src}", file=sys.stderr)
    save(extract(iter_records(src), plan(match_id)), match_id)
    return out


def save(df: pd.DataFrame, match_id: int) -> None:
    """One file per period (keeps each file < 25 MB)."""
    cols = ["frame25", "period", "player_id"] + [f"{j}_{a}" for j in JOINTS for a in "xyz"]
    for per, g in df[cols].groupby("period"):
        g.to_parquet(BUILD_DIR / str(match_id) / f"skeletons_p{int(per)}.parquet", index=False,
                     compression="zstd", compression_level=19)


def load(match_id: int) -> pd.DataFrame:
    parts = sorted((BUILD_DIR / str(match_id)).glob("skeletons_p*.parquet"))
    if not parts:
        return pd.DataFrame()
    return joints_m(pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True))


if __name__ == "__main__":  # chunked use on low-memory / time-limited machines
    # python -m dbl.pose.skeletons <match_id> <src.jsonl> <start_byte> <end_byte> <out.parquet> [plan.json]
    mid, src, a, b, out = int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    keep = json.loads(Path(sys.argv[6]).read_text(encoding="utf-8")) if len(sys.argv) > 6 else plan(mid)
    extract(iter_byte_range(src, a, b), keep).to_parquet(out, index=False, compression="zstd")
    print("DONE", out)
