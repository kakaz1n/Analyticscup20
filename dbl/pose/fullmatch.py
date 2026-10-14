"""Full-match skeleton store for the 3D match viewer.

Streams the raw body-pose file once and keeps every resolved skeleton at
12.5 fps (every 2nd pose frame), 23 joints, int16 centimetres. Written as
10-minute blocks so each file stays small and the viewer server can load
only what it needs:

    data/build/<match>/full/pose_<period>_<block>.parquet

Rows: frame25, player_id, then <joint>_x/_y/_z for JOINTS (raw SkillCorner
pitch coordinates; the viewer does not normalise orientation -- it shows the
match as broadcast).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BUILD_DIR, POSE_FPS
from .extract import iter_records
from .skeletons import JOINTS

STEP = 2                      # keep every 2nd pose frame -> 12.5 fps
BLOCK_S = 600                 # 10-minute files
BLOCK_F = BLOCK_S * POSE_FPS  # in pose frames
COLS = ["frame25", "period", "player_id"] + [f"{j}_{a}" for j in JOINTS for a in "xyz"]


def full_dir(match_id: int) -> Path:
    return BUILD_DIR / str(match_id) / "full"


def extract(records) -> pd.DataFrame:
    cols = defaultdict(list)
    for rec in records:
        f, per = rec.get("frame"), rec.get("period")
        if per is None or f is None or f % STEP:
            continue
        for p in rec.get("player_data") or []:
            j = p.get("joints")
            if not j:
                continue
            cols["frame25"].append(f)
            cols["period"].append(per)
            cols["player_id"].append(p["player_id"])
            for n in JOINTS:
                xyz = (j.get(n) or {}).get("xyz")
                if xyz is None:
                    xyz = (np.nan, np.nan, np.nan)
                cols[n + "_x"].append(xyz[0]); cols[n + "_y"].append(xyz[1]); cols[n + "_z"].append(xyz[2])
    df = pd.DataFrame(cols, columns=COLS)
    df["frame25"] = df.frame25.astype("int64")
    df["period"] = df.period.astype("int8")
    df["player_id"] = df.player_id.astype("int64")
    for c in COLS[3:]:
        df[c] = (df[c].astype("float64") * 100).round().fillna(-32768).clip(-32768, 32767).astype("int16")
    return df


def save_blocks(df: pd.DataFrame, match_id: int) -> list[Path]:
    out = full_dir(match_id)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    df = df.drop_duplicates(["frame25", "player_id"]).sort_values(["frame25", "player_id"])
    for (per, blk), g in df.groupby([df.period, df.frame25 // BLOCK_F]):
        p = out / f"pose_{int(per)}_{int(blk):03d}.parquet"
        g[COLS].to_parquet(p, index=False, compression="zstd", compression_level=19)
        paths.append(p)
    return paths


def build_full(match_id: int, src: Path | str, force: bool = False) -> Path:
    out = full_dir(match_id)
    if list(out.glob("pose_*.parquet")) and not force:
        return out
    print(f"  full-match skeletons from {src} (12.5 fps)", file=sys.stderr)
    save_blocks(extract(iter_records(src)), match_id)
    return out


def iter_byte_range(path: Path, start: int, end: int):
    with open(path, "rb") as fh:
        fh.seek(start)
        if start:
            fh.readline()
        while fh.tell() <= end:
            line = fh.readline()
            if not line:
                break
            yield json.loads(line)


if __name__ == "__main__":
    # Chunked mode for time-limited machines:
    #   python -m dbl.pose.fullmatch part <match> <src.jsonl> <start_byte> <end_byte> <out_part.parquet>
    #   python -m dbl.pose.fullmatch merge <match> <part1.parquet> [<part2> ...]
    if sys.argv[1] == "part":
        mid, src, a, b, out = int(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
        extract(iter_byte_range(src, a, b)).to_parquet(out, index=False, compression="zstd")
        print("DONE", out)
    elif sys.argv[1] == "merge":
        mid = int(sys.argv[2])
        df = pd.concat([pd.read_parquet(p) for p in sys.argv[3:]], ignore_index=True)
        print("DONE", len(save_blocks(df, mid)), "blocks")
