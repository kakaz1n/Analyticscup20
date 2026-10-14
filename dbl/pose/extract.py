"""Stream a SkillCorner body-pose file into a compact per-player-frame feature table.

A full match is ~3.3 GB of JSON (45M joint observations). We never hold it in
memory: each 25 fps record is reduced to ~15 interpretable numbers per player
(heading of head / shoulders / hips, knee flexion, hip height, stance), gated by
SkillCorner's per-joint p90 error. Output: ``data/build/<match>/pose_features.parquet``.

Conventions
-----------
* Headings are ground-plane angles in degrees, (-180, 180], 0 = +x, CCW positive,
  in raw SkillCorner pitch coordinates (orientation normalisation happens later).
* "Forward" for a left/right joint pair is the left->right axis rotated +90 deg
  (same convention as SkillCorner's ``pose_orientation.py``).
"""
from __future__ import annotations

import gzip
import json
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BUILD_DIR, TH

try:
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

J = ["nose", "lEar", "rEar", "lEye", "rEye", "lShoulder", "rShoulder", "lHip", "rHip",
     "midHip", "lKnee", "rKnee", "lAnkle", "rAnkle", "neck"]
JI = {n: i for i, n in enumerate(J)}


def iter_records(path: Path | str) -> Iterator[dict]:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            for line in fh:
                yield _loads(line)
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = [n for n in zf.namelist() if n.endswith(".jsonl")][0]
            with zf.open(member) as fh:
                for line in fh:
                    yield _loads(line)
    else:
        with open(path, "rb") as fh:
            for line in fh:
                yield _loads(line)


def _heading(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Facing direction of a left/right joint pair -> (deg, pair width m)."""
    dx, dy = right[:, 0] - left[:, 0], right[:, 1] - left[:, 1]
    fx, fy = -dy, dx
    return np.degrees(np.arctan2(fy, fx)), np.hypot(dx, dy)


def _circ_mean(*angles_deg: np.ndarray) -> np.ndarray:
    s = sum(np.sin(np.radians(a)) for a in angles_deg)
    c = sum(np.cos(np.radians(a)) for a in angles_deg)
    return np.degrees(np.arctan2(s, c))


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at b (deg) for 3D points."""
    u, v = a - b, c - b
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def features_from_block(xyz: np.ndarray, err: np.ndarray) -> dict[str, np.ndarray]:
    """xyz: (N, len(J), 3), err: (N, len(J)) with NaN for missing joints."""
    g = lambda n: xyz[:, JI[n]]  # noqa: E731
    e = lambda *ns: np.nanmax(np.stack([err[:, JI[n]] for n in ns], 1), 1)  # noqa: E731
    gate = TH.max_joint_err_cm

    sh_deg, sh_w = _heading(g("lShoulder"), g("rShoulder"))
    hp_deg, hp_w = _heading(g("lHip"), g("rHip"))
    sh_ok = (e("lShoulder", "rShoulder") <= gate) & (sh_w >= TH.shoulder_width_m[0]) & (sh_w <= TH.shoulder_width_m[1])
    hp_ok = (e("lHip", "rHip") <= gate) & (hp_w >= TH.hip_width_m[0]) & (hp_w <= TH.hip_width_m[1])
    torso = np.where(sh_ok & hp_ok, _circ_mean(sh_deg, hp_deg), np.where(sh_ok, sh_deg, np.where(hp_ok, hp_deg, np.nan)))

    # head: ear-axis perpendicular blended with ear-midpoint -> nose vector
    ear_deg, ear_w = _heading(g("lEar"), g("rEar"))
    ear_mid = (g("lEar") + g("rEar")) / 2
    nv = g("nose") - ear_mid
    nose_deg = np.degrees(np.arctan2(nv[:, 1], nv[:, 0]))
    head_err = e("nose", "lEar", "rEar")
    head_ok = (head_err <= gate) & (ear_w > 0.06) & (ear_w < 0.30)
    head = np.where(head_ok, _circ_mean(ear_deg, nose_deg), np.nan)

    knee_l = _angle3(g("lHip"), g("lKnee"), g("lAnkle"))
    knee_r = _angle3(g("rHip"), g("rKnee"), g("rAnkle"))
    leg_ok = e("lHip", "lKnee", "lAnkle", "rHip", "rKnee", "rAnkle") <= gate * 1.3  # legs are noisier
    knee = np.where(leg_ok, (knee_l + knee_r) / 2, np.nan)

    la, ra = g("lAnkle"), g("rAnkle")
    stance_vec = ra[:, :2] - la[:, :2]
    stance_w = np.where(leg_ok, np.hypot(stance_vec[:, 0], stance_vec[:, 1]), np.nan)
    # stagger: how much the feet are split along the torso facing direction (side-on cue)
    t = np.radians(torso)
    stagger = np.where(leg_ok, np.abs(stance_vec[:, 0] * np.cos(t) + stance_vec[:, 1] * np.sin(t)), np.nan)

    hip_h = np.where(e("midHip") <= gate, g("midHip")[:, 2], np.nan)
    return {
        "torso_deg": torso, "shoulder_deg": np.where(sh_ok, sh_deg, np.nan),
        "hip_deg": np.where(hp_ok, hp_deg, np.nan), "head_deg": head,
        "knee_deg": knee, "hip_h": hip_h, "stance_w": stance_w, "stagger": stagger,
        "head_err": head_err, "torso_err": e("lShoulder", "rShoulder", "lHip", "rHip"),
    }


def extract(path: Path | str, block: int = 20000, progress_every: int = 20000) -> pd.DataFrame:
    meta_rows: list[tuple] = []
    xyz_buf: list[np.ndarray] = []
    err_buf: list[np.ndarray] = []
    out: list[pd.DataFrame] = []
    nJ = len(J)

    def flush():
        if not meta_rows:
            return
        m = np.array(meta_rows, dtype="float64")
        f = features_from_block(np.stack(xyz_buf), np.stack(err_buf))
        df = pd.DataFrame({"frame25": m[:, 0].astype("int64"), "period": m[:, 1].astype("int8"),
                           "player_id": m[:, 2].astype("int64"), "x": m[:, 3], "y": m[:, 4]})
        for k, v in f.items():
            df[k] = v.astype("float32")
        out.append(df)
        meta_rows.clear(); xyz_buf.clear(); err_buf.clear()

    for i, rec in enumerate(iter_records(path)):
        if progress_every and i and i % progress_every == 0:
            print(f"    {i} pose frames", file=sys.stderr)
        per = rec.get("period")
        if per is None:
            continue
        for p in rec.get("player_data") or []:
            joints = p.get("joints")
            if not joints:
                continue
            xyz = np.full((nJ, 3), np.nan)
            err = np.full(nJ, np.nan)
            for n, k in JI.items():
                jj = joints.get(n)
                if jj:
                    xyz[k] = jj["xyz"]
                    err[k] = jj.get("p90_mae_cm") or np.nan
            meta_rows.append((rec["frame"], per, p["player_id"], p.get("x") or np.nan, p.get("y") or np.nan))
            xyz_buf.append(xyz)
            err_buf.append(err)
            if len(meta_rows) >= block:
                flush()
    flush()
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["frame10"] = (df["frame25"] * 0.4).astype("float32")  # tracking clock
    return df


def build_pose_features(match_id: int, src: Path | str, force: bool = False) -> Path:
    dest = BUILD_DIR / str(match_id) / "pose_features.parquet"
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  extracting pose features from {src}", file=sys.stderr)
    df = extract(src)
    save_pose_features(df, dest)
    return dest


def save_pose_features(df: pd.DataFrame, dest: Path) -> None:
    """Round to 0.1 deg / 1 mm (far below the ~10 cm joint error) and zstd-compress: ~4x smaller."""
    df = df.copy()
    for c in ("x", "y", "torso_deg", "shoulder_deg", "hip_deg", "head_deg", "knee_deg", "head_err", "torso_err"):
        if c in df:
            df[c] = df[c].round(1)
    for c in ("hip_h", "stance_w", "stagger"):
        if c in df:
            df[c] = df[c].round(3)
    df.to_parquet(dest, index=False, compression="zstd", compression_level=19)
