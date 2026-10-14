"""Paths, match list and every tunable threshold in one place.

All thresholds are deliberately exposed: the framework is a *coaching lens*,
and coaches disagree on what "side-on" or "low enough" means. Change them here
(or via environment variables) and rebuild.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DBL_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # SkillCorner files as downloaded
POSE_RAW_DIR = DATA_DIR / "pose_raw"  # Hugging Face body-pose zips
BUILD_DIR = DATA_DIR / "build"      # parquet outputs of the pipeline

# The only two matches with full-match body pose in the open data release.
MATCHES: dict[int, str] = {
    1925299: "Brisbane Roar v Perth Glory (2024-12-21)",
    1996435: "Sydney FC v Adelaide United (2025-02-01)",
}

GITHUB_MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"
GITHUB_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"
HF_POSE = "https://huggingface.co/datasets/SkillCorner/opendata-bodypose/resolve/main/raw"
POSE_SAMPLE_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/bodypose/sample_1925299_phase406.jsonl.gz"

POSE_FPS = 25
TRACKING_FPS = 10


@dataclass(frozen=True)
class Thresholds:
    # --- pose quality gates (see SkillCorner src/features/pose_orientation.py)
    max_joint_err_cm: float = 18.0        # p90 error gate for joints used in an angle
    shoulder_width_m: tuple = (0.15, 0.60)
    hip_width_m: tuple = (0.10, 0.50)

    # --- SEE: perception
    fov_half_deg: float = 60.0            # ball/man counted "in view" within +/- this of head heading
    check_away_deg: float = 75.0          # head turned this far from the ball = a shoulder check
    check_min_frames: int = 3             # ...for at least this many 25 fps frames (0.12 s)
    scan_window_s: float = 3.0            # look-back window for shoulder checks

    # --- SHAPE: body orientation
    side_on_min_deg: float = 25.0         # torso vs line-to-carrier: below = square
    side_on_max_deg: float = 70.0         # above = turned / chasing
    open_body_min_deg: float = 45.0       # torso vs own-goal axis at run start: "open" if within

    # --- STANCE: readiness
    ready_hip_drop_m: float = 0.03        # hip lower than own speed-adjusted norm by this = loaded
    ready_knee_extra_deg: float = 5.0     # ...or knees this much more flexed than the norm

    # --- REACT
    react_speed_mps: float = 2.5          # goal-ward speed that counts as having reacted
    react_max_s: float = 2.0
    recover_window_s: float = 3.0

    # --- context
    engage_window_s: float = 0.4          # pose summarised over [start - w, start + w/2]
    marker_max_dist_m: float = 12.0       # beyond this a defender is not "marking" the receiver


TH = Thresholds()
