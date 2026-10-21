"""Fetch the SkillCorner open data for the two body-pose matches.

Tracking / events / phases / match sheet come from the SkillCorner GitHub
repository (tracking is stored with Git LFS, so we use the media URL).
Body pose comes from the Hugging Face Hub (~620 MB zip per match). If your
network blocks huggingface.co, download the two zips in a browser and drop
them in ``data/pose_raw/`` -- the pipeline picks them up from there.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from ..config import GITHUB_MEDIA, GITHUB_RAW, HF_POSE, MATCHES, POSE_RAW_DIR, POSE_SAMPLE_URL, RAW_DIR

POSE_SHA256 = {
    1925299: "0415665098d594fb5681815000f7a6312037788a3237910a9f41bb8de5c336a0",
    1996435: "64bcab2be9776172d062521d9a97e0e6f391fc9cacb331f9e42a6fe88cf8434b",
}
POSE_SIZE = {1925299: 623408231, 1996435: 644327811}


def _fetch(url: str, dest: Path, expected_size: int | None = None) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)
    return dest


def match_files(match_id: int) -> dict[str, Path]:
    d = RAW_DIR / str(match_id)
    return {
        "match": d / f"{match_id}_match.json",
        "tracking": d / f"{match_id}_tracking_extrapolated.jsonl",
        "events": d / f"{match_id}_dynamic_events.csv",
        "phases": d / f"{match_id}_phases_of_play.csv",
    }


def pose_file(match_id: int) -> Path:
    return POSE_RAW_DIR / f"{match_id}.jsonl.zip"


def sample_pose_file() -> Path:
    return POSE_RAW_DIR / "sample_1925299_phase406.jsonl.gz"


def download_match(match_id: int, with_pose: bool = True) -> None:
    files = match_files(match_id)
    for kind, path in files.items():
        base = GITHUB_MEDIA if kind == "tracking" else GITHUB_RAW
        _fetch(f"{base}/{match_id}/{path.name}", path)
    if with_pose:
        try:
            _fetch(f"{HF_POSE}/{match_id}.jsonl.zip", pose_file(match_id), POSE_SIZE[match_id])
        except Exception as exc:  # network policy, proxy, ...
            print(
                f"  !! could not download body pose for {match_id} ({exc}).\n"
                f"     Download {HF_POSE}/{match_id}.jsonl.zip manually into {POSE_RAW_DIR}",
                file=sys.stderr,
            )


def download_all(with_pose: bool = True) -> None:
    _fetch(POSE_SAMPLE_URL, sample_pose_file())
    for mid in MATCHES:
        print(f"match {mid}", file=sys.stderr)
        download_match(mid, with_pose)


def verify_pose(match_id: int) -> bool:
    h = hashlib.sha256()
    with pose_file(match_id).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest() == POSE_SHA256[match_id]
