"""Command line: `python -m dbl <command>`.

  download   fetch SkillCorner tracking/events/phases (+ body pose if reachable)
  pose       extract body-pose features from data/pose_raw/<match>.jsonl[.zip]
  build      build defensive moments + evidence for both matches
  skeletons  keep raw 3D skeletons around every moment (for the 3D viewer)
  all        download + pose + build + skeletons + fullmatch + patterns + gait + validate
  patterns   coach every dangerous moment -> error patterns + docs/TRAINING.md
  gait       running clips for the simulated defenders
  validate   test the simulation's danger score against real passes and shots
  fullmatch  full-match skeletons at 12.5 fps (for the match viewer)
  viewer     open the full-match 3D viewer (http://localhost:8765)
  ask "..."  ask a question (Gemini if GEMINI_API_KEY is set, offline otherwise)
  report     regenerate docs/RESULTS.md and the README figures
  app        launch the Streamlit interface
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import BUILD_DIR, MATCHES, POSE_RAW_DIR, ROOT


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        import os
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cmd_download(_):
    from .data.download import download_all
    download_all(with_pose=True)


def _pose_source(mid: int) -> Path | None:
    for cand in (POSE_RAW_DIR / f"{mid}.jsonl", POSE_RAW_DIR / f"{mid}.jsonl" / f"{mid}.jsonl",
                 POSE_RAW_DIR / f"{mid}.jsonl.zip"):
        if cand.is_file():
            return cand
    return None


def cmd_pose(args):
    from .pose.extract import build_pose_features
    for mid in MATCHES:
        src = _pose_source(mid)
        if src is None:
            print(f"!! no body-pose file for {mid} in {POSE_RAW_DIR} (see README: Data)", file=sys.stderr)
            continue
        print(f"match {mid}: pose features", file=sys.stderr)
        build_pose_features(mid, src, force=args.force)


def cmd_skeletons(args):
    from .pose.skeletons import build_skeletons
    for mid in MATCHES:
        src = _pose_source(mid)
        if src is None:
            print(f"!! no body-pose file for {mid} in {POSE_RAW_DIR}", file=sys.stderr)
            continue
        print(f"match {mid}: 3D skeletons", file=sys.stderr)
        build_skeletons(mid, src, force=args.force)


def cmd_build(_):
    from .moments.build import build_match
    from .store import MomentStore
    from .shape import build_shape
    for mid in MATCHES:
        print(f"match {mid}: moments", file=sys.stderr)
        build_match(mid)
        print(f"match {mid}: team shape (2 Hz)", file=sys.stderr)
        build_shape(mid)
    print("evidence (bootstrap)...", file=sys.stderr)
    ev = MomentStore().all_evidence(use_cache=False)
    print(ev[["id", "rate_exposed", "rate_other", "adj_diff", "adj_diff_ci"]].to_string(index=False))
    print(f"outputs in {BUILD_DIR}", file=sys.stderr)


def cmd_all(args):
    cmd_download(args)
    cmd_pose(args)
    cmd_build(args)
    cmd_skeletons(args)
    cmd_full(args)
    args.workers = getattr(args, "workers", 2)
    cmd_patterns(args)
    cmd_gait(args)
    cmd_validate(args)


def cmd_ask(args):
    _load_env()
    from .query.agent import ask
    a = ask(" ".join(args.question))
    print(f"[{a.backend}] {a.text}")
    if a.data:
        import json
        print(json.dumps(a.data, indent=1, ensure_ascii=False, default=str)[:4000])
    for c in a.calls:
        print(f"  tool: {c['tool']}({c['args']})", file=sys.stderr)


def cmd_viewer(args):
    from .viewer.server import serve
    serve(args.port, open_browser=not args.no_browser)


def cmd_full(args):
    from .pose.fullmatch import build_full
    for mid in MATCHES:
        src = _pose_source(mid)
        if src is None:
            print(f"!! no body-pose file for {mid} in {POSE_RAW_DIR}", file=sys.stderr)
            continue
        print(f"match {mid}: full-match skeletons", file=sys.stderr)
        build_full(mid, src, force=args.force)


def cmd_report(_):
    from .report import build_results
    from .viz.figures import build_figures
    build_figures()
    print(build_results())


def cmd_patterns(args):
    from .patterns import build_patterns, write_training_doc, _load
    for m in MATCHES:
        print(f"[{m}] coaching every dangerous moment (+ control frames)…", flush=True)
        build_patterns(m, workers=args.workers, force=args.force)
    _load.cache_clear()
    print("wrote", write_training_doc())


def cmd_gait(args):
    from .gait import build_gait
    for m in MATCHES:
        lib = build_gait(m, force=args.force)
        print(f"[{m}] gait clips for {len(lib)} players")


def cmd_validate(args):
    import json
    from .validate import validate, validate_marking, validate_replay
    from .config import BUILD_DIR
    w = getattr(args, "workers", 2)
    res = {"marking": validate_marking(workers=w, force=getattr(args, "force", False)),
           "danger": validate(workers=w, force=getattr(args, "force", False)),
           "replay": validate_replay()}
    (BUILD_DIR / "validation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    m, d = res["marking"], res["danger"]
    print(f"marking term: targeted option freer {m['targeted_within_pass']['free']['auc']} (within pass), "
          f"completion free/half/tight {m['completion_by_cover']}")
    print(f"danger vs shot in 10 s, out of match: danger {d['out_of_match_auc']['danger']['auc']}, "
          f"ball only {d['out_of_match_auc']['ball']['auc']}")
    r = res["replay"]
    print(f"replay kinematics: accel p99 sim {r['accel_mps2']['simulated']['p99']} vs real {r['accel_mps2']['real']['p99']} m/s2; "
          f"speed max sim {r['speed_mps']['simulated']['max']} vs real {r['speed_mps']['real']['max']} m/s")


def cmd_app(_):
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py")], check=False)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):      # never crash on a console/pipe that cannot print "→", "…"
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(prog="dbl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    p = sub.add_parser("pose"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_pose)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    p = sub.add_parser("skeletons"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_skeletons)
    p = sub.add_parser("all"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_all)
    p = sub.add_parser("ask"); p.add_argument("question", nargs="+"); p.set_defaults(fn=cmd_ask)
    sub.add_parser("app").set_defaults(fn=cmd_app)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    p = sub.add_parser("patterns"); p.add_argument("--force", action="store_true"); p.add_argument("--workers", type=int, default=2)
    p.set_defaults(fn=cmd_patterns)
    p = sub.add_parser("validate"); p.add_argument("--force", action="store_true"); p.add_argument("--workers", type=int, default=2)
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("gait"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_gait)
    p = sub.add_parser("viewer"); p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true"); p.set_defaults(fn=cmd_viewer)
    p = sub.add_parser("fullmatch"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_full)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
