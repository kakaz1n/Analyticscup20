"""Local server for the full-match 3D viewer.

    python -m dbl viewer            # opens http://localhost:8765

Endpoints (all read-only, local only):
    /api/matches                          available matches
    /api/meta?match=ID                    teams, colours, players, periods, bones, timeline events
    /api/pose?match=ID&from=F&to=F        binary skeletons for tracking frames [from, to] (10 fps clock)
    /api/track?match=ID&from=F&to=F       tracking positions + ball (JSON)
Static viewer files are served from dbl/viewer/static/.
"""
from __future__ import annotations

import json
import struct
import threading
import webbrowser
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from ..config import BUILD_DIR, MATCHES, POSE_FPS, TRACKING_FPS
from ..data.load import load_meta
from ..pose.fullmatch import BLOCK_F, full_dir
from ..pose.skeletons import BONES, JOINTS

STATIC = Path(__file__).parent / "static"
_lock = threading.Lock()


def available_matches() -> list[int]:
    return [m for m in MATCHES if list(full_dir(m).glob("pose_*.parquet"))]


@lru_cache(maxsize=4)
def _tracking(match_id: int):
    xy = pd.read_parquet(BUILD_DIR / str(match_id) / "players_xy.parquet", columns=["frame", "player_id", "x", "y"])
    fr = pd.read_parquet(BUILD_DIR / str(match_id) / "frames.parquet")
    return xy.sort_values("frame").reset_index(drop=True), fr.sort_values("frame").reset_index(drop=True)


@lru_cache(maxsize=8)
def _block(match_id: int, period: int, blk: int) -> pd.DataFrame | None:
    p = full_dir(match_id) / f"pose_{period}_{blk:03d}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def pose_window(match_id: int, f10_from: int, f10_to: int) -> bytes:
    a, b = int(f10_from * POSE_FPS / TRACKING_FPS), int(np.ceil(f10_to * POSE_FPS / TRACKING_FPS))
    parts = []
    with _lock:
        for blk in range(a // BLOCK_F, b // BLOCK_F + 1):
            for per in (1, 2):
                df = _block(match_id, per, blk)
                if df is not None:
                    parts.append(df[(df.frame25 >= a) & (df.frame25 <= b)])
    df = pd.concat(parts) if parts else pd.DataFrame(columns=["frame25", "player_id"])
    n, nj = len(df), len(JOINTS)
    head = struct.pack("<ii", n, nj)
    if n == 0:
        return head
    f = df.frame25.to_numpy(np.int32)
    pid = df.player_id.to_numpy(np.int32)
    jt = df.iloc[:, 3:].to_numpy(np.int16)
    return head + f.tobytes() + pid.tobytes() + np.ascontiguousarray(jt).tobytes()


def track_window(match_id: int, f10_from: int, f10_to: int) -> dict:
    xy, fr = _tracking(match_id)
    w = xy[(xy.frame >= f10_from) & (xy.frame <= f10_to)]
    f = fr[(fr.frame >= f10_from) & (fr.frame <= f10_to)]
    players = {}
    for pid, g in w.groupby("player_id"):
        players[int(pid)] = [g.frame.tolist(), np.round(g.x, 2).tolist(), np.round(g.y, 2).tolist()]
    ball = [f.frame.tolist(), np.round(f.ball_x.astype(float), 2).fillna(-999).tolist(),
            np.round(f.ball_y.astype(float), 2).fillna(-999).tolist(),
            np.round(f.ball_z.astype(float), 2).fillna(0).tolist(), f.poss_group.fillna("").tolist()]
    return {"players": players, "ball": ball}


def _events(match_id: int, meta) -> list[dict]:
    out = []
    d = BUILD_DIR / str(match_id)

    def rd(name):
        p = d / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    runs, passes, duels, recs = rd("runs"), rd("passes"), rd("duels"), rd("recoveries")

    def tb(v) -> bool:
        return bool(v) if v is not None and pd.notna(v) else False

    fwd = {"run_ahead_of_the_ball", "behind", "cross_receiver", "overlap", "underlap", "pulling_half_space"}
    for r in runs.itertuples():
        if r.subtype not in fwd:
            continue
        flags = [x for x, ok in (("beaten in behind", tb(r.beaten_in_behind)), ("blind side", tb(r.blind_side)),
                                 ("square", getattr(r, "body_shape", None) == "square"), ("dangerous", tb(r.dangerous))) if ok]
        out.append(dict(f=int(r.frame), kind="run", key=tb(r.beaten_in_behind) or tb(r.dangerous),
                        label=f"Run ({r.subtype.replace('_', ' ')}): {r.attacker} vs {r.defender}" + (f" · {', '.join(flags)}" if flags else ""),
                        defender=int(r.defender_id), attacker=int(r.attacker_id)))
    for r in passes.itertuples():
        if not (tb(r.line_break) or tb(r.lead_to_shot)):
            continue
        dv = getattr(r, "vision", None)
        out.append(dict(f=int(r.frame), kind="pass", key=tb(r.lead_to_shot),
                        label=f"Line-breaking pass {r.passer} → {r.attacker}" + (f" · marker {r.defender}: {dv}" if isinstance(dv, str) else ""),
                        defender=int(r.defender_id) if pd.notna(r.defender_id) else None, attacker=int(r.attacker_id)))
    for r in duels.itertuples():
        if not (tb(r.regain) or tb(r.beaten)):
            continue
        out.append(dict(f=int(r.frame), kind="duel", key=tb(r.beaten),
                        label=f"{'Regain' if tb(r.regain) else 'Beaten'}: {r.defender} on {r.attacker}"
                              + (f" · {r.body_shape}" if isinstance(getattr(r, 'body_shape', None), str) else ""),
                        defender=int(r.defender_id), attacker=int(r.attacker_id)))
    for r in recs.itertuples():
        out.append(dict(f=int(r.frame), kind="recovery", key=not tb(r.recovered),
                        label=f"Recovery {r.defender}: {'back goal-side' if tb(r.recovered) else 'not recovered'}",
                        defender=int(r.defender_id), attacker=int(r.attacker_id) if pd.notna(r.attacker_id) else None))
    return sorted(out, key=lambda e: e["f"])


def _clockmap(match_id: int) -> list[list[int]]:
    """[match_second, period, tracking_frame] samples from Dynamic Events, to turn 'mm:ss' in answers into frames."""
    from ..data.load import load_events
    ev = load_events(match_id)
    ev = ev.dropna(subset=["minute_start", "second_start", "frame_start"])
    ev = ev.assign(sec=(ev.minute_start * 60 + ev.second_start).astype(int))
    g = ev.groupby(["period", "sec"]).frame_start.min().reset_index()
    return [[int(r.sec), int(r.period), int(r.frame_start)] for r in g.itertuples()]


@lru_cache(maxsize=4)
def match_meta(match_id: int) -> dict:
    meta = load_meta(match_id)
    raw = meta.raw
    _, fr = _tracking(match_id)
    periods = [{"period": int(p), "start": int(g.frame.min()), "end": int(g.frame.max())}
               for p, g in fr.dropna(subset=["period"]).groupby("period")]
    pl = meta.players
    players = [dict(id=int(r.player_id), name=r.name, number=int(r.number) if pd.notna(r.number) else None,
                    team=int(r.team_id), role=r.role, gk=bool(r.is_gk)) for r in pl.itertuples()]
    teams = {}
    for side in ("home", "away"):
        t, kit = raw[f"{side}_team"], raw.get(f"{side}_team_kit") or {}
        teams[side] = dict(id=t["id"], name=t["short_name"], acronym=t.get("acronym"),
                           color=kit.get("jersey_color") or ("#2563eb" if side == "home" else "#dc2626"),
                           number_color=kit.get("number_color") or "#ffffff")
    return dict(id=match_id, title=meta.title, date=raw.get("date_time"), pitch=list(meta.pitch),
                home_side=list(raw.get("home_team_side") or []),
                teams=teams, players=players, periods=periods, joints=list(JOINTS),
                bones=[[JOINTS.index(a), JOINTS.index(b)] for a, b in BONES],
                fps=dict(pose=POSE_FPS, tracking=TRACKING_FPS), events=_events(match_id, meta),
                clockmap=_clockmap(match_id))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, *args):  # quiet
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if not (u.path.startswith("/api/") or u.path.startswith("/training/")):
            return super().do_GET()
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/matches":
                body = [{"id": m, "title": match_meta(m)["title"]} for m in available_matches()]
                return self._send(json.dumps(body).encode(), "application/json")
            m = int(q["match"])
            if u.path == "/api/meta":
                return self._send(json.dumps(_clean(match_meta(m)), default=str).encode(), "application/json")
            if u.path == "/api/insights":
                return self._send(json.dumps(_clean(match_insights(m)), default=_jsonable).encode(), "application/json")
            if u.path == "/api/series":
                from ..shape import series
                return self._send(json.dumps(_clean(series(m))).encode(), "application/json")
            if u.path == "/api/goals":
                return self._send(json.dumps(_clean(match_goals(m)), default=_jsonable).encode(), "application/json")
            if u.path == "/api/simulate":
                res = _simulate_cached(m, q.get("goal"), q.get("frame"), q.get("team") or None, q.get("focus"),
                                       q.get("posture", "1") != "0", q.get("player"))
                return self._send(json.dumps(_clean(res), default=_jsonable).encode(), "application/json")
            if u.path == "/api/replay":
                from ..replay import replay, replay_goal
                sec = float(q.get("seconds", 5))
                cv = q.get("mode") == "clairvoyant"
                res = replay_goal(m, int(q["goal"]), sec, cv) if q.get("goal") else \
                    replay(m, int(q["frame"]), sec, q.get("team") or None, int(q["focus"]) if q.get("focus") else None, cv)
                return self._send(json.dumps(_clean(res), default=_jsonable).encode(), "application/json")
            if u.path == "/training/sheet":
                from ..session import session_sheet_html
                return self._send(session_sheet_html(m, q["team"]).encode(), "text/html; charset=utf-8")
            if u.path == "/api/drill":
                from ..session import drill_board
                res = drill_board(m, int(q["frame"]), q["team"], int(q["player"]), q["tag"])
                return self._send(json.dumps(_clean(res), default=_jsonable).encode(), "application/json")
            if u.path == "/api/gait":
                from ..gait import gait_json
                return self._send(gait_json(m), "application/json")
            if u.path == "/api/training":
                from ..patterns import training_plan
                from ..data.load import load_meta
                mt = load_meta(m)
                res = {"plans": [training_plan(mt.team_name(t)) for t in (mt.home_id, mt.away_id)]}
                return self._send(json.dumps(_clean(res), default=_jsonable).encode(), "application/json")
            if u.path == "/api/players":
                return self._send(json.dumps(_clean(player_cards(m)), default=_jsonable).encode(), "application/json")
            a, b = int(q["from"]), int(q["to"])
            if u.path == "/api/pose":
                return self._send(pose_window(m, a, b), "application/octet-stream")
            if u.path == "/api/track":
                return self._send(json.dumps(_clean(track_window(m, a, b))).encode(), "application/json")
            return self._send(b'{"error":"unknown endpoint"}', "application/json", 404)
        except Exception as exc:  # noqa: BLE001
            return self._send(json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(), "application/json", 500)


    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/reset":
            from ..query.agent import MEMORY
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            MEMORY.reset(body.get("session", ""))
            return self._send(b'{"ok":true}', "application/json")
        if u.path == "/api/whatif":
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                from ..coach import whatif
                res = whatif(int(body["match"]), int(body["frame"]), body.get("edits") or [],
                             int(body["focus"]) if body.get("focus") else None, body.get("team") or None)
                return self._send(json.dumps(_clean(res), default=_jsonable).encode(), "application/json")
            except Exception as exc:  # noqa: BLE001
                return self._send(json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(), "application/json", 500)
        if u.path != "/api/ask":
            return self._send(b'{"error":"unknown endpoint"}', "application/json", 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            ans = ask_with_context(body)
            return self._send(json.dumps(_clean(ans), default=_jsonable, ensure_ascii=False).encode(), "application/json")
        except Exception as exc:  # noqa: BLE001
            return self._send(json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(), "application/json", 500)


@lru_cache(maxsize=64)
def _simulate_cached(m, goal, frame, team, focus, posture, player):
    from ..simulate import simulate, simulate_goal
    only = [int(player)] if player else None
    if goal:
        return simulate_goal(m, int(goal), posture=posture, only=only)
    return simulate(m, int(frame), team, int(focus) if focus else None, posture=posture, only=only)


def _clean(o):
    """Recursively replace NaN/inf (invalid JSON for browsers) with None and numpy scalars with Python ones."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


@lru_cache(maxsize=4)
def match_insights(match_id: int) -> dict:
    from ..query.tools import store
    from ..shape import insights
    return insights(match_id, store())


@lru_cache(maxsize=4)
def match_goals(match_id: int) -> list[dict]:
    from ..goals import find_goals, goal_breakdown
    from ..query.tools import store
    return [goal_breakdown(match_id, g["goal_number"], store()) for g in find_goals(match_id)]


@lru_cache(maxsize=4)
def player_cards(match_id: int) -> dict:
    """Per-player SEE-SHAPE-STANCE-REACT profile (both matches pooled) for the viewer's player card."""
    from ..query.tools import store
    pp = store().player_profile(min_moments=1)
    ids = {p["id"] for p in match_meta(match_id)["players"]}
    pp = pp[pp.defender_id.isin(ids)]
    return {int(r.defender_id): {k: v for k, v in r._asdict().items() if k != "Index"} for r in pp.itertuples()}


def ask_with_context(body: dict) -> dict:
    """Question from the viewer + what the analyst is looking at -> Gemini (or offline) answer."""
    from ..cli import _load_env
    from ..query.agent import ask
    _load_env()
    m, f = body.get("match"), body.get("frame")
    ctx = []
    if m is not None:
        meta = match_meta(int(m))
        ctx.append(f"match_id={m} ({meta['title']})")
    if f is not None:
        ctx.append(f"tracking frame={int(f)} (clock {body.get('clock', '?')})")
    if body.get("selected"):
        pid = int(body["selected"])
        p = next((x for x in match_meta(int(m))["players"] if x["id"] == pid), None)
        if p:
            ctx.append(f"selected player={p['name']} (id {pid}, #{p['number']}, {p['role']})")
    if body.get("live"):
        ctx.append("live overlay state=" + json.dumps(body["live"], default=_jsonable)[:1500])
    if m is not None and f is not None:
        # pre-compute what the analyst is looking at, so the model needs few (or no) tool calls
        from ..query import tools as T
        from ..goals import find_goals
        try:
            ctx.append("defensive_state=" + json.dumps(T.defensive_state(int(m), int(f)), default=_jsonable))
            mw = T.moments_window(int(m), int(f) - 80, int(f) + 40)
            ctx.append("moments_in_window(-8s..+4s)=" + json.dumps({k: v[:6] for k, v in mw.items() if v},
                                                                    default=_jsonable, ensure_ascii=False)[:3500])
            ctx.append("goals=" + json.dumps([{k: g[k] for k in ("goal_number", "time", "frame", "team", "scorer", "assist",
                                                                  "score_after")} for g in find_goals(int(m))],
                                             default=_jsonable, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            ctx.append(f"(context tools failed: {exc})")
    q = body.get("question", "").strip()
    text = (f"[CONTEXT] {'; '.join(ctx)}\n\n" if ctx else "") + q
    hist = body.get("history") or []
    a = ask(text, hist[-6:], session_id=body.get("session"))
    focus = a.focus
    if focus and focus.get("select_name") and m is not None:
        p = next((x for x in match_meta(int(m))["players"] if x["name"] == focus["select_name"]), None)
        focus["select"] = p["id"] if p else None
    return {"text": a.text, "backend": a.backend, "calls": a.calls, "data": a.data, "focus": focus}


_server: ThreadingHTTPServer | None = None


def start(port: int = 8765, background: bool = True) -> int:
    """Start the viewer server (idempotent). Returns the port."""
    global _server
    if _server is not None:
        return _server.server_address[1]
    try:
        _server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:  # already running (e.g. another Streamlit rerun) -> reuse
        return port
    if background:
        threading.Thread(target=_server.serve_forever, daemon=True).start()
    return port


def serve(port: int = 8765, open_browser: bool = True) -> None:
    start(port, background=True)
    url = f"http://localhost:{port}/"
    print(f"3D match viewer on {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
