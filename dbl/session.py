"""Printable training-session sheet built from the match: each drill starts from a REAL situation of the game.

For every priority of ``patterns.training_plan`` the most avoidable moment of the focus player is re-coached, and
the real positions of all players (attack red, defence in kit colour) plus the corrected positions (green arrows)
are drawn on a pitch diagram: "set the cones here, play it from here". Load parameters (sets × reps, work:rest,
space, numbers, RPE) and the microcycle day follow common periodisation practice for a one-game week.
"""
from __future__ import annotations

import html
from functools import lru_cache

import numpy as np

from .coach import PostureScene
from .data.load import load_meta
from .patterns import TAGS, training_plan, GRID_FAST, TORSO_FAST

LOAD = {
    "goal_side":     dict(day="MD-4", sets="4 × 6 reps", work="8–10 s", rest="40 s", space="40 × 50 m (half pitch)", players="2v2 + GK", rpe="7–8"),
    "tight_marking": dict(day="MD-3", sets="4 × 3 min", work="3 min", rest="1 min", space="40 × 35 m (box + D)", players="4v4 + 2 servers + GK", rpe="7"),
    "press_ball":    dict(day="MD-4", sets="6 × 90 s", work="90 s", rest="90 s", space="25 × 20 m", players="3v3 + 2 gates", rpe="8"),
    "line_drop":     dict(day="MD-3", sets="4 × 4 min", work="4 min", rest="1 min", space="52 × 68 m (half pitch)", players="8v6", rpe="6"),
    "line_step":     dict(day="MD-3", sets="4 × 4 min", work="4 min", rest="1 min", space="52 × 68 m (half pitch)", players="8v6", rpe="6"),
    "tuck_in":       dict(day="MD-3", sets="4 × 4 min", work="4 min", rest="90 s", space="50 × 40 m", players="6v6 + 2 neutrals", rpe="7"),
    "widen":         dict(day="MD-3", sets="4 × 3 min", work="3 min", rest="1 min", space="45 × 68 m", players="5v6", rpe="6–7"),
    "cover_space":   dict(day="MD-4", sets="5 × 2 min", work="2 min", rest="1 min", space="30 × 30 m", players="3v2 + 1", rpe="7"),
    "scan":          dict(day="MD-2", sets="4 × 2 min", work="2 min", rest="30 s", space="15 × 15 m", players="4v4 + 3", rpe="5–6"),
    "open_body":     dict(day="MD-2", sets="3 × 6 reps", work="6–8 s", rest="45 s", space="20 × 30 m", players="1v1 + server", rpe="8"),
    "get_low":       dict(day="MD-4", sets="3 × 6 reps", work="8 s", rest="40 s", space="10 × 20 m", players="1v1", rpe="8"),
}


@lru_cache(maxsize=64)
def scenario(match_id: int, frame: int, team: str, player_id: int) -> dict:
    """Real positions at the moment + the coach's corrected positions (normalised: own goal on the left)."""
    sc = PostureScene(match_id, frame, team)
    sc.grid, sc.torso_grid = GRID_FAST, TORSO_FAST
    x1, y1, *_ = sc.optimise(sweeps=2)
    meta = load_meta(match_id)
    pl = meta.players.set_index("player_id")
    def num(p):
        v = pl.number.get(p) if p in pl.index else None
        return int(v) if v is not None and v == v else ""
    D = [{"id": int(r.player_id), "n": num(r.player_id), "x": float(r.x), "y": float(r.y), "x1": float(x1[i]), "y1": float(y1[i]),
          "focus": int(r.player_id) == player_id} for i, r in sc.D.iterrows()]
    A = [{"id": int(r.player_id), "n": num(r.player_id), "x": float(r.x), "y": float(r.y)} for _, r in sc.A.iterrows()]
    G = [{"x": float(r.x), "y": float(r.y)} for _, r in sc.GK.iterrows()]
    ball = [float(sc.ball[0]), float(sc.ball[1])] if sc.ball is not None else None
    import json
    from .config import RAW_DIR
    raw = json.loads((RAW_DIR / str(match_id) / f"{match_id}_match.json").read_text(encoding="utf-8"))
    side = "home" if meta.team_name(meta.home_id) == team else "away"
    kit = (raw.get(f"{side}_team_kit") or {}).get("jersey_color") or "#2563eb"
    return {"D": D, "A": A, "GK": G, "ball": ball, "L": sc.L, "W": sc.W, "kit": kit}


def pitch_svg(sc: dict, w=520) -> str:
    L, W = sc["L"], sc["W"]
    xs = [p["x"] for p in sc["D"] + sc["A"]] + ([sc["ball"][0]] if sc["ball"] else [])
    x0 = max(-L / 2 - 2, min(xs) - 8); x1 = min(L / 2 + 2, max(max(xs) + 8, x0 + 45))
    ys = [p["y"] for p in sc["D"] + sc["A"]] + ([sc["ball"][1]] if sc["ball"] else [])
    yt = min(W / 2 + 2, max(ys) + 8); yb = max(-W / 2 - 2, min(ys) - 8)
    if yt - yb < 30:
        mid = (yt + yb) / 2; yt, yb = mid + 15, mid - 15
    s = w / (x1 - x0); h = (yt - yb) * s
    X = lambda x: (x - x0) * s  # noqa: E731
    Y = lambda y: (yt - y) * s  # noqa: E731
    o = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" style="background:#2f7d3a;border-radius:6px;max-height:420px">',
         '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4ade80"/></marker></defs>']
    ln = 'stroke="#e5f3e8" stroke-width="1.6" fill="none"'
    o.append(f'<rect x="{X(-L/2):.1f}" y="{Y(W/2):.1f}" width="{L*s:.1f}" height="{W*s:.1f}" {ln}/>')
    o.append(f'<line x1="{X(0):.1f}" y1="{Y(W/2):.1f}" x2="{X(0):.1f}" y2="{Y(-W/2):.1f}" {ln}/>')
    o.append(f'<circle cx="{X(0):.1f}" cy="{Y(0):.1f}" r="{9.15*s:.1f}" {ln}/>')
    for sg in (-1, 1):
        gx = sg * L / 2
        o.append(f'<rect x="{min(X(gx), X(gx - sg*16.5)):.1f}" y="{Y(20.16):.1f}" width="{16.5*s:.1f}" height="{40.32*s:.1f}" {ln}/>')
        o.append(f'<rect x="{min(X(gx), X(gx - sg*5.5)):.1f}" y="{Y(9.16):.1f}" width="{5.5*s:.1f}" height="{18.32*s:.1f}" {ln}/>')
        o.append(f'<rect x="{min(X(gx), X(gx + sg*1.5)):.1f}" y="{Y(3.66):.1f}" width="{1.5*s:.1f}" height="{7.32*s:.1f}" fill="#e5f3e8"/>')
    for p in sc["D"]:
        if np.hypot(p["x1"] - p["x"], p["y1"] - p["y"]) > 1.0:
            o.append(f'<line x1="{X(p["x"]):.1f}" y1="{Y(p["y"]):.1f}" x2="{X(p["x1"]):.1f}" y2="{Y(p["y1"]):.1f}" stroke="#4ade80" '
                     f'stroke-width="{2.6 if p["focus"] else 1.6}" stroke-dasharray="5 3" marker-end="url(#ar)"/>')
            o.append(f'<circle cx="{X(p["x1"]):.1f}" cy="{Y(p["y1"]):.1f}" r="{0.9*s:.1f}" fill="none" stroke="#4ade80" stroke-width="2"/>')
    for p in sc["A"]:
        o.append(f'<circle cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="{1.0*s:.1f}" fill="#dc2626" stroke="#fff" stroke-width="1"/>'
                 f'<text x="{X(p["x"]):.1f}" y="{Y(p["y"])+3.5:.1f}" font-size="{max(8, 1.1*s):.0f}" text-anchor="middle" fill="#fff" font-weight="700">{p["n"]}</text>')
    for p in sc["D"]:
        o.append(f'<circle cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="{1.0*s:.1f}" fill="{html.escape(sc["kit"])}" stroke="{"#facc15" if p["focus"] else "#111"}" '
                 f'stroke-width="{3 if p["focus"] else 1}"/><text x="{X(p["x"]):.1f}" y="{Y(p["y"])+3.5:.1f}" font-size="{max(8, 1.1*s):.0f}" text-anchor="middle" fill="#111" font-weight="700">{p["n"]}</text>')
    for g in sc["GK"]:
        o.append(f'<circle cx="{X(g["x"]):.1f}" cy="{Y(g["y"]):.1f}" r="{1.0*s:.1f}" fill="#111827" stroke="#fff"/>')
    if sc["ball"]:
        o.append(f'<circle cx="{X(sc["ball"][0]):.1f}" cy="{Y(sc["ball"][1]):.1f}" r="{0.55*s:.1f}" fill="#fff" stroke="#000"/>')
    o.append("</svg>")
    return "".join(o)


def session_sheet_html(match_id: int, team: str) -> str:
    tp = training_plan(team)
    meta = load_meta(match_id)
    opp = [meta.team_name(t) for t in (meta.home_id, meta.away_id) if meta.team_name(t) != team]
    e = html.escape
    blocks, total = [], 15 + 20
    for b in tp.get("session", []):
        d, ld = b["drill"], LOAD.get(b["tag"], {})
        ex = next((x for x in b["examples"] if x["player_id"] == b["players"][0]["player_id"]), b["examples"][0] if b["examples"] else None)
        svg, cap = "", ""
        if ex:
            try:
                svg = pitch_svg(scenario(int(ex["match_id"]), int(ex["frame"]), team, int(ex["player_id"])))
                cap = (f"Real situation, {e(ex.get('time') or '')} ({e(ex['kind'])}): {e(ex['defender'])} highlighted. "
                       f"Red = attack, kit colour = {e(team)}, green arrows = where the coach puts the defenders. "
                       f"Start the drill from these positions; own goal on the left.")
            except Exception as exc:  # noqa: BLE001
                cap = f"(scenario unavailable: {e(str(exc))})"
        board = ""
        if ex:
            from urllib.parse import urlencode
            qs = urlencode({"match": ex["match_id"], "frame": ex["frame"], "team": team, "player": ex["player_id"], "tag": b["tag"]})
            board = (f'<div class="noprint" style="margin-top:8px"><a href="/?{urlencode({"match": ex["match_id"], "f": ex["frame"], "cam": "tactical", "drill": qs})}" '
                     f'target="_blank" style="display:inline-block;background:#0f172a;color:#fff;padding:7px 12px;border-radius:8px;text-decoration:none">'
                     f'▶ Watch this drill in the 3D viewer (tactical)</a></div>')
        total += b["minutes"]
        blocks.append(f"""
<section class="blk">
  <h2>{b['priority']}. {e(d['name'])} <span class="tag">{e(b['principle'])}</span><span class="day">{e(ld.get('day',''))} · {b['minutes']}'</span></h2>
  <p class="why">Fixes <b>{e(b['pattern'].lower())}</b> — {e(b['evidence'])}. Focus: {', '.join(e(p['defender']) for p in b['players'][:3])}.</p>
  <div class="grid"><div>{svg}<p class="cap">{cap}</p></div>
  <div><table>
    <tr><th>Objective</th><td>{e(d['objective'])}</td></tr><tr><th>Set-up</th><td>{e(d['setup'])}</td></tr>
    <tr><th>Rules</th><td>{e(d['rules'])}</td></tr>
    <tr><th>Load</th><td>{e(ld.get('sets',''))} · work {e(ld.get('work',''))} / rest {e(ld.get('rest',''))} · RPE {e(ld.get('rpe',''))}</td></tr>
    <tr><th>Space · players</th><td>{e(ld.get('space',''))} · {e(ld.get('players',''))}</td></tr>
    <tr><th>Coaching points</th><td><ul>{''.join(f'<li>{e(c)}</li>' for c in d['coaching'])}</ul></td></tr>
    <tr><th>Progression</th><td>{e(d['progression'])}</td></tr>
    <tr><th>Check next match</th><td>{e(d['kpi'])}</td></tr></table></div></div>
  {board}
</section>""")
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(team)} — defensive session</title><style>
body{{font:14px/1.45 Inter,system-ui,sans-serif;color:#0f172a;margin:24px auto;max-width:1100px;padding:0 16px;background:#fff}}
h1{{font-size:24px;margin:0}} .sub{{color:#475569;margin:4px 0 14px}} h2{{font-size:17px;margin:18px 0 4px}}
.tag{{font-size:11px;background:#fef3c7;color:#92400e;border-radius:6px;padding:2px 6px;margin-left:8px;vertical-align:middle}}
.day{{float:right;font-size:13px;color:#334155}} .why{{color:#334155;margin:2px 0 8px}}
.grid{{display:grid;grid-template-columns:1.1fr 1fr;gap:16px}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th{{text-align:left;vertical-align:top;color:#475569;width:120px;padding:4px 6px}} td{{padding:4px 6px}}
tr{{border-bottom:1px solid #e2e8f0}} ul{{margin:0;padding-left:18px}} .cap{{font-size:12px;color:#475569}}
.outline{{background:#f1f5f9;border-radius:8px;padding:10px 12px}} .blk{{break-inside:avoid;border-top:2px solid #e2e8f0;padding-top:6px}}
@media print{{body{{margin:0}} .noprint{{display:none}}}}
</style></head><body>
<button class="noprint" onclick="print()" style="float:right">Print / PDF</button>
<h1>{e(team)} — defensive training session</h1>
<div class="sub">Built from the simulations of {tp.get('dangerous_moments_coached', 0)} dangerous moments vs {e(', '.join(opp))} (SkillCorner tracking + body pose).
Total ≈ {total}' · squad 18 + 2 GK · one-game week: tactical-intensive day MD-4/MD-3, reaction day MD-2.</div>
<div class="outline"><b>Session:</b> {' → '.join(e(x) for x in tp.get('session_outline', []))}</div>
{''.join(blocks)}
<p class="cap" style="margin-top:18px">{e(tp.get('note',''))}</p></body></html>"""


# ------------------------------------------------------------------ animated drill board ("prancheta")
FORMAT = {  # (defenders, attackers) used on the board for each drill
    "goal_side": (2, 2), "tight_marking": (4, 4), "press_ball": (3, 3), "line_drop": (6, 7), "line_step": (6, 7),
    "tuck_in": (6, 6), "widen": (5, 6), "cover_space": (3, 3), "scan": (4, 4), "open_body": (1, 1), "get_low": (1, 1),
}


@lru_cache(maxsize=32)
def drill_board(match_id: int, frame: int, team: str, player_id: int, tag: str) -> dict:
    """Everything the tactics board needs to animate a drill built on a real moment:
    5 s around the moment (3 s before, 2 s after) — attackers and ball as they really moved, the selected defenders as
    they really moved ("what happened") and as the better-defence replay moves them ("what we train")."""
    from .replay import replay
    from .simulate import _data
    from .patterns import DRILLS
    meta, fr, xy, _ = _data(match_id)
    start = int(frame) - 30
    rp = replay(match_id, start, 5.0, team, None)
    team_id = meta.home_id if meta.team_name(meta.home_id) == team else meta.away_id
    s = 1.0 if meta.defends_left(team_id, int(fr.loc[rp["start"], "period"])) else -1.0   # own goal on the left
    pl = meta.players.set_index("player_id")
    frames = [f["frame"] for f in rp["frames"]]
    key = min(range(len(frames)), key=lambda k: abs(frames[k] - frame))
    snap = xy.loc[[frames[key]]].reset_index()
    snap["team"] = snap.player_id.map(pl.team_id)
    snap["gk"] = snap.player_id.map(pl.is_gk).fillna(False).astype(bool)
    fx, fy = snap.set_index("player_id").loc[player_id, ["x", "y"]] if player_id in set(snap.player_id) else (0.0, 0.0)
    nd, na = FORMAT.get(tag, (4, 4))
    d_all = snap[(snap.team == team_id) & ~snap.gk]
    a_all = snap[(snap.team != team_id) & ~snap.gk]
    dsel = list(d_all.assign(d=np.hypot(d_all.x - fx, d_all.y - fy)).sort_values("d").player_id[:nd])
    if player_id not in dsel:
        dsel = [player_id] + dsel[:-1]
    bx, by = fr.loc[frames[key], ["ball_x", "ball_y"]]
    a_sorted = a_all.assign(d=np.hypot(a_all.x - fx, a_all.y - fy), db=np.hypot(a_all.x - bx, a_all.y - by))
    carrier = list(a_sorted.sort_values("db").player_id[:1])
    asel = carrier + [p for p in a_sorted.sort_values("d").player_id if p not in carrier][:max(0, na - 1)]
    gk = snap[(snap.team == team_id) & snap.gk].player_id.tolist()[:1]
    num = lambda p: int(pl.number.get(p)) if p in pl.index and pl.number.get(p) == pl.number.get(p) else ""  # noqa: E731
    name = lambda p: str(pl.name.get(p, ""))  # noqa: E731
    out = []
    for k, f in enumerate(frames):
        sn = xy.loc[[f]].set_index("player_id") if f in xy.index else None
        sim = {d["player_id"]: d for d in rp["frames"][k]["defenders"]}
        pos = lambda p: [round(float(sn.loc[p, "x"]) * s, 2), round(float(sn.loc[p, "y"]) * s, 2)] if sn is not None and p in sn.index else None  # noqa: E731
        b = fr.loc[f, ["ball_x", "ball_y"]] if f in fr.index else None
        out.append({"t": round((f - frame) / 10, 1),
                    "A": [pos(p) for p in asel], "D": [pos(p) for p in dsel],
                    "S": [[round(sim[p]["x"] * s, 2), round(sim[p]["y"] * s, 2), round(sim[p]["torso"] + (0 if s > 0 else 180), 1)]
                          if p in sim else pos(p) for p in dsel],
                    "G": [pos(p) for p in gk],
                    "ball": [round(float(b.ball_x) * s, 2), round(float(b.ball_y) * s, 2)] if b is not None and b.ball_x == b.ball_x else None})
    kit = scenario(match_id, frame, team, player_id)["kit"]
    pts = [q for f in out for q in (f["A"] + f["D"] + f["S"]) if q]
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    d = DRILLS.get(tag, {})
    return {"match_id": match_id, "frame": frame, "team": team, "tag": tag, "kit": kit, "s": s, "start": int(rp["start"]),
            "end": int(rp["end"]), "defender_ids": [int(p) for p in dsel], "attacker_ids": [int(p) for p in asel], "gk_ids": [int(p) for p in gk],
            "drill": d, "load": LOAD.get(tag, {}), "format": f"{nd}v{na}" + (" + GK" if gk else ""),
            "defenders": [{"id": int(p), "n": num(p), "name": name(p), "focus": int(p) == int(player_id)} for p in dsel],
            "attackers": [{"id": int(p), "n": num(p), "name": name(p), "carrier": p in carrier} for p in asel],
            "area": [max(-52.5, min(xs) - 5), max(-34, min(ys) - 5), min(52.5, max(xs) + 5), min(34, max(ys) + 5)],
            "frames": out, "key_t": 0.0, "L": 105.0, "W": 68.0,
            "danger": {"real": [f["danger_real"] for f in rp["frames"]], "sim": [f["danger_sim"] for f in rp["frames"]]}}
