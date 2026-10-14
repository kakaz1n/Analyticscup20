"""3D moment viewer: real SkillCorner skeletons on a pitch, the focal defender's
field-of-view cone projected from his head, the ball, and a 2.4 s animation.

Coordinates are defender-normalised (defending team defends the goal at x = -L/2).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import POSE_FPS, TH, TRACKING_FPS
from ..pose.skeletons import BONES, JOINTS, load
from .pitch import _pitch_lines

DEF, ATT, FOCUS_D, FOCUS_A, BALL = "#2a78d6", "#d03b2f", "#f59e0b", "#7c2d12", "#111827"
FEET = ("lAnkle", "rAnkle", "lHeel", "rHeel", "lBigToe", "rBigToe")


@lru_cache(maxsize=2)
def skeletons(match_id: int) -> pd.DataFrame:
    return load(match_id)


def _frame_players(sk: pd.DataFrame, f: int, s: float) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    """{player_id: (len(JOINTS), 3) array} for one pose frame (normalised, feet on the ground) + ball xyz."""
    rows = sk[sk.frame25 == f]
    out, ball = {}, None
    for _, r in rows.iterrows():
        if r.player_id == -1:
            ball = np.array([r.nose_x * s, r.nose_y * s, r.nose_z])
            continue
        a = np.array([[r[f"{j}_x"] * s, r[f"{j}_y"] * s, r[f"{j}_z"]] for j in JOINTS], dtype=float)
        feet = [JOINTS.index(j) for j in FEET]
        g = np.nanmin(a[feet, 2]) if np.isfinite(a[feet, 2]).any() else 0.0
        a[:, 2] -= g if np.isfinite(g) else 0.0
        out[int(r.player_id)] = a
    return out, ball


def _bones(players: dict[int, np.ndarray], ids) -> tuple[list, list, list]:
    xs, ys, zs = [], [], []
    for pid in ids:
        a = players[pid]
        for b0, b1 in BONES:
            p, q = a[JOINTS.index(b0)], a[JOINTS.index(b1)]
            if np.isfinite(p).all() and np.isfinite(q).all():
                xs += [p[0], q[0], None]; ys += [p[1], q[1], None]; zs += [p[2], q[2], None]
    return xs, ys, zs


def _head(a: np.ndarray) -> tuple[np.ndarray, float] | None:
    le, re, no = (a[JOINTS.index(j)] for j in ("lEar", "rEar", "nose"))
    if not (np.isfinite(le).all() and np.isfinite(re).all() and np.isfinite(no).all()):
        return None
    d = re - le
    perp = np.degrees(np.arctan2(d[0], -d[1]))  # left->right ear axis rotated +90
    mid = (le + re) / 2
    nv = no - mid
    nose = np.degrees(np.arctan2(nv[1], nv[0]))
    h = np.degrees(np.arctan2(np.sin(np.radians(perp)) + np.sin(np.radians(nose)),
                              np.cos(np.radians(perp)) + np.cos(np.radians(nose))))
    return mid, h


def _cone(apex: np.ndarray, heading: float, r: float = 9.0, half: float = TH.fov_half_deg, n: int = 18):
    """Field-of-view fan from the head down to the grass: Mesh3d vertices + triangles."""
    ang = np.radians(np.linspace(heading - half, heading + half, n))
    xs = np.r_[apex[0], apex[0] + r * np.cos(ang)]
    ys = np.r_[apex[1], apex[1] + r * np.sin(ang)]
    zs = np.r_[apex[2], np.full(n, 0.02)]
    i = np.zeros(n - 1, int); j = np.arange(1, n); k = np.arange(2, n + 1)
    return xs, ys, zs, i, j, k


def build_scene(ctx, row: dict, animate: bool = True, window: tuple[float, float] = (20, 13)):
    import plotly.graph_objects as go

    team_name = row["defending_team"]
    team = ctx.meta.home_id if ctx.meta.team_name(ctx.meta.home_id) == team_name else ctx.meta.away_id
    s = ctx.flip(team, int(row["period"]))
    sk = skeletons(int(row["match_id"]))
    if sk.empty:
        raise FileNotFoundError("no skeletons_p*.parquet for this match (run `python -m dbl skeletons`)")
    key = int(round(float(row["frame"]) * POSE_FPS / TRACKING_FPS))
    have = set(sk.frame25[(sk.frame25 >= key - 32) & (sk.frame25 <= key + 32)].unique())
    clip = sorted({key + o for o in range(-30, 31, 4) if key + o in have} | ({key} & have))  # own 6.25 fps clip
    if key not in have and have:
        key = min(have, key=lambda f: abs(f - key))
        clip = sorted(set(clip) | {key})
    frames = clip if animate and clip else [key]
    dfd, att = row.get("defender_id"), row.get("attacker_id")
    dfd = int(dfd) if pd.notna(dfd) else None
    att = int(att) if pd.notna(att) else None
    defenders = {pid for pid, t in ctx.team_of.items() if t == team}

    # tracking positions at the key frame for players without a skeleton (context dots)
    snap = ctx.players_at(int(row["frame"]), team, int(row["period"]))
    pts = [(float(row.get("x", 0)), float(row.get("y", 0)))]
    for kx, ky in (("runner_x", "runner_y"), ("receiver_x", "receiver_y"), ("carrier_x", "carrier_y")):
        if pd.notna(row.get(kx)):
            pts.append((float(row[kx]), float(row[ky])))
    bx, by = ctx.ball_at(int(row["frame"]), team, int(row["period"]))
    if np.isfinite(bx):
        pts.append((bx, by))
    cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])  # frame the action, not just the defender
    wx, wy = window
    box = dict(x=[cx - wx, cx + wx], y=[cy - wy, cy + wy])

    def traces(f):
        players, ball = _frame_players(sk, f, s)
        players = {p: a for p, a in players.items()
                   if box["x"][0] - 5 < np.nanmean(a[:, 0]) < box["x"][1] + 5 and box["y"][0] - 5 < np.nanmean(a[:, 1]) < box["y"][1] + 5}
        out = []
        for ids, col, w, nm in (
            ([p for p in players if p in defenders and p != dfd], DEF, 4, "defenders"),
            ([p for p in players if p not in defenders and p != att], ATT, 4, "attackers"),
            ([p for p in players if p == dfd], FOCUS_D, 8, f"defender: {row.get('defender')}"),
            ([p for p in players if p == att], FOCUS_A, 8, f"attacker: {row.get('attacker')}"),
        ):
            x, y, z = _bones(players, ids)
            out.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=col, width=w), name=nm,
                                    hoverinfo="skip", showlegend=True))
        # head cone of the focal defender
        cone = go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], color=FOCUS_D, opacity=0.25, name="field of view",
                         hoverinfo="skip", showlegend=True)
        if dfd in players and (h := _head(players[dfd])) is not None:
            xs, ys, zs, i, j, k = _cone(h[0], h[1])
            cone = go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=FOCUS_D, opacity=0.25,
                             name="field of view (±60°)", hoverinfo="skip", showlegend=True)
        out.append(cone)
        b = ball if ball is not None and np.isfinite(ball).all() else np.array([np.nan] * 3)
        out.append(go.Scatter3d(x=[b[0]], y=[b[1]], z=[max(b[2], 0.11) if np.isfinite(b[2]) else None], mode="markers",
                                marker=dict(size=6, color=BALL, line=dict(color="white", width=1)), name="ball"))
        return out

    static = []
    # grass + pitch lines inside the window
    gx, gy = box["x"], box["y"]
    static.append(go.Mesh3d(x=[gx[0], gx[1], gx[1], gx[0]], y=[gy[0], gy[0], gy[1], gy[1]], z=[0, 0, 0, 0],
                            i=[0, 0], j=[1, 2], k=[2, 3], color="#86c995", opacity=1, hoverinfo="skip",
                            showlegend=False, flatshading=True))
    lx, ly = [], []
    for xs, ys in _pitch_lines(ctx.L, ctx.W):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        inside = (xs >= gx[0]) & (xs <= gx[1]) & (ys >= gy[0]) & (ys <= gy[1])
        xs, ys = np.where(inside, xs, np.nan), np.where(inside, ys, np.nan)
        lx += list(xs) + [None]; ly += list(ys) + [None]
    static.append(go.Scatter3d(x=lx, y=ly, z=[0.01 if v is not None else None for v in lx], mode="lines",
                               line=dict(color="white", width=3), hoverinfo="skip", showlegend=False))
    # context dots from tracking (players outside the pose clip)
    ctxp = snap[snap.x.between(*gx) & snap.y.between(*gy)]
    static.append(go.Scatter3d(x=ctxp.x, y=ctxp.y, z=np.full(len(ctxp), 0.05), mode="markers",
                               marker=dict(size=3, color=[DEF if d else ATT for d in ctxp.defending], opacity=0.45),
                               text=ctxp.player_id.map(ctx.name_of), hovertemplate="%{text}<extra>tracking</extra>",
                               name="tracking position"))
    key_f = min(frames, key=lambda f: abs(f - key))
    fig = go.Figure(data=static + traces(key_f))
    n_static = len(static)
    if animate and len(frames) > 1:
        fig.frames = [go.Frame(data=traces(f), traces=list(range(n_static, n_static + 6)),
                               name=f"{(f - key) / POSE_FPS:+.2f}s") for f in frames]
        steps = [dict(method="animate", label=fr.name,
                      args=[[fr.name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                 for fr in fig.frames]
        fig.update_layout(
            updatemenus=[dict(type="buttons", x=0.02, y=0.02, xanchor="left", yanchor="bottom", direction="left",
                              buttons=[dict(label="▶ play", method="animate",
                                            args=[None, dict(frame=dict(duration=160, redraw=True), fromcurrent=True)]),
                                       dict(label="⏸", method="animate",
                                            args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
            sliders=[dict(active=frames.index(key_f), steps=steps, x=0.15, len=0.83, y=0.02,
                          currentvalue=dict(prefix="t = ", font=dict(size=12)))])
    zmax = 6
    fig.update_layout(
        height=640, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
        scene=dict(
            xaxis=dict(range=gx, visible=False), yaxis=dict(range=gy, visible=False), zaxis=dict(range=[0, zmax], visible=False),
            aspectmode="manual", aspectratio=dict(x=1, y=(gy[1] - gy[0]) / (gx[1] - gx[0]), z=zmax / (gx[1] - gx[0])),
            camera=dict(eye=dict(x=-0.78, y=-0.42, z=0.30), center=dict(x=0, y=0, z=-0.12)),
        ),
        title=dict(text=f"{row.get('time', '')} · {row.get('defender')} vs {row.get('attacker')}", x=0.01, font=dict(size=14)),
    )
    return fig
