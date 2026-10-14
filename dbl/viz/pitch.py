"""Draw a defensive moment: every player, the ball, and -- for players with body
pose -- the torso direction (arrow) and the head's field of view (cone).

Coordinates are defender-normalised (the defending team defends the LEFT goal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TH
from ..geometry import circmean
from ..moments.context import MatchContext

DEF_COLOR, ATT_COLOR, FOCUS_COLOR, BALL_COLOR = "#2563eb", "#dc2626", "#f59e0b", "#111827"


def moment_snapshot(ctx: MatchContext, row: dict) -> dict:
    """Collect positions + pose headings at the moment's frame."""
    team_name = row["defending_team"]
    team = ctx.meta.home_id if ctx.meta.team_name(ctx.meta.home_id) == team_name else ctx.meta.away_id
    per, f = int(row["period"]), int(row["frame"])
    snap = ctx.players_at(f, team, per)
    bx, by = ctx.ball_at(f, team, per)
    heads, torsos = [], []
    for pid in snap.player_id:
        pw = ctx.pose_window(int(pid), f - 1, f + 1, team, per)
        heads.append(np.nan if pw is None else pw.head)
        torsos.append(np.nan if pw is None else pw.torso)
    snap = snap.assign(head=heads, torso=torsos)
    snap["name"] = snap.player_id.map(ctx.name_of)
    return {"players": snap, "ball": (bx, by), "L": ctx.L, "W": ctx.W,
            "focus": {row.get("defender_id"), row.get("attacker_id")}, "row": row}


def _cone(x, y, heading, r=9.0, half=TH.fov_half_deg, n=16):
    a = np.radians(np.linspace(heading - half, heading + half, n))
    return np.r_[x, x + r * np.cos(a), x], np.r_[y, y + r * np.sin(a), y]


def _pitch_lines(L, W):
    hl, hw = L / 2, W / 2
    segs = [([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw]), ([0, 0], [-hw, hw])]
    for s in (-1, 1):
        segs.append(([s * hl, s * (hl - 16.5), s * (hl - 16.5), s * hl], [-20.16, -20.16, 20.16, 20.16]))
        segs.append(([s * hl, s * (hl - 5.5), s * (hl - 5.5), s * hl], [-9.16, -9.16, 9.16, 9.16]))
    t = np.linspace(0, 2 * np.pi, 60)
    segs.append((9.15 * np.cos(t), 9.15 * np.sin(t)))
    return segs


def plotly_moment(snapshot: dict, show_all_cones: bool = True, title: str = ""):
    import plotly.graph_objects as go

    P, (bx, by), L, W = snapshot["players"], snapshot["ball"], snapshot["L"], snapshot["W"]
    fig = go.Figure()
    for xs, ys in _pitch_lines(L, W):
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines", line=dict(color="#9ca3af", width=1),
                                 hoverinfo="skip", showlegend=False))
    for _, p in P.iterrows():
        focus = p.player_id in snapshot["focus"]
        if not np.isnan(p["head"]) and (show_all_cones or focus) and p.defending:
            cx, cy = _cone(p.x, p.y, p["head"])
            fig.add_trace(go.Scatter(x=cx, y=cy, fill="toself", mode="lines", line=dict(width=0),
                                     fillcolor="rgba(245,158,11,0.35)" if focus else "rgba(37,99,235,0.12)",
                                     hoverinfo="skip", showlegend=False))
        if not np.isnan(p["torso"]):
            t = np.radians(p["torso"])
            fig.add_annotation(x=p.x + 2.2 * np.cos(t), y=p.y + 2.2 * np.sin(t), ax=p.x, ay=p.y,
                               xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowsize=1, arrowwidth=2,
                               arrowcolor=FOCUS_COLOR if focus else (DEF_COLOR if p.defending else ATT_COLOR))
    for side, col, nm in ((True, DEF_COLOR, "defending"), (False, ATT_COLOR, "attacking")):
        g = P[P.defending == side]
        fig.add_trace(go.Scatter(
            x=g.x, y=g.y, mode="markers", name=nm,
            marker=dict(size=[16 if pid in snapshot["focus"] else 11 for pid in g.player_id], color=col,
                        line=dict(color=[FOCUS_COLOR if pid in snapshot["focus"] else "white" for pid in g.player_id], width=3)),
            text=g.name, hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[bx], y=[by], mode="markers", name="ball",
                             marker=dict(size=9, color=BALL_COLOR, line=dict(color="white", width=1))))
    fig.update_layout(title=title, height=520, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="#f0fdf4",
                      xaxis=dict(range=[-L / 2 - 3, L / 2 + 3], visible=False),
                      yaxis=dict(range=[-W / 2 - 3, W / 2 + 3], visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.02))
    return fig


def mpl_moment(snapshot: dict, ax=None, xlim=None, ylim=None, title: str = "", label_focus: bool = True,
               show_all_cones: bool = True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    P, (bx, by), L, W = snapshot["players"], snapshot["ball"], snapshot["L"], snapshot["W"]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor("#f0fdf4")
    for xs, ys in _pitch_lines(L, W):
        ax.plot(xs, ys, color="#9ca3af", lw=1, zorder=1)
    if xlim and ylim:
        P = P[P.x.between(xlim[0] + 1.5, xlim[1] - 1.5) & P.y.between(ylim[0] + 1.5, ylim[1] - 1.5)]
    for _, p in P.iterrows():
        focus = p.player_id in snapshot["focus"]
        if not np.isnan(p["head"]) and p.defending and (show_all_cones or focus):
            cx, cy = _cone(p.x, p.y, p["head"])
            ax.add_patch(Polygon(np.c_[cx, cy], closed=True, lw=0, zorder=2,
                                 fc=(0.96, 0.62, 0.04, 0.35) if focus else (0.15, 0.39, 0.92, 0.10)))
        if not np.isnan(p["torso"]):
            t = np.radians(p["torso"])
            ax.annotate("", xy=(p.x + 2.4 * np.cos(t), p.y + 2.4 * np.sin(t)), xytext=(p.x, p.y), zorder=4,
                        annotation_clip=True,
                        arrowprops=dict(arrowstyle="-|>", lw=1.8,
                                        color=FOCUS_COLOR if focus else (DEF_COLOR if p.defending else ATT_COLOR)))
        ax.scatter(p.x, p.y, s=140 if focus else 70, zorder=5, clip_on=True, c=DEF_COLOR if p.defending else ATT_COLOR,
                   edgecolors=FOCUS_COLOR if focus else "white", linewidths=2.5 if focus else 1)
        if focus and label_focus:
            ax.text(p.x + 0.9, p.y + 1.1, p["name"], fontsize=9, zorder=6, weight="bold")
    ax.scatter(bx, by, s=45, c=BALL_COLOR, edgecolors="white", zorder=6)
    ax.set_xlim(*(xlim or (-L / 2 - 2, L / 2 + 2)))
    ax.set_ylim(*(ylim or (-W / 2 - 2, W / 2 + 2)))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    return ax


__all__ = ["moment_snapshot", "plotly_moment", "mpl_moment", "circmean", "pd"]
