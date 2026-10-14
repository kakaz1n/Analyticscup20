"""The two README figures (the Analytics Cup allows at most two figures/tables)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import ROOT  # noqa: E402
from ..moments.context import MatchContext  # noqa: E402
from ..store import MomentStore  # noqa: E402
from .pitch import moment_snapshot, mpl_moment  # noqa: E402

COLORS = {"SEE": "#2a78d6", "SHAPE": "#eb6834", "STANCE": "#1baf7a", "REACT": "#eda100"}
INK, MUTED = "#1f2937", "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED})


def fig_evidence(store: MomentStore, path):
    ev = store.all_evidence().iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    for i, r in ev.iterrows():
        lo, hi = r.adj_diff_ci
        c = COLORS[r.principle]
        sig = lo > 0 or hi < 0
        ax.plot([lo * 100, hi * 100], [i, i], color=c, lw=2, solid_capstyle="round", zorder=2, alpha=1 if sig else .45)
        ax.scatter(r.adj_diff * 100, i, s=60, color=c if sig else "white", edgecolor=c, lw=2, zorder=3)
        ax.text(hi * 100 + 1.2, i, f"{r.adj_diff * 100:+.1f} pp", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(ev)))
    ax.set_yticklabels([f"{r.principle} · {r.cue}\n{r.exposure_label} → {r.outcome.replace('_', ' ')}"
                        f"  (n={r.n_exposed}/{r.n_other})" for r in ev.itertuples()], fontsize=7.6, color=INK)
    ax.set_xlabel("Stratified difference in outcome rate, percentage points (95% bootstrap CI)", color=INK)
    ax.grid(axis="x", color="#e5e7eb", lw=.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(min(-25, ev.adj_diff_ci.map(lambda c: c[0]).min() * 100 - 2), 40)
    ax.set_title("Which coaching cues show up in the outcomes? (filled = CI excludes 0)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _pick(store: MomentStore):
    r = store.tables["runs"]
    base = r[r.forward_run & (r.pose_frames > 0) & r.runner_distance.between(3.5, 7.5) & (r.blind_side.notna())]
    bad = base[(base.body_shape == "square") & (base.beaten_in_behind == 1)].sort_values("reaction_lag_s", ascending=False)
    good = base[(base.body_shape == "open") & (base.beaten_in_behind == 0) & (base.runner_in_view == 1)
                & (base.runner_x - base.x).between(1.5, 6) & (base.dangerous == 0)]

    def first_off_ball(cands):
        for _, row in cands.iterrows():
            ctx = MatchContext(int(row.match_id))
            snap = moment_snapshot(ctx, row.to_dict())
            bx, by = snap["ball"]
            if np.hypot(bx - row.runner_x, by - row.runner_y) > 8:
                return row.to_dict()
        return cands.iloc[0].to_dict()

    b = first_off_ball(bad.head(15))
    gs = good[good.subtype == b["subtype"]]
    g = first_off_ball((gs if len(gs) else good).sort_values("runner_distance", ascending=False).head(15))
    return g, b


def fig_moments(store: MomentStore, path):
    good, bad = _pick(store)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    for ax, row, tag in ((axes[0], good, "OPEN body, runner in view"), (axes[1], bad, "SQUARE body → beaten in behind")):
        ctx = MatchContext(int(row["match_id"]))
        snap = moment_snapshot(ctx, row)
        cx, cy = row["x"], row["y"]
        mpl_moment(snap, ax=ax, xlim=(cx - 20, cx + 16), ylim=(cy - 13, cy + 13), show_all_cones=False,
                   title=f"{tag}\n{row['defending_team']} · {row['time']} · {row['defender']} vs {row['attacker']} "
                         f"· reaction lag {row.get('reaction_lag_s', np.nan):.1f}s")
    fig.text(0.01, 0.01, "Blue = defending team (defends the left goal), red = attackers, black = ball. "
             "Arrows: torso direction. Cone: the marked defender's head field of view (±60°).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_figures(store: MomentStore | None = None):
    store = store or MomentStore()
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    fig_evidence(store, out / "fig1_evidence.png")
    fig_moments(store, out / "fig2_moments.png")
    return out


if __name__ == "__main__":
    print(build_figures())
