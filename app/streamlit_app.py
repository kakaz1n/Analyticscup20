"""Defensive Body Language -- analyst interface.

Run:  python -m dbl app   (or: streamlit run app/streamlit_app.py)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dbl.cli import _load_env  # noqa: E402
from dbl.config import MATCHES  # noqa: E402
from dbl.framework import METRICS, PRINCIPLES  # noqa: E402
from dbl.store import MomentStore  # noqa: E402

_load_env()
st.set_page_config(page_title="Defensive Body Language", layout="wide")


@st.cache_resource
def get_store() -> MomentStore:
    return MomentStore()


@st.cache_resource
def get_ctx(match_id: int):
    from dbl.moments.context import MatchContext
    return MatchContext(match_id)


s = get_store()
if s.tables["duels"].empty:
    st.error("No built data found. Run `python -m dbl all` first (see README).")
    st.stop()

st.title("Defensive Body Language")
st.caption("SEE · SHAPE · STANCE · REACT — what defenders' bodies tell us, from SkillCorner tracking + body pose "
           f"({len(MATCHES)} A-League 2024/25 matches).")

tab_over, tab_team, tab_play, tab_mom, tab_match, tab_att, tab_ask = st.tabs(
    ["Framework", "Teams", "Players", "Moments", "Match 3D", "Attack patterns", "Ask (Gemini)"])


@st.cache_resource
def viewer_port() -> int | None:
    from dbl.viewer.server import available_matches, start
    return start(8765) if available_matches() else None


def viewer_url(match_id=None, frame=None, focus=None, cam=None) -> str:
    q = {k: v for k, v in dict(match=match_id, f=frame, focus=focus, cam=cam).items() if v is not None}
    return f"http://localhost:{viewer_port()}/?" + "&".join(f"{k}={int(v) if isinstance(v, float) else v}" for k, v in q.items())

# ------------------------------------------------------------------ framework
with tab_over:
    cols = st.columns(4)
    for c, (k, v) in zip(cols, PRINCIPLES.items()):
        c.markdown(f"**{k}**  \n{v}")
    st.subheader("Does the cue matter? (stratified risk difference, 95% bootstrap CI)")
    ev = s.all_evidence()
    show = ev.assign(
        exposed=lambda d: d.rate_exposed.map("{:.0%}".format) + " (n=" + d.n_exposed.astype(str) + ")",
        other=lambda d: d.rate_other.map("{:.0%}".format) + " (n=" + d.n_other.astype(str) + ")",
        adjusted=lambda d: d.adj_diff.map("{:+.1%}".format) + "  [" +
        d.adj_diff_ci.map(lambda c: f"{c[0]:+.1%}, {c[1]:+.1%}") + "]",
    )[["principle", "cue", "exposure_label", "outcome", "exposed", "other", "adjusted", "strata"]]
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("Two matches: these are associations to guide video review, not causal effects. "
               "CI excluding 0 = consistent pattern in this sample.")
    with st.expander("Metric dictionary"):
        st.dataframe(pd.DataFrame(METRICS).T[["principle", "cue", "table", "definition"]], use_container_width=True)

# ---------------------------------------------------------------------- teams
with tab_team:
    st.dataframe(s.team_profile(), use_container_width=True, hide_index=True)
    st.caption("Rates are 0-1. reaction_lag_s = mean seconds the defender reacts after the runner (forward runs).")

# -------------------------------------------------------------------- players
with tab_play:
    pp = s.player_profile(min_moments=10)
    c1, c2 = st.columns([1, 3])
    team = c1.selectbox("Team", ["All"] + s.teams, key="pt")
    sort = c1.selectbox("Sort by", [c for c in pp.columns if pp[c].dtype.kind == "f"], index=0)
    if team != "All":
        pp = pp[pp.team == team]
    c2.dataframe(pp.sort_values(sort, ascending=False), use_container_width=True, hide_index=True, height=520)

# -------------------------------------------------------------------- moments
with tab_mom:
    fam = st.radio("Moment family", ["runs", "passes", "duels", "recoveries"], horizontal=True)
    df = s.tables[fam]
    f1, f2, f3, f4 = st.columns(4)
    t = f1.selectbox("Defending team", ["All"] + s.teams, key="mt")
    if t != "All":
        df = df[df.defending_team == t]
    dnames = sorted(df.defender.dropna().unique())
    dsel = f2.selectbox("Defender", ["All"] + dnames)
    if dsel != "All":
        df = df[df.defender == dsel]
    cat_cols = [c for c in ("body_shape", "vision", "subtype", "phase_oop", "third", "shows", "source") if c in df]
    cc = f3.selectbox("Filter column", ["-"] + cat_cols)
    if cc != "-":
        vals = f4.multiselect("Values", sorted(df[cc].dropna().astype(str).unique()))
        if vals:
            df = df[df[cc].astype(str).isin(vals)]
    q = st.text_input("Advanced filter (pandas query)", placeholder="e.g. blind_side == 1 and beaten_in_behind == 1")
    if q:
        try:
            df = df.query(q, engine="python")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Query error: {exc}")
    st.write(f"**{len(df)}** moments")
    key_cols = [c for c in ["match_id", "time", "defending_team", "defender", "attacker", "subtype", "vision",
                            "shoulder_checks", "body_shape", "body_angle", "shows", "loaded_stance", "blind_side",
                            "reaction_lag_s", "receiver_pressure", "line_break", "beaten_in_behind", "dangerous",
                            "regain", "carrier_outcome", "recovered", "event_id"] if c in df]
    ev_sel = st.dataframe(df[key_cols], use_container_width=True, hide_index=True, height=300,
                          on_select="rerun", selection_mode="single-row")
    rows = ev_sel.selection.rows if ev_sel and ev_sel.selection else []
    if rows:
        row = df.iloc[rows[0]].to_dict()
        from dbl.viz.pitch import moment_snapshot, plotly_moment
        from dbl.viz.scene3d import build_scene
        ctx = get_ctx(int(row["match_id"]))
        a, b = st.columns([3, 1])
        mode = a.radio("View", ["3D skeletons", "2D pitch"], horizontal=True, label_visibility="collapsed")
        try:
            if mode == "3D skeletons":
                a.plotly_chart(build_scene(ctx, row), use_container_width=True)
                a.caption("Real SkillCorner skeletons (body pose, 25 fps → 6.25 fps clip, ±1.2 s). Orange = focal "
                          "defender, brown = attacker; the cone is his head's ±60° field of view. Drag to rotate, ▶ to play.")
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            snap = moment_snapshot(ctx, row)
            a.plotly_chart(plotly_moment(snap, title=f"{row['time']} · {row.get('defender')} vs {row.get('attacker')}"),
                           use_container_width=True)
        if viewer_port():
            b.link_button("▶ Watch in full-match 3D viewer",
                          viewer_url(int(row["match_id"]), int(row["frame"]),
                                     int(row["defender_id"]) if pd.notna(row.get("defender_id")) else None, "follow"))
        b.markdown("**Moment detail**")
        keep = set(METRICS) | {"receiver_pressure", "carrier_outcome", "dangerous", "line_break",
                               "beaten_in_behind", "regain", "phase_oop", "head_to_ball_deg", "head_to_man_deg",
                               "head_to_runner_deg", "marking_distance", "distance_m", "knee_extra_deg"}
        detail = {k: (round(float(v), 2) if isinstance(v, float) else v) for k, v in row.items() if k in keep and pd.notna(v)}
        b.json(json.loads(json.dumps(detail, default=lambda o: round(float(o), 2) if hasattr(o, "__float__") else str(o))),
               expanded=True)
        b.caption("Defending team always defends the left goal (-x). 2D view: cones = head field of view, "
                  "arrows = torso direction.")
    else:
        st.info("Select a row to see the moment on the pitch with every defender's field of view.")

# ------------------------------------------------------------------ match 3D
with tab_match:
    if not viewer_port():
        st.info("Full-match skeletons not built yet: run `python -m dbl fullmatch` (needs the body-pose files).")
    else:
        st.caption("The whole match in 3D: real body-pose skeletons (12.5 fps, interpolated) for every player the "
                   "broadcast sees; translucent 'ghosts' from tracking for the rest. Timeline ticks = framework moments "
                   "(click to jump). Cameras: broadcast, tactical, behind goal, follow player, **player's eyes**.")
        st.link_button("Open full screen", viewer_url())
        import streamlit.components.v1 as components
        components.iframe(viewer_url(), height=820)

# ------------------------------------------------------------ attack patterns
with tab_att:
    st.markdown("How attackers exploit defenders' bodies: runs started on the **blind side**, against **square** "
                "defenders, and passes to receivers whose marker could not see ball and man.")
    r = s.tables["runs"]
    r = r[r.forward_run]
    att = st.selectbox("Attacking team", sorted(r.attacking_team.unique()))
    ra = r[r.attacking_team == att]
    c1, c2 = st.columns(2)
    c1.markdown("**Runners: blind-side starts and outcomes**")
    c1.dataframe(ra.groupby("attacker").agg(runs=("event_id", "size"), blind_side=("blind_side", "mean"),
                                           vs_square=("square_body", "mean"), got_in_behind=("beaten_in_behind", "mean"),
                                           dangerous=("dangerous", "mean"), received=("received", "mean"))
                 .query("runs >= 3").sort_values("runs", ascending=False).round(2), use_container_width=True)
    c2.markdown("**Where: channel × blind side**")
    c2.dataframe(ra.groupby(["channel", "blind_side"]).agg(runs=("event_id", "size"), dangerous=("dangerous", "mean"))
                 .round(2), use_container_width=True)
    p = s.tables["passes"]
    pa = p[(p.attacking_team == att) & p.vision.notna()]
    st.markdown("**Receivers targeted while their marker could not see ball and man**")
    st.dataframe(pa.groupby("attacker").agg(passes=("event_id", "size"), marker_not_dual=("dual_vision", lambda x: 1 - x.mean()),
                                            line_break=("line_break", "mean"), free=("receiver_free", "mean"))
                 .query("passes >= 5").sort_values("passes", ascending=False).round(2), use_container_width=True)

# -------------------------------------------------------------------------- ask
with tab_ask:
    from dbl.query.agent import ask, gemini_available
    st.caption(("Gemini connected — answers are computed by tool calls on the tables (shown below each answer)."
                if gemini_available() else
                "Offline mode: set GEMINI_API_KEY in .env for free-form questions. Keyword answers still work."))
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
    qtxt = st.chat_input("e.g. Which Sydney defenders get caught square when runs start? Show the moments.")
    if qtxt:
        st.session_state.chat.append({"role": "user", "text": qtxt})
        with st.chat_message("user"):
            st.markdown(qtxt)
        with st.chat_message("assistant"):
            with st.spinner("Computing..."):
                hist = [{"role": "model" if m["role"] == "assistant" else "user", "text": m["text"]}
                        for m in st.session_state.chat[:-1]][-6:]
                if "sid" not in st.session_state:
                    import uuid
                    st.session_state.sid = uuid.uuid4().hex
                ans = ask(qtxt, hist, session_id=st.session_state.sid)
            st.markdown(ans.text)
            if ans.data:
                rows = ans.data.get("rows") or ans.data.get("findings")
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if ans.calls:
                with st.expander(f"{len(ans.calls)} tool call(s) · {ans.backend}"):
                    st.code(json.dumps(ans.calls, indent=1, ensure_ascii=False, default=str))
        st.session_state.chat.append({"role": "assistant", "text": ans.text})
