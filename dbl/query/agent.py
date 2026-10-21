"""Natural-language access to the framework.

* With GEMINI_API_KEY set (and `google-genai` installed): Gemini answers by calling
  the typed tools in ``tools.py`` (automatic function calling). The trace of
  tool calls is returned with the answer so every number can be audited.
* Without a key: a deterministic keyword router answers the common questions
  with the same tools, so the project still runs out of the box.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from . import tools as T

SYSTEM = """You are a defensive-analysis assistant for football coaches and analysts.
You answer ONLY from the tools, which expose SkillCorner tracking + body-pose data for two A-League
2024/25 matches (Brisbane Roar v Perth Glory, Sydney FC v Adelaide United), organised by the
SEE-SHAPE-STANCE-REACT framework:
- SEE: does the defender see ball and man, check his shoulder, avoid ball-watching?
- SHAPE: is the body side-on/open, showing the carrier the right way?
- STANCE: is he 'loaded' (lower than his own norm at that speed) in duels?
- REACT: how fast does he react to runs and recover after being beaten?
Team-shape tools (match_insights, shape_by_phase, defensive_state, moments_window) add the classic analyst
view: block type, back-line height, compactness (whole block and around the ball), space between lines, free
attackers near goal, pressure on the ball, PPDA, counter-pressing, line breaks conceded.
When the user is watching the 3D viewer, the message starts with a [CONTEXT] block that ALREADY contains the
match_id, tracking frame, clock, selected player, live overlay state, defensive_state at that frame, the framework
moments in the surrounding seconds and the list of goals. Answer from it directly; call tools only for information
that is not there (e.g. goal_breakdown for a goal, player_profiles for a player's match profile). Never loop over
defensive_state frame by frame. If the context shows the ball out of play or no moments, say so and describe the
closest moments instead.
For any question about a goal (how it happened, who is to blame / guilty / responsible), call goals(match_id)
then goal_breakdown(match_id, goal_number) and answer from it: the sequence in 3-5 steps, the shape problem, then
the defenders in order of responsibility with their concrete reasons, then the coaching lesson. Say clearly that the
points are indicators from tracking + body pose, not a verdict (instructions from the coach are unknown).
For "how could it be avoided / prevented / what should they have done", call simulate_goal_prevention (goals) or
simulate_defence (any frame) and explain the key_fix first, then the other moves, with before→after numbers.
Always split the effect: reduction_parts.position_only vs posture_only (body, head, stance) — say honestly when
posture adds little. For "what should HE have done" about one player, call coach_defender(match_id, frame, player_id).
For recurring errors / what to train / training plan / drills, call error_patterns(team) or training_plan(team)
and answer with the priorities, the evidence (moments, lift, link to real shots), the players and the drill.
When asked whether a simulation or replay is realistic / trustworthy / validated, call simulation_validity() and report both what is supported and what is not.
To show how the WHOLE defence should have MOVED over time (up to 5 s, attack unchanged), call
replay_better_defence(match_id, goal_number=N) or (match_id, frame=F): explain mean / end reduction, the defenders
who defended most differently and any ball_reachable events; the viewer then plays the simulated defence.
Use at most 6 tool calls, then ALWAYS write the final answer in text.
Structure explanations like an analyst: 1) what the defence looked like (shape), 2) the key individual action
(body orientation / vision / reaction), 3) the consequence, 4) the coaching correction.
Rules:
1. Never invent numbers. Call tools; if a column is unclear call describe_framework or list_columns.
2. Always state sample size n next to any rate and say when n < 20 is too small to conclude.
3. Two matches only: present associations, not causal claims.
4. Translate results into concrete coaching language (what to correct, which moments to review: give minute + event_id).
5. Answer in the user's language. Be concise; use a small table when comparing."""


@dataclass
class Answer:
    text: str
    calls: list[dict] = field(default_factory=list)
    backend: str = "offline"
    data: dict | None = None
    focus: dict | None = None  # where the viewer should go: {frame, select, play, label}


def goal_focus(match_id: int, goal_number: int) -> dict | None:
    """Viewer target for a goal: start of the build-up (max 15 s before), most responsible defender selected."""
    try:
        b = T.goal_breakdown(int(match_id), int(goal_number))
        g = b["goal"]
        seq = b.get("sequence") or []
        f0 = max(seq[0]["frame"], g["frame"] - 150) if seq else g["frame"] - 120
        top = (b.get("responsibility") or [{}])[0].get("defender")
        return {"frame": int(f0), "select_name": top, "play": True,
                "label": f"Goal {g['goal_number']} build-up ({g['time']} {g['scorer']})"}
    except Exception:  # noqa: BLE001
        return None


def focus_from_calls(calls: list[dict]) -> dict | None:
    for c in reversed(calls):
        a = c.get("args") or {}
        if c.get("tool") == "replay_better_defence" and "match_id" in a:
            if int(a.get("goal_number") or 0) > 0:
                return {"replay": {"goal": int(a["goal_number"])}, "label": f"better-defence replay of goal {a['goal_number']}"}
            if a.get("frame"):
                return {"replay": {"frame": int(a["frame"])}, "label": "better-defence replay"}
        if c.get("tool") in ("simulate_goal_prevention", "simulate_defence") and "match_id" in a:
            try:
                if c["tool"] == "simulate_goal_prevention":
                    from ..goals import find_goals
                    g = find_goals(int(a["match_id"]))[int(a["goal_number"]) - 1]
                    return {"sim": {"goal": int(a["goal_number"])}, "label": f"simulation of goal {a['goal_number']}"}
                return {"sim": {"frame": int(a["frame"])}, "label": "simulation"}
            except Exception:  # noqa: BLE001
                return None
        if c.get("tool") == "goal_breakdown" and "match_id" in a and "goal_number" in a:
            return goal_focus(a["match_id"], a["goal_number"])
        if c.get("tool") == "open_moment" and "match_id" in a:
            try:
                d = T.open_moment(a.get("table", "runs"), a["event_id"], int(a["match_id"]))
                return {"frame": int(d.get("frame", 0)) - 30, "select_name": d.get("defender"), "play": True,
                        "label": f"{d.get('time', '')} {d.get('defender', '')} vs {d.get('attacker', '')}"}
            except Exception:  # noqa: BLE001
                return None
    return None


# ------------------------------------------------------------------ conversation memory
class Memory:
    """Per-conversation memory kept on the local server.

    * gemini: the full Gemini conversation (questions, tool calls AND tool results, answers) so follow-ups such as
      "and the second goal?", "why him?", "how could it be avoided?" are answered with the data already retrieved.
    * topic: what the conversation is about (match, goal number, frame, player) for the offline router.
    """
    MAX_CONTENTS = 40

    def __init__(self):
        self.sessions: dict[str, dict] = {}

    def get(self, sid: str | None) -> dict:
        if not sid:
            return {"gemini": [], "topic": {}, "turns": []}
        return self.sessions.setdefault(sid, {"gemini": [], "topic": {}, "turns": []})

    def reset(self, sid: str) -> None:
        self.sessions.pop(sid, None)

    @staticmethod
    def trim(contents: list) -> list:
        """Keep the last MAX_CONTENTS items, cutting only at a plain user question (never between a tool call
        and its result), and shrink very large tool results."""
        if len(contents) > Memory.MAX_CONTENTS:
            cut = len(contents) - Memory.MAX_CONTENTS
            while cut < len(contents):
                c = contents[cut]
                if c.role == "user" and any(getattr(p, "text", None) for p in (c.parts or [])):
                    break
                cut += 1
            contents = contents[cut:]
        return contents


MEMORY = Memory()


def gemini_available() -> bool:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except ImportError:
        return False


def ask_gemini(question: str, history: list[dict] | None = None, model: str | None = None,
               session: dict | None = None) -> Answer:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    contents = []
    if session is not None and session["gemini"]:
        contents = list(session["gemini"])          # full memory incl. tool calls/results
    else:
        for turn in history or []:                   # stateless fallback: text-only history from the client
            contents.append(types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM, tools=T.TOOLS, temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=8),
    )
    mdl = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    resp = client.models.generate_content(model=mdl, contents=contents, config=cfg)
    hist = list(resp.automatic_function_calling_history or [])
    calls = []
    for c in hist:
        for p in c.parts or []:
            if p.function_call:
                calls.append({"tool": p.function_call.name, "args": dict(p.function_call.args or {})})
    text = _text_of(resp)
    if not text:
        # The model stopped on a tool call (call budget exhausted) -> force a final written answer
        # from everything it already retrieved, with tools disabled.
        convo = [c for c in (hist if hist else contents) if c is not None]
        while convo and convo[-1].role == "model" and any(getattr(p, "function_call", None) for p in (convo[-1].parts or [])):
            convo.pop()  # an unanswered tool call cannot end the conversation
        convo = convo + [types.Content(role="user", parts=[types.Part(text=(
            "You have all the data you need from the tool results above. Do not call any tool. "
            "Now write the final answer to my question."))])]
        resp2 = client.models.generate_content(
            model=mdl, contents=convo,
            config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.2))
        text = _text_of(resp2)
    text = text or "(Gemini returned no text — try rephrasing)"
    if session is not None:
        convo = [c for c in (hist if hist else contents) if c is not None]
        while convo and convo[-1].role == "model" and any(getattr(p, "function_call", None) for p in (convo[-1].parts or [])):
            convo.pop()
        convo.append(types.Content(role="model", parts=[types.Part(text=text)]))
        session["gemini"] = Memory.trim(convo)
    return Answer(text=text, calls=calls, backend="gemini", focus=focus_from_calls(calls))


def _text_of(resp) -> str:
    try:
        if resp.text:
            return resp.text
    except Exception:  # noqa: BLE001
        pass
    out = []
    for c in getattr(resp, "candidates", None) or []:
        for p in (getattr(c, "content", None) and c.content.parts) or []:
            if getattr(p, "text", None):
                out.append(p.text)
    return "\n".join(out).strip()


# ------------------------------------------------------------------ offline
_METRIC_WORDS = {
    "blind": ("runs", "blind_side"), "cego": ("runs", "blind_side"), "ball-watch": ("runs", "blind_side"),
    "shoulder": ("passes", "shoulder_checks"), "ombro": ("passes", "shoulder_checks"), "scan": ("passes", "shoulder_checks"),
    "vision": ("passes", "dual_vision"), "visão": ("passes", "dual_vision"), "see": ("passes", "dual_vision"),
    "side-on": ("duels", "side_on"), "de lado": ("duels", "side_on"), "body": ("runs", "square_body"),
    "corpo": ("runs", "square_body"), "square": ("runs", "square_body"),
    "low": ("duels", "loaded_stance"), "stance": ("duels", "loaded_stance"), "base": ("duels", "loaded_stance"),
    "react": ("runs", "reaction_lag_s"), "reação": ("runs", "reaction_lag_s"), "reacao": ("runs", "reaction_lag_s"),
    "recover": ("recoveries", "recovered"), "recupera": ("recoveries", "recovered"),
    "behind": ("runs", "beaten_in_behind"), "costas": ("runs", "beaten_in_behind"),
    "regain": ("duels", "regain"), "roubada": ("duels", "regain"), "desarme": ("duels", "regain"),
}


def explain_moment_offline(match_id: int, frame: int) -> Answer:
    """Deterministic 'explain this moment' (no LLM): shape snapshot + framework moments in a ±6 s window."""
    st = T.defensive_state(match_id, frame)
    mw = T.moments_window(match_id, frame - 60, frame + 60)
    lines = []
    if "line_height" in st:
        lines.append(f"**Shape** — {st['defending_team']} defending ({str(st.get('phase_oop', '')).replace('_', ' ')}): "
                     f"back line {st['line_height']} m from goal, block {st['length']} × {st['width']} m "
                     f"(area {st.get('area', '?')} m²), ball-zone compactness {st.get('ball_zone_area', '?')} m², "
                     f"line gap {st.get('line_gap', '?')} m, {st.get('behind_ball', '?')} outfielders behind the ball, "
                     f"{st.get('press_n', 0)} within 5 m of the ball, **{st.get('free_attackers', 0)} free attacker(s)** near goal.")
    else:
        lines.append("Ball not in play at this moment.")
    for kind, rows in mw.items():
        for r in rows[:4]:
            bits = [f"{k}={v}" for k, v in r.items() if k not in ("time", "event_id", "defending_team", "defender", "attacker")
                    and v not in (None, False, "nan")]
            kname = {"duels": "duel", "passes": "pass", "runs": "run", "recoveries": "recovery"}[kind]
            lines.append(f"- {r.get('time')} **{kname}** {r.get('defender', '')} vs {r.get('attacker', '')}: " + ", ".join(bits[:6]))
    lines.append("_Offline summary — add GEMINI_API_KEY for a coach-style explanation._")
    return Answer("\n".join(lines), [{"tool": "defensive_state", "args": {"match_id": match_id, "frame": frame}},
                                     {"tool": "moments_window", "args": {"match_id": match_id, "frame_from": frame - 60,
                                                                         "frame_to": frame + 60}}])


def validity_answer_offline(v: dict) -> Answer:
    if "error" in v:
        return Answer(v["error"], [{"tool": "simulation_validity", "args": {}}])
    L = ["**Can the simulation be trusted?**"]
    m, d, r = v.get("marking"), v.get("danger"), v.get("replay")
    if m:
        c = m["completion_by_cover"]
        L.append(f"- Supported: the marking term matches real passing — passes to attackers it rates free are completed "
                 f"{c['free']:.0%} vs {c['tight']:.0%} for tightly marked ones; the chosen option is the freer one "
                 f"(AUC {m['targeted_within_pass']['free']['auc']}); agreement with SkillCorner xPass ρ={m['corr_free_vs_xpass']}.")
    if r:
        L.append(f"- Supported: simulated defenders run {r['distance_5s_m']['simulated_mean']} m in 5 s vs "
                 f"{r['distance_5s_m']['real_mean']} m real, top speed {r['speed_mps']['simulated']['max']} vs "
                 f"{r['speed_mps']['real']['max']} m/s, acceleration ≤ {r['accel_mps2']['simulated']['max']} m/s².")
    if d:
        o = d["out_of_match_auc"]
        L.append(f"- Not supported: the total danger does not predict shots better than ball location "
                 f"(out-of-match AUC {o['danger']['auc']} vs {o['ball']['auc']}): reductions are of the objective, not of shot probability.")
    linked = [k for k, o in (v.get("error_patterns_vs_real_shots") or {}).items() if o.get("linked")]
    if linked:
        L.append("- Error patterns linked to real shots: " + ", ".join(linked) + ".")
    L.append("- The exact % of a goal's replay depends on the control weight (see RESULTS.md §5.3); the direction is stable.")
    return Answer("\n".join(L), [{"tool": "simulation_validity", "args": {}}])


def training_answer_offline(plans: list[dict]) -> Answer:
    L = []
    for tp in plans:
        if "error" in tp:
            L.append(f"**{tp['team']}**: {tp['error']}")
            continue
        L.append(f"**{tp['team']} — error patterns from {tp['dangerous_moments_coached']} simulated dangerous moments**")
        for b in tp["session"]:
            L.append(f"{b['priority']}. **{b['pattern']}** ({b['principle']}) — {b['evidence']}. Focus: "
                     + ", ".join(p["defender"] for p in b["players"][:3]) + f". Drill: *{b['drill']['name']}* — {b['drill']['objective']} "
                     f"Track: {b['drill']['kpi']}.")
        L.append("Session: " + " → ".join(tp["session_outline"]))
        L.append("")
    L.append("_Model-derived priorities; the full drills are in the Match report tab and docs/TRAINING.md._")
    return Answer("\n".join(L), [{"tool": "training_plan", "args": {"team": tp.get("team")}} for tp in plans])


def replay_answer_offline(r: dict, key: dict) -> Answer:
    if "error" in r:
        return Answer(r["error"])
    g = r.get("goal")
    L = [f"**Better-defence replay — {r['defending_team']}**, {r['seconds']} s"
         + (f" before goal {g['goal_number']} ({g['scorer']}, {g['time']})" if g else "")
         + ": the attack replays exactly as it happened; every defender is re-steered every 0.2 s within physical "
           f"limits ({r['limits']['speed_mps']:.0f} m/s, {r['limits']['accel_mps2']:.0f} m/s², body {r['limits']['body_turn_dps']}°/s).",
         f"**Danger −{r['mean_reduction']:.0%} on average, −{r['end_reduction']:.0%} at the end**"
         f" (free attackers near goal at the end {r['end_real']['free_attackers']} → {r['end_sim']['free_attackers']}"
         + (f"; nearest defender to {r['focus_attacker']} {r['end_real'].get('focus_marker_dist')} → {r['end_sim'].get('focus_marker_dist')} m" if r.get("focus_attacker") else "")
         + ").", "**Who defends differently**"]
    for d in r["differences"][:5]:
        L.append(f"- {d['defender']}: ends {d['end_gap_m']} m from where he really was, on {d['ends_on']} "
                 f"({d['ends_dist_m']} m, {'goal-side' if d['goal_side'] else 'not goal-side'}), body {d['shape_end']}")
    for e in r["events"][:3]:
        L.append(f"- ⚠ {e['defender']} could reach the ball (frame {e['frame']}): from there the real play would have changed.")
    L.append("_Green bodies in the viewer = the simulated defence; press play to watch it against the real attack._")
    return Answer("\n".join(L), [{"tool": "replay_better_defence", "args": key}], focus={"replay": key, "label": "better-defence replay"})


def sim_answer_offline(r: dict, sim_key: dict) -> Answer:
    if "error" in r:
        return Answer(r["error"])
    b, a = r["before"], r["after"]
    head = (f"**Simulation — {r['defending_team']}** at the {r.get('moment', 'selected moment')}"
            + (f" of goal {r['goal']['goal_number']} ({r['goal']['scorer']}, {r['goal']['time']})" if r.get("goal") else "")
            + f": each defender could move up to {r['max_move_m']} m (reacting {r['reaction_window_s']} s earlier).")
    rp = r.get("reduction_parts")
    L = [head, f"**Danger −{r['danger_reduction']:.0%}**" + (
        f" (position alone −{rp['position_only']:.0%}, posture alone −{rp['posture_only']:.0%})" if rp else ""),
         f"- free attackers near goal: {b['free_attackers']} → {a['free_attackers']}"]
    if "focus_marker_dist" in b:
        L.append(f"- nearest defender to {r.get('focus_attacker')}: {b['focus_marker_dist']} m → {a['focus_marker_dist']} m")
    if "carrier_pressure_dist" in b:
        L.append(f"- pressure on the ball carrier: {b['carrier_pressure_dist']} m → {a['carrier_pressure_dist']} m")
    L.append(f"- open shot lane: {b['lane_open']:.2f} → {a['lane_open']:.2f}; attack-controlled space near goal: "
             f"{b['attack_space_m2']:.0f} → {a['attack_space_m2']:.0f} m²")
    if r.get("key_fix"):
        k = r["key_fix"]
        L.append(f"**Key fix:** {k['defender']} — {k['why']} ({k['move_m']} m).")
    L.append("**Other corrections**")
    for m_ in r["moves"][1:6]:
        L.append(f"- {m_['defender']}: {m_['why']} ({m_['move_m']} m)")
    L.append("_Green figures on the pitch = corrected positions; arrows = the movement needed._")
    return Answer("\n".join(L), [{"tool": "simulate", "args": sim_key}], focus={"sim": sim_key, "label": "simulation"})


def goal_answer_offline(match_id: int, n: int) -> Answer:
    b = T.goal_breakdown(match_id, n)
    g = b["goal"]
    L = [f"**Goal {n}: {g['scorer']} ({g['team']}) at {g['time']}** — {g['score_after']}"
         + (f", assisted by {g['assist']}" if g.get("assist") else "") + "."]
    L.append("**Sequence**")
    for s_ in b["sequence"][-6:]:
        L.append(f"- {s_['time']} {s_['player']} {s_['action']}" + (f" → {s_['to']}" if isinstance(s_.get('to'), str) else "")
                 + (f" (breaks {s_['line_break']})" if s_.get("line_break") else ""))
    sh = b.get("shape_at_shot") or {}
    if "line_height" in sh:
        L.append(f"**Shape at the shot** — {g['conceding']}: back line {sh['line_height']} m from goal, "
                 f"{sh.get('behind_ball', '?')} behind the ball, {sh.get('free_attackers', 0)} free attacker(s) near goal.")
    a = b.get("at_shot") or {}
    if a.get("defender"):
        L.append(f"**At the shot** — nearest defender {a['defender']} {a['distance_m']} m from the scorer "
                 f"({'goal-side' if a['goal_side'] else 'not goal-side'}), {a['defenders_in_shot_line']} defender(s) in the shot line, "
                 f"shot from {a['shot_distance_m']} m.")
    L.append("**Responsibility indicators** (not a verdict)")
    for r in b["responsibility"]:
        L.append(f"- **{r['defender']}** ({r['points']} pts): " + "; ".join(x.split(': ', 1)[-1] for x in r["reasons"]))
    return Answer("\n".join(L), [{"tool": "goals", "args": {"match_id": match_id}},
                                 {"tool": "goal_breakdown", "args": {"match_id": match_id, "goal_number": n}}],
                  focus=goal_focus(match_id, n))


def ask_offline(question: str) -> Answer:
    ctx = re.match(r"\[CONTEXT\](.*?)\n\n", question, re.S)
    mm = None
    if ctx:
        question = question[ctx.end():]
        mm, ff = re.search(r"match_id=(\d+)", ctx.group(1)), re.search(r"tracking frame=(\d+)", ctx.group(1))
    if re.search(r"realis|realist|trust|confi[aá]|valid|reliab|how good is the simul|quão real|é real", question.lower()):
        return validity_answer_offline(T.simulation_validity())
    if re.search(r"train|treino|drill|exerc|padr[õo]|pattern|recurr|recorrent|erros? (de|da|do)|mistakes", question.lower()):
        q = question.lower()
        from ..config import MATCHES
        from ..data.load import load_meta
        mid = int(mm.group(1)) if mm else next(iter(MATCHES))
        mt = load_meta(mid)
        teams = [mt.team_name(mt.home_id), mt.team_name(mt.away_id)]
        pick = [tm for tm in teams if tm.lower().split()[0] in q] or teams
        return training_answer_offline([T.training_plan(tm) for tm in pick])
    if re.search(r"replay|moviment|movement|sequ[eê]nc|5 ?s\b|segundos|seconds|over time|ao longo", question.lower()):
        q = question.lower()
        from ..config import MATCHES
        mid = int(mm.group(1)) if mm else next(iter(MATCHES))
        num = re.search(r"(?:goal|gol)\s*(\d)", q)
        gl = T.goals(mid)["goals"]
        if (re.search(r"goal|gol", q) or not (ctx and ff)) and gl:
            n = min(int(num.group(1)), len(gl)) if num else 1
            for word, k in (("second", 2), ("segundo", 2), ("third", 3), ("terceiro", 3), ("last", len(gl)), ("último", len(gl))):
                if word in q:
                    n = min(k, len(gl))
            return replay_answer_offline(T.replay_better_defence(mid, n), {"goal": n})
        if ctx and ff:
            return replay_answer_offline(T.replay_better_defence(mid, 0, int(ff.group(1))), {"frame": int(ff.group(1))})
    if re.search(r"avoid|evit|prevent|impedir|simul|stop it|parar", question.lower()):
        from ..config import MATCHES
        mid = int(mm.group(1)) if mm else next(iter(MATCHES))
        q = question.lower()
        gl = T.goals(mid)["goals"]
        n = 1
        for word, k in (("second", 2), ("segundo", 2), ("2nd", 2), ("third", 3), ("terceiro", 3), ("3rd", 3),
                        ("fourth", 4), ("quarto", 4), ("fifth", 5), ("quinto", 5), ("last", len(gl)), ("último", len(gl))):
            if word in q:
                n = min(k, len(gl))
        num = re.search(r"(?:goal|gol)\s*(\d)", q)
        if num:
            n = min(int(num.group(1)), len(gl))
        if re.search(r"goal|gol", q) and gl:
            return sim_answer_offline(T.simulate_goal_prevention(mid, n), {"goal": n})
        if ctx and ff:
            return sim_answer_offline(T.simulate_defence(mid, int(ff.group(1))), {"frame": int(ff.group(1))})
    if re.search(r"\bgoal|\bgol|guilt|culpa|blame|respons", question.lower()):
        q = question.lower()
        from ..config import MATCHES
        mid = int(mm.group(1)) if mm else next(iter(MATCHES))
        gl = T.goals(mid)["goals"]
        if gl:
            n = 1
            for word, k in (("second", 2), ("segundo", 2), ("2nd", 2), ("third", 3), ("terceiro", 3), ("3rd", 3),
                            ("fourth", 4), ("quarto", 4), ("fifth", 5), ("quinto", 5), ("last", len(gl)), ("último", len(gl))):
                if word in q:
                    n = min(k, len(gl))
            num = re.search(r"(?:goal|gol)\s*(\d)", q)
            if num:
                return goal_answer_offline(mid, min(int(num.group(1)), len(gl)))
            if re.search(r"\beach\b|\ball\b|todos|cada|goals\b|gols\b", q):
                parts = [goal_answer_offline(mid, k) for k in range(1, len(gl) + 1)]
                return Answer("\n\n".join(p.text for p in parts), [c for p in parts for c in p.calls], focus=parts[0].focus)
            return goal_answer_offline(mid, n)
    if ctx:
        if mm and ff and re.search(r"explain|expli|aconte|happen|lance|moment|situa|analis|analys|should|deveria|done|feito|correct|corrig",
                                   question.lower()):
            return explain_moment_offline(int(mm.group(1)), int(ff.group(1)))
        if mm and re.search(r"exploit|explor|attack|ataque|padr|pattern", question.lower()):
            import json as _json
            data = T.aggregate("runs", ["blind_side", "beaten_in_behind", "dangerous", "received"], ["attacking_team", "attacker"],
                               where_json=_json.dumps({"forward_run": True, "match_id": int(mm.group(1))}), min_n=4,
                               sort_by="dangerous")
            return Answer("Forward runs by attacker in this match (rates 0-1, n = runs): who started on defenders' blind "
                          "side, got in behind, and made dangerous runs.",
                          [{"tool": "aggregate", "args": {"table": "runs", "group_by": ["attacking_team", "attacker"]}}],
                          data=data)
        if mm and re.search(r"insight|relat|report|resum|summary|bloco|block|shape|forma", question.lower()):
            data = T.match_insights(int(mm.group(1)))
            txt = "\n".join(f"**{t}**\n" + "\n".join(f"- {b}" for b in bs) for t, bs in data["bullets"].items())
            return Answer(txt, [{"tool": "match_insights", "args": {"match_id": int(mm.group(1))}}])
    q = question.lower()
    s = T.store()
    team = next((t for t in s.teams if t.lower().split()[0] in q), "")
    names = s.players.drop_duplicates("player_id")
    player = next((n for n in names.name if n and n.lower().split(".")[-1].strip() in q and len(n) > 3), "")
    hit = next((v for k, v in _METRIC_WORDS.items() if k in q), None)
    if re.search(r"evid|prova|finding|achado|framework|correla", q):
        data = T.framework_evidence()
        lines = [f"- {f['cue']}: {f['exposure_label']} -> {f['outcome']} {f['rate_exposed']:.0%} (n={f['n_exposed']}) "
                 f"vs {f['rate_other']:.0%} (n={f['n_other']}), adj. diff {f['adj_diff']:+.1%} "
                 f"[{f['adj_diff_ci'][0]:+.1%}, {f['adj_diff_ci'][1]:+.1%}]"
                 for f in data["findings"]]
        return Answer("Framework evidence (2 matches, associations):\n" + "\n".join(lines),
                      [{"tool": "framework_evidence", "args": {}}], data=data)
    if player:
        data = T.player_profiles(min_moments=1, limit=100)
        row = [r for r in data["rows"] if r["defender"] == player]
        return Answer(f"Profile for {player}:", [{"tool": "player_profiles", "args": {}}], data={"rows": row})
    if hit:
        table, metric = hit
        group = ["defending_team"] if not team else ["defender"]
        where = {"defending_team": team} if team else {}
        if table == "runs":
            where["forward_run"] = True
        data = T.aggregate(table, [metric], group, where_json=__import__("json").dumps(where), min_n=5, sort_by=metric)
        return Answer(f"{metric} by {group[0]}" + (f" ({team})" if team else "") + " — rates in 0-1, with n:",
                      [{"tool": "aggregate", "args": {"table": table, "metrics": [metric], "group_by": group, "where": where}}],
                      data=data)
    data = T.team_profiles()
    return Answer("Offline mode (no GEMINI_API_KEY): showing the team defensive body-language profile. "
                  "Try questions with: blind side, shoulder checks, vision, side-on, stance, reaction, recovery, "
                  "a team or a player name, or 'evidence'.", [{"tool": "team_profiles", "args": {}}], data=data)


_ORD = (("first", 1), ("primeiro", 1), ("1st", 1), ("second", 2), ("segundo", 2), ("2nd", 2), ("third", 3),
        ("terceiro", 3), ("3rd", 3), ("fourth", 4), ("quarto", 4), ("4th", 4), ("fifth", 5), ("quinto", 5), ("5th", 5))


def _resolve_followup(question: str, topic: dict) -> str:
    """Offline follow-ups: 'and the second?', 'how to avoid it?', 'why him?' inherit the previous topic."""
    ctx = re.match(r"\[CONTEXT\](.*?)\n\n", question, re.S)
    prefix, question = (question[:ctx.end()], question[ctx.end():]) if ctx else ("", question)
    q = question.lower()
    mentions_goal = re.search(r"\bgoal|\bgol", q)
    ordinal = next((k for w, k in _ORD if re.search(rf"\b{w}\b", q)), None)
    if topic.get("goal") and not mentions_goal:
        if ordinal or re.search(r"\b(it|this|that|isso|esse|este|ele|nele|him|dele)\b", q) or \
                re.search(r"avoid|evit|prevent|impedir|simul|culpa|guilt|blame|respons|happen|aconte", q):
            n = ordinal or topic["goal"]
            question = f"{question} (goal {n})"
    return prefix + question


def _update_topic(topic: dict, question: str, ans: Answer) -> None:
    for c in ans.calls:
        a = c.get("args") or {}
        if "goal_number" in a:
            topic["goal"] = int(a["goal_number"])
        if "goal" in a:
            topic["goal"] = int(a["goal"])
        if "match_id" in a:
            topic["match"] = int(a["match_id"])
    m = re.search(r"\(goal (\d)\)", question)
    if m:
        topic["goal"] = int(m.group(1))


def ask(question: str, history: list[dict] | None = None, session_id: str | None = None) -> Answer:
    sess = MEMORY.get(session_id) if session_id else None
    if gemini_available():
        try:
            a = ask_gemini(question, history, session=sess)
            if sess is not None:
                _update_topic(sess["topic"], question, a)
            return a
        except Exception as exc:  # network, quota, bad key -> degrade gracefully
            q2 = _resolve_followup(question, sess["topic"]) if sess else question
            a = ask_offline(q2)
            if sess is not None:
                _update_topic(sess["topic"], q2, a)
            a.text = f"(Gemini unavailable: {exc.__class__.__name__}: {exc}; offline answer)\n\n" + a.text
            return a
    q2 = _resolve_followup(question, sess["topic"]) if sess else question
    a = ask_offline(q2)
    if sess is not None:
        _update_topic(sess["topic"], q2, a)
    return a
