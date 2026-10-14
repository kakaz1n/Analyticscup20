"""The SEE-SHAPE-STANCE-REACT framework: coaching cues -> measurable pose metrics.

Every metric below is (1) a cue coaches already shout from the touchline,
(2) computed from SkillCorner body pose + tracking at a moment defined by
SkillCorner Dynamic Events, and (3) linked to an outcome the analyst cares about.
"""
from __future__ import annotations

PRINCIPLES = {
    "SEE": "Perception: does the defender gather the information he needs (ball, man, space behind)?",
    "SHAPE": "Body orientation: is the body positioned to act on that information (side-on, open, showing the right way)?",
    "STANCE": "Readiness: is the defender loaded (lower / more flexed than his own norm at that speed) when it matters?",
    "REACT": "Response: how fast does the body turn and move when the threat appears, and does he recover after being beaten?",
}

# name -> metadata. table = which moment family carries it.
METRICS: dict[str, dict] = {
    # ---------------------------------------------------------------- SEE
    "vision": dict(principle="SEE", table="passes", kind="category",
                   cue="See ball and man", values=["ball_and_man", "ball_only", "man_only", "neither"],
                   definition="At the pass moment, are the ball and the receiver each within +/-60 deg of the receiver's "
                              "marker's head heading (ear axis + nose vector)?"),
    "shoulder_checks": dict(principle="SEE", table="passes", kind="count", cue="Check your shoulder",
                            definition="Number of times the receiver's marker turned his head >75 deg away from the ball "
                                       "for >=0.12 s in the 3 s before the pass."),
    "ball_man_split_deg": dict(principle="SEE", table="passes", kind="deg", cue="Get in a position to see both",
                               definition="Angle between ball and receiver as seen from the marker. >120 deg: impossible to "
                                          "see both without turning -- a positioning problem, not a perception one."),
    "blind_side": dict(principle="SEE", table="runs", kind="bool", cue="Don't ball-watch",
                       definition="Run started outside the nearest defender's field of view while he was looking at the ball."),
    "runner_in_view": dict(principle="SEE", table="runs", kind="bool", cue="Know where your runner is",
                           definition="Runner within +/-60 deg of the defender's head heading at run start."),
    # -------------------------------------------------------------- SHAPE
    "body_shape": dict(principle="SHAPE", table="duels|runs", kind="category",
                       cue="Get side-on / open your body",
                       definition="Duels: torso vs line to carrier (<25 square, 25-70 side-on, >70 turned). "
                                  "Runs: torso vs own goal (<45 turned, 45-135 open, >135 square/flat)."),
    "body_angle": dict(principle="SHAPE", table="duels", kind="deg", cue="Side-on in the duel",
                       definition="Absolute angle between torso heading (shoulders+hips) and the line to the carrier."),
    "shows": dict(principle="SHAPE", table="duels", kind="category", cue="Show him outside / inside",
                  values=["outside", "inside", "none"],
                  definition="Side the defender's chest opens toward (the side he invites the carrier), mapped to "
                             "touchline (outside) vs centre (inside)."),
    "torso_to_goal_deg": dict(principle="SHAPE", table="runs", kind="deg", cue="Open up to your own goal",
                              definition="Angle between torso and the own-goal direction at run start (0 = facing own goal)."),
    # ------------------------------------------------------------- STANCE
    "loaded_stance": dict(principle="STANCE", table="duels", kind="bool", cue="Get low / on your toes",
                          definition="Hip >=3 cm lower or knees >=5 deg more flexed than the player's own norm at the "
                                     "same running speed (gait-adjusted)."),
    "hip_drop_m": dict(principle="STANCE", table="duels", kind="m", cue="Get low",
                       definition="Speed-adjusted hip-height residual (positive = lower than usual)."),
    "approach_decel": dict(principle="STANCE", table="duels", kind="m/s", cue="Slow down before you arrive",
                           definition="Peak speed in the last 1 s minus speed at engagement (tracking)."),
    # -------------------------------------------------------------- REACT
    "reaction_lag_s": dict(principle="REACT", table="runs", kind="s", cue="React to the run",
                           definition="Time between the runner and the defender each first reaching 2.5 m/s goal-ward."),
    "recovered": dict(principle="REACT", table="recoveries", kind="bool", cue="Recover goal-side",
                      definition="After being dribbled / run in behind, back between threat and goal within 3 s."),
    "turned_within_1s": dict(principle="REACT", table="recoveries", kind="bool", cue="Turn and run",
                             definition="Torso within 60 deg of own goal within 1 s of being beaten."),
}

OUTCOMES: dict[str, dict] = {
    "receiver_free": dict(table="passes", definition="Receiver under no pressure at all at reception (SkillCorner overall_pressure_start = no_pressure)."),
    "line_break": dict(table="passes", definition="Pass broke at least one defensive line (SkillCorner)."),
    "pass_success": dict(table="passes", definition="Pass completed."),
    "lead_to_shot": dict(table="passes|runs", definition="Possession led to a shot."),
    "beaten_in_behind": dict(table="runs", definition="Runner started level/behind the defender and finished goal-side of him."),
    "dangerous": dict(table="runs", definition="SkillCorner 'dangerous' run."),
    "received": dict(table="runs", definition="Runner received the ball."),
    "regain": dict(table="duels", definition="Engagement ended in a direct/indirect regain."),
    "carrier_progressed": dict(table="duels", definition="Pressed carrier still shot, broke a line or passed forward."),
    "beaten": dict(table="duels", definition="Defender beaten by the carrier's possession or movement."),
    "recovered": dict(table="recoveries", definition="Back goal-side within 3 s."),
}

# The pre-registered comparisons the README reports (exposure, outcome, strata).
FINDINGS = [
    dict(id="checks_line_break", principle="SEE", table="passes", cue="Check your shoulder",
         exposure="checked", exposure_label=">=1 shoulder check in 3 s", outcome="line_break",
         strata=["third", "marking_band"], better="lower"),
    dict(id="dual_vision_free", principle="SEE", table="passes", cue="See ball and man",
         exposure="dual_vision", exposure_label="ball AND man in view", outcome="receiver_free",
         strata=["third", "marking_band"], better="lower", filter="ball_man_split_deg <= 120"),
    dict(id="open_body_runs", principle="SHAPE", table="runs", cue="Open your body to the runner",
         exposure="square_body", exposure_label="square/flat at run start", outcome="beaten_in_behind",
         strata=["third"], better="lower", filter="forward_run"),
    dict(id="blind_side_runs", principle="SEE", table="runs", cue="Don't ball-watch",
         exposure="blind_side", exposure_label="run started on the blind side", outcome="dangerous",
         strata=["subtype"], better="lower", filter="forward_run"),
    dict(id="loaded_regain", principle="STANCE", table="duels", cue="Get low",
         exposure="loaded_stance", exposure_label="loaded stance", outcome="regain",
         strata=["subtype", "third"], better="higher"),
    dict(id="side_on_progress", principle="SHAPE", table="duels", cue="Side-on in the duel",
         exposure="side_on", exposure_label="side-on", outcome="carrier_progressed",
         strata=["subtype", "distance_band"], better="lower"),
    dict(id="turn_recover", principle="REACT", table="recoveries", cue="Turn and run",
         exposure="turned_within_1s", exposure_label="turned within 1 s", outcome="recovered",
         strata=["source"], better="higher"),
]
