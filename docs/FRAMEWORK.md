# Defensive Body Language — full framework reference

This document is the long-form companion to the README (which is limited to 1,000 words). Metric definitions and
thresholds mirror `dbl/framework.py` and `dbl/config.py`.

## 0. The framework is the 3D viewer with Gemini inside it

The deliverable for analysts is not a set of metrics: it is a **loop played on the real match**.

| Step | What the analyst does | What the system provides | Where |
|---|---|---|---|
| **Watch** | Plays the full match in 3D, picks a camera (incl. *Player's eyes*), selects a defender | Real skeletons at 12.5 fps, live overlays (shape, marking, lanes, space), event timeline | §11 |
| **Ask** | Types a question about what is on screen; follows up | Gemini with the frame as context, 17 read-only tools, conversation memory; answers jump the viewer to the cited moment | §12 |
| **Diagnose** | Reads the computed answer | Team shape layer (§12), body-language layer SEE-SHAPE-STANCE-REACT (§2–5), goal responsibility (§13) | §2–§13 |
| **Fix** | Asks "how could it be avoided?" | Counterfactual positions + posture reachable within 1.5 s, a 5 s better-defence replay, an editor | §14 |
| **Train** | Plans the week | Recurring error patterns from all simulated dangerous moments → drills + KPIs | §15 |

![Watch + Ask: goal 1 explained in the viewer](img/viewer_gemini_goal.png)

*Watch + Ask + Diagnose: the analyst asks about the first goal; the answer (sequence, shape at the shot,
responsibility indicators) comes from tools, and the viewer is already at the build-up.*

![Fix: follow-up simulation](img/viewer_gemini_simulation.png)

*Fix: the follow-up "How could it be avoided?" (no goal named — the chat remembers) runs the simulation and draws
the corrected positions on the pitch.*

Everything numeric is computed by the Python package; the LLM only selects tools, reads their JSON and writes the
explanation. The same tools power a deterministic offline mode, so the framework works without an API key.

## 1. Why body pose changes defensive analysis

Classical defensive metrics (PPDA, pressures, compactness, line height, pitch control) describe *positions*. Coaching feedback is mostly about *orientation and readiness*: where the defender is looking, how his hips are turned, whether he is loaded to move. Those were invisible until body pose. The framework keeps the coach's vocabulary and makes each cue a number measured at a defined moment, linked to an outcome, and reviewable on the pitch.

## 2. Principles

- **SEE** — Perception: does the defender gather the information he needs (ball, man, space behind)?
- **SHAPE** — Body orientation: is the body positioned to act on that information (side-on, open, showing the right way)?
- **STANCE** — Readiness: is the defender loaded (lower / more flexed than his own norm at that speed) when it matters?
- **REACT** — Response: how fast does the body turn and move when the threat appears, and does he recover after being beaten?

## 3. Metrics

### SEE
- **`vision`** (table `passes`) — cue: *See ball and man*. At the pass moment, are the ball and the receiver each within +/-60 deg of the receiver's marker's head heading (ear axis + nose vector)?
- **`shoulder_checks`** (table `passes`) — cue: *Check your shoulder*. Number of times the receiver's marker turned his head >75 deg away from the ball for >=0.12 s in the 3 s before the pass.
- **`ball_man_split_deg`** (table `passes`) — cue: *Get in a position to see both*. Angle between ball and receiver as seen from the marker. >120 deg: impossible to see both without turning -- a positioning problem, not a perception one.
- **`blind_side`** (table `runs`) — cue: *Don't ball-watch*. Run started outside the nearest defender's field of view while he was looking at the ball.
- **`runner_in_view`** (table `runs`) — cue: *Know where your runner is*. Runner within +/-60 deg of the defender's head heading at run start.

### SHAPE
- **`body_shape`** (table `duels|runs`) — cue: *Get side-on / open your body*. Duels: torso vs line to carrier (<25 square, 25-70 side-on, >70 turned). Runs: torso vs own goal (<45 turned, 45-135 open, >135 square/flat).
- **`body_angle`** (table `duels`) — cue: *Side-on in the duel*. Absolute angle between torso heading (shoulders+hips) and the line to the carrier.
- **`shows`** (table `duels`) — cue: *Show him outside / inside*. Side the defender's chest opens toward (the side he invites the carrier), mapped to touchline (outside) vs centre (inside).
- **`torso_to_goal_deg`** (table `runs`) — cue: *Open up to your own goal*. Angle between torso and the own-goal direction at run start (0 = facing own goal).

### STANCE
- **`loaded_stance`** (table `duels`) — cue: *Get low / on your toes*. Hip >=3 cm lower or knees >=5 deg more flexed than the player's own norm at the same running speed (gait-adjusted).
- **`hip_drop_m`** (table `duels`) — cue: *Get low*. Speed-adjusted hip-height residual (positive = lower than usual).
- **`approach_decel`** (table `duels`) — cue: *Slow down before you arrive*. Peak speed in the last 1 s minus speed at engagement (tracking).

### REACT
- **`reaction_lag_s`** (table `runs`) — cue: *React to the run*. Time between the runner and the defender each first reaching 2.5 m/s goal-ward.
- **`recovered`** (table `recoveries`) — cue: *Recover goal-side*. After being dribbled / run in behind, back between threat and goal within 3 s.
- **`turned_within_1s`** (table `recoveries`) — cue: *Turn and run*. Torso within 60 deg of own goal within 1 s of being beaten.

## 4. Outcomes

- **`receiver_free`** (`passes`): Receiver under no pressure at all at reception (SkillCorner overall_pressure_start = no_pressure).
- **`line_break`** (`passes`): Pass broke at least one defensive line (SkillCorner).
- **`pass_success`** (`passes`): Pass completed.
- **`lead_to_shot`** (`passes|runs`): Possession led to a shot.
- **`beaten_in_behind`** (`runs`): Runner started level/behind the defender and finished goal-side of him.
- **`dangerous`** (`runs`): SkillCorner 'dangerous' run.
- **`received`** (`runs`): Runner received the ball.
- **`regain`** (`duels`): Engagement ended in a direct/indirect regain.
- **`carrier_progressed`** (`duels`): Pressed carrier still shot, broke a line or passed forward.
- **`beaten`** (`duels`): Defender beaten by the carrier's possession or movement.
- **`recovered`** (`recoveries`): Back goal-side within 3 s.

## 5. Pre-registered comparisons

Each finding compares the outcome rate when the cue is present vs absent, stratified (Mantel–Haenszel-style weighted risk difference) by the listed context variables, with a 95% percentile bootstrap CI over moments. Results are written to `data/build/evidence.json` by `python -m dbl build`.

- `checks_line_break` (SEE): exposure *>=1 shoulder check in 3 s* → outcome `line_break`; strata ['third', 'marking_band']; filter `-`.
- `dual_vision_free` (SEE): exposure *ball AND man in view* → outcome `receiver_free`; strata ['third', 'marking_band']; filter `ball_man_split_deg <= 120`.
- `open_body_runs` (SHAPE): exposure *square/flat at run start* → outcome `beaten_in_behind`; strata ['third']; filter `forward_run`.
- `blind_side_runs` (SEE): exposure *run started on the blind side* → outcome `dangerous`; strata ['subtype']; filter `forward_run`.
- `loaded_regain` (STANCE): exposure *loaded stance* → outcome `regain`; strata ['subtype', 'third']; filter `-`.
- `side_on_progress` (SHAPE): exposure *side-on* → outcome `carrier_progressed`; strata ['subtype', 'distance_band']; filter `-`.
- `turn_recover` (REACT): exposure *turned within 1 s* → outcome `recovered`; strata ['source']; filter `-`.

![Evidence](fig1_evidence.png)

*Stratified risk differences with 95 % bootstrap CIs for the pre-registered comparisons (numbers in
[RESULTS.md](RESULTS.md)).*

## 6. Thresholds (dbl/config.py)

- `max_joint_err_cm` = 18.0
- `shoulder_width_m` = (0.15, 0.6)
- `hip_width_m` = (0.1, 0.5)
- `fov_half_deg` = 60.0
- `check_away_deg` = 75.0
- `check_min_frames` = 3
- `scan_window_s` = 3.0
- `side_on_min_deg` = 25.0
- `side_on_max_deg` = 70.0
- `open_body_min_deg` = 45.0
- `ready_hip_drop_m` = 0.03
- `ready_knee_extra_deg` = 5.0
- `react_speed_mps` = 2.5
- `react_max_s` = 2.0
- `recover_window_s` = 3.0
- `engage_window_s` = 0.4
- `marker_max_dist_m` = 12.0

## 7. Pipeline

1. **Pose features** (`dbl/pose/extract.py`): stream each 3.3 GB JSON match, reduce each player-frame (25 fps) to torso/shoulder/hip/head headings, knee angle, hip height, stance width and foot stagger. Every angle is gated by the p90 joint error and plausible widths (adapted from SkillCorner's `pose_orientation.py`).
2. **Normalisation** (`dbl/moments/context.py`): coordinates and headings rotated so the defending team always defends x = −L/2. Velocities from Savitzky–Golay smoothing (SkillCorner recommends smoothing raw speeds).
3. **Posture norms**: match-wide hip-height and knee-angle medians per 0.5 m/s speed bin + per-player offsets; STANCE uses the residual.
4. **Marking**: Hungarian assignment of outfield defenders to attackers on distance to a point 2 m goal-side of each attacker (max 12 m).
5. **Moments** (`dbl/moments/build.py`): duels (on-ball engagements), passes (receiver's marker), runs (assigned/nearest defender), recoveries (after a beaten duel or a run getting in behind).
6. **Store & evidence** (`dbl/store.py`), **tools** for the LLM (`dbl/query/tools.py`), **agent** (`dbl/query/agent.py`), **app** (`app/streamlit_app.py`).

## 8. How analysts use it

- **Individual development**: Players tab → sort by `square_at_run` or `blind_side` → Moments tab filtered to that defender → clips for the session (minute + event id).
- **After an error**: Moments → recoveries → did he turn within 1 s, how long to get goal-side? Compare with his own baseline.
- **Opponent preparation**: Attack patterns → which runners start on blind sides, in which channel, and against which of our defenders.
- **Team principles**: Teams tab → dual vision, shoulder checks, receivers left free, beaten in behind; Ask tab for ad-hoc questions.

## 9. Limitations

- Two matches (~5k moments); associations, not causal effects; CIs resample moments not matches.
- Head yaw is a proxy for gaze; ±60° is a convention, not a measured visual field.
- Pose only for players visible in the broadcast; 59% of passes have both a marker and his pose.
- Reverse causality is possible (e.g. tighter marking situations may induce more checks); strata reduce but do not remove it.

## 10. References

- Bekkers, J. (2025). *Pressing Intensity: an intuitive measure for pressing in soccer.* arXiv:2501.04712.
- Spearman, W. (2018). *Beyond Expected Goals.* MIT Sloan Sports Analytics Conference.
- Fernández, J. & Bornn, L. (2018). *Wide Open Spaces.* MIT Sloan SAC.
- Groom, S. et al. (2026). *A Machine Learning Framework for Off Ball Defensive Role and Performance Evaluation in Football.* arXiv:2601.00748.
- U.S. Soccer Federation — *Visual exploratory behaviour* (SSAC 2026) open code.
- Wade, A. *Principles of Team Play / Defensive Play* (pressure, cover, balance, compactness, control & restraint).
- SkillCorner open data & body-pose tutorials (`src/features/pose_orientation.py`).

## 11. Full-match 3D viewer

`python -m dbl viewer` (or the *Match 3D* tab) serves a Three.js viewer from a local server (`dbl/viewer/`).

- **Data**: `python -m dbl fullmatch` keeps every resolved skeleton at 12.5 fps (23 joints, int16 cm, 10-minute parquet blocks, ~78 MB per match). The browser streams 30-second windows and interpolates to the display frame rate.
- **Who is drawn**: players with body pose at that instant are full skeletons (~14 of 22 on average, those visible in the broadcast). Everyone else is a translucent "ghost" placed from tracking.
- **Cameras**: broadcast (follows the ball), tactical (top-down), behind goal, follow player, **player's eyes** (camera at the selected player's head, looking along his head yaw — what he could see), free orbit.
- **Analysis layer**: the timeline shows framework moments (runs, line-breaking passes, duels, recoveries; "key" = beaten in behind / dangerous / led to shot / beaten / not recovered). Click a tick to jump to 3 s before and select the defender; pop-ups narrate moments while playing. The selected player gets a ±60° field-of-view cone projected from his head.
- **Deep links**: `/?match=1925299&f=30110&focus=560989&cam=follow&play=1` (f = tracking frame, 10 fps). The Moments tab in the app links straight into this view.
- **Moment clips** (Streamlit *Moments* tab) use the same skeletons for ±1.2 s around one moment:

![3D moment clips](fig2_moments_3d.png)

*Left: open body, runner in view, no lag. Right: square and ball-watching; the runner (brown) starts outside the
cone, reaction 2.0 s late, beaten in behind.*

![Player's eyes](img/viewer_players_eyes.png)

*Player's eyes: camera at the selected defender's head along his head yaw — what he could see.*

- **Typical workflow**: highest `square_at_run` defender → his square-and-beaten moments → open each in *Player's eyes* to see what he saw → clips for the session.

## 12. Defensive analysis layer (team shape, live overlays, report, Gemini)

What professional analysts read off the tactical camera, rebuilt from SkillCorner tracking + Dynamic Events and
shown on top of the 3D match (all computed, no LLM):

| Analyst question | Metric (dbl/shape.py, viewer/analysis.js) | Reference |
|---|---|---|
| How high / deep do they defend? | back-line height (deepest 4 outfielders, m from own goal), block height; block type mix from SkillCorner phases (high / medium / low block) | SkillCorner Phases of Play |
| Are they compact? | block length × width, convex-hull area of the 10 outfielders | Frencken et al. (2011); Low et al. (2020) |
| Compact **where it matters**? | ball-zone compactness: hull of the 5 defenders nearest the ball — the only compactness linked to defensive success | Forcher et al. / Univ. Vienna (2023), *Is a compact organization important for defensive success?* |
| Space between the lines | distance back line → next 3 players; alert > 15 m | coaching standard |
| Numbers behind the ball | outfielders goal-side of the ball; alert ≤ 5 | coaching standard |
| Pressure on the ball | nearest defender to the ball carrier, defenders within 5 m; carrier ring green/amber/red | Bekkers (2025) *Pressing Intensity* |
| Marking / free men | nearest defender per attacker (cyan = goal-side, rose = not); red pulsing ring = attacker inside 35 m with no defender within 5 m | Groom et al. (2026); Song et al. (2026) |
| Passing lanes | lanes from the carrier to team-mates, blocked if a defender is within 1.6 m (cover shadow); count of open forward lanes | AnalyticsFC (2021) *pass prevention* |
| Space conceded | simplified time-to-intercept pitch control (0.7 s reaction, 7 m/s, σ = 0.45): red = attack, blue = defence; **strong red = space the attack controls behind the defensive line** (+ m² shown live) | Spearman (2018); Fernández & Bornn (2018) |
| Team lines | each team's defence / midfield / attack lines in its kit colour (split at the two largest depth gaps, ≥2 players per line) | — |
| Offside line | pink dashed line at the second-last defender | — |
| Pressing intensity (match) | PPDA from SkillCorner engagements ending in regain / disruption / foul in the opponent's own two-thirds; passes per pressing engagement; counter-presses; pressing chains ending in a regain | Trainor (PPDA); Hudl/StatsBomb definitions — **no StatsBomb data used** |
| Penetration conceded | line-breaking passes conceded (any line / last line), shots conceded | SkillCorner Dynamic Events |

![Tactical camera with all overlays](img/viewer_live_overlays.png)

*All four overlays and the Live defence tab, 6 s before Perth's goal: back line 15 m deeper than usual, 4 free
attackers inside 35 m, nobody pressing the ball.*

**Live alerts** fire when a threshold is crossed (gap between lines, free attackers, not compact around the ball,
no pressure on the ball, ≥3 open forward lanes, ≤5 behind the ball). Each numeric tile is compared with the
same team's median in organised defence in that match (⚠ when 20% worse).

**Match report** (tab): per-team bullets (block mix, line height, compactness, free-man rate, PPDA, counter-press,
line breaks conceded, body-language profile), a KPI table and a clickable list of moments to review.

**Gemini in the viewer** (`POST /api/ask`): the question is sent with a `[CONTEXT]` block (match, tracking frame,
clock, selected player, and the live overlay state). Gemini can only call the typed tools (`defensive_state`,
`moments_window`, `match_insights`, `shape_by_phase`, `find_moments`, `aggregate`, `player_profiles`,
`framework_evidence`, ...) and is instructed to answer like an analyst: shape → key individual action → consequence →
coaching correction. Every tool call is listed under the answer. Without `GEMINI_API_KEY` (or if the Google API is
unreachable) a deterministic route answers *Explain this moment*, *What should he have done?*, *Match report* and
*How were they exploited?* from the same tools.

## 13. Goal-concession analysis ("how did it happen, who is responsible?")

`dbl/goals.py` finds every goal from SkillCorner's own flags (`game_interruption_after = goal_for`, `lead_to_goal`;
falls back to the score change) — validated against the match sheet's scorers. For each goal it rebuilds the
attacking chain (passes, line breaks, pressure on the ball), the conceding team's shape at the assist and the shot,
the situation at the shot (nearest defender to the scorer, distance, goal-side, defenders in the shot line) and a
**responsibility ranking** with explicit reasons from the framework moments in the build-up (beaten in behind,
vision failure at the reception, marking distance, no shoulder check, blind side, square body, late reaction, lost
duel, no recovery, scorer unmarked at the shot; weights doubled when the scorer/assister is involved). It is an
evidence-backed starting point for video review, not a verdict: tracking and pose cannot see the coach's instructions.

![Goal card in the match report](img/viewer_match_report.png)

In the viewer: white triangles on the timeline mark goals (click = watch the build-up with the most responsible
defender selected); the *Match report* tab shows a card per goal; Gemini has `goals` and `goal_breakdown` tools and
the *Goals & who's responsible* quick question.

## 14. "How could it have been avoided?" — counterfactual simulation

`dbl/simulate.py` re-positions the defending outfield players at a chosen frame (by default the assist pass of a
goal) to minimise a danger objective, with **attackers and ball held fixed** and each defender constrained to where he
could physically have been: a disc around his position 1.5 s earlier with radius 6 m/s × 1.5 s = 9 m.

Objective = marking danger (threat-weighted attackers left uncovered; scorer and ball carrier weighted more;
not goal-side counts 60 %) + open ball→goal lane + attack-controlled threat-weighted space within 32 m of goal
(time-to-intercept pitch control) + a small movement penalty (prefer minimal corrections). Optimised by coordinate
descent on a polar grid inside each reachable disc. Output: before → after (free attackers, distance to the scorer,
pressure on the ball, shot lane, space near goal), % danger reduction, and the corrections ranked by marginal value
(**key fix** = the single most valuable move).

Example — goal 1, Brisbane 0–1 Perth (68:25): at the assist, Walid Shour stepping 7.5 m goal-side onto D. Williams is
the key fix; the position-only correction cuts modelled danger by 52 % (free attackers 5 → 2, nearest defender to
the scorer 8.0 → 0.7 m, shot lane closed); the posture-aware model (§14.1) gives −46 %, almost all of it from position. Viewer: *🧪 How to stop it* on each goal card, *🧪 Simulate better defending here* in
Live defence, or ask Gemini; green figures and dashed arrows show the corrected positions.

![Simulation card](img/viewer_simulation.png)

Limits: attackers would also react; the model rates configurations, it does not replay the match. In the spirit of
positional-optimisation work from the first Analytics Cup (Shah, 2026), implemented independently.

### 14.1 Posture-aware coaching (position + body + head + stance) — `dbl/coach.py`

The positional model above ignores how the defender stands. The coach adds the body, using each defender's pose at
that frame (torso heading, head heading, stance vs his own speed-adjusted norm), and lets the search change it.
Posture enters the danger through the three effects the framework measured — no hand-tuned weights:

| Principle | Mechanism in the model | Calibration (from `evidence.json` / moment tables) |
|---|---|---|
| SEE | an attacker is fully covered only if the defender sees **ball and man** (both within ±60° of the head); otherwise coverage × k_vision | dual vision → receiver unpressured 14 % vs 30 % (adjusted) ⇒ k_vision = 0.82 |
| SHAPE | body shape to the own goal sets the reaction lag to runs; while he reacts a runner gains `speed × lag` m, and he reaches space later | mean lag at run start: open 0.49 s, square 1.07 s (turned capped at open: its 0.10 s is confounded by defenders already retreating) |
| STANCE | pressure on the ball carrier × k_unloaded when not loaded | loaded stance → regain 19 % vs 14 % (adjusted) ⇒ k_unloaded = 0.75 |

Search: alternating coordinate descent per defender — position on the reach grid, then posture (16 torso headings
plus the directions to the ball / nearest attackers / their bisectors; head within ±75° of the torso; stance). A
posture correction is only kept if it is worth ≥ 0.5 % of the danger on its own, so the coach does not ask for
gratuitous turns. Output adds `reduction_parts` (posture only / position only / both), per-defender
`gain_position` / `gain_posture`, torso/head/stance before → after and the coaching sentence.

Results on the six goals (RESULTS.md §4): posture alone is worth 0–7 % (largest for Sydney's opener, where two
Adelaide defenders could have had ball and man in view); position dominates everywhere. That is a finding, not a
flaw: at the moment of the assist, body orientation matters less than being in the right place; its value shows
earlier, at run starts (§5).

**What-if editor** (`whatif()`, `POST /api/whatif`): scores any configuration the analyst drags/rotates in the 3D
viewer (positions clamped to the reach disc, head clamped to ±75° of the torso) in ~50 ms, returning danger vs what
happened and, per defender, whether he sees ball and man, his body shape, reaction and stance.

### 14.2 Better-defence replay — re-simulating the movement of all defenders (≤ 5 s) — `dbl/replay.py`

The static coach answers *"where should they have been at the assist?"*. The replay answers *"how should the whole
defence have moved during the last seconds?"*. The attack and the ball replay **exactly as they happened**. Every
defending outfield player starts where he really was, with his real velocity, and is then steered every 0.2 s:

- **Physics (per player).**
  - Each player has his own top speed from this match (99.5th percentile, 5.5–8.5 m/s).
  - Speed also depends on the running direction relative to the body: forwards 100 %, sideways 70 %, back-pedalling
    50 %.
  - Acceleration is ≤ 5 m/s². The body turns ≤ 180°/s; the head turns ≤ 360°/s and stays within ±75° of the body.
  - Above 4 m/s a player faces where he runs (no sideways sprinting).
- **Football structure.**
  - The back line (centre-backs and full-backs) stays a line: ≤ 2.5 m from the others' median depth, in its
    left-to-right order.
  - Every player stays within ~8 m of his real zone at that instant, so roles are kept.
- **Decision.**
  - Each step runs coordinate descent over the defenders (2 sweeps): the velocity change, then the posture.
  - The cost is the posture-aware danger of §14.1 now plus 0.6 × the danger 0.6 s ahead.
  - **No clairvoyance (default).** A defender decides from what was visible **0.4 s earlier** (≈ the measured
    reaction of an open defender), with attackers and ball extrapolated at constant velocity.
  - *Perfect anticipation* lets decisions see the real future. It is an upper bound, not a realistic standard.
- **Evaluation.** The simulated defence is always scored against the **real** attack at each instant. The output
  gives:
  - the danger curve, real vs simulated;
  - the mean reduction and the reduction at the shot;
  - free attackers;
  - the defenders who would have played most differently (end gap, the man they end on, goal-side);
  - `ball_reachable` events.
- **Honest boundary.** Once a simulated defender can reach the ball (below 1.8 m), the real attack would no longer
  have happened as recorded. Those moments are flagged instead of being continued as if nothing changed.

- **Smooth control.** A velocity change costs 0.005 per (m/s)², so players do not swerve for marginal gains, and
  the direction cap of the body never brakes harder than the acceleration limit.

Results (RESULTS.md §4–5), danger reduction at the shot (averaged over the 5 s in brackets):

| Goal (conceding team) | Read-late (default) | Perfect anticipation |
|---|---|---|
| Brisbane–Perth, Williams (Brisbane) | −48 % (−26 %) | −59 % (−33 %) |
| Sydney 1, Lolley (Adelaide) | −48 % (−26 %) | −31 % (−24 %) |
| Sydney 2, Klimala (Adelaide) | −46 % (−24 %) | −32 % (−33 %) |
| Adelaide, Pierias (Sydney) | −9 % (−5 %) | −13 % (−6 %) |
| Sydney 3, Segecic (Adelaide) | −61 % (−39 %) | −56 % (−44 %) |
| Sydney 4, Cáceres (Adelaide) | −34 % (−33 %) | −45 % (−37 %) |

The search is greedy (one step at a time), so read-late can beat perfect anticipation. Treat both as
well-organised alternatives to teach from, not optima.

**Validation (RESULTS.md §5).**
- *Kinematics.* Simulated defenders run as far as the real ones in the same 5 s (18 m vs 17 m) at similar speeds
  (p99 7.4 vs 7.6 m/s, never above 7.5). Their acceleration stays ≤ 5.3 m/s², but they change pace more often than
  real players (p90 4.4 vs 2.2 m/s²).
- *Sensitivity.* The one free weight (cost of a velocity change) moves the percentage at the shot by up to ~30 points
  on some goals: the direction is stable, the exact figure is not. The table in RESULTS §5.3 shows all values.
- *What the danger means.* Its marking term matches real passing: passes to attackers it rates free are completed
  98 % of the time vs 73 % for tightly marked ones, and the option actually chosen is the freer one (AUC 0.57). It
  agrees with SkillCorner's xPass (ρ = 0.61). The total danger does not predict shots better than the ball location
  (out-of-match AUC 0.75 vs 0.80), so a reduction is a reduction of the objective, not of shot probability.

Rendering: simulated defenders move with **their own running clips** (`dbl/gait.py`: per player, the smoothest
1.3 s stretch in each speed band, taken in place from the full-match skeletons). The clip is played at the
simulated speed and advanced by the distance covered, so the feet match the ground speed.

Viewer: *🎬 Better defence (5 s)* (goal card), *🎬 Re-play 5 s* (simulation card) or *🎬 Better defence, next 5 s*
(Live defence) plays the match with the simulated defenders as green bodies. They reuse each player's own running
animation, moved and turned. Arrows show the gap to where they really were. *Hide real defenders* also re-runs the
live analysis (lines, free men, space) on the simulated defence.

## 15. From simulations to training — error patterns (`dbl/patterns.py`)

1. **Moments.** Every dangerous situation each team faced is collected and merged within 1.5 s: opponent
   line-breaking passes, passes in possessions that led to a shot, dangerous runs, runs in behind, and duels lost.
   That gives 115–172 per team. A **control** sample of 60 ordinary out-of-possession frames per team, ≥ 3 s away from
   any of them, is added.
2. **Coach.** The posture-aware coach (§14.1, faster grid) is run on every moment. Each defender's correction is split
   into position gain and posture gain, then tagged from its geometry:
   - **Position**, relative to the nearest attacker after the correction:
     - *no pressure on the ball*: the carrier is ≥ 1.5 m closer;
     - *lost goal-side*: not goal-side → goal-side, < 4 m;
     - *too loose*: ≥ 1.5 m tighter, ends < 3 m;
     - back four *too high* (corrected ≥ 1.5 m deeper) or *too deep* (≥ 1.5 m higher);
     - *too wide* (tucks in ≥ 2 m) or *too narrow* (widens ≥ 2 m);
     - otherwise *not covering space / lane*.
   - **Posture:**
     - *ball-watching*: a head turn of ≥ 20° brings an extra attacker into view with the ball;
     - *square body* → open;
     - *upright* → loaded.
   - Gains are shared equally among the tags of the same kind. Gains under 1 % of the moment's danger are ignored.
3. **Pattern.** Per team and tag, the output gives:
   - moments with the error / dangerous moments (the rate);
   - the **lift** = rate in dangerous moments / rate in control frames, with a 400-sample bootstrap CI over moments;
   - the **avoidable danger** (summed gains) and its share of the danger in those moments;
   - **excess avoidable** = avoidable × (1 − 1/lift), the part attributable to the error being more frequent when
     danger happens;
   - players, third / channel / phase / situation, and examples (taken in the drill's zone when the team has one:
     the final third for marking drills);
   - the **real outcome** (`tag_outcomes`): over all dangerous moments of both matches, how often the attack really
     shot within 10 s when the coach flags the error vs when it does not, stratified by pitch third (Mantel-Haenszel
     weights, bootstrap CI). Five error types are linked to real shots: no pressure on the ball (+14 pp),
     ball-watching (+13 pp), too narrow (+12 pp), line too deep (+8 pp), too loose (+7 pp). "Lost goal-side",
     "line too high", "not covering space" and "too wide" are not.
4. **Training plan.**
   - The top 4 patterns are kept (≥ 10 moments): first those that are both specific to danger (lift CI > 1) and
     linked to real shots, then either, then by excess avoidable.
   - Each is mapped to a drill from an 11-drill catalogue (`DRILLS`, one per error type). A drill gives the objective,
     setup, rules, coaching points in the framework's principles, the progression and the KPI to track with this
     framework in the next match.
   - The session is a scanning-rondo warm-up, the 4 drills, and a conditioned game.

**What it found.** Dangerous moments are only modestly more "fixable" than ordinary defending: the coach removes 14–20 %
of the modelled danger in dangerous moments vs 12–16 % in control frames. The errors that separate them are therefore
the signal:
- **Perth:** too loose on his man (lift 1.6), back line too deep (2.0), ball-watching (2.0).
- **Adelaide:** back line too deep (2.1) and ball-watching (1.9), both linked to real shots.
- **Brisbane:** too loose on his man (1.7) and too narrow (1.6).
- **Sydney:** too loose on his man (1.9), too narrow (1.6) and back line too deep (1.8).

"Too loose" is flagged often for every team because the danger objective values marking; it stays in the plans
because it is both more frequent in dangerous moments (lift CI > 1) and linked to real shots (+7 pp).

Limits: two matches, a model-derived objective, and drills that are coaching suggestions — not validated
interventions. The KPI attached to each drill is how the plan should be checked.

**Session sheet** (`dbl/session.py`, `/training/sheet`). For each drill the focus player's most avoidable moment is
re-coached. Its real positions (attack, defence, ball, GK) and the corrections are drawn as a pitch diagram, cropped
to the action, so the staff can set the drill up from a situation the players lived. Load follows common one-game-week
periodisation:
- MD-4: intensive / duels — recovery race, first defender, 1v1 stance, cover & balance;
- MD-3: extensive / tactical — box marking, line control, compactness, shifting;
- MD-2: speed / reaction — scanning, opening the hips.

Each drill gets sets × reps, work:rest, space, numbers and target RPE.

**3D tactical drill mode** (`session.drill_board`, `/api/drill`, `drill.js`). For each priority it takes the example moment and
selects the drill's format around the focus defender: the ball carrier plus the nearest attackers, and the focus
defender plus his nearest team-mates (e.g. 4v4 for box marking, 2v2 for the recovery race, 6v7 for line control). It
then plays 5 s (3 s before, 2 s after) inside the 3D viewer, on a tactical camera framed on the drill area:
- the attack and the ball exactly as in the match;
- the defenders either as they really moved, or as the better-defence replay (§14.2) moves them, i.e. the behaviour
  the drill teaches.

Drill area, cones and gates follow the drill (gates for pressing, central gates for compactness, the 12 m band for
line control). Captions show the set-up and then the coaching points at the key moment, alongside the danger of the
situation, real vs corrected.
