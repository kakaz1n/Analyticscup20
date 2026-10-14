# Data

## Source and rules

- **SkillCorner open data** for the PySport × SkillCorner Analytics Cup 2.0 — A-League 2024/25
  ([github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata),
  [huggingface.co/datasets/SkillCorner/opendata-bodypose](https://huggingface.co/datasets/SkillCorner/opendata-bodypose)).
- Competition rule: only the provided SkillCorner data may be used (no video, no other providers). **No StatsBomb or
  other data is used**; "StatsBomb-style" KPIs (PPDA, counter-pressing, pressure regains) are rebuilt from
  SkillCorner Dynamic Events.
- The data is **not redistributed** in this repository (`data/` is git-ignored). Credit: SkillCorner.

## What is used

| Layer | Rate | Used for |
|---|---|---|
| Tracking (`*_tracking_extrapolated.jsonl`) | 10 fps | positions, velocities (Savitzky-Golay), team shape, marking, pitch control |
| Body pose (29 joints, `*.jsonl.zip`) | 25 fps | head/torso/hip headings, knee flexion, hip height, stance; 3D skeletons |
| Dynamic Events (`*_dynamic_events.csv`) | event | moments (on-ball engagements, player possessions/passes, off-ball runs), outcomes (regain, line break, pressure, EPV, dangerous…) |
| Phases of Play (`*_phases_of_play.csv`) | phase | team in possession, block type (high/medium/low), transitions |
| Match sheet (`*_match.json`) | – | squads, roles, pitch size, sides, kits, score/scorers (validation) |

Only the two matches with full-match body pose are analysed: **1925299** (Brisbane Roar 0–1 Perth Glory) and
**1996435** (Sydney FC 4–1 Adelaide United).

## Coverage & quality

| | 1925299 | 1996435 |
|---|---|---|
| Tracking frames (in play) | 61k | 58k |
| Pose player-frames (25 fps) | 1.54 M | 1.57 M |
| Players with a skeleton at a given instant | ~14 / 22 | ~12 / 22 |
| Moments (duels / passes / runs) | 935 / 912 / 498 | 1,025 / 877 / 515 |

- **Pose exists only for players visible on the broadcast.** Off-screen players are shown as tracking "ghosts".
- Joint estimates carry a p90 error (median ~11 cm); every angle is gated (≤ 18 cm, plausible shoulder/hip widths).
- `z` is accurate relative to the body, not the pitch: skeletons are stood on the grass by their lowest foot joint.
- **Validation**: above 4 m/s the torso heading is within 8.7° (median) of the running direction (94 % within 45°).
- Goals detected from SkillCorner flags match the match sheet's scorers in both matches (test).
- Tracking player identities are ~97 % accurate (SkillCorner); speeds are smoothed as SkillCorner recommends.

## Known limitations

- Two matches: results are associations, not causal effects; CIs resample moments, not matches.
- Head yaw is a proxy for gaze (±60° field of view is a convention).
- PPDA rebuilt from SkillCorner engagements is not comparable with event-provider PPDA.
- The simulation holds attackers fixed; it rates safer configurations, it does not replay the match.
