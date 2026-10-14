# /brag plan — Defensive Body Language

**What it is:** a defensive-analysis framework you watch, question and correct in 3D — full A-League matches replayed with real SkillCorner skeletons, an integrated Gemini analyst, and a counterfactual "how to stop it" simulation.
**For:** football analysts and coaches reviewing defending (PySport × SkillCorner Analytics Cup 2.0).
**Angle:** start inside a defender's head, end with the fix. Watch → Ask → Diagnose → Fix, all on the real goal (Brisbane 0–1 Perth, 68:25).
**Hook:** first-person "Player's eyes" camera + "What did the defender see?"
**Highlights:** (1) full match in 3D with real skeletons; (2) ask in plain English, get the computed answer and the viewer jumps to the moment; (3) the follow-up "How could it be avoided?" draws reachable corrected positions — danger −52% (real model output); (4) both teams' shapes, free men and space behind, live.
**Punchline:** "Watch. Ask. Diagnose. Fix."
**Tone:** polished / cinematic. Dark navy UI (#0b1220), amber accent (#f59e0b), sim green (#4ade80), Inter / Inter Display — the viewer's own palette and font.
**Truthfulness:** every frame is the real viewer, captured frame-exactly (`window.__manual` + `window.__step`). The chat answers in the video come from the deterministic offline route of the same tools (the capture machine cannot reach the Gemini API); numbers are the framework's real outputs. No personal data; players appear only as they do in the public match data.

| # | Scene | Dur | Picture | Text |
|---|---|---|---|---|
| A | Hook | 3.13 s | Player's eyes, Brisbane #8, seconds before the goal | What did the defender see? |
| B | Reveal | 3.73 s | Follow cam on the scorer's run, skeletons | Full matches in 3D. / Real SkillCorner body pose — skeletons, not dots. |
| C | Ask | 4.37 s | Real UI: question typed, answer arrives, viewer jumps to the build-up | “How did the first goal happen and who is responsible?” |
| D | Fix | 4.37 s | Follow-up typed → green corrected positions animate | “How could it be avoided?” · Danger −52% |
| E | Shape | 2.50 s | Tactical: both teams' lines in kit colours, free attackers, space | Both shapes. Free men. Space behind. |
| F | Outro | 3.73 s | Blurred pitch | Watch. Ask. Diagnose. Fix. · SEE · SHAPE · STANCE · REACT · Analytics Cup 2.0 · repo |
| | **Total** | **21.8 s** | Cuts on the beat (96 bpm, A minor) | |
