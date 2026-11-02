# Installation & reproduction

Everything runs locally with open-source software (Python, pandas, Streamlit, Three.js). An internet connection is
needed once to download the data; Gemini is optional.

## 0. Run it online (GitHub Codespaces)

[Open the repository in Codespaces](https://codespaces.new/kakaz1n/Analyticscup20) (a GitHub account is enough; the
free monthly quota covers it). The container (`.devcontainer/`) installs the requirements, downloads the SkillCorner open
data from its official sources (GitHub + Hugging Face; nothing is redistributed by this repo), builds everything once
(~45 min on 4 cores, log in `data-build.log`) and starts the viewer; port 8765 opens in the browser. For Gemini, add a
Codespaces secret `GEMINI_API_KEY` (the container asks for it; optional).

The viewer binds to 127.0.0.1 by default; `python -m dbl viewer --host 0.0.0.0` (or `DBL_HOST=0.0.0.0`) exposes it
inside a container.

## 1. Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.10 | tested on 3.10 and 3.13 |
| RAM | 4 GB | pose extraction streams the 3.3 GB files, it never loads them whole |
| Disk | ~3 GB free | raw pose zips 1.3 GB + build outputs ~0.3 GB |
| Browser | any WebGL browser | Chrome / Edge / Firefox / Safari for the 3D viewer |

## 2. Install

```bash
git clone https://github.com/kakaz1n/Analyticscup20.git
cd Analyticscup20
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs numpy, pandas, scipy, pyarrow, matplotlib, plotly, streamlit, orjson, google-genai and
pytest. `pip install -e .` additionally installs the `dbl` console command (same as `python -m dbl`).

## 3. Data

The project uses only the SkillCorner open data released for the Analytics Cup 2.0 (A-League 2024/25) — the two
matches that have full-match body pose:

| match_id | Fixture |
|---|---|
| 1925299 | Brisbane Roar 0–1 Perth Glory (2024-12-21) |
| 1996435 | Sydney FC 4–1 Adelaide United (2025-02-01) |

```bash
python -m dbl download
```

downloads into `data/raw/<match_id>/` the match sheet, Dynamic Events, Phases of Play and tracking (GitHub, Git LFS
media URL) and the two body-pose archives into `data/pose_raw/` from the Hugging Face Hub (~620 MB each, SHA-256
checked in `dbl/data/download.py`).

**If huggingface.co is blocked** on your network (corporate proxies often do this), open these two links in a browser
and save the files into `data/pose_raw/`:

- https://huggingface.co/datasets/SkillCorner/opendata-bodypose/resolve/main/raw/1925299.jsonl.zip
- https://huggingface.co/datasets/SkillCorner/opendata-bodypose/resolve/main/raw/1996435.jsonl.zip

Zipped or unzipped both work (`1925299.jsonl.zip`, `1925299.jsonl`, or a folder `1925299.jsonl/1925299.jsonl`).
You can also point the project to another data folder with `DBL_DATA_DIR=/path/to/data`.

## 4. Build

```bash
python -m dbl pose        # body-pose features per player-frame (25 fps)          ~2-3 min per match
python -m dbl build       # defensive moments, team shape, evidence (bootstrap)      ~6 min
python -m dbl skeletons   # 3D clips around every moment (Moments tab)             ~2 min per match
python -m dbl fullmatch   # full-match skeletons at 12.5 fps (3D match viewer)       ~2 min per match
python -m dbl patterns    # coach every dangerous moment → error patterns + docs/TRAINING.md   ~5 min
python -m dbl gait        # running clips per player for the simulated defenders              ~20 s per match
python -m dbl validate    # test the simulation against real passes/shots + kinematics  ~10 min
python -m dbl report      # regenerate docs/RESULTS.md and README figures            ~1 min
# or everything at once:
python -m dbl all
```

Outputs go to `data/build/` (see [ARCHITECTURE.md](ARCHITECTURE.md#build-artefacts)). Every step is idempotent and
skips work already done (`--force` to redo pose/skeleton steps).

## 5. Run

```bash
python -m dbl viewer      # full-match 3D viewer  -> http://localhost:8765
python -m dbl app         # Streamlit analyst app -> http://localhost:8501 (includes the viewer in "Match 3D")
python -m dbl ask "Who ball-watches most at Sydney FC?"
pytest                    # unit + integration tests
```

## 6. Gemini (optional)

```bash
cp .env.example .env      # Windows: copy .env.example .env
# edit .env:
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-2.5-flash
```

`.env` is git-ignored. Without a key — or if the Google API is unreachable — every question is answered by the
deterministic offline router using the same tools, and the answer says so.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `!! no body-pose file` | Put the two Hugging Face files in `data/pose_raw/` (section 3). |
| Viewer shows only translucent "ghosts" | Full-match skeletons not built: `python -m dbl fullmatch`. Pose also only exists for players visible on the broadcast (~14 of 22 at a time). |
| `Gemini unavailable: ProxyError 403` | Your network blocks `generativelanguage.googleapis.com`; the offline answer is shown instead. |
| `400 INVALID_ARGUMENT … NaN` | Fixed: all tool results are sanitised to strict JSON (`dbl/query/tools.py::_json_safe`). Update the code. |
| Port 8765 busy | `python -m dbl viewer --port 8800` |
| Streamlit "No built data found" | Run `python -m dbl build` first. |
| Slow 3D in a VM / remote desktop | Software WebGL; use a machine with a GPU, or the Tactical camera. |
