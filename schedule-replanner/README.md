# Godrej Activity Replanner

A construction **delay-impact analysis** tool built around the Critical Path Method (CPM). A planner selects an activity, applies a delay, and immediately sees whether project completion moves, which downstream activities are affected, and whether the critical path changes.

The idea comes from internship work at **Godrej Properties** (understanding how a slip on one tower activity can affect handover). The activity table in this repository is a **synthetic demo network** with typical tower sequences (clearance → structure → finishes → handover). It is **not** a live project export, and the dates should not be treated as real site performance.

## Problem statement

On a multi-tower job, the same activity name appears in more than one location (for example “Tiling” in T1 and T3). A useful replanner must:

- Identify the **unique activity**, not just the name.
- Recalculate using **dependency constraints**, not by adding the same number of days to every descendant.
- Keep CPM math **deterministic** and separate from any language model.

## Features

- Load and validate activities from CSV (IDs, durations, predecessors, cycles).
- CPM forward pass, backward pass, total float, and critical path.
- Delay analysis that preserves the baseline table.
- Streamlit dashboard: overview, searchable table, impact summary, Gantt-style timeline, dependency graph, CSV export.
- Optional local AI assistant (Ollama) that can **only** call allow-listed scheduler tools. The app runs without Ollama.

## Architecture

```
User (Streamlit)
        │
        ▼
     app.py  ── delay button / tables / charts / CSV download
        │
        ▼
  scheduler.py  ← source of truth for all dates
        ▲
        │
     agent.py  (optional) ── Ollama tool loop, never executes arbitrary code
        │
        ▼
 data/activities.csv
```

| File | Role |
|---|---|
| `scheduler.py` | Graph + CPM + delay impact. No LLM. |
| `data/activities.csv` | Demo activities with unique IDs. |
| `app.py` | Streamlit UI. |
| `agent.py` | Optional Ollama assistant and safe tool dispatch. |
| `tests/` | Unit tests, including hand-calculated CPM examples. |
| `.env.example` | Optional Ollama settings (no secrets). |

## Technologies

| Tool | Why |
|---|---|
| Python | Interview-friendly, easy to explain. |
| Pandas | Load CSV and compare schedules. |
| NetworkX | Dependency graph, cycle checks, topological order. |
| Streamlit | Fast local demo UI. |
| Plotly | Gantt-style bars and network diagram. |
| python-dotenv / requests | Optional local Ollama configuration and HTTP. |

## Calendar convention (version 1)

- Durations are **calendar days** (weekends count).
- An activity starting on date `S` with duration `D` occupies `S` through `S+(D-1)` inclusive.
- A successor may start the next day (`S+D`).
- `baseline_start` is a not-earlier-than constraint.
- Finish-to-start links only. No resource leveling, overtime, or crashing.

## Installation (Windows PowerShell)

From the project folder (`schedule-replanner` in this workspace):

```powershell
cd D:\RP\schedule-replanner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Optional AI (not required for the dashboard):

```powershell
copy .env.example .env
# Install Ollama from https://ollama.com/download then:
ollama pull llama3.2
```

## Tests

```powershell
cd D:\RP\schedule-replanner
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

Expected regression checks (do not change these just to pass):

- Baseline project finish: **8 May 2026**
- `T3-TIL` delayed by 5 days: finish **13 May 2026**
- `T1-TIL` delayed by 5 days: project finish unchanged (float)

## Launch the dashboard

```powershell
cd D:\RP\schedule-replanner
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Then open **http://localhost:8501**

## How to run a delay-impact analysis

1. In the sidebar, select activity `T3-TIL` (Tiling, Tower T3).
2. Set delay to `5` calendar days.
3. Click **Run delay-impact analysis**.
4. Open **Delay impact**: completion should move from 2026-05-08 to 2026-05-13.
5. Confirm Tower T1 handover did not move.
6. Download `revised_schedule.csv`.

Second demo: select `T1-TIL`, delay 5 days, run again. Completion should stay 2026-05-08.

## Interview demonstration script (about 3 minutes)

1. **Context:** “This is a CPM delay-impact demo inspired by internship planning work. The dataset is synthetic.”
2. **Unique IDs:** Search “Tiling” in the table — T1, T2, and T3 exist, so we never delay the wrong tower.
3. **Baseline:** Overview shows completion 8 May 2026 and critical path through Tower T3 then external works.
4. **Critical delay:** `T3-TIL` + 5 days → finish 13 May, painting/handover/external works move.
5. **Float:** `T1-TIL` + 5 days → finish unchanged.
6. **Engine:** Dates come from `scheduler.py` (forward/backward pass). The LLM, if used, only calls tools.
7. **Limits:** No resource leveling, no recovery plan, calendar days only.

## Optional Ollama setup

The **AI assistant** tab talks to a local model. It may only call:

- `get_project_status`
- `get_critical_path`
- `find_activity`
- `get_activity_details`
- `analyze_delay`

If Ollama is not installed, the tab shows setup steps and the rest of the app still works.

```powershell
ollama pull llama3.2
# Ollama should be running at http://127.0.0.1:11434
```

Example questions once the model is ready:

- What is the current project completion date?
- Which activities are on the critical path?
- What happens if T3-TIL is delayed by five days?
- Which downstream activities are affected?

The UI only treats an answer as grounded if a scheduler tool actually ran.

## Limitations and future work

- No resource leveling or crew constraints.
- No crashing, overtime, or recovery optimization.
- Calendar days only; no working-day calendar or holidays.
- Finish-to-start only (no SS/FF lags).
- Demo data, not a live Godrej programme.

Possible next steps: working-day calendars, multiple delay events, earned-value views, and a proper recovery suggestion module that stays separate from CPM math.

## Secrets

Do not commit `.env` or `.streamlit/secrets.toml`. No API keys are required for the core scheduler or dashboard.
