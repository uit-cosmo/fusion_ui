# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

**Shot Explorer** — a Streamlit web app that gives the research group
point-and-click access to fusion diagnostic data (Alcator C-Mod GPI imaging and
Langmuir probes) and to the analyses in `imaging_methods`, `fusion_scripts` and
`velocity_estimation`. It runs as a hosted service on the group server, next to
the data.

**Read `docs/PLAN.md` before starting work.** It is the source of truth for the
architecture, the database schema, the phase order, and what has been decided.
This file holds only the conventions that apply while writing code.

## Current state

Phase 00 (skeleton, deployed) — **not started**. Nothing in `fusion_ui/` is
implemented yet; the package directories are placeholders.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env        # then edit for this machine
.venv/bin/streamlit run fusion_ui/app.py
```

`imaging_methods` is not on PyPI — install it editable from a local checkout
(`/home/sosno/Git/imaging-methods` on the dev machine).

## Conventions

- **No hardcoded paths.** Everything machine-specific goes through
  `fusion_ui/config.py`, which reads `.env` with shell environment variables
  taking precedence — the same pattern as `fusion_scripts/config.py`.
- **The discharge DB is read-only.** `experimental_database` and its
  `plasma_discharges.json` are consumed, never written. All state this app
  generates goes in its own SQLite file (schema in `docs/PLAN.md`).
- **Every analysis is a `PlotSpec`.** Once the registry lands in phase 02, no
  plot gets wired into a page directly. The contract will be documented here.
- **Never load a full time axis.** APD files are ~500 MB and 583k samples;
  always slice to the discharge DB's `t_start..t_end` (or a centred 0.2 s window
  when there is no metadata).
- **Decimate before handing a 1D trace to Plotly.** Use the shared min/max
  envelope helper in `core/decimate.py` — striding drops spikes.
- **Interactive plots are Plotly.** Movies stay matplotlib, rendered to mp4 and
  served with `st.video`.
- **Development uses one small shot.** Do not loop over the data tree to "check
  something" — it is tens of terabytes and growing.

## Testing

```bash
pytest                 # all tests
pytest tests/ -v
```

Streamlit pages are smoke-tested with `streamlit.testing.v1.AppTest`.
