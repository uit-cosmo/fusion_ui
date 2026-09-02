# Shot Explorer

A browser UI over the group's fusion diagnostic data. Pick a shot and a plot,
get the plot; pick a scalar and an x-axis, get it across every shot.

It runs as a hosted service on the group server, next to the data, and drives
the analysis code in [`imaging_methods`](https://github.com/Sosnowsky/imaging-methods),
[`fusion_scripts`](https://github.com/Sosnowsky/fusion_scripts) and
`velocity_estimation`.

## Views

- **Shot browser** — filterable table of every shot on disk, joined to the
  discharge metadata (f_GW, I_p, n̄_e, confinement mode).
- **Single shot** — a shot, a diagnostic, a plot type and its parameters, in.
  A figure out: raw frames and time series, velocity fields, conditional
  averages, movies.
- **Multi shot** — one scalar (velocity, duration time, event rate, …) plotted
  against shot number, Greenwald fraction, or line-averaged density across a
  selection of shots.

## Status

Phase 00 is in: the config module, the SQLite schema, the `rescan` catalog job,
the shot browser and the deployment path. The single-shot and multi-shot views
land in phases 01 and 04. See [`docs/PLAN.md`](docs/PLAN.md) for the
architecture, the database schema, and the phase order.

## Development

```bash
python -m venv .venv
# imaging_methods, experimental_database, fusion_scripts, velocity_estimation
# and fppanalysis are not on PyPI — install each editable from its checkout
# first, then the pinned versions below are already satisfied.
.venv/bin/pip install -e ../imaging-methods -e ../experimental_database \
  -e ../fusion_scripts -e ../velocity-estimation -e ../fpp-analysis-tools
.venv/bin/pip install -e ".[dev]"
cp .env.example .env          # then edit for your machine
.venv/bin/fusion-ui init-db
.venv/bin/fusion-ui rescan    # index the data tree
.venv/bin/streamlit run fusion_ui/app.py
```

## Commands

```bash
fusion-ui init-db    # create or migrate the app's SQLite database
fusion-ui rescan     # index the data tree into the `shots` table (cron this)
fusion-ui status     # resolved paths and index counts — run after deploying
```

New shots appear in the browser only after a rescan; the server runs it on cron
and the browser page has a button for it.

## Tests

```bash
.venv/bin/pytest
```

The suite uses a temporary tree of empty files and never touches the real data.

## Deployment

Streamlit on loopback behind nginx (TLS + shared-password basic auth), run by
systemd. Unit and site config live in [`deploy/`](deploy/); the full recipe,
including the websocket-proxy pitfall, is in `docs/PLAN.md`.
