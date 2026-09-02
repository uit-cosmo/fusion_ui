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

Under construction. See [`docs/PLAN.md`](docs/PLAN.md) for the architecture,
the database schema, and the phase order.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env          # then edit for your machine
.venv/bin/streamlit run fusion_ui/app.py
```

`imaging_methods` is not on PyPI — install it editable from a local checkout.

## Deployment

Streamlit on loopback behind nginx (TLS + shared-password basic auth), run by
systemd. Unit and site config live in [`deploy/`](deploy/); the full recipe,
including the websocket-proxy pitfall, is in `docs/PLAN.md`.
