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

Phases 00-02 are in: the config module, the SQLite schema, the `rescan` catalog
job, the shot browser, the single-shot view, and the plot registry that view is
built on — parameter forms generated from a dataclass, a content-addressed
result cache, and the scalar store the multi-shot view will read. The
multi-shot view lands in phase 04. See [`docs/PLAN.md`](docs/PLAN.md) for the
architecture, the database schema, and the phase order.

## Adding a plot

One file in `fusion_ui/plots/`, one line in its `__init__.py`. No page changes,
no new storage, and it appears in the single-shot view immediately.

```python
from dataclasses import dataclass
import plotly.graph_objects as go
import xarray as xr
from fusion_ui.core import registry


@dataclass
class MyParams:
    """
    threshold: What counts as an event, in standard deviations.
    """
    threshold: float = 2.5


def compute(ds, params):          # -> xr.Dataset. Cached; omit for a live view.
    return xr.Dataset({"y": ("t", ...)})


def render(result, params, target):   # -> go.Figure, or draw and return None
    return go.Figure(go.Scatter(y=result["y"].values))


def scalars(result):              # -> {name: value} or {(x, y, name): value}
    return {"my_number": float(result["y"].mean())}


SPEC = registry.register(
    registry.PlotSpec(
        key="my_plot",            # permanent: it is part of the cache key
        label="My plot",
        diagnostics=("apd",),
        params=MyParams,
        render=render,
        compute=compute,
        scalars=scalars,
    )
)
```

The parameter form, the cache key, the netCDF blob, the run ledger and the
scalar rows all follow from that. The rules the contract depends on are in
[`CLAUDE.md`](CLAUDE.md).

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
fusion-ui init-db          # create or migrate the app's SQLite database
fusion-ui rescan           # index the data tree into `shots` (cron this)
fusion-ui import-results   # seed `scalars` from density_scan/results.json (once)
fusion-ui status           # resolved paths, index and result counts
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
