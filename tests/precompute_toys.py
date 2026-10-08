"""Toy cached specs for the precompute tests, and a driver for the CLI.

Instant computes on the tiny APD fixture: one unchained spec, a chain shaped
like phase 06's (a batch-only bank with two products built on it, whose
parameter classes share an ``averages`` field), and a few that misbehave on
purpose -- interrupting themselves, or blocking until they are interrupted.

Importing this module registers nothing, so that the toys never reach another
test's page; :func:`register` and :func:`unregister` do, around the tests that
use them. A pool worker is a fresh interpreter, which knows a spec only once it
imports a module that registers it (``precompute.execute``'s ``modules``):
that is :data:`REGISTERING_MODULE`.
"""

import os
import signal
import time
from dataclasses import dataclass, field

import xarray as xr

from fusion_ui.core import registry

#: Where ``toy_block`` drops a file named after each shot it starts on.
MARKERS = "TOY_MARKERS"


def _render(result, params, target):
    return None


def _thread_niceness():
    """The lowest niceness among this process's threads (Linux)."""
    try:
        threads = [int(name) for name in os.listdir("/proc/self/task")]
    except OSError:
        threads = [0]
    return min(os.getpriority(os.PRIO_PROCESS, thread) for thread in threads)


def _probe(ds):
    """What the process that computed saw."""
    return {
        "niceness": os.getpriority(os.PRIO_PROCESS, 0),
        "least_thread_niceness": _thread_niceness(),
        "pid": os.getpid(),
        "in_memory": int(ds["frames"].variable._in_memory),
        "samples": int(ds.sizes["time"]),
    }


# --- one plot, no chain ------------------------------------------------------


@dataclass
class ToyParams:
    scale: float = 1.0
    refx: int = 0
    refy: int = 0


def mean_compute(ds, params):
    out = xr.Dataset({"mean": ds["frames"].mean("time") * params.scale})
    out.attrs.update(_probe(ds))
    return out


def mean_scalars(result):
    return {"mean": float(result["mean"].mean()), (1, 2, "pixel_mean"): 1.0}


def whole_compute(ds, params):
    """``mean_compute`` on a record that needs to be whole: what it was given."""
    from fusion_ui.core import loader

    out = mean_compute(ds, params)
    start, end = loader.discharge_window(ds)
    out.attrs.update(window_start=start, window_end=end)
    out.attrs.update(first=float(ds.time[0]), last=float(ds.time[-1]))
    return out


# --- a chain: a batch-only bank, and two products on it ----------------------


@dataclass
class Averages:
    threshold: float = 2.5
    window: int = 4
    single_counting: bool = True


@dataclass
class BankParams:
    averages: Averages = field(default_factory=Averages)


@dataclass
class Tracking:
    step: int = 1


@dataclass
class FieldsParams:
    averages: Averages = field(default_factory=Averages)
    tracking: Tracking = field(default_factory=Tracking)


@dataclass
class BlobsParams:
    averages: Averages = field(default_factory=Averages)
    neighbour_step: int = 1


def bank_compute(ds, params):
    window = params.averages.window
    bank = ds["frames"].isel(time=slice(0, window)).mean("time")
    out = xr.Dataset({"bank": bank * params.averages.threshold})
    out.attrs.update(_probe(ds))
    return out


def fields_compute(ds, params, upstream):
    return xr.Dataset({"field": upstream["bank"] * params.tracking.step})


def blobs_compute(ds, params, upstream):
    return xr.Dataset({"blob": upstream["bank"] + params.neighbour_step})


def bank_of(params):
    return BankParams(averages=params.averages)


# --- misbehaving ------------------------------------------------------------


def _sleep_until_interrupted(seconds):
    # Short sleeps: the signal handler runs between bytecodes.
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.01)


def interrupt_compute(ds, params):
    """Ctrl-C while computing: SIGINT to this very process."""
    os.kill(os.getpid(), signal.SIGINT)
    _sleep_until_interrupted(10)
    raise AssertionError("the interrupt never arrived")


def interrupt_scalars(result):
    """Ctrl-C while the store writes: scalars() runs after the run row is
    committed and before the scalars are."""
    os.kill(os.getpid(), signal.SIGINT)
    _sleep_until_interrupted(0.2)
    return {"mean": float(result["mean"].mean()), (0, 0, "pixel_mean"): 1.0}


def _mark(name):
    folder = os.environ.get(MARKERS)
    if folder:
        with open(os.path.join(folder, name), "w") as marker:
            marker.write(str(os.getpid()))


def stubborn_compute(ds, params):
    """Deaf to Ctrl-C, as a long call into C would be: it notes each interrupt
    and carries on, for at most a minute."""
    shot = str(ds.attrs["shot_number"])
    _mark(shot)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            time.sleep(0.01)
        except KeyboardInterrupt:
            _mark(f"{shot}.interrupted")
    return mean_compute(ds, params)


def crash_compute(ds, params):
    """A worker that dies outright (the OOM killer, a segfault)."""
    os._exit(3)


def block_compute(ds, params):
    """Mark that it started, then wait to be interrupted (at most a minute)."""
    _mark(str(ds.attrs["shot_number"]))
    _sleep_until_interrupted(60)
    return mean_compute(ds, params)


SPECS = [
    registry.PlotSpec(
        key="toy_mean",
        label="Toy mean",
        diagnostics=("apd",),
        params=ToyParams,
        render=_render,
        compute=mean_compute,
        scalars=mean_scalars,
    ),
    registry.PlotSpec(
        key="toy_whole",
        label="Toy whole record",
        diagnostics=("apd",),
        params=ToyParams,
        render=_render,
        compute=whole_compute,
        scalars=mean_scalars,
        whole_record=True,
    ),
    registry.PlotSpec(
        key="toy_bank",
        label="Toy bank",
        diagnostics=("apd",),
        params=BankParams,
        render=_render,
        compute=bank_compute,
        batch_only=True,
    ),
    registry.PlotSpec(
        key="toy_fields",
        label="Toy fields",
        diagnostics=("apd",),
        params=FieldsParams,
        render=_render,
        compute=fields_compute,
        requires="toy_bank",
        upstream_params=bank_of,
    ),
    registry.PlotSpec(
        key="toy_blobs",
        label="Toy blobs",
        diagnostics=("apd",),
        params=BlobsParams,
        render=_render,
        compute=blobs_compute,
        requires="toy_bank",
        upstream_params=bank_of,
    ),
    registry.PlotSpec(
        key="toy_interrupt",
        label="Toy interrupt",
        diagnostics=("apd",),
        params=ToyParams,
        render=_render,
        compute=interrupt_compute,
    ),
    registry.PlotSpec(
        key="toy_interrupt_on_write",
        label="Toy interrupt on write",
        diagnostics=("apd",),
        params=ToyParams,
        render=_render,
        compute=mean_compute,
        scalars=interrupt_scalars,
    ),
    registry.PlotSpec(
        key="toy_block",
        label="Toy block",
        diagnostics=("apd",),
        params=ToyParams,
        render=_render,
        compute=block_compute,
    ),
    registry.PlotSpec(
        key="toy_crash",
        label="Toy crash",
        diagnostics=("apd",),
        params=ToyParams,
        render=_render,
        compute=crash_compute,
    ),
    registry.PlotSpec(
        key="toy_stubborn",
        label="Toy stubborn",
        diagnostics=("apd",),
        params=ToyParams,
        render=_render,
        compute=stubborn_compute,
    ),
]

#: What a pool worker imports to know the toys: it registers them on import.
REGISTERING_MODULE = "tests.precompute_toys_registered"


def register():
    """Register every toy not registered yet."""
    for spec in SPECS:
        if spec.key not in registry.REGISTRY:
            registry.register(spec)


def unregister():
    """Take the toys out again, so no other test's page lists them."""
    for spec in SPECS:
        registry.REGISTRY.pop(spec.key, None)


#: How a test starts :func:`drive` in a process of its own. Imported by name
#: rather than run as ``__main__``, so that the toys are the same classes in
#: the driver and in its workers.
DRIVER = (
    "import sys; from tests.precompute_toys import drive; sys.exit(drive(sys.argv[1:]))"
)


def drive(argv):
    """``fusion-ui`` with the toys registered, in this process and in every
    pool worker -- for the tests that need a process of their own (a process
    group to send Ctrl-C to, or a niceness to inherit)."""
    from fusion_ui import cli
    from fusion_ui.core import precompute

    register()
    # The toys alone: a worker importing every real plot as well would take
    # seconds longer to start, for specs these tests never run.
    precompute.WORKER_MODULES = (REGISTERING_MODULE,)
    return cli.main(argv)
