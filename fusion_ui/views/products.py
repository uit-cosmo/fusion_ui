"""The products of a shot under some settings: which exist, which do not, and what fills them.

The Fields page never computes. It reads the ledger and the blobs a batch job left, and for what is
missing it names the command that fills it. Everything here is read-only: it hashes parameters with
:func:`fusion_ui.core.params_ui.hash_params` (which writes nothing, unlike ``store.record_params``),
asks the ledger with ``store.find_run`` and loads blobs through a callable the page supplies, so the
page can cache them on (path, mtime).

**Settings** mean a ``method_fields`` parameter set: the default, or any other one in the ledger,
shown as its difference from the default. The other products follow from it. ``pixel_averages`` is
exactly what ``method_fields`` is chained to (``upstream_params``). Any other product built on the
same bank (``blob_parameters``, and whatever is added to :data:`KEYS` after it) keeps its own
defaults except for what it shares with the velocity fields: the 2DCA settings (``averages``) and the
contour's ``neighbour_step``. That is the one place this module assumes anything of the products'
parameters beyond what the registry says, and it leaves a default alone where a name is not there.
"""

import copy
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from fusion_ui.core import params_ui, precompute, store, versions

KEYS = ("pixel_averages", "method_fields", "blob_parameters")

OK = "ok"
FAILED = "failed"  # a run was recorded as failed
MISSING = "missing"  # nothing in the ledger
UNREADABLE = "unreadable"  # an ok row whose blob is gone or cannot be read
UNREGISTERED = "unregistered"  # the spec itself is not registered

#: Roughly what a command costs, for the person deciding whether to run it now. From the plan.
COST = {
    "pixel_averages": "35–70 min on one core",
    "method_fields": "1–3 min",
    "blob_parameters": "about a minute",
}


@dataclass(frozen=True)
class Product:
    """One product of one shot under one set of parameters."""

    key: str
    spec: Any
    params: Any
    params_hash: Optional[str]
    run: Any  # the ``runs`` row, or None
    state: str
    dataset: Any = None
    #: Why the result may no longer be what a recompute gives (``store.stale_runs``' reason).
    stale: Optional[str] = None
    #: A sentence when fusion_scripts has moved on since the result was computed.
    code_note: Optional[str] = None
    #: What fills it, when it is not ok.
    command: Optional[str] = None
    #: What recomputes it, when it is stale.
    refresh: Optional[str] = None

    @property
    def ok(self):
        return self.state == OK and self.dataset is not None


# ---------------------------------------------------------------------------
# Specs and parameters
# ---------------------------------------------------------------------------


def specs(registry):
    """``({key: spec}, [missing keys])`` over :data:`KEYS`, from the registry."""
    found = {key: registry.REGISTRY[key] for key in KEYS if key in registry.REGISTRY}
    return found, [key for key in KEYS if key not in found]


def params_from_json(spec, text):
    """The parameters a ``param_sets.params_json`` text names, as ``spec``'s own dataclass."""
    body = json.loads(text)
    return params_ui.from_dict(spec.params, body["params"]["values"])


def _same_as_default(plot, params, spec):
    return (
        params_ui.hash_params(plot, params)[0]
        == params_ui.hash_params(plot, spec.params())[0]
    )


def related_params(found, method_params):
    """``{key: params}`` of the products that go with ``method_params``, the ``method_fields`` settings.

    ``pixel_averages`` is what ``method_fields`` is chained to, whatever its fields are called.
    Every other product keeps its own defaults except for what it shares with the velocity fields,
    the 2DCA settings (``averages``) and the contour's ``neighbour_step``. At the default settings
    every product is at its own default, so a field that is named differently on two specs cannot
    move a default product off its default key.
    """
    method_spec = found["method_fields"]
    out = {"method_fields": method_params}
    if "pixel_averages" in found:
        out["pixel_averages"] = method_spec.upstream_params(method_params)
    at_default = _same_as_default("method_fields", method_params, method_spec)
    for key, spec in found.items():
        if key in out or key == "pixel_averages":
            continue
        base = spec.params()
        if at_default:
            out[key] = base
            continue
        changes = {}
        names = {f.name for f in dataclasses.fields(base)}
        if "averages" in names and hasattr(method_params, "averages"):
            changes["averages"] = copy.deepcopy(method_params.averages)
        step = neighbour_step(method_params, None)
        if "neighbour_step" in names and step is not None:
            changes["neighbour_step"] = step
        out[key] = dataclasses.replace(base, **changes)
    return out


def neighbour_step(method_params, default=1):
    """The contour level's neighbour step the velocity fields were computed with."""
    return getattr(getattr(method_params, "tracking", None), "neighbour_step", default)


# ---------------------------------------------------------------------------
# Settings: the parameter sets of method_fields in the ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """A ``method_fields`` parameter set, for the settings picker."""

    hash: str
    params_json: Optional[str]
    is_default: bool
    diff: tuple  # ((dotted path, default value, this value), ...)
    shots: int  # shots of this machine with a good run under it
    label: str


def _flatten(values, prefix=""):
    out = {}
    for name, value in values.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def params_diff(default_text, other_text):
    """``((path, default, other), …)`` for every leaf where two ``params_json`` texts differ."""
    default = _flatten(json.loads(default_text)["params"]["values"])
    other = _flatten(json.loads(other_text)["params"]["values"])
    absent = "(absent)"
    return tuple(
        (path, default.get(path, absent), other.get(path, absent))
        for path in sorted(set(default) | set(other))
        if default.get(path, absent) != other.get(path, absent)
    )


def describe(diff, limit=3):
    """A settings label from its difference from the default: ``averages.window = 40``."""
    if not diff:
        return "default"
    parts = [f"{path} = {value}" for path, _, value in diff[:limit]]
    if len(diff) > limit:
        parts.append(f"+{len(diff) - limit} more")
    return "; ".join(parts)


def _shots_with_good_run(conn, machine, plot, params_hash):
    return {
        row["shot"]
        for row in conn.execute(
            "SELECT DISTINCT shot FROM runs WHERE machine = ? AND diagnostic = 'apd'"
            " AND preprocessed = 1 AND plot = ? AND params_hash = ? AND status = 'ok'",
            (machine, plot, params_hash),
        )
    }


def good_shots(conn, machine, params_hash):
    """The shots with a good ``method_fields`` run under ``params_hash``."""
    return _shots_with_good_run(conn, machine, "method_fields", params_hash)


def settings(conn, method_spec, machine):
    """The settings picker's options: the default first, then every other set the ledger has a run for."""
    default_hash, default_text = params_ui.hash_params(
        "method_fields", method_spec.params()
    )
    out = [
        Setting(
            hash=default_hash,
            params_json=default_text,
            is_default=True,
            diff=(),
            shots=len(good_shots(conn, machine, default_hash)),
            label="default",
        )
    ]
    others = conn.execute(
        "SELECT p.hash, p.params_json FROM param_sets p WHERE p.plot = 'method_fields' AND p.hash != ?"
        " AND EXISTS (SELECT 1 FROM runs r WHERE r.params_hash = p.hash AND r.machine = ?"
        "             AND r.diagnostic = 'apd')"
        " ORDER BY p.created_at, p.hash",
        (default_hash, machine),
    ).fetchall()
    for row in others:
        try:
            diff = params_diff(default_text, row["params_json"])
            rebuilt = params_from_json(method_spec, row["params_json"])
        except (ValueError, KeyError, TypeError):
            continue  # a parameter set this code cannot read is not offered
        if params_ui.hash_params("method_fields", rebuilt)[0] != row["hash"]:
            # Stored under parameters this version no longer has exactly (a field added or removed):
            # rebuilt, it would hash to another key and the page would look for its products there.
            continue
        out.append(
            Setting(
                hash=row["hash"],
                params_json=row["params_json"],
                is_default=False,
                diff=diff,
                shots=len(good_shots(conn, machine, row["hash"])),
                label=describe(diff),
            )
        )
    return out


# ---------------------------------------------------------------------------
# The shots the page offers
# ---------------------------------------------------------------------------


def preprocessed_shots(conn, machine):
    """``[(shot, path), …]`` of the preprocessed APD files indexed for ``machine``, by shot number."""
    return [
        (row["shot"], row["path"])
        for row in conn.execute(
            "SELECT shot, path FROM shots WHERE machine = ? AND diagnostic = 'apd' AND preprocessed = 1"
            " ORDER BY shot",
            (machine,),
        )
    ]


# ---------------------------------------------------------------------------
# What exists
# ---------------------------------------------------------------------------


def commit_note(run, current):
    """A sentence when the run was made under another fusion_scripts commit than the current checkout's."""
    stored = versions.commit(versions.parse(run["code_version"]).get("fusion_scripts"))
    now = versions.commit((current or {}).get("fusion_scripts"))
    if stored and now and stored != now:
        return f"computed under fusion_scripts {stored}, the checkout is now at {now}"
    return None


def stale_reasons(conn):
    """``{run id: reason}`` for every run ``store.stale_runs`` lists. Needs the plots imported."""
    return {run["id"]: run["stale"] for run in store.stale_runs(conn)}


def collect(conn, target, found, method_params, load, stale=None, current=None):
    """``{key: Product}`` for the three products of ``target`` under ``method_params``.

    ``load(run)`` returns a run's dataset or ``None``; the page caches it. ``stale`` is
    :func:`stale_reasons`' map and ``current`` ``versions.code_version()``'s. A product whose spec is
    not registered is :data:`UNREGISTERED` with nothing to run.
    """
    stale = stale or {}
    wanted = related_params(found, method_params) if "method_fields" in found else {}
    products = {}
    for key in KEYS:
        spec, params = found.get(key), wanted.get(key)
        if spec is None or params is None:
            products[key] = Product(key, spec, params, None, None, UNREGISTERED)
            continue
        digest, _ = params_ui.hash_params(key, params)
        run = store.find_run(conn, target, key, digest)
        if run is None:
            products[key] = Product(
                key,
                spec,
                params,
                digest,
                None,
                MISSING,
                command=precompute.command(spec, target, params),
            )
            continue
        if run["status"] == "failed":
            products[key] = Product(
                key,
                spec,
                params,
                digest,
                run,
                FAILED,
                command=precompute.command(spec, target, params, "--retry-failed"),
            )
            continue
        dataset = (
            load(run) if run["blob_path"] and os.path.exists(run["blob_path"]) else None
        )
        if dataset is None:
            products[key] = Product(
                key,
                spec,
                params,
                digest,
                run,
                UNREADABLE,
                command=precompute.command(spec, target, params, "--force"),
            )
            continue
        reason = stale.get(run["id"])
        products[key] = Product(
            key,
            spec,
            params,
            digest,
            run,
            OK,
            dataset=dataset,
            stale=reason,
            code_note=commit_note(run, current),
            refresh=(
                precompute.command(spec, target, params, "--force") if reason else None
            ),
        )
    return products
