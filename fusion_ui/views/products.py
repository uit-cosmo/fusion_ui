"""The products of a shot under some settings: which exist, which do not, and what fills them.

The Fields page never computes. It reads the ledger and the blobs a batch job left, and for what is
missing it names the command that fills it. Everything here is read-only: parameters are hashed
with ``params_ui.hash_params`` (which writes nothing), the ledger is asked with ``store.find_run``
and blobs are loaded through a callable the page supplies, so the page can cache them.

**Settings** mean a ``method_fields`` parameter set: the default, or any other one in the ledger.
The other products follow from it (:func:`related_params`).
"""

import copy
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from fusion_ui import config
from fusion_ui.core import params_ui, precompute, store, versions

#: In dependency order.
KEYS = ("pixel_averages", "method_fields", "blob_parameters")

OK = "ok"
FAILED = "failed"  # a run was recorded as failed
MISSING = "missing"  # nothing in the ledger
UNREADABLE = "unreadable"  # an ok row whose blob is gone or cannot be read

#: Roughly what a command costs, for the person deciding whether to run it now.
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
    params_hash: str
    run: Any  # the ``runs`` row, or None
    state: str
    dataset: Any = None
    #: Why the result may no longer be what a recompute gives (``store.stale_runs``' reason).
    stale: Optional[str] = None
    #: A sentence when the result was computed under other code than the checkout's (:func:`commit_note`).
    code_note: Optional[str] = None
    #: What fills it, when it is not ok.
    command: Optional[str] = None

    @property
    def ok(self):
        return self.state == OK


def specs(registry):
    """``({key: spec}, [missing keys])`` over :data:`KEYS`, from the registry."""
    found = {key: registry.REGISTRY[key] for key in KEYS if key in registry.REGISTRY}
    return found, [key for key in KEYS if key not in found]


def params_from_json(spec, text):
    """The parameters a ``param_sets.params_json`` text names, as ``spec``'s own dataclass."""
    body = json.loads(text)
    return params_ui.from_dict(spec.params, body["params"]["values"])


def related_params(found, method_params):
    """``{key: params}`` of the products that go with ``method_params``, the ``method_fields`` settings.

    ``pixel_averages`` is what ``method_fields`` is chained to. ``blob_parameters`` shares the 2DCA
    settings (``averages``) and the contour's ``neighbour_step`` with the velocity fields and keeps
    its own defaults for the rest, so a blob run made with other ``blobs`` settings is not found.
    """
    return {
        "pixel_averages": found["method_fields"].upstream_params(method_params),
        "method_fields": method_params,
        "blob_parameters": dataclasses.replace(
            found["blob_parameters"].params(),
            averages=copy.deepcopy(method_params.averages),
            neighbour_step=neighbour_step(method_params),
        ),
    }


def neighbour_step(method_params, default=1):
    """The contour level's neighbour step the velocity fields were computed with."""
    return getattr(getattr(method_params, "tracking", None), "neighbour_step", default)


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


def good_shots(conn, machine, params_hash):
    """The shots with a good ``method_fields`` run under ``params_hash``."""
    return {
        row["shot"]
        for row in conn.execute(
            "SELECT DISTINCT shot FROM runs WHERE machine = ? AND diagnostic = 'apd'"
            " AND preprocessed = 1 AND plot = 'method_fields' AND params_hash = ? AND status = 'ok'",
            (machine, params_hash),
        )
    }


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
            # Its fields are not this version's: rebuilt, it would hash to another key, where the
            # page would look for its products.
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


def commit_note(run, current):
    """A sentence about the code a run was computed under, or ``None``. Information only.

    A run whose ``code_version`` names ``fusion_scripts`` was computed when the methods lived
    there, and says so. Any other run compares ``imaging_methods``' commit with the checkout's.
    """
    stored = versions.parse(run["code_version"])
    if "fusion_scripts" in stored:
        before = versions.commit(stored["fusion_scripts"]) or stored["fusion_scripts"]
        return (
            f"computed under fusion_scripts {before}, before the methods moved to imaging_methods, "
            "whose regression on the server found the stored products bit-equal"
        )
    then = versions.commit(stored.get("imaging_methods"))
    now = versions.commit((current or {}).get("imaging_methods"))
    if then and now and then != now:
        return f"computed under imaging_methods {then}, the checkout is now at {now}"
    return None


def stale_reasons(conn):
    """``{run id: reason}`` for every run ``store.stale_runs`` lists. Needs the plots imported."""
    return {run["id"]: run["stale"] for run in store.stale_runs(conn)}


def refresh_command(target):
    """The command that recomputes the stale results of ``target``'s shot in place.

    Every product is named, with ``--stale``: that recomputes exactly the stale results of the
    shot, upstream first, each with the parameters it was stored with. Not ``--force`` on one
    product, which recomputes it on whatever lies beneath it, stale or not.
    """
    parts = ["fusion-ui", "precompute", *KEYS, "--stale", "--shot", str(target.shot)]
    if target.machine != config.MACHINE:
        parts += ["--machine", target.machine]
    return " ".join(parts)


def collect(conn, target, found, method_params, load, stale=None, current=None):
    """``{key: Product}`` for the three products of ``target`` under ``method_params``.

    ``load(run)`` returns a run's dataset or ``None``; the page caches it. ``stale`` is
    :func:`stale_reasons`' map and ``current`` ``versions.code_version()``'s.
    """
    stale = stale or {}
    wanted = related_params(found, method_params)
    products = {}
    for key in KEYS:
        spec, params = found[key], wanted[key]
        digest, _ = params_ui.hash_params(key, params)
        run = store.find_run(conn, target, key, digest)
        if run is None:
            state, flags = MISSING, ()
        elif run["status"] == "failed":
            state, flags = FAILED, ("--retry-failed",)
        else:
            exists = run["blob_path"] and os.path.exists(run["blob_path"])
            dataset = load(run) if exists else None
            if dataset is not None:
                products[key] = Product(
                    key,
                    spec,
                    params,
                    digest,
                    run,
                    OK,
                    dataset=dataset,
                    stale=stale.get(run["id"]),
                    code_note=commit_note(run, current),
                )
                continue
            state, flags = UNREADABLE, ("--force",)
        products[key] = Product(
            key,
            spec,
            params,
            digest,
            run,
            state,
            command=precompute.command(spec, target, params, *flags),
        )
    return products
