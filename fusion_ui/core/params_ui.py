"""Parameter dataclasses: the canonical form, the cache key, and the widgets.

Every analysis declares its parameters as a dataclass, often one that embeds
``imaging_methods`` classes directly. One walk over that tree gives

* :func:`canonical`, a nested dict of JSON-safe scalars, and :func:`hash_params`,
  the sha1 of it that keys ``param_sets`` and ``runs``;
* :func:`form`, the Streamlit widget panel,

so the form and the key cannot drift apart. :func:`from_dict`,
:func:`from_canonical` and :func:`with_values` go the other way.

Every leaf is coerced **by its annotation**, and anything unrecognised raises,
because the upstream defaults do not match their own annotations and a cache key
that disagrees with itself fails silently:

* ``size_penalty: float = 5`` is an int; a number input returns ``5.0``, which
  would be a second hash for the same parameter set;
* ``size_max: float = None`` means "use the full grid extent", which ``0.0``
  does not;
* JSON has one number type, so an ``int`` comes back as ``1000.0`` and a strict
  ``isinstance(radius, int)`` downstream refuses it;
* ``velocity_estimation``'s option classes carry ``@dataclass`` but declare their
  own ``__init__``, so they have no fields and would hash as ``{}``.
"""

import copy
import dataclasses
import hashlib
import json
import math
import typing

# ---------------------------------------------------------------------------
# Field metadata that the params classes do not carry, keyed by
# (dataclass name, field name)
# ---------------------------------------------------------------------------

#: ``str`` fields that are really enumerations: a selectbox, not a text input.
#: A spec adds its own (``raw.py``, ``thomson_profiles.py``).
CHOICES = {
    ("PositionFilterParams", "window_type"): (
        "hann",
        "hamming",
        "blackman",
        "bartlett",
        "boxcar",
        "gaussian",
    ),
    ("VelocityParams", "estimator"): ("central_diff", "lsq"),
    ("PdfParams", "estimator"): ("histogram", "kde"),
}

#: Number fields whose ``None`` means "work it out": they get an "automatic"
#: checkbox beside the number input, because ``0.0`` is another instruction.
OPTIONAL = {
    ("GaussFitParams", "size_max"),
    ("TdeVelocityParams", "max_threshold"),
    ("PsdParams", "cutoff"),
}

#: Help text for fields whose class docstring has none. Takes precedence over
#: the docstring.
HELP = {
    ("TaudEstimationParams", "cutoff"): "Frequency cutoff for the PSD fit, in rad/s.",
    ("TaudEstimationParams", "nperseg"): (
        "Welch segment length, in samples. Sets the lowest frequency that can "
        "be estimated; for run-normalised data, start at the normalisation "
        "window length."
    ),
    ("PositionFilterParams", "window_size"): (
        "Length of the smoothing filter on the position signal."
    ),
    ("PositionFilterParams", "window_type"): (
        "Shape of the smoothing window, resolved with scipy.signal.windows."
    ),
    ("PositionFilterParams", "mask_distance"): (
        "How far the tracked structure may drift from the reference pixel and "
        "still count, in pixel widths."
    ),
    ("PositionFilterParams", "mask_signal_factor"): (
        "Only track while the field's spatial maximum is at least this "
        "fraction of its largest value -- it is what keeps the tail of a "
        "decaying average out of the velocity."
    ),
    ("PositionFilterParams", "require_within_boundaries"): (
        "Drop times where the contour touches the edge of the array. Worth "
        "switching on for a structure comparable in size to the field of view, "
        "where a clipped contour's centroid follows the edge and not the blob."
    ),
}


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def _hints(cls):
    """``{field name: annotation}`` for ``cls``, with string annotations resolved."""
    return typing.get_type_hints(cls)


def _unwrap_optional(annotation):
    """``T`` for ``Optional[T]`` and ``T | None``; anything else as it is."""
    if typing.get_origin(annotation) is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _dataclass_fields(annotation):
    """The fields of ``annotation`` if it is a dataclass, else ``None``.

    Raises on a dataclass with no fields, which would give a blank form and an
    empty, colliding hash without complaint.
    """
    if not (isinstance(annotation, type) and dataclasses.is_dataclass(annotation)):
        return None
    fields = dataclasses.fields(annotation)
    if not fields:
        raise TypeError(
            f"{annotation.__module__}.{annotation.__qualname__} is a dataclass "
            "with no fields -- it declares its own __init__, so it cannot be "
            "walked. Wrap the settings you need in a real dataclass."
        )
    return fields


def _annotated_fields(cls):
    """``[(field name, annotation with Optional removed)]`` for dataclass ``cls``."""
    hints = _hints(cls)
    return [(f.name, _unwrap_optional(hints[f.name])) for f in _dataclass_fields(cls)]


def _type_name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def _join(prefix, name):
    return f"{prefix}.{name}" if prefix else name


def _check_instance(params):
    if isinstance(params, type) or not dataclasses.is_dataclass(params):
        raise TypeError(f"expected a dataclass instance, got {params!r}")


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------


def _leaf(annotation, value, path):
    """One field's JSON-safe value, coerced by its declared type."""
    nested = _dataclass_fields(annotation) is not None
    # ``None`` is meaningful whatever the annotation claims.
    if value is None:
        return None
    if nested:
        if not dataclasses.is_dataclass(value):
            raise TypeError(f"{path}: expected {annotation.__name__}, got {value!r}")
        return _walk(value, path)

    # bool before int: bool is a subclass of int, and a True landing in an int
    # field is a mistake, not a 1.
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path}: expected a bool, got {value!r}")
        return value
    if isinstance(value, bool):
        raise TypeError(f"{path}: got a bool for a {annotation!r} field")

    if annotation is int:
        coerced = int(value)
        if coerced != value:
            raise TypeError(f"{path}: {value!r} is not an exact integer")
        return coerced
    if annotation is float:
        coerced = float(value)
        if not math.isfinite(coerced):
            raise ValueError(f"{path}: {value!r} is not finite")
        return coerced
    if annotation is str:
        return str(value)

    raise TypeError(
        f"{path}: no canonical form for {annotation!r} (value {value!r}). "
        "Add a rule to fusion_ui.core.params_ui._leaf rather than letting it "
        "fall into the hash unchecked."
    )


def _walk(instance, prefix=""):
    # The fields of the instance's own class, not of the annotation: a field
    # declared as one dataclass can hold an instance of another.
    return {
        name: _leaf(annotation, getattr(instance, name), _join(prefix, name))
        for name, annotation in _annotated_fields(type(instance))
    }


def canonical(params):
    """``{"__type__": ..., "values": {...}}`` for a params dataclass instance.

    The class name is recorded because values alone do not identify a parameter
    set: a subclass that adds nothing has the same values as its base.
    """
    _check_instance(params)
    return {"__type__": _type_name(type(params)), "values": _walk(params)}


def _dumps(obj, indent=None):
    # allow_nan=False so a NaN raises here rather than being written as a bare
    # ``NaN`` literal, which is not valid JSON.
    return json.dumps(
        obj,
        sort_keys=True,
        allow_nan=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
    )


def hash_params(plot_key, params):
    """``(sha1_hex, params_json)`` for one plot's parameter set.

    The plot key is hashed with the parameters: ``param_sets.hash`` is the
    primary key and ``plot`` an ordinary column, so two plots with the same
    parameter class and values would otherwise collide on it.
    """
    body = {"plot": plot_key, "params": canonical(params)}
    return hashlib.sha1(_dumps(body).encode("utf-8")).hexdigest(), _dumps(
        body, indent=2
    )


# ---------------------------------------------------------------------------
# The inverse, for parameters this app stored
# ---------------------------------------------------------------------------


def _rebuild(annotation, value, path):
    nested = _dataclass_fields(annotation) is not None
    if value is None:
        return None
    if nested:
        return from_dict(annotation, value, path)
    if annotation is bool:
        return bool(value)
    if annotation is int:
        coerced = int(value)
        if coerced != value:
            raise TypeError(f"{path}: {value!r} is not an exact integer")
        return coerced
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)
    raise TypeError(f"{path}: no rule to rebuild {annotation!r} from {value!r}")


def from_dict(params_cls, values, prefix=""):
    """Rebuild a params instance from :func:`canonical`'s ``values`` dict.

    Coerced by annotation, since JSON gives an ``int`` field back as a float. A
    field ``values`` does not name keeps its default, and a name the class does
    not have is ignored, so a set stored by another version of the class still
    reads.
    """
    return params_cls(
        **{
            name: _rebuild(annotation, values[name], _join(prefix, name))
            for name, annotation in _annotated_fields(params_cls)
            if name in values
        }
    )


# ---------------------------------------------------------------------------
# The strict inverse, for parameters a person wrote down
#
# A parameter file passed to `fusion-ui precompute --params-json` may have been
# typed by hand, and there from_dict's rules mislead: bool("false") is True,
# float("2.5") turns a string into a number, and a misspelt field is dropped.
# These raise instead, naming the dotted path.
# ---------------------------------------------------------------------------


def _strict_leaf(annotation, value, path):
    """One leaf of a hand-written parameter set, checked against its annotation."""
    if annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected true or false, got {value!r}")
        return value
    if isinstance(value, bool):
        raise ValueError(f"{path}: got {value!r} for a {annotation.__name__} field")
    if annotation is int:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError(f"{path}: expected a whole number, got {value!r}")
    if annotation is float:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{path}: expected a finite number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected a string, got {value!r}")
        return value
    raise TypeError(f"{path}: no rule to rebuild {annotation!r} from {value!r}")


def _strict_tree(params_cls, values, prefix, base):
    """``params_cls`` built from ``values``, every name checked.

    ``base`` is ``None`` for a complete parameter set, where every field must
    be given; otherwise the instance whose fields ``values`` replaces, leaving
    the rest as they are.
    """
    where = prefix or "the top level"
    if not isinstance(values, dict):
        raise ValueError(f"{where}: expected an object of fields, got {values!r}")
    annotations = dict(_annotated_fields(params_cls))
    unknown = sorted(set(values) - set(annotations))
    if unknown:
        raise ValueError(
            f"{where}: {params_cls.__name__} has no field "
            + ", ".join(repr(name) for name in unknown)
            + f" (it has {', '.join(annotations)})"
        )
    if base is None:
        missing = [name for name in annotations if name not in values]
        if missing:
            raise ValueError(
                f"{where}: {params_cls.__name__} is missing "
                + ", ".join(repr(name) for name in missing)
            )
    kwargs = {}
    for name, value in values.items():
        path = _join(prefix, name)
        annotation = annotations[name]
        nested = _dataclass_fields(annotation) is not None
        if value is None:
            kwargs[name] = None
        elif not nested:
            kwargs[name] = _strict_leaf(annotation, value, path)
        else:
            current = None if base is None else getattr(base, name)
            kwargs[name] = _strict_tree(annotation, value, path, current)
    if base is None:
        return params_cls(**kwargs)
    return dataclasses.replace(base, **kwargs)


def from_canonical(params_cls, canonical_form, plot_key=None):
    """The ``params_cls`` instance a canonical form describes, strictly.

    ``canonical_form`` is what :func:`canonical` returns (``{"__type__",
    "values"}``) or what ``param_sets.params_json`` stores (``{"plot",
    "params"}``), as a dict or as its JSON text. It must name ``params_cls`` as
    its ``__type__``, ``plot_key`` as its plot when one is given, and every
    field at every level and no other. What comes back hashes to the key the
    store records for those parameters.
    """
    form = canonical_form
    if isinstance(form, (str, bytes)):
        form = json.loads(form)
    if not isinstance(form, dict):
        raise ValueError(f"expected a canonical parameter set, got {form!r}")
    if "plot" in form or "params" in form:
        if "params" not in form or set(form) - {"plot", "params"}:
            raise ValueError(
                "a stored parameter set has exactly the keys 'plot' and 'params', "
                f"not {sorted(form)}"
            )
        if plot_key is not None and form.get("plot") != plot_key:
            raise ValueError(
                f"this is the parameter set of plot {form.get('plot')!r}, "
                f"not of {plot_key!r}"
            )
        form = form["params"]
    if not isinstance(form, dict) or set(form) != {"__type__", "values"}:
        raise ValueError(
            "a canonical parameter set has exactly the keys '__type__' and "
            f"'values', not {sorted(form) if isinstance(form, dict) else form!r}"
        )
    expected = _type_name(params_cls)
    if form["__type__"] != expected:
        raise ValueError(
            f"this parameter set is a {form['__type__']}, but the plot's "
            f"parameters are a {expected}"
        )
    return _strict_tree(params_cls, form["values"], "", None)


def with_values(params, values):
    """A copy of ``params`` with the fields named in ``values`` replaced.

    ``values`` is part of a canonical ``values`` tree, e.g. ``{"averages":
    {"window": 30}}``. Checked as strictly as :func:`from_canonical` checks a
    whole set, except that a field may be left out to keep its value. A nested
    field whose value is ``None`` has nothing to keep, so a change to it must
    give all of its fields. ``params`` itself is not modified.
    """
    _check_instance(params)
    # A deep copy, so the fields left alone are not shared with the caller's.
    return _strict_tree(type(params), values, "", copy.deepcopy(params))


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------


def _docstring_help(cls):
    """``{field name: text}`` from the entries of ``cls``'s docstring.

    An entry is ``name:`` or ``name1, name2:``, each name optionally followed by
    ``(type)``, and runs until a blank line.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    out, current = {}, []
    for raw in (cls.__doc__ or "").splitlines():
        line = raw.strip().lstrip("-").strip()
        head, sep, tail = line.partition(":")
        candidates = [
            part.split("(")[0].strip() for part in head.split(",") if part.strip()
        ]
        if sep and candidates and all(c in names for c in candidates):
            current = candidates
            for name in current:
                out[name] = tail.strip()
        elif current and line:
            for name in current:
                out[name] = f"{out[name]} {line}".strip()
        elif not line:
            current = []
    return out


def help_for(cls, field_name, path=None):
    """The help text of field ``field_name`` of ``cls``, or ``None``: :data:`HELP`
    first, then the class docstring. ``path`` is not used."""
    return (
        HELP.get((cls.__name__, field_name))
        or _docstring_help(cls).get(field_name)
        or None
    )


def _label(name):
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


def _choices_for(cls, field_name, path, spec, ds, chosen):
    """Allowed values for a ``str`` field: the spec's own answer for the opened
    file first, then the static table."""
    if spec is not None and spec.choices is not None and ds is not None:
        dynamic = spec.choices(ds, path, chosen)
        if dynamic is not None:
            return tuple(dynamic)
    return CHOICES.get((cls.__name__, field_name))


def _seed(state, key, value):
    """Put a widget's starting value in session state, then let the key own it.

    Passing both ``value=`` and ``key=`` makes Streamlit warn; seeding once and
    passing only ``key`` does not.
    """
    if key not in state:
        state[key] = value


def _scalar_widget(
    container, state, key, annotation, current, label, help_text, choices
):
    if annotation is bool:
        _seed(state, key, bool(current))
        return container.checkbox(label, key=key, help=help_text)

    if choices:
        options = list(choices)
        if key in state and state[key] not in options:
            # The file changed under a remembered choice (another shot's
            # quantities): fall back rather than raise.
            del state[key]
        _seed(state, key, current if current in options else options[0])
        return container.selectbox(label, options, key=key, help=help_text)

    if annotation is str:
        _seed(state, key, "" if current is None else str(current))
        return container.text_input(label, key=key, help=help_text)

    if annotation is int:
        _seed(state, key, int(current or 0))
        return int(container.number_input(label, step=1, key=key, help=help_text))

    _seed(state, key, float(current or 0.0))
    # "%g", not Streamlit's two decimals: the values span 1e6 (a frequency
    # cutoff) to 2e-5 (a duration time), and %.2f renders the latter as 0.00.
    return float(container.number_input(label, format="%g", key=key, help=help_text))


def _optional_widget(container, state, key, annotation, current, name, help_text):
    """A number input with an "automatic" checkbox, for an :data:`OPTIONAL` field."""
    auto_key = f"{key}.__auto__"
    _seed(state, auto_key, current is None)
    auto = container.checkbox(f"{_label(name)}: automatic", key=auto_key, help=help_text)
    value = _scalar_widget(
        container, state, key, annotation, current or 0.0, name, None, None
    )
    return None if auto else value


def _fields(cls, defaults, prefix, key_prefix, container, state, spec, ds, chosen):
    values = {}
    for name, annotation in _annotated_fields(cls):
        path = _join(prefix, name)
        current = getattr(defaults, name, None)
        help_text = help_for(cls, name)

        if _dataclass_fields(annotation) is not None:
            group = container.expander(_label(name).title(), expanded=not prefix)
            if help_text:
                group.caption(help_text)
            values[name] = _fields(
                annotation,
                current if current is not None else annotation(),
                path,
                key_prefix,
                group,
                state,
                spec,
                ds,
                chosen,
            )
            continue

        key = f"{key_prefix}.{path}"
        if (cls.__name__, name) in OPTIONAL:
            value = _optional_widget(
                container, state, key, annotation, current, name, help_text
            )
        else:
            value = _scalar_widget(
                container,
                state,
                key,
                annotation,
                current,
                _label(name),
                help_text,
                _choices_for(cls, name, path, spec, ds, chosen),
            )
        values[name] = value
        chosen[path] = value
    return cls(**values)


def form(params_cls, key_prefix, container, spec=None, ds=None):
    """Draw the widget panel for ``params_cls`` in ``container`` and return a
    filled instance.

    Widgets are keyed by ``key_prefix`` plus the dotted field path, so two specs
    on one page cannot collide and a parameter survives switching shots.
    """
    import streamlit as st

    return _fields(
        params_cls,
        params_cls(),
        "",
        key_prefix,
        container,
        st.session_state,
        spec,
        ds,
        {},
    )


def seed_session_state(state, key_prefix, params, prefix=""):
    """Put a params instance's leaves into ``state`` under the keys :func:`form`
    reads, so the form opens on exactly these parameters."""
    cls = type(params)
    for name, annotation in _annotated_fields(cls):
        path = _join(prefix, name)
        value = getattr(params, name)
        if _dataclass_fields(annotation) is not None:
            if value is not None:
                seed_session_state(state, key_prefix, value, path)
            continue
        key = f"{key_prefix}.{path}"
        if (cls.__name__, name) in OPTIONAL:
            state[f"{key}.__auto__"] = value is None
            state[key] = 0.0 if value is None else value
        else:
            state[key] = value


def panel(spec, target, ds=None):
    """``(params, ready)``: the sidebar parameter panel for one spec on one target.

    A live spec's widgets take effect on the next rerun. A cached spec's go
    inside ``st.form``, so a nudged slider cannot start an analysis: ``ready``
    is ``False`` until the button has been pressed for this target. The
    parameters persist across targets. A batch-only spec is never computed from
    a page, so its button reads **Show**.
    """
    import streamlit as st

    key_prefix = f"params.{spec.key}"
    if not spec.cached:
        return form(spec.params, key_prefix, st.sidebar, spec=spec, ds=ds), True

    ready_key = f"ready.{spec.key}.{target.key}"
    # The form is entered as a context manager and the widgets go through
    # `st`: passing the sidebar itself would place them outside the form.
    with st.sidebar.form(f"form.{spec.key}"):
        params = form(spec.params, key_prefix, st, spec=spec, ds=ds)
        label = "Show" if spec.batch_only else "Compute"
        if st.form_submit_button(label, use_container_width=True):
            st.session_state[ready_key] = True
    return params, bool(st.session_state.get(ready_key))
