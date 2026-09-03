"""Parameter dataclasses: the canonical form, the cache key, and the widgets.

Every analysis declares its parameters as a dataclass -- often one that embeds
``imaging_methods`` classes directly, so the form and the analysis can never
disagree about what a knob is called. This module turns such a tree into three
things:

* :func:`canonical` -- a nested dict of JSON-safe scalars, coerced **by
  annotation**, which is what gets hashed;
* :func:`hash_params` -- the sha1 that keys ``param_sets`` and ``runs``;
* :func:`form` -- the Streamlit widget panel, walked from the same fields.

One walk produces all three, so the form and the key cannot drift apart.

Why the coercion is not optional
--------------------------------

The upstream defaults do not match their own annotations, and a cache key that
silently disagrees with itself is the failure mode that makes the whole tool
untrustworthy. So every leaf is forced through its declared type and anything
unrecognised raises rather than guessing:

``size_penalty: float = 5``
    An ``int`` under a ``float`` annotation. ``json.dumps`` writes ``5``, but a
    number input returns ``5.0`` -- two hashes for one parameter set.
``radius: int = 1000``
    ``imaging_methods.run_norm_ds`` does a strict ``isinstance(radius, int)``.
    A JSON round-trip yields ``1000.0`` and hard-fails at compute time.
``size_max: float = None``
    The annotation lies; it is really ``float | None``, and ``0.0`` is *not*
    the same as "use the full grid extent".
``velocity_estimation.EstimationOptions``
    Carries ``@dataclass`` but declares its own ``__init__``, so it has zero
    fields: ``asdict`` returns ``{}``. Walking one would produce an empty,
    colliding hash with no error at all -- hence the explicit guard.
"""

import dataclasses
import hashlib
import json
import math
import typing
from enum import Enum

# ---------------------------------------------------------------------------
# Field metadata that upstream does not carry
#
# imaging_methods has no ``field(metadata=...)`` anywhere: the allowed values
# for its string fields exist only in prose, and seven of the twenty-one leaves
# have no docstring at all. Rather than patch a dependency, the knowledge we
# need for a usable form lives here, keyed by (dataclass name, field name).
# ---------------------------------------------------------------------------

#: ``str`` fields that are really enumerations -> a selectbox instead of a
#: free-text input. ``window_type`` is resolved with ``getattr(scipy.signal.
#: windows, name)``; ``estimator`` is validated at runtime by
#: ``get_averaged_velocity_from_position``, which raises on anything else.
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
}

#: Fields annotated ``float`` (or ``int``) whose ``None`` means "work it out" --
#: they get a "default" checkbox next to the number input, because ``0.0`` is a
#: different instruction entirely.
OPTIONAL = {
    ("GaussFitParams", "size_max"),
    ("TdeVelocityParams", "max_threshold"),
}

#: Help text, by dotted path from the top of whichever tree is being walked.
#: Takes precedence over the docstring parse below.
HELP = {
    "preprocessing.radius": "Window length of the running normalisation, in samples.",
    "two_dca.refx": "X index of the reference pixel (0 .. nx-1).",
    "two_dca.refy": "Y index of the reference pixel (0 .. ny-1).",
    "taud_estimation.cutoff": "Frequency cutoff for the PSD fit, in rad/s.",
    "taud_estimation.nperseg": (
        "Welch segment length, in samples. Sets the lowest frequency that can "
        "be estimated; for run-normalised data, start at the normalisation "
        "window length."
    ),
    "position_filter.window_size": "Length of the smoothing filter on the position signal.",
    # PositionFilterParams and ContouringParams carry no docstring at all, so
    # without these five entries the busiest panel in the app is unlabelled.
    "position_filter.window_type": (
        "Shape of the smoothing window, resolved with scipy.signal.windows."
    ),
    "position_filter.mask_distance": (
        "How far the tracked structure may drift from the reference pixel and "
        "still count, in pixel widths."
    ),
    "position_filter.mask_signal_factor": (
        "Only track while the field's spatial maximum is at least this "
        "fraction of its largest value -- it is what keeps the tail of a "
        "decaying average out of the velocity."
    ),
    "position_filter.require_within_boundaries": (
        "Drop times where the contour touches the edge of the array. Worth "
        "switching on for a structure comparable in size to the field of view, "
        "where a clipped contour's centroid follows the edge and not the blob."
    ),
    "contouring.threshold_factor": (
        "Contour level, as a fraction of the maximum amplitude over the whole "
        "event. 0.3 is what the APD analyses use; 0.5 is the synthetic default."
    ),
    # velocity_2dca_tde.TdeParams has no upstream docstring at all -- it is
    # this app's own wrapper around a bare pair of arguments upstream never
    # grouped into a class.
    "tde.gauss_convolve": (
        "Smooth each pixel's lag trace with a Gaussian before locating its "
        "maximum. Off by default, matching upstream."
    ),
    "tde.sigma": (
        "Standard deviation of that smoothing Gaussian, in samples. Only "
        "used when gauss_convolve is on."
    ),
    "cross_corr.mask_signal_factor": (
        "Mask signal factor for the cross-correlation track only. The 2D "
        "cross-correlation sits on a pedestal well above zero rather than "
        "decaying to it like the conditional average does, so this fraction-"
        "of-max floor needs its own value to bind at all."
    ),
}


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

_NONE = type(None)


def _hints(cls):
    """``{field name: annotation}`` for ``cls``, resolving string annotations.

    ``imaging_methods`` does not use ``from __future__ import annotations``, so
    ``fields(cls)[i].type`` is already the live class -- but a params class
    written later might, and a string annotation would sail straight past every
    ``is`` check below.
    """
    try:
        return typing.get_type_hints(cls)
    except Exception:  # noqa: BLE001 - a forward reference we cannot resolve
        return {f.name: f.type for f in dataclasses.fields(cls)}


def _unwrap_optional(annotation):
    """``(inner, optional)`` -- ``Optional[T]`` and ``T | None`` both give T."""
    if typing.get_origin(annotation) is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not _NONE]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _dataclass_fields(annotation):
    """The fields of ``annotation`` if it is a dataclass, else ``None``.

    Raises on a dataclass with no fields: ``velocity_estimation``'s options
    classes look like dataclasses but declare their own ``__init__``, so
    ``fields()`` is empty and both the form and the hash would come out blank
    without complaint.
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


def _type_name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------


def _leaf(annotation, value, path):
    """One field's JSON-safe value, coerced by its declared type."""
    annotation, _ = _unwrap_optional(annotation)

    if _dataclass_fields(annotation) is not None:
        if value is None:
            return None
        if not dataclasses.is_dataclass(value):
            raise TypeError(f"{path}: expected {annotation.__name__}, got {value!r}")
        return _walk(value, path)

    # ``None`` is meaningful whatever the annotation claims -- see size_max.
    if value is None:
        return None

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        member = annotation(value) if not isinstance(value, annotation) else value
        return member.name
    if isinstance(value, Enum):
        return value.name

    # bool before int: bool is a subclass of int, so the order matters and a
    # True landing in an int field is a mistake, not a 1.
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
    cls = type(instance)
    # On the instance's own class, not just on the annotation: a field declared
    # as one dataclass can hold an instance of another, and the empty-fields
    # trap has to be caught wherever it turns up.
    fields = _dataclass_fields(cls)
    hints = _hints(cls)
    out = {}
    for field in fields:
        path = f"{prefix}.{field.name}" if prefix else field.name
        out[field.name] = _leaf(
            hints.get(field.name, field.type), getattr(instance, field.name), path
        )
    return out


def canonical(params):
    """``{"__type__": ..., "values": {...}}`` for a params dataclass instance.

    The class name is recorded because values alone do not identify a parameter
    set: ``fusion_scripts.SEParameters`` extends ``MethodParameters``, so their
    defaults share a prefix and would otherwise be indistinguishable.
    """
    if isinstance(params, type) or not dataclasses.is_dataclass(params):
        raise TypeError(f"expected a dataclass instance, got {params!r}")
    return {"__type__": _type_name(type(params)), "values": _walk(params)}


def payload(plot_key, params):
    """What actually gets hashed: the parameters *and* the plot they belong to.

    ``param_sets.hash`` is the primary key while ``plot`` is an ordinary column,
    so hashing the parameters alone would let two plots with identical defaults
    -- easy, since several specs use bare ``TwoDcaParams()`` -- collide on that
    key with conflicting ``plot`` values.
    """
    return {"plot": plot_key, "params": canonical(params)}


def _dumps(obj, indent=None):
    # allow_nan=False so a NaN from a widget raises here rather than being
    # written as a bare ``NaN`` literal that is not valid JSON.
    return json.dumps(
        obj,
        sort_keys=True,
        allow_nan=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
    )


def hash_params(plot_key, params):
    """``(sha1_hex, params_json)`` for one plot's parameter set."""
    body = payload(plot_key, params)
    return hashlib.sha1(_dumps(body).encode("utf-8")).hexdigest(), _dumps(
        body, indent=2
    )


# ---------------------------------------------------------------------------
# The inverse
# ---------------------------------------------------------------------------


def _rebuild(annotation, value, path):
    annotation, _ = _unwrap_optional(annotation)

    if _dataclass_fields(annotation) is not None:
        return None if value is None else from_dict(annotation, value)
    if value is None:
        return None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation[value]
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

    Coercion by annotation matters as much on the way in as on the way out: JSON
    has one number type, so ``radius`` comes back as ``1000.0`` and trips
    ``run_norm_ds``'s ``isinstance(radius, int)`` check at compute time.
    """
    hints = _hints(params_cls)
    kwargs = {}
    for field in dataclasses.fields(params_cls):
        if field.name not in values:
            continue
        path = f"{prefix}.{field.name}" if prefix else field.name
        kwargs[field.name] = _rebuild(
            hints.get(field.name, field.type), values[field.name], path
        )
    return params_cls(**kwargs)


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------


#: ``name`` or ``name1, name2`` optionally followed by ``(type)``, then a colon.
#: The shape imaging_methods uses in its class docstrings -- loosely: some
#: entries name three fields at once, and seven of the twenty-one leaves have no
#: entry at all, which is why HELP above exists.
def _docstring_help(cls):
    text = cls.__doc__ or ""
    names = {f.name for f in dataclasses.fields(cls)}
    out, current = {}, []
    for raw in text.splitlines():
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


def _help_for(cls, field_name, path):
    if path in HELP:
        return HELP[path]
    return _docstring_help(cls).get(field_name) or None


def _label(name):
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


def _choices_for(cls, field_name, path, spec, ds, chosen):
    """Allowed values for a ``str`` field, dynamic first.

    A probe's quantities are a property of the file that was opened, not of any
    annotation, so a spec may answer for its own fields before the static table
    is consulted.
    """
    if spec is not None and spec.choices is not None and ds is not None:
        dynamic = spec.choices(ds, path, chosen)
        if dynamic is not None:
            return tuple(dynamic)
    return CHOICES.get((cls.__name__, field_name))


def _seed(state, key, value):
    """Put a widget's starting value in session state, then let the key own it.

    Passing both ``value=`` and ``key=`` makes Streamlit warn and makes which
    one wins version-dependent; seeding once and passing only ``key`` does not.
    """
    if key not in state:
        state[key] = value
    return state[key]


def _scalar_widget(
    container, state, key, annotation, current, label, help_text, choices
):
    import streamlit as st  # noqa: F401 - imported for the session-state type only

    if annotation is bool:
        _seed(state, key, bool(current))
        return container.checkbox(label, key=key, help=help_text)

    if annotation is str or choices:
        options = list(choices) if choices else None
        if options is not None:
            if key in state and state[key] not in options:
                # The file changed under a remembered choice -- an ASP shot with
                # a different set of quantities. Fall back rather than raise.
                del state[key]
            _seed(state, key, current if current in options else options[0])
            return container.selectbox(label, options, key=key, help=help_text)
        _seed(state, key, "" if current is None else str(current))
        return container.text_input(label, key=key, help=help_text)

    if annotation is int:
        _seed(state, key, int(current or 0))
        return int(container.number_input(label, step=1, key=key, help=help_text))

    _seed(state, key, float(current or 0.0))
    # "%g" rather than Streamlit's default two decimals: these values span
    # 1e6 (a frequency cutoff) to 2e-5 (a duration time), and %.2f renders
    # both as an unusable 0.00.
    return float(container.number_input(label, format="%g", key=key, help=help_text))


def _optional_widget(container, state, key, annotation, current, label, help_text):
    """A number input with an "automatic" toggle, for a field whose ``None``
    means "work it out" -- ``size_max`` being the one that exists today. A plain
    number input would turn that into ``0.0``, which is a different instruction."""
    auto_key = f"{key}.__auto__"
    _seed(state, auto_key, current is None)
    auto = container.checkbox(
        f"{_label(label)}: automatic", key=auto_key, help=help_text
    )
    value = _scalar_widget(
        container, state, key, annotation, current or 0.0, label, None, None
    )
    return None if auto else value


def _fields(cls, defaults, prefix, key_prefix, container, state, spec, ds, chosen):
    hints = _hints(cls)
    values = {}
    for field in _dataclass_fields(cls):
        annotation, _ = _unwrap_optional(hints.get(field.name, field.type))
        path = f"{prefix}.{field.name}" if prefix else field.name
        current = getattr(defaults, field.name, None)
        help_text = _help_for(cls, field.name, path)

        if _dataclass_fields(annotation) is not None:
            group = container.expander(_label(field.name).title(), expanded=not prefix)
            if help_text:
                group.caption(help_text)
            values[field.name] = _fields(
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
        if (cls.__name__, field.name) in OPTIONAL:
            value = _optional_widget(
                container, state, key, annotation, current, field.name, help_text
            )
        else:
            value = _scalar_widget(
                container,
                state,
                key,
                annotation,
                current,
                _label(field.name),
                help_text,
                _choices_for(cls, field.name, path, spec, ds, chosen),
            )
        values[field.name] = value
        chosen[path] = value
    return cls(**values)


def form(params_cls, key_prefix, defaults=None, container=None, spec=None, ds=None):
    """Draw the widget panel for ``params_cls`` and return a filled instance.

    Widgets are keyed by ``key_prefix`` plus the dotted field path, so two specs
    on one page cannot collide and a parameter survives switching shots.
    """
    import streamlit as st

    if container is None:
        container = st.sidebar
    if defaults is None:
        defaults = params_cls()
    return _fields(
        params_cls,
        defaults,
        "",
        key_prefix,
        container,
        st.session_state,
        spec,
        ds,
        {},
    )


def seed_session_state(state, key_prefix, params, prefix=""):
    """Put a params instance's leaves into ``state`` under the widget keys.

    :func:`form` walks the same tree and reads each leaf back out of ``state``
    keyed by ``key_prefix`` plus the dotted field path, so seeding those keys is
    what lets the multi-shot view open the single-shot view on exactly the
    parameters that produced a stored result. Mirrors ``_fields``' key scheme,
    including the ``.__auto__`` toggle for :data:`OPTIONAL` fields.
    """
    cls = type(params)
    hints = _hints(cls)
    for field in _dataclass_fields(cls):
        annotation, _ = _unwrap_optional(hints.get(field.name, field.type))
        path = f"{prefix}.{field.name}" if prefix else field.name
        value = getattr(params, field.name)
        if _dataclass_fields(annotation) is not None:
            if value is None:
                continue
            seed_session_state(state, key_prefix, value, path)
            continue
        key = f"{key_prefix}.{path}"
        if (cls.__name__, field.name) in OPTIONAL:
            state[f"{key}.__auto__"] = value is None
            state[key] = value or 0.0
        else:
            state[key] = value


def panel(spec, target, ds=None, container=None):
    """``(params, ready)`` -- the parameter panel for one spec on one target.

    A live spec's widgets sit loose in the sidebar and take effect on the next
    rerun: they are cheap and interactive, and waiting for a button would make
    a frame slider useless.

    A cached spec's widgets go inside ``st.form``, so nothing is committed until
    Compute is pressed -- a nudged slider must not be able to start a
    four-minute analysis. ``ready`` is ``False`` until it has been pressed for
    this target; the parameters themselves persist across targets, so moving to
    the next shot is one click rather than a re-entry of every value.
    """
    import streamlit as st

    if container is None:
        container = st.sidebar
    key_prefix = f"params.{spec.key}"

    if not spec.cached:
        return (
            form(spec.params, key_prefix, container=container, spec=spec, ds=ds),
            True,
        )

    ready_key = f"ready.{spec.key}.{target.key}"
    with container.form(f"form.{spec.key}"):
        params = form(spec.params, key_prefix, container=st, spec=spec, ds=ds)
        if st.form_submit_button("Compute", use_container_width=True):
            st.session_state[ready_key] = True
    return params, bool(st.session_state.get(ready_key))
