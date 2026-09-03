"""Pure helpers for the multi-shot view.

The page reads scalars from the store and metadata from the catalog; the work
of turning per-pixel rows into one number per shot, and of joining that to the
discharge descriptor, lives here so it can be tested without a Streamlit
runtime. Nothing in this module touches Streamlit, the database or the
filesystem.

A scalar value is per-pixel (``x`` / ``y``), and the multi-shot scatter needs
one number per shot, so the page offers a choice of how to collapse the grid.
Which choice is right is a research judgement -- mean over the array, a fixed
reference pixel, a median over a region and the pixel of maximum signal all
give different answers -- which is why the choice is shown in the axis label
rather than silently baked in.
"""

import pandas as pd

#: How a grid of per-pixel values becomes one number per shot, and what the
#: page calls each option.
AGGREGATES = {
    "mean": "mean over pixels",
    "median": "median over pixels",
    "maximum": "maximum over pixels",
    "pixel": "fixed pixel",
}

#: The x-axes the page offers: key -> (column in the shot table, axis label).
X_AXES = {
    "shot": ("shot", "shot number"),
    "f_GW": ("f_GW", "Greenwald fraction f_GW"),
    "n_e_bar": ("n_e_bar", "line-averaged density n̄_e [10²⁰ m⁻³]"),
    "I_p": ("I_p", "plasma current I_p [MA]"),
}


def distinct_names(frame):
    """The scalar names present in ``frame``, sorted."""
    return sorted(frame["name"].dropna().unique())


def distinct_sources(frame, name):
    """The ``(plot, params_hash, diagnostic, preprocessed)`` tuples carrying ``name``.

    A scalar name can be written by several parameter sets -- and, for the
    seeded rows, by the import rather than by a live analysis -- so picking a
    name is not enough to say *which* values to plot. Each distinct source is
    one entry in the page's source picker.
    """
    sub = frame[frame["name"] == name]
    columns = ["plot", "params_hash", "diagnostic", "preprocessed"]
    return [
        tuple(row)
        for row in sub[columns]
        .drop_duplicates()
        .sort_values("plot", kind="stable")
        .itertuples(index=False)
    ]


def _finite(frame):
    values = pd.to_numeric(frame["value"], errors="coerce")
    return frame[values.notna()]


def aggregate(frame, source, name, how, pixel=None):
    """One value per ``(machine, shot)`` for one source, collapsed over pixels.

    ``source`` is a ``(plot, params_hash, diagnostic, preprocessed)`` tuple as
    returned by :func:`distinct_sources`. ``how`` is a key of
    :data:`AGGREGATES`; ``pixel`` is an ``(x, y)`` tuple required when ``how``
    is ``"pixel"``. Shot-level scalars (``x = y = -1``) already have one value
    per shot and are returned unchanged whatever ``how`` is.
    """
    plot, params_hash, diagnostic, preprocessed = source
    sub = frame[
        (frame["name"] == name)
        & (frame["plot"] == plot)
        & (frame["params_hash"] == params_hash)
        & (frame["diagnostic"] == diagnostic)
        & (frame["preprocessed"] == preprocessed)
    ]
    empty = pd.DataFrame(columns=["machine", "shot", "value"])
    sub = _finite(sub)
    if sub.empty:
        return empty

    if (sub["x"] == -1).all() and (sub["y"] == -1).all():
        return (
            sub[["machine", "shot", "value"]].drop_duplicates().reset_index(drop=True)
        )

    if how == "pixel":
        x, y = pixel
        return sub[(sub["x"] == x) & (sub["y"] == y)][
            ["machine", "shot", "value"]
        ].reset_index(drop=True)

    group = sub.groupby(["machine", "shot"], as_index=False)["value"]
    if how == "median":
        return group.median()
    if how == "maximum":
        return group.max()
    return group.mean()


def pixel_choices(frame, source, name):
    """The ``(xs, ys)`` a fixed-pixel picker may offer for one source."""
    plot, params_hash, diagnostic, preprocessed = source
    sub = frame[
        (frame["name"] == name)
        & (frame["plot"] == plot)
        & (frame["params_hash"] == params_hash)
        & (frame["diagnostic"] == diagnostic)
        & (frame["preprocessed"] == preprocessed)
    ]
    sub = sub[(sub["x"] != -1) | (sub["y"] != -1)]
    return sorted(sub["x"].unique()), sorted(sub["y"].unique())


def is_shot_level(frame, source, name):
    """Whether every value for ``source``/``name`` is a shot-level scalar."""
    plot, params_hash, diagnostic, preprocessed = source
    sub = frame[
        (frame["name"] == name)
        & (frame["plot"] == plot)
        & (frame["params_hash"] == params_hash)
        & (frame["diagnostic"] == diagnostic)
        & (frame["preprocessed"] == preprocessed)
    ]
    return (sub["x"] == -1).all() and (sub["y"] == -1).all()


def with_metadata(aggregated, shot_table):
    """Join per-shot values to the discharge metadata the axes read.

    The shot table is keyed by ``(machine, shot)``; a value for a shot that has
    since left the index (or belongs to a different machine) comes back with
    missing metadata and is dropped by whichever axis needs it.
    """
    columns = ["machine", "shot", "f_GW", "n_e_bar", "I_p", "mode", "has_metadata"]
    return aggregated.merge(
        shot_table[columns], on=["machine", "shot"], how="left", validate="many_to_one"
    )
