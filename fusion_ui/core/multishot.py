"""Pure helpers for the multi-shot view.

They turn per-pixel scalar rows into one number per shot and join that to the
discharge metadata. Nothing here touches Streamlit, the database or the
filesystem.

How a grid of pixels becomes one number is a judgement (the mean, the median,
the maximum and a fixed pixel all differ), so the page offers the choice and
names it on the axis.
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

_SOURCE_COLUMNS = ["plot", "params_hash", "diagnostic", "preprocessed"]


def distinct_names(frame):
    """The scalar names present in ``frame``, sorted."""
    return sorted(frame["name"].dropna().unique())


def distinct_sources(frame, name):
    """The ``(plot, params_hash, diagnostic, preprocessed)`` tuples carrying ``name``.

    Several plots and parameter sets can write the same scalar name, so a name
    alone does not say which values to plot.
    """
    sub = frame[frame["name"] == name]
    return [
        tuple(row)
        for row in sub[_SOURCE_COLUMNS]
        .drop_duplicates()
        .sort_values(_SOURCE_COLUMNS, kind="stable")
        .itertuples(index=False)
    ]


def _rows(frame, source, name):
    """The rows of ``frame`` that hold ``name`` from ``source``."""
    keep = frame["name"] == name
    for column, value in zip(_SOURCE_COLUMNS, source):
        keep &= frame[column] == value
    return frame[keep]


def _shot_level(rows):
    """Whether every row is a shot-level scalar, stored at ``x = y = -1``."""
    return (rows["x"] == -1).all() and (rows["y"] == -1).all()


def aggregate(frame, source, name, how, pixel=None):
    """One value per ``(machine, shot)`` for one source, collapsed over pixels.

    ``how`` is a key of :data:`AGGREGATES`; ``pixel`` is the ``(x, y)`` that
    ``"pixel"`` needs. Values that are not finite numbers are dropped.
    Shot-level scalars are already one value per shot and come back unchanged
    whatever ``how`` is.
    """
    sub = _rows(frame, source, name)
    sub = sub[pd.to_numeric(sub["value"], errors="coerce").notna()]
    if sub.empty:
        return pd.DataFrame(columns=["machine", "shot", "value"])

    if _shot_level(sub):
        return (
            sub[["machine", "shot", "value"]].drop_duplicates().reset_index(drop=True)
        )

    if how == "pixel":
        if pixel is None:
            raise TypeError("`pixel` (an (x, y) tuple) is required when `how='pixel'`")
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
    """The ``(x, y)`` pairs that occur for one source, sorted.

    The page chains its two selectboxes on these, so that nobody can pick a
    pair the data does not have.
    """
    sub = _rows(frame, source, name)
    sub = sub[(sub["x"] != -1) | (sub["y"] != -1)]
    return sorted({(int(x), int(y)) for x, y in zip(sub["x"], sub["y"])})


def is_shot_level(frame, source, name):
    """Whether every value for ``source``/``name`` is a shot-level scalar."""
    return _shot_level(_rows(frame, source, name))


def with_metadata(aggregated, shot_table):
    """Join per-shot values to the discharge metadata the axes read.

    A value for a shot that is not in the index comes back with missing
    metadata, and the axis that needs it drops it.
    """
    columns = ["machine", "shot", "f_GW", "n_e_bar", "I_p", "mode", "has_metadata"]
    return aggregated.merge(
        shot_table[columns], on=["machine", "shot"], how="left", validate="many_to_one"
    )


def clicked_shot(point):
    """``(machine, shot)`` for one selected scatter point, or ``None``.

    ``plotly_express`` appends the ``hover_data`` columns to ``customdata``
    after ``[machine, shot]``, so the first two are read by position.
    """
    try:
        machine, shot = point["customdata"][:2]
        return machine, int(shot)
    except (TypeError, KeyError, ValueError):
        return None
