"""Where a click on a multi-shot point goes.

A point on the scatter is one shot's number from one *source*, ``(plot, params_hash, diagnostic,
preprocessed)``. Most sources open the single-shot page. The two products the Fields page draws,
``method_fields`` and ``blob_parameters``, open that page instead when it has settings to open on.
It reads the request once from ``st.session_state["fields.open"]``::

    {"shot": 1160616027, "settings": <a method_fields params hash>, "pixel": (x, y)}

The Fields page's settings are a ``method_fields`` parameter set, and the ``blob_parameters`` it
shows are the ones :func:`fusion_ui.views.products.related_params` derives from them. A
``blob_parameters`` point therefore has no settings of its own: :func:`settings_for` finds the
``method_fields`` ones it belongs to by asking that mapping forwards. A blob run that no settings go
with opens the single-shot page, on its exact run.

Pure: nothing here touches Streamlit, the database or the filesystem.
"""

from fusion_ui.core import params_ui
from fusion_ui.views import products as prod

#: The sources whose points open the Fields page.
FIELDS_PLOTS = ("method_fields", "blob_parameters")


def opens_fields(plot):
    """Whether a point from ``plot`` opens the Fields page rather than the single-shot page."""
    return plot in FIELDS_PLOTS


def selection(machine, shot, source):
    """The shared selection contract (``st.session_state["selection"]``) for a clicked point."""
    _, _, diagnostic, preprocessed = source
    return {
        "machine": machine,
        "shot": int(shot),
        "diagnostic": diagnostic,
        "preprocessed": bool(preprocessed),
    }


def related_blob_hash(found, setting):
    """The ``blob_parameters`` params hash that goes with a ``method_fields`` setting, or ``None``.

    ``found`` is ``{key: spec}`` as ``products.specs`` returns it, and ``setting`` a settings picker
    option (``hash`` and ``params_json``). ``None`` when the set cannot be rebuilt as it was stored:
    rebuilt, it would hash to another key.
    """
    try:
        params = prod.params_from_json(found["method_fields"], setting.params_json)
        if params_ui.hash_params("method_fields", params)[0] != setting.hash:
            return None
        related = prod.related_params(found, params)["blob_parameters"]
    except (KeyError, TypeError, ValueError):
        return None
    return params_ui.hash_params("blob_parameters", related)[0]


def settings_for(plot, params_hash, found, options, good=()):
    """The ``method_fields`` params hash the Fields page opens on for a point, or ``None``.

    - A ``method_fields`` point: its own parameters.
    - A ``blob_parameters`` point: the settings among ``options`` whose related blob parameters hash
      to the point's. Several can, since the blob parameters read only the 2DCA settings and the
      neighbour step. The first that has a good ``method_fields`` run on the shot (``good``, a
      collection of hashes) is chosen, so that the page has velocity fields to draw; otherwise the
      first match. ``options`` are the settings picker's, in its order, default first.
    - ``None`` when nothing goes with it: a ``blob_parameters`` run whose own settings no
      ``method_fields`` set maps to, and any other plot. The click then goes to the single-shot
      page, which shows that exact run.
    """
    if plot == "method_fields":
        return params_hash
    if plot != "blob_parameters":
        return None
    matching = [
        option.hash
        for option in options
        if related_blob_hash(found, option) == params_hash
    ]
    for digest in matching:
        if digest in good:
            return digest
    return matching[0] if matching else None


def fields_request(shot, settings, how, pixel):
    """What ``st.session_state["fields.open"]`` is set to, for the Fields page to read once.

    ``settings`` is the ``method_fields`` hash :func:`settings_for` found. The pixel goes only when
    the scatter's aggregate is a fixed pixel (``how == "pixel"``, a key of ``multishot.AGGREGATES``).
    """
    request = {"shot": int(shot), "settings": settings}
    if how == "pixel" and pixel is not None:
        request["pixel"] = (int(pixel[0]), int(pixel[1]))
    return request
