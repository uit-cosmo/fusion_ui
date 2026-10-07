"""Where a click on a multi-shot point goes.

A point on the scatter is one shot's number from one *source*, ``(plot, params_hash, diagnostic,
preprocessed)``. Most sources open the single-shot page, on the plot that made the number with the
parameters that made it. The two products the Fields page draws, ``method_fields`` and
``blob_parameters``, open that page instead when it has settings to open on: on the shot, the
settings and (for a fixed-pixel scatter) the pixel the point is about. It reads the request once from
``st.session_state["fields.open"]`` (``pages/5_fields.py``)::

    {"shot": 1160616027, "settings": <a method_fields params hash>, "pixel": (x, y)}

The Fields page's **settings are a ``method_fields`` parameter set**. The ``blob_parameters`` it shows
are the ones that go with them (:func:`fusion_ui.views.products.related_params`: the 2DCA settings and
the contour's neighbour step carry over, everything else is the blob product's own default). A
``blob_parameters`` point therefore has no settings of its own, and :func:`settings_for` finds the
``method_fields`` ones it belongs to by asking that mapping forwards, because the mapping is the one
place that knows what goes with what.

**A blob run that no settings go with opens the single-shot page**, like every other source, on its
exact run. The Fields page shows the blob parameters of the settings it is on, and for such a run
those are other numbers than the one clicked (its fit settings are in no ``method_fields`` set); the
single-shot page restores the run's own parameters and draws it from the cache.

Pure: nothing here touches Streamlit, the database or the filesystem. The page reads the ledger
(:func:`fusion_ui.views.products.settings` and ``good_shots``) and hands over what it found.
"""

from fusion_ui.core import params_ui
from fusion_ui.views import products as prod

#: The sources whose points open the Fields page.
FIELDS_PLOTS = ("method_fields", "blob_parameters")


def opens_fields(plot):
    """Whether a point from ``plot`` opens the Fields page rather than the single-shot page."""
    return plot in FIELDS_PLOTS


def selection(machine, shot, source):
    """The shared selection contract (``st.session_state["selection"]``) for a clicked point.

    What every page follows, so it is set whichever page the click opens.
    """
    _, _, diagnostic, preprocessed = source
    return {
        "machine": machine,
        "shot": int(shot),
        "diagnostic": diagnostic,
        "preprocessed": bool(preprocessed),
    }


def related_blob_hash(found, setting):
    """The ``blob_parameters`` params hash that goes with a ``method_fields`` setting, or ``None``.

    ``found`` is ``{key: spec}`` as :func:`fusion_ui.views.products.specs` returns it, and ``setting``
    a settings picker option: the ``hash`` of a ``method_fields`` parameter set and its ``params_json``.
    ``None`` when there is no ``blob_parameters`` spec, and when the set cannot be rebuilt as it was
    stored (a field has been added or removed since): rebuilt, it would hash to another key, and the
    Fields page would not find what it was computed under.
    """
    try:
        params = prod.params_from_json(found["method_fields"], setting.params_json)
        if params_ui.hash_params("method_fields", params)[0] != setting.hash:
            return None
        related = prod.related_params(found, params).get("blob_parameters")
    except (KeyError, TypeError, ValueError):
        return None
    if related is None:
        return None
    return params_ui.hash_params("blob_parameters", related)[0]


def settings_for(plot, params_hash, found, options, good=()):
    """The ``method_fields`` params hash the Fields page opens on for a point, or ``None``.

    - A ``method_fields`` point: its own parameters.
    - A ``blob_parameters`` point: the settings among ``options`` whose related blob parameters hash to
      the point's. Several settings can: the blob parameters read only the 2DCA settings and the
      neighbour step, so two ``method_fields`` sets that differ in the tracking filters or the TDE give
      the same blob run. The first that has a good ``method_fields`` run on the shot (``good``, a
      collection of hashes) is chosen, so that the page has velocity fields to draw beside the blob
      parameters; the options come in the picker's order, default first, so the default wins when it
      is among them. When none has a run on the shot the first match is still chosen, so that the blob
      parameters are the ones shown.
    - ``None`` when nothing goes with it, which is a ``blob_parameters`` run whose own settings no
      ``method_fields`` set maps to (the ellipse fit or the duration time fit were changed, or its 2DCA
      settings were never run with the velocity fields), and any other plot. The click then goes to the
      single-shot page, which shows that exact run: the Fields page would show the blob parameters of
      whatever settings it is on, which are not the ones that were clicked.

    ``options`` are the settings picker's, in its order: objects with ``hash`` and ``params_json``.
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

    ``settings`` is the ``method_fields`` hash :func:`settings_for` found: a point it finds none for
    does not open the Fields page, so there is no request without one. The pixel goes only when the
    scatter's aggregate is a fixed pixel: any other aggregate has no pixel to open. ``how`` is a key of
    ``multishot.AGGREGATES``.
    """
    request = {"shot": int(shot), "settings": settings}
    if how == "pixel" and pixel is not None:
        request["pixel"] = (int(pixel[0]), int(pixel[1]))
    return request
