"""Fields: every method's velocity field for a shot, and the average behind each pixel's number.

Reads the three products a batch job leaves in the result store (``pixel_averages``,
``method_fields``, ``blob_parameters``) and draws them. **This page never computes.** It asks the
ledger with ``store.find_run`` and loads blobs through a ``st.cache_data`` keyed on (path, mtime),
so flipping through the pixels of a shot never reloads the 12 MB bank; what is missing it names,
with the exact ``fusion-ui precompute`` command that fills it, because a 40-minute analysis must
never start inside the process that serves the group.

The page draws nothing itself: it walks ``views.SHOT_VIEWS`` and ``views.PIXEL_VIEWS``. A new
figure is a pure builder in ``fusion_ui/views/`` plus an entry in one of the two lists.

Session state another page may set before it switches here (the multi-shot jump does):

``fields.open``
    **A request to open the page somewhere, consumed once**:
    ``{"shot": 1160616027, "settings": <a method_fields params hash>, "pixel": (x, y)}``. Every
    entry is optional; an unknown shot or hash is ignored. It always wins, however often it is
    sent, because it is taken out of the session state when it is read: a jump to the shot the page
    was already remembering still lands there, which a comparison of states could not tell.
``selection``
    The shared selection contract, ``{"machine", "shot", "diagnostic", "preprocessed"}``. A
    selection that differs from the last one this page saw picks the shot (the shot browser's);
    otherwise the page keeps the shot it was last on.
``fields.settings``
    The ``method_fields`` parameters hash in use: the widget's own state. Anything unknown is the
    default.
``fields.pixel``
    ``(x, y)`` of the pixel the pixel level shows, or ``None``. Set by a click on any panel and by
    the previous/next buttons; kept when the shot or the settings change, and dropped when it is
    not on the array.

Every other key is view state of this page (the cuts, the lags of the strip) and never a parameter:
moving one recomputes nothing and mints no ``param_sets`` row.
"""

import dataclasses
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fusion_ui.plots  # noqa: F401 - importing the package registers every spec
from fusion_ui import config, ui, views
from fusion_ui.core import params_ui, precompute, registry, rundays, store, versions
from fusion_ui.views import products as prod
from fusion_ui.views.bundle import Bundle, Cuts
from fusion_ui.views.geometry import pixel_from_event, reading_order, step_pixel

st.set_page_config(page_title="Fields · Shot Explorer", layout="wide")

NC_MIME = "application/x-netcdf"


# ---------------------------------------------------------------------------
# Reading: the ledger and the blobs, never a compute
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading the stored result…", max_entries=24)
def load_blob(path, mtime_ns):
    """A stored dataset, in memory. ``mtime_ns`` is the cache key's: a rewritten blob is a new entry."""
    return store.load_result(None, {"blob_path": path})


def load(run):
    path = run["blob_path"]
    return load_blob(path, os.stat(path).st_mtime_ns)


@st.cache_data(ttl=60, show_spinner=False)
def current_versions():
    """The checkouts' commits as they are now, read at most once a minute."""
    return versions.code_version()


def when(created_at):
    """A ledger timestamp to the second: microseconds are kept for ordering runs, not for reading."""
    try:
        return datetime.fromisoformat(created_at).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return created_at


# ---------------------------------------------------------------------------
# View state that outlives a visit to another page
# ---------------------------------------------------------------------------

MEMORY = "fields._memory"
SETTINGS, DAY, SHOT = "fields.settings", "fields.day", "fields.shot"


def restore():
    """Put back the widget values the page remembered, before any widget is drawn.

    Streamlit drops the state of a widget that was not drawn in a run, so leaving this page -- the
    link to the dead-pixel view does -- would send the person back to the first shot with the cuts
    and the lags reset. Plain session-state keys survive, so the values are kept in one.
    """
    for key, value in st.session_state.get(MEMORY, {}).items():
        if key not in st.session_state:
            st.session_state[key] = value


def remember(key):
    """Keep the value of the widget ``key`` for :func:`restore`. Call it after the widget is drawn."""
    if key in st.session_state:
        st.session_state.setdefault(MEMORY, {})[key] = st.session_state[key]


# ---------------------------------------------------------------------------
# The sidebar: settings, run day and shot, cuts
# ---------------------------------------------------------------------------


def settings_label(setting):
    shots = f"{setting.shots} shot{'' if setting.shots == 1 else 's'}"
    return f"{setting.label} \u00b7 {shots}"


def pick_settings(container, conn, method_spec):
    """The ``method_fields`` parameter set in use: the default, or any other the ledger has a run for."""
    options = prod.settings(conn, method_spec, config.MACHINE)
    by_hash = {setting.hash: setting for setting in options}
    if st.session_state.get(SETTINGS) not in by_hash:
        # The default parameter set comes first.
        st.session_state[SETTINGS] = options[0].hash
    chosen = container.selectbox(
        "Settings",
        list(by_hash),
        format_func=lambda digest: settings_label(by_hash[digest]),
        key=SETTINGS,
        help=(
            "The default parameter set, or any other a batch job computed method_fields with, shown as "
            "its difference from the default. A shot with faster dynamics has its own, e.g. a shorter "
            "2DCA window."
        ),
    )
    remember(SETTINGS)
    setting = by_hash[chosen]
    if not setting.is_default:
        container.caption(
            "Differs from the default: "
            + "; ".join(
                f"`{path}` {value} (default {default})"
                for path, default, value in setting.diff
            )
        )
    return setting


def apply_request():
    """Open where another page asked: ``st.session_state["fields.open"]``, read once and removed.

    Returns the shot it names, if any. The settings go straight into the picker's state, which
    :func:`pick_settings` checks against what exists, and the pixel is this page's own. The selection
    is marked as seen: a jump sets it as well, and the request is the one that counts.
    """
    request = st.session_state.pop("fields.open", None)
    if not request:
        return None
    selection = st.session_state.get("selection")
    if selection:
        st.session_state["fields.seen_selection"] = dict(selection)
    if request.get("settings"):
        st.session_state[SETTINGS] = request["settings"]
    if request.get("pixel") is not None:
        st.session_state["fields.pixel"] = tuple(request["pixel"])
    return int(request["shot"]) if request.get("shot") is not None else None


def wanted_shot(known, asked):
    """The shot to put the pickers on in this run, or ``None`` to leave them where the person has them.

    A request wins; then a selection that differs from the last one this page saw (the shot
    browser's); and on a first visit, whatever the pickers are seeded with.
    """
    if asked in known:
        return asked
    selection = st.session_state.get("selection")
    if selection and selection != st.session_state.get("fields.seen_selection"):
        st.session_state["fields.seen_selection"] = dict(selection)
        if selection.get("shot") in known:
            return selection["shot"]
    return None


def pick_shot(day_box, shot_box, shots, good, asked=None):
    """``(shot, path)``: run day first, then a shot of it with the computed ones before the rest.

    The pickers are keyed and their state is set here, never left to a default index: a widget keeps
    what the person chose while its default does not move, so a default could not tell it to go
    somewhere else.
    """
    by_day = {}
    for shot, path in shots:
        by_day.setdefault(rundays.day_of(shot), []).append((shot, path))
    days = sorted(by_day)
    known = {shot for shot, _ in shots}

    wanted = wanted_shot(known, asked)
    if wanted is None and st.session_state.get(DAY) not in days:  # a first visit
        computed = sorted(good & known)
        wanted = computed[0] if computed else min(known)
    if wanted is not None:
        st.session_state[DAY], st.session_state[SHOT] = rundays.day_of(wanted), wanted

    def day_label(day):
        computed = sum(1 for shot, _ in by_day[day] if shot in good)
        return f"{day} \u00b7 {computed}/{len(by_day[day])} computed"

    day = day_box.selectbox("Run day", days, format_func=day_label, key=DAY)
    ordered = [s for s, _ in by_day[day] if s in good] + [
        s for s, _ in by_day[day] if s not in good
    ]
    if st.session_state.get(SHOT) not in ordered:
        st.session_state[SHOT] = ordered[0]
    shot = shot_box.selectbox(
        "Shot",
        ordered,
        format_func=lambda s: f"{s}" if s in good else f"{s} \u00b7 not computed",
        key=SHOT,
    )
    remember(DAY)
    remember(SHOT)
    return shot, dict(by_day[day])[shot]


def pick_cuts(container):
    """The view cuts. View state: moving one recomputes nothing."""
    container.markdown("**View cuts**")
    for key, default in (
        ("fields.min_lags", 8),
        ("fields.min_events", 200),
        ("fields.interior", False),
    ):
        st.session_state.setdefault(key, default)
    min_lags = container.number_input(
        "Minimum lags",
        min_value=0,
        max_value=200,
        step=1,
        key="fields.min_lags",
        help=(
            "Fewest lags a 2DCA track's slope may rest on to be drawn. A slope through two or three "
            "lags is a secant, not a fit, and the wildest velocities come from those pixels."
        ),
    )
    min_events = container.number_input(
        "Minimum events",
        min_value=0,
        max_value=100000,
        step=50,
        key="fields.min_events",
        help="Fewest events a conditional average may rest on, for the methods built on it.",
    )
    interior = container.checkbox(
        "Interior pixels only",
        key="fields.interior",
        help="Leave out the pixels on the array's border, where the 2DCA track leaves the view in a few lags.",
    )
    for key in ("fields.min_lags", "fields.min_events", "fields.interior"):
        remember(key)
    return Cuts(int(min_lags), int(min_events), bool(interior))


# ---------------------------------------------------------------------------
# The products: what exists, and the command for what does not
# ---------------------------------------------------------------------------


def show_command(product):
    """The command that fills a product, and the parameters to save when they are not its defaults."""
    st.code(product.command, language="bash")
    if not precompute.is_default(product.spec, product.params):
        st.caption(f"with these parameters saved as `{precompute.PARAMS_FILE}`:")
        st.code(params_ui.hash_params(product.key, product.params)[1], language="json")


def show_product(product, target):
    label = f"**{product.key}**"
    cost = prod.COST.get(product.key, "")
    if product.state == prod.UNREGISTERED:
        st.error(
            f"{label} is not registered in this deployment, so the page cannot read it.",
            icon="⚠️",
        )
    elif product.state == prod.MISSING:
        st.warning(
            f"{label} is not computed for {target.shot} with these settings ({cost}). Nothing is computed "
            "here; fill it from the command line on the server:",
            icon="⏳",
        )
        show_command(product)
    elif product.state == prod.FAILED:
        st.error(f"{label} failed in batch: {product.run['error']}", icon="⚠️")
        show_command(product)
    elif product.state == prod.UNREADABLE:
        st.warning(
            f"{label} is in the ledger but its result cannot be read. Recompute it from the command line:",
            icon="⚠️",
        )
        show_command(product)
    else:
        run = product.run
        seconds = f" in {run['seconds']:.0f} s" if run["seconds"] is not None else ""
        badges = []
        if product.stale:
            badges.append(f":orange-badge[stale: {product.stale}]")
        if product.code_note:
            badges.append(f":orange-badge[{product.code_note}]")
        st.markdown(
            f"{label} · computed {when(run['created_at'])}{seconds} · "
            f"`{run['code_version'] or 'version unknown'}` · params `{product.params_hash[:12]}` "
            + " ".join(badges)
        )
        if product.stale:
            st.caption(
                f"A recompute would differ from this result. Refresh it with `{product.refresh}`."
            )


def show_products(products, target):
    """Each product's state, with what fills the ones that are missing. Open when anything needs saying."""
    needs_saying = [p for p in products.values() if not p.ok or p.stale or p.code_note]
    with st.expander(
        "Products for this shot and these settings", expanded=bool(needs_saying)
    ):
        for product in products.values():
            show_product(product, target)


# ---------------------------------------------------------------------------
# Views: the two lists, drawn the same way
# ---------------------------------------------------------------------------


def draw_control(view, control, bundle):
    """One widget of a view. Its state is set before it is drawn and remembered after, never defaulted.

    The key carries the bank's hash, so the lags of a bank with another window are another widget
    and a slider is never out of its own bounds. A number that starts empty is the one exception:
    empty is its value (the builder works it out), and Streamlit keeps that without a default.
    """
    key = f"fields.{view.key}.{control.key}.{bundle.tag}"
    low, high, step = control.bounds if control.bounds else (None, None, None)
    if control.kind == "number":
        value = st.number_input(
            control.label,
            min_value=low,
            max_value=high,
            value=None,
            step=step,
            placeholder=control.placeholder or None,
            key=key,
            help=control.help or None,
        )
    else:
        st.session_state.setdefault(
            key, control.default if control.default is not None else ""
        )
        if control.kind == "radio":
            value = st.radio(
                control.label,
                control.options,
                format_func=lambda option: control.labels.get(option, option),
                horizontal=True,
                key=key,
                help=control.help or None,
            )
        elif control.kind == "slider":
            value = st.slider(
                control.label,
                min_value=low,
                max_value=high,
                step=step,
                key=key,
                help=control.help or None,
            )
        else:
            value = st.text_input(
                control.label,
                placeholder=control.placeholder or None,
                key=key,
                help=control.help or None,
            )
    remember(key)
    return value


def draw_result(result, key):
    """A builder's result: a figure, a table, a sentence, or a list of them."""
    items = result if isinstance(result, list) else [result]
    figures = [item for item in items if isinstance(item, go.Figure)]
    side_by_side = len(items) == 2 and len(figures) == 2
    columns = st.columns([3, 2]) if side_by_side else [st] * len(items)
    for index, (item, container) in enumerate(zip(items, columns)):
        if isinstance(item, go.Figure):
            container.plotly_chart(item, use_container_width=True, key=f"{key}.{index}")
        elif isinstance(item, pd.DataFrame):
            container.dataframe(item, hide_index=True, use_container_width=True)
        elif item:
            container.info(str(item), icon="ℹ️")


def draw_view(view, bundle, products):
    """One entry of a view list: its products, its controls, its builder."""
    st.subheader(view.title)
    absent = [key for key in view.reads if not products[key].ok]
    if absent:
        st.info(
            f"This needs {', '.join(absent)}, which is not computed with these settings. "
            "The commands that fill it are under Products above.",
            icon="⏳",
        )
        return
    values = {}
    controls = view.controls(bundle) if view.controls else ()
    if controls:
        for control, column in zip(controls, st.columns(len(controls))):
            with column:
                values[control.key] = draw_control(view, control, bundle)
    try:
        result = view.build(bundle, **values)
    except Exception as error:  # noqa: BLE001 - a plot bug must not take the page down
        st.error(f"{type(error).__name__}: {error}", icon="⚠️")
        st.caption(f"Drawing {view.title.lower()} failed.")
        return
    if view.caption:
        st.caption(view.caption)
    if view.selectable and isinstance(result, go.Figure):
        generation = st.session_state.get("fields.generation", 0)
        event = st.plotly_chart(
            result,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=f"fields.chart.{view.key}.{generation}",
        )
        picked = pixel_from_event(event)
        if picked is not None and picked != st.session_state.get("fields.pixel"):
            # Remount the chart under a new key: its selection lives in widget state, and the old one
            # would otherwise be read again on the next run (as core.multipixel.selector does).
            st.session_state["fields.pixel"] = picked
            st.session_state["fields.generation"] = generation + 1
            st.rerun()
        return
    draw_result(result, f"fields.{view.key}.{bundle.tag}")


# ---------------------------------------------------------------------------
# The mask and the downloads
# ---------------------------------------------------------------------------


def has_raw(conn, shot):
    return (
        conn.execute(
            "SELECT 1 FROM shots WHERE machine = ? AND shot = ? AND diagnostic = 'apd' AND preprocessed = 0",
            (config.MACHINE, shot),
        ).fetchone()
        is not None
    )


def show_mask(conn, bundle, shot):
    """The mask the products were computed with, its source, and the way to the evidence."""
    dead = bundle.dead
    source = bundle.mask_source or "source not recorded"
    left, right = st.columns([4, 2])
    left.caption(
        f"Dead-pixel mask: **{source}** · {int(dead.sum())} of {dead.size} pixels dead"
    )
    if not has_raw(conn, shot):
        right.caption(
            "No raw file is indexed for this shot, so there is no dead-pixel view to open."
        )
    elif right.button(
        "Open the dead-pixel view of the raw file",
        key="fields.dead_pixels",
        help="The single-shot page, raw version, on the PDF and spectrum of every pixel that the mask was made from.",
    ):
        st.session_state["selection"] = {
            "machine": config.MACHINE,
            "shot": shot,
            "diagnostic": "apd",
            "preprocessed": False,
        }
        st.session_state["spec.apd"] = registry.get("dead_pixels")
        st.switch_page("pages/2_single_shot.py")


def show_downloads(products, shot):
    ready = [p for p in products.values() if p.ok]
    if not ready:
        return
    # Narrow columns: a button is as wide as its label, and spread across the page they read as a table.
    columns = st.columns([2] * len(ready) + [max(1, 9 - 2 * len(ready))])
    for product, column in zip(ready, columns):
        path = product.run["blob_path"]
        column.download_button(
            f"Download {product.key} (.nc)",
            # Deferred: the 12 MB bank is read when the button is pressed, not on every rerun.
            data=lambda path=path: Path(path).read_bytes(),
            file_name=f"apd_{shot}_{product.key}_{product.params_hash[:8]}.nc",
            mime=NC_MIME,
            key=f"fields.download.{product.key}",
        )


# ---------------------------------------------------------------------------
# The pixel level's navigation
# ---------------------------------------------------------------------------


def pixel_header(bundle):
    x, y = bundle.pixel
    st.subheader(f"Pixel (x={x}, y={y})")
    R, Z = bundle.grid
    order = reading_order(R, Z)
    previous, following, close, _ = st.columns([1, 1, 1, 5])
    new = None
    if previous.button("◀ Previous", key="fields.previous"):
        new = step_pixel(order, bundle.dead, (x, y), -1)
    if following.button("Next ▶", key="fields.next"):
        new = step_pixel(order, bundle.dead, (x, y), +1)
    if close.button("Close", key="fields.close"):
        st.session_state["fields.pixel"] = None
        st.rerun()
    if new is not None and new != (x, y):
        st.session_state["fields.pixel"] = new
        st.rerun()
    st.caption(
        f"R = {R[y, x]:.4f} m, Z = {Z[y, x]:.4f} m · previous and next walk the live pixels in reading "
        "order, top row first"
    )


# ---------------------------------------------------------------------------


def main():
    st.session_state.setdefault("selection", None)
    st.session_state.setdefault("shot_selection", [])
    st.session_state.setdefault("fields.pixel", None)
    restore()
    asked = apply_request()

    st.title("Fields")
    st.caption(
        "Every method's velocity field, and the average behind each pixel's number, read from what "
        "batch jobs stored. Nothing is computed on this page."
    )
    conn = ui.get_connection()
    found, unregistered = prod.specs(registry)
    if "method_fields" not in found:
        st.error(
            "The Fields page needs the `method_fields` product, which is not registered in this "
            f"deployment (missing: {', '.join(unregistered)}).",
            icon="⚠️",
        )
        return
    shots = prod.preprocessed_shots(conn, config.MACHINE)
    if not shots:
        st.info(
            "No preprocessed APD shot is indexed for this machine. The products are computed from the "
            "preprocessed record: preprocess a shot, then run a rescan on the Shot browser page.",
            icon="ℹ️",
        )
        return

    # The sidebar is built in the order a person reads it, and filled in the order it depends on:
    # the shots offered first are the ones computed under the settings, so the settings come first.
    day_box, shot_box, settings_box, cuts_box = (
        st.sidebar.container() for _ in range(4)
    )
    setting = pick_settings(settings_box, conn, found["method_fields"])
    good = prod.good_shots(conn, config.MACHINE, setting.hash)
    shot, path = pick_shot(day_box, shot_box, shots, good, asked)
    cuts = pick_cuts(cuts_box)

    target = registry.Target(
        machine=config.MACHINE,
        shot=shot,
        diagnostic="apd",
        preprocessed=True,
        path=path,
        t_start=float("nan"),
        t_end=float("nan"),
        window_source="none",
    )
    method_params = prod.params_from_json(found["method_fields"], setting.params_json)
    products = prod.collect(
        conn,
        target,
        found,
        method_params,
        load,
        stale=prod.stale_reasons(conn),
        current=current_versions(),
    )

    bank = products["pixel_averages"]
    bundle = Bundle(
        shot=shot,
        bank=bank.dataset,
        fields=products["method_fields"].dataset,
        blobs=products["blob_parameters"].dataset,
        cuts=cuts,
        neighbour_step=prod.neighbour_step(method_params),
        tag=(bank.params_hash or "none")[:8],
    )
    # The pixel in view is kept across shots and settings, which share one array, even through a shot
    # with nothing computed. It is dropped only when it is not on the array the products are on.
    pixel = None
    remembered = st.session_state.get("fields.pixel")
    if bundle.reference() is not None and remembered is not None:
        ny, nx = bundle.shape
        x, y = int(remembered[0]), int(remembered[1])
        if 0 <= x < nx and 0 <= y < ny:
            pixel = (x, y)
        else:
            st.session_state["fields.pixel"] = None
    bundle = dataclasses.replace(bundle, pixel=pixel)

    st.subheader(f"Shot {shot}")
    if bundle.reference() is not None:
        show_mask(conn, bundle, shot)
    show_products(products, target)
    show_downloads(products, shot)

    for view in views.SHOT_VIEWS:
        draw_view(view, bundle, products)

    if bundle.reference() is None:
        return
    if bundle.pixel is None:
        if products["method_fields"].ok:
            st.info(
                "Click a pixel in any panel to see the average behind its numbers.",
                icon="\U0001f446",
            )
        return
    pixel_header(bundle)
    for view in views.PIXEL_VIEWS:
        draw_view(view, bundle, products)


main()
