"""The paper's cut on the Fields page: ``reliable()`` of ``apd_check/figures.py``, as a checkbox.

The rule (``views.reliable``) is held to the paper's own ``reliable()`` on a real ``method_fields``
result: the three real specs run on J3's synthetic record through the store (``tests/real_fixtures``), and
the same fields go through both. ``decorrelation.apd_check.figures`` cannot be imported in the app's venv
(it imports ``figure_provenance``, a plotting helper the venv does not have, and nothing may be added to
it), so the paper's function is read out of the file with ``ast`` and run as written: the source of
``reliable()`` (``figures.py:127``) and the three constants it reads, nothing else of the module. A change
to the paper's rule fails these tests; a rename or a move of the function fails them loudly.

The panels' use of the rule is tested on the synthetic products of ``tests/fields_fixtures`` (a 10 x 9
array with dead pixels and failed fits), which has what a 3 x 3 record does not: many interior pixels.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fusion_ui.views import methods, panels, reliable
from fusion_ui.views.bundle import Bundle, Cuts
from fusion_ui.views.methods import METHODS, Status
from tests import fields_fixtures as ff
from tests import real_fixtures as real

FIGURES = "decorrelation.apd_check.figures"
#: What the paper's rule reads of a ``method_fields``, as its source lists it: stated here, and not read off
#: ``reliable.FINITE``, so that a name dropped from the rule is a name still tried.
READ = (
    "nlags_com",
    "nevents",
    "vr_com",
    "vz_com",
    "vr3_tde",
    "vz3_tde",
    "vr2_tde",
    "vz2_tde",
)


@pytest.fixture(scope="module")
def products(tmp_path_factory):
    return real.deployment(tmp_path_factory).products()


@pytest.fixture(scope="module")
def fields(products):
    return products["method_fields"]


def _assigned(node):
    """The names an assignment binds, ``MIN_LAGS, MIN_EVENTS = 8, 200`` included."""
    names = set()
    for target in node.targets:
        for part in ast.walk(target):
            if isinstance(part, ast.Name):
                names.add(part.id)
    return names


@pytest.fixture(scope="module")
def paper():
    """The paper's ``reliable()``, run as written: ``paper(fields, min_lags, min_events) -> (y, x) bool``.

    ``MIN_LAGS`` is a constant the function reads when it is called and ``min_events`` an argument, so the
    page's two thresholds reach it by the one and the other.
    """
    spec = importlib.util.find_spec(FIGURES)
    assert (
        spec is not None and spec.origin
    ), f"{FIGURES} is not installed, so there is no paper's rule to hold the page's to"
    path = Path(spec.origin)
    wanted = {"MIN_LAGS", "MIN_EVENTS", "TRACK"}
    body = [
        node
        for node in ast.parse(path.read_text()).body
        if (isinstance(node, ast.FunctionDef) and node.name == "reliable")
        or (isinstance(node, ast.Assign) and _assigned(node) & wanted)
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    missing = (wanted | {"reliable"}) - set(namespace)
    assert not missing, f"{path} no longer defines {sorted(missing)}"

    def call(fields, min_lags, min_events):
        namespace["MIN_LAGS"] = min_lags
        return np.asarray(namespace["reliable"](fields, min_events=min_events))

    return SimpleNamespace(
        call=call,
        TRACK=namespace["TRACK"],
        MIN_LAGS=namespace["MIN_LAGS"],
        MIN_EVENTS=namespace["MIN_EVENTS"],
        path=path,
    )


def rule(fields, min_lags, min_events):
    return reliable.rule(fields, fields["dead"].shape, min_lags, min_events)


def with_values(fields, name, value, at=((1, 1),)):
    """A copy of ``fields`` with ``name`` set to ``value`` at the pixels ``at`` (``(y, x)``)."""
    out = fields.copy(deep=True)
    data = np.array(out[name].values, dtype=float)
    for y, x in at:
        data[y, x] = value
    out[name] = out[name].copy(data=data)
    return out


# -- the rule is the paper's ---------------------------------------------------------------------


def test_the_cuts_default_to_the_papers_constants_and_the_track_is_the_centroid(paper):
    assert (Cuts().min_lags, Cuts().min_events) == (paper.MIN_LAGS, paper.MIN_EVENTS)
    assert paper.TRACK == "_" + reliable.TRACK == "_com"
    assert Cuts().paper is False  # the per-method cuts stay the default


def test_the_rule_is_the_papers_reliable_on_a_real_method_fields(fields, paper):
    """Every pair of thresholds that matters: each count of lags there is, and every count of events
    there is, one below and one above it. The pixels left in are the paper's, to the pixel.
    """
    lags = np.asarray(fields["nlags_com"].values)
    events = np.asarray(fields["nevents"].values)
    min_lags = range(0, int(np.nanmax(lags)) + 3)
    found = {int(v) for v in events[np.isfinite(events)]}
    min_events = sorted({0, 10_000} | {e + d for e in found for d in (-1, 0, 1)})
    kept = set()
    for lag in min_lags:
        for count in min_events:
            expected = paper.call(fields, lag, count)
            np.testing.assert_array_equal(
                rule(fields, lag, count).ok, expected, f"{lag} lags, {count} events"
            )
            kept.add(int(expected.sum()))
    # The comparison is not between two empty masks: some pairs keep a pixel and some keep none.
    assert 0 in kept and max(kept) >= 1


def test_each_clause_of_the_papers_rule_is_this_rules(fields, paper):
    """One pixel just passing (the interior one), then each clause failed in turn: the lags, the events,
    each of the six velocities, and the border. Both rules drop it, and keep the rest as it was.
    """
    lags, events = int(fields["nlags_com"].values[1, 1]), int(
        fields["nevents"].values[1, 1]
    )
    assert (
        rule(fields, lags, events).ok[1, 1] and paper.call(fields, lags, events)[1, 1]
    )

    def agree(changed, **thresholds):
        t = {"min_lags": lags, "min_events": events, **thresholds}
        ours = rule(changed, t["min_lags"], t["min_events"]).ok
        np.testing.assert_array_equal(
            ours, paper.call(changed, t["min_lags"], t["min_events"])
        )
        return ours

    assert not agree(fields, min_lags=lags + 1)[1, 1]
    assert not agree(fields, min_events=events + 1)[1, 1]
    for name in READ:
        assert not agree(with_values(fields, name, np.nan))[1, 1], name
    assert not agree(with_values(fields, "nlags_com", lags - 1))[1, 1]
    assert not agree(with_values(fields, "nevents", events - 1))[1, 1]
    # The border: the same good numbers on a corner are not enough.
    corner = fields
    for name in READ:
        value = float(fields[name].values[1, 1])
        corner = with_values(corner, name, value, at=((0, 0),))
    ours = agree(corner)
    assert not ours[0, 0] and ours[1, 1]
    # The maximum and the 2DCC are not read: a pixel without them is as reliable as it was.
    for name in ("vr_max", "vz_2dcc", "vr3_catde", "cc_tde"):
        assert agree(with_values(fields, name, np.nan))[1, 1], name


def test_the_rule_is_the_papers_on_a_larger_array_with_dead_pixels_and_failed_fits(
    paper,
):
    stand_ins = ff.make_products(failed=((7, 4),), no_events=((6, 7),)).fields
    kept = set()
    for lag in (0, 5, 8, 18, 19, 20):
        for count in (0, 100, 450, 700, 900, 10_000):
            expected = paper.call(stand_ins, lag, count)
            np.testing.assert_array_equal(
                rule(stand_ins, lag, count).ok, expected, f"{lag} lags, {count} events"
            )
            kept.add(int(expected.sum()))
    assert 0 in kept and max(kept) > 10


def test_a_pixel_says_which_clauses_of_the_rule_it_fails(fields):
    lags, events = int(fields["nlags_com"].values[1, 1]), int(
        fields["nevents"].values[1, 1]
    )
    prefix = reliable.PREFIX

    def why(changed=fields, min_lags=lags, min_events=events, y=1, x=1):
        return rule(changed, min_lags, min_events).reason[y, x]

    assert why() == ""  # a pixel it keeps says nothing
    assert (
        why(min_lags=lags + 1) == f"{prefix}2DCA centroid on {lags} lags < {lags + 1}"
    )
    assert why(min_events=events + 1) == f"{prefix}{events} events < {events + 1}"
    assert (
        why(with_values(fields, "nlags_com", 1))
        == f"{prefix}2DCA centroid on 1 lag < {lags}"
    )
    assert why(with_values(fields, "vr3_tde", np.nan)) == f"{prefix}no 3TDE velocity"
    assert why(with_values(fields, "vz2_tde", np.nan)) == f"{prefix}no 2TDE velocity"
    assert (
        why(with_values(fields, "vz_com", np.nan))
        == f"{prefix}no 2DCA centroid velocity"
    )
    # A method with two missing components is one missing velocity.
    both = with_values(with_values(fields, "vr3_tde", np.nan), "vz3_tde", np.nan)
    assert why(both) == f"{prefix}no 3TDE velocity"
    # A count of lags that is not there is said once: by the velocity it would have given, or alone.
    assert (
        why(with_values(fields, "nlags_com", np.nan))
        == f"{prefix}2DCA centroid lags unknown"
    )
    assert (
        why(with_values(with_values(fields, "nlags_com", np.nan), "vr_com", np.nan))
        == f"{prefix}no 2DCA centroid velocity"
    )
    assert why(with_values(fields, "nevents", np.nan)) == f"{prefix}no event count"
    # Several at once, in the order the paper lists them, and the border last.
    several = with_values(fields, "nlags_com", lags - 1)
    several = with_values(several, "nevents", events - 1)
    several = with_values(several, "vr3_tde", np.nan)
    assert (
        why(several)
        == f"{prefix}2DCA centroid on {lags - 1} lags < {lags}, {events - 1} events < {events},"
        " no 3TDE velocity"
    )
    assert why(y=0, x=0).endswith("edge pixel")  # the border is said last
    corner = fields
    for name in READ:
        corner = with_values(
            corner, name, float(fields[name].values[1, 1]), at=((0, 0),)
        )
    assert why(corner, y=0, x=0) == f"{prefix}edge pixel"


def test_a_variable_the_product_lacks_keeps_nothing_where_the_papers_code_would_raise(
    fields,
):
    stripped = fields.drop_vars("vr3_tde")
    assert not rule(stripped, 0, 0).ok.any()
    assert "no 3TDE velocity" in rule(stripped, 0, 0).reason[1, 1]
    nothing = reliable.rule(None, (3, 3), 0, 0)
    assert not nothing.ok.any() and nothing.reason.shape == (3, 3)


# -- the panels use it -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stand_ins():
    return ff.make_products(failed=((7, 4),), no_events=((6, 7),))


def bundle_of(stand_ins, **kwargs):
    return Bundle(
        shot=1160616027,
        bank=stand_ins.bank,
        fields=stand_ins.fields,
        blobs=stand_ins.blobs,
        **kwargs,
    )


def traces(figure, kind, panel=None):
    found = [t for t in figure.data if t.meta and t.meta.get("kind") == kind]
    return [t for t in found if panel is None or t.meta.get("panel") == panel]


def argwhere_xy(mask):
    return {(int(x), int(y)) for y, x in np.argwhere(mask)}


def test_with_the_papers_cut_every_panel_shows_the_pixels_it_keeps_and_no_others(
    stand_ins,
):
    cuts = Cuts(min_lags=8, min_events=450, paper=True)
    bundle = bundle_of(stand_ins, cuts=cuts)
    kept = reliable.rule(stand_ins.fields, bundle.shape, 8, 450)
    plain = bundle_of(stand_ins, cuts=Cuts(min_lags=8, min_events=450))
    live = ~stand_ins.geometry.dead
    assert kept.ok.any() and not kept.ok.all()
    for method in METHODS:
        p, own = methods.panel(bundle, method), methods.panel(plain, method)
        finite = np.isfinite(p.vr) & np.isfinite(p.vz)
        # The same pixels in every panel, wherever the panel's own method found a number there.
        np.testing.assert_array_equal(p.ok, kept.ok & finite & live, method.key)
        np.testing.assert_array_equal(
            p.status == Status.CUT, live & finite & ~kept.ok, method.key
        )
        # What the method itself did is its own: failures and dead pixels do not move with the cut.
        for status in (Status.FAILED, Status.DEAD):
            np.testing.assert_array_equal(p.status == status, own.status == status)
        # Every pixel it leaves out says it was the paper's rule, and the rule's reason.
        left_out = p.status == Status.CUT
        assert all(r.startswith(reliable.PREFIX) for r in p.reason[left_out])
        np.testing.assert_array_equal(p.reason[left_out], kept.reason[left_out])
        assert not any(p.reason[p.status == Status.OK])


def test_under_the_papers_cut_the_two_minimums_reach_every_panel_and_the_border_is_always_out(
    stand_ins,
):
    # More events than any pixel has: per method, the 2DCC and the TDEs off the record are not cut by
    # them; under the paper's rule, which asks every pixel of every panel for them, nothing is left.
    many = Cuts(min_events=10_000)
    for key in ("2dcc", "tde3", "tde2"):
        assert methods.panel(
            bundle_of(stand_ins, cuts=many), methods.METHOD_BY_KEY[key]
        ).ok.any()
    for method in METHODS:
        p = methods.panel(
            bundle_of(stand_ins, cuts=Cuts(min_events=10_000, paper=True)), method
        )
        assert not p.ok.any(), method.key
    # Every fit in the fixture rests on 19 lags: a minimum of 20 leaves nothing.
    for method in METHODS:
        p = methods.panel(
            bundle_of(stand_ins, cuts=Cuts(min_lags=20, paper=True)), method
        )
        assert not p.ok.any(), method.key
    # The border is out whether or not the checkbox for it is on.
    off = [
        methods.panel(bundle_of(stand_ins, cuts=Cuts(paper=True)), m).status
        for m in METHODS
    ]
    on = [
        methods.panel(
            bundle_of(stand_ins, cuts=Cuts(paper=True, interior_only=True)), m
        ).status
        for m in METHODS
    ]
    for a, b in zip(off, on):
        np.testing.assert_array_equal(a, b)
    ny, nx = bundle_of(stand_ins).shape
    edge = ~reliable.interior((ny, nx))
    assert all(
        not methods.panel(bundle_of(stand_ins, cuts=Cuts(paper=True)), m).ok[edge].any()
        for m in METHODS
    )


def test_the_per_method_cuts_are_what_they_were_and_the_default(stand_ins):
    """``paper`` is one more field with a default: nothing that does not set it can tell."""
    assert Cuts(8, 200, False) == Cuts(8, 200, False, False) == Cuts()
    for method in METHODS:
        a = methods.panel(bundle_of(stand_ins), method)
        b = methods.panel(bundle_of(stand_ins, cuts=Cuts(paper=False)), method)
        np.testing.assert_array_equal(a.status, b.status)
        np.testing.assert_array_equal(a.reason, b.reason)
    # No products at all is not an error under either cut.
    for cuts in (Cuts(), Cuts(paper=True)):
        assert methods.panel(Bundle(shot=1, cuts=cuts), METHODS[0]).status.size == 0


def test_the_figure_says_which_cut_is_on_and_draws_what_it_keeps(stand_ins):
    per_method = panels.velocity_panels(bundle_of(stand_ins))
    cuts = Cuts(min_lags=8, min_events=450, paper=True)
    paper = panels.velocity_panels(bundle_of(stand_ins, cuts=cuts))
    assert per_method.layout.meta["cut"] == "method"
    assert paper.layout.meta["cut"] == "paper"

    def said(figure):
        return [a.text for a in figure.layout.annotations if a.text.startswith("Cut")]

    assert said(per_method) == [methods.describe_cut(Cuts())]
    assert said(paper) == [methods.describe_cut(cuts)]
    assert "paper's reliable()" in said(paper)[0] and "450 events" in said(paper)[0]
    assert "per method" in said(per_method)[0] and "interior" not in said(per_method)[0]
    assert said(
        panels.velocity_panels(bundle_of(stand_ins, cuts=Cuts(interior_only=True)))
    )[0].endswith("interior pixels only")
    # The pixels each panel draws are the rule's, and the ones it leaves out are named for the rule.
    kept = reliable.rule(stand_ins.fields, bundle_of(stand_ins).shape, 8, 450)
    for method in METHODS:
        dots = traces(paper, "pixels", method.key)
        finite = np.isfinite(methods.panel(bundle_of(stand_ins), method).vr)
        drawn = {(int(c[0]), int(c[1])) for t in dots for c in t.customdata}
        assert drawn == argwhere_xy(kept.ok & finite), method.key
    names = {t.name for t in paper.data if t.name}
    assert "not reliable (the paper's cut)" in names and "cut by the view" not in names
    cutting = panels.velocity_panels(bundle_of(stand_ins, cuts=Cuts(min_events=700)))
    assert "cut by the view" in {t.name for t in cutting.data if t.name}
    # Maps and arrows alike carry it.
    assert (
        panels.velocity_panels(bundle_of(stand_ins, cuts=cuts), mode="vr").layout.meta[
            "cut"
        ]
        == "paper"
    )
