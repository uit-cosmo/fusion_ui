"""The paper's cut on the Fields page: ``reliable()`` of ``imaging_methods.decorrelation``, as a checkbox.

``views.reliable.rule`` is held to the library's own ``reliable`` on a real ``method_fields`` result
(the three real specs run on a synthetic record through the store, ``tests/real_fixtures``) and on
the synthetic products of ``tests/fields_fixtures`` (a 10 x 9 array with dead pixels and failed
fits, which has what a 3 x 3 record does not: many interior pixels).
"""

import numpy as np
import pytest

from imaging_methods import decorrelation as library
from fusion_ui.views import methods, panels, reliable
from fusion_ui.views.bundle import Bundle, Cuts
from fusion_ui.views.methods import METHODS, Status
from tests import fields_fixtures as ff
from tests import real_fixtures as real

#: What the paper's rule reads of a ``method_fields``, as its source lists it: stated here, and not
#: read off ``reliable.FINITE``, so that a name dropped from the rule is a name still tried.
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
def fields(tmp_path_factory):
    return real.deployment(tmp_path_factory).products()["method_fields"]


@pytest.fixture(scope="module")
def stand_ins():
    return ff.make_products(failed=((7, 4),), no_events=((6, 7),))


def paper(fields, min_lags, min_events):
    """The library's verdict, ``(y, x)`` bool."""
    return np.asarray(
        library.reliable(fields, min_lags=min_lags, min_events=min_events)
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


def test_the_cuts_default_to_the_papers_constants_and_the_track_is_the_centroid():
    assert (Cuts().min_lags, Cuts().min_events) == (
        library.MIN_LAGS,
        library.MIN_EVENTS,
    )
    assert library.TRACK == "_com"
    assert Cuts().paper is False  # the per-method cuts stay the default


def test_the_rule_is_the_papers_reliable_whatever_the_thresholds(fields, stand_ins):
    """On the real fields, each count of lags there is, and every count of events there is, one
    below and one above it; on the larger synthetic array, thresholds either side of what it holds.
    The pixels left in are the paper's, and every pixel left out has a reason while every pixel
    kept has none."""
    lags = np.asarray(fields["nlags_com"].values)
    events = np.asarray(fields["nevents"].values)
    found = {int(v) for v in events[np.isfinite(events)]}
    every_count = sorted({0, 10_000} | {e + d for e in found for d in (-1, 0, 1)})
    for data, min_lags, min_events in (
        (fields, range(0, int(np.nanmax(lags)) + 3), every_count),
        (stand_ins.fields, (0, 5, 8, 18, 19, 20), (0, 100, 450, 700, 900, 10_000)),
    ):
        kept = set()
        for lag in min_lags:
            for count in min_events:
                expected = paper(data, lag, count)
                ours = rule(data, lag, count)
                where = f"{lag} lags, {count} events"
                np.testing.assert_array_equal(ours.ok, expected, where)
                np.testing.assert_array_equal(ours.reason == "", expected, where)
                kept.add(int(expected.sum()))
        # Not a comparison of two empty masks: some pairs keep a pixel and some keep none.
        assert 0 in kept and max(kept) >= 1


def test_each_clause_of_the_papers_rule_is_this_rules(fields):
    """One pixel just passing (the interior one), then each clause failed in turn: the lags, the events,
    each of the six velocities, and the border. Both rules drop it, and keep the rest as it was.
    """
    lags = int(fields["nlags_com"].values[1, 1])
    events = int(fields["nevents"].values[1, 1])
    assert rule(fields, lags, events).ok[1, 1] and paper(fields, lags, events)[1, 1]

    def agree(changed, min_lags=lags, min_events=events):
        ours = rule(changed, min_lags, min_events).ok
        np.testing.assert_array_equal(ours, paper(changed, min_lags, min_events))
        return ours

    assert not agree(fields, min_lags=lags + 1)[1, 1]
    assert not agree(fields, min_events=events + 1)[1, 1]
    for name in READ:
        assert not agree(with_values(fields, name, np.nan))[1, 1], name
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


def test_a_pixel_says_which_clauses_of_the_rule_it_fails(fields):
    lags = int(fields["nlags_com"].values[1, 1])
    events = int(fields["nevents"].values[1, 1])
    prefix = reliable.PREFIX

    def why(changed=fields, min_lags=lags, min_events=events, y=1, x=1):
        return rule(changed, min_lags, min_events).reason[y, x]

    assert why() == ""  # a pixel it keeps says nothing
    assert why(with_values(fields, "vr3_tde", np.nan)) == f"{prefix}no 3TDE velocity"
    # A count of lags that is not there is said once: by the velocity it would have given, or alone.
    assert "lags unknown" in why(with_values(fields, "nlags_com", np.nan))
    assert (
        why(with_values(with_values(fields, "nlags_com", np.nan), "vr_com", np.nan))
        == f"{prefix}no 2DCA centroid velocity"
    )
    # Several at once, in the order the paper lists them, and the border last.
    several = with_values(fields, "nlags_com", lags - 1)
    several = with_values(several, "nevents", events - 1)
    several = with_values(several, "vr3_tde", np.nan)
    assert (
        why(several)
        == f"{prefix}2DCA centroid on {lags - 1} lags < {lags}, {events - 1} events < {events},"
        " no 3TDE velocity"
    )
    assert why(y=0, x=0).endswith("edge pixel")


def test_a_variable_the_product_lacks_keeps_nothing_where_the_papers_code_would_raise(
    fields,
):
    stripped = fields.drop_vars("vr3_tde")
    assert not rule(stripped, 0, 0).ok.any()
    assert "no 3TDE velocity" in rule(stripped, 0, 0).reason[1, 1]
    nothing = reliable.rule(None, (3, 3), 0, 0)
    assert not nothing.ok.any() and nothing.reason.shape == (3, 3)


# -- the panels use it -----------------------------------------------------------------------------


def bundle_of(stand_ins, **kwargs):
    return Bundle(
        shot=1160616027,
        bank=stand_ins.bank,
        fields=stand_ins.fields,
        blobs=stand_ins.blobs,
        **kwargs,
    )


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
        # Every pixel it leaves out carries the rule's reason.
        left_out = p.status == Status.CUT
        np.testing.assert_array_equal(p.reason[left_out], kept.reason[left_out])
        assert not any(p.reason[p.status == Status.OK])


def test_under_the_papers_cut_the_two_minimums_reach_every_panel_and_the_border_is_always_out(
    stand_ins,
):
    # More events than any pixel has: per method, the 2DCC and the TDEs off the record are not cut by
    # them; under the paper's rule, which asks every pixel of every panel for them, nothing is left.
    many = bundle_of(stand_ins, cuts=Cuts(min_events=10_000))
    for key in ("2dcc", "tde3", "tde2"):
        assert methods.panel(many, methods.METHOD_BY_KEY[key]).ok.any()
    edge = ~reliable.interior(many.shape)
    for method in METHODS:
        for cuts in (
            Cuts(min_events=10_000, paper=True),
            Cuts(min_lags=20, paper=True),
        ):
            assert not methods.panel(bundle_of(stand_ins, cuts=cuts), method).ok.any()
        # The border is out whether or not the checkbox for it is on.
        off = methods.panel(bundle_of(stand_ins, cuts=Cuts(paper=True)), method)
        on = methods.panel(
            bundle_of(stand_ins, cuts=Cuts(paper=True, interior_only=True)), method
        )
        np.testing.assert_array_equal(off.status, on.status)
        assert not off.ok[edge].any()


def test_no_products_at_all_is_not_an_error_under_either_cut():
    for cuts in (Cuts(), Cuts(paper=True)):
        assert methods.panel(Bundle(shot=1, cuts=cuts), METHODS[0]).status.size == 0


def test_the_figure_says_which_cut_is_on_and_draws_what_it_keeps(stand_ins):
    per_method = panels.velocity_panels(bundle_of(stand_ins))
    cuts = Cuts(min_lags=8, min_events=450, paper=True)
    paper_figure = panels.velocity_panels(bundle_of(stand_ins, cuts=cuts))
    assert per_method.layout.meta["cut"] == "method"
    assert paper_figure.layout.meta["cut"] == "paper"

    def said(figure):
        return [a.text for a in figure.layout.annotations if a.text.startswith("Cut")]

    assert said(per_method) == [methods.describe_cut(Cuts())]
    assert said(paper_figure) == [methods.describe_cut(cuts)]
    assert said(per_method) != said(paper_figure)
    # The pixels each panel draws are the rule's, and the ones it leaves out are named for the rule.
    kept = reliable.rule(stand_ins.fields, bundle_of(stand_ins).shape, 8, 450)
    for method in METHODS:
        dots = [
            t
            for t in paper_figure.data
            if t.meta
            and t.meta.get("kind") == "pixels"
            and t.meta["panel"] == method.key
        ]
        finite = np.isfinite(methods.panel(bundle_of(stand_ins), method).vr)
        drawn = {(int(c[0]), int(c[1])) for t in dots for c in t.customdata}
        expected = {(int(x), int(y)) for y, x in np.argwhere(kept.ok & finite)}
        assert drawn == expected, method.key
    names = {t.name for t in paper_figure.data if t.name}
    assert "not reliable (the paper's cut)" in names and "cut by the view" not in names
