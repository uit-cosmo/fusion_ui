"""Each statistic's compute against a trace with a planted answer.

The blob-fixture philosophy applied to statistics: a sine peaks where the sine
is, an autocorrelation is 1 at zero lag, a shifted copy peaks at the shift.
"""

import numpy as np
import pytest

import fusion_ui.stats  # noqa: F401 - registers every spec
from fusion_ui.core import statistics, traces

DT = 1e-6
N = 20000


def _trace(values, dt=DT):
    return traces.Trace(
        ref=traces.TraceRef("cmod", 1, "apd", False, ("pixel", 0, 0)),
        time=np.arange(len(values)) * dt,
        value=np.asarray(values, dtype=float),
        dt=dt,
        coords={},
        label="test",
    )


def _sine(freq=5000.0, n=N, seed=3):
    rng = np.random.default_rng(seed)
    time = np.arange(n) * DT
    return np.sin(2 * np.pi * freq * time) + 0.1 * rng.normal(size=n)


def _ornstein_uhlenbeck(tau, n=10000, seed=11):
    rng = np.random.default_rng(seed)
    decay = np.exp(-DT / tau)
    out = np.zeros(n)
    for index in range(1, n):
        out[index] = decay * out[index - 1] + rng.normal()
    return out


def test_registry_holds_four_specs_in_plan_order():
    assert [s.key for s in statistics.all_specs()] == ["pdf", "psd", "acf", "ccf"]
    assert statistics.get("ccf").pairwise
    assert not statistics.get("pdf").pairwise
    with pytest.raises(ValueError):
        statistics.register(statistics.get("pdf"))


def test_psd_of_a_sine_peaks_at_that_sines_frequency():
    from fusion_ui.stats import psd

    result = psd.compute(
        _trace(_sine()), psd.PsdParams(nperseg=2000, cutoff=None, fit=False)
    )
    peak = float(result["omega"].values[int(np.argmax(result["psd"].values))])
    assert peak / (2 * np.pi) == pytest.approx(5000.0, rel=0.05)


def test_acf_at_zero_lag_is_one():
    from fusion_ui.stats import acf

    result = acf.compute(
        _trace(_sine()), acf.AcfParams(max_lag=1e-4, biased=False, fit=False)
    )
    lags = result["lag"].values
    assert result["acf"].values[int(len(lags) // 2)] == pytest.approx(1.0)


def test_acf_fit_recovers_a_planted_correlation_time():
    from fusion_ui.stats import acf

    planted = 40e-6
    result = acf.compute(
        _trace(_ornstein_uhlenbeck(planted)),
        acf.AcfParams(max_lag=4e-4, biased=False, fit=True),
    )
    assert float(result["taud"]) == pytest.approx(planted, rel=0.5)
    assert "acf_fit" in result


def test_ccf_against_a_shifted_copy_peaks_at_the_shift():
    from fusion_ui.stats import ccf

    values = _sine()
    shift = 37
    result = ccf.compute(
        _trace(np.roll(values, shift)),
        _trace(values),
        ccf.CcfParams(max_lag=2e-4, biased=False),
    )
    assert float(result["peak_lag"]) == pytest.approx(shift * DT, abs=2 * DT)
    assert float(result["peak_value"]) == pytest.approx(1.0, abs=0.05)


def test_pdf_of_standard_normal_samples_integrates_to_one_and_peaks_near_zero():
    from fusion_ui.stats import pdf

    values = np.random.default_rng(3).normal(size=N)
    for estimator in ("histogram", "kde"):
        result = pdf.compute(
            _trace(values),
            pdf.PdfParams(bins=64, estimator=estimator, standardise=True,
                          log_y=True),
        )
        assert np.trapz(result["pdf"].values, result["value"].values) == \
            pytest.approx(1.0, rel=0.05)
        assert abs(float(result["value"].values[int(np.argmax(result["pdf"].values))])) < 0.2


def test_standardise_changes_the_axis_and_bins_changes_the_length():
    from fusion_ui.stats import pdf

    values = 100.0 + 5.0 * np.random.default_rng(4).normal(size=N)
    raw = pdf.compute(
        _trace(values),
        pdf.PdfParams(bins=64, estimator="histogram", standardise=False,
                      log_y=True),
    )
    standard = pdf.compute(
        _trace(values),
        pdf.PdfParams(bins=64, estimator="histogram", standardise=True,
                      log_y=True),
    )
    assert float(raw["value"].values.min()) > 50.0
    assert float(standard["value"].values.max()) < 5.0
    few = pdf.compute(
        _trace(values),
        pdf.PdfParams(bins=32, estimator="histogram", standardise=True,
                      log_y=True),
    )
    assert len(standard["value"]) == 64 and len(few["value"]) == 32


def test_pdf_rejects_an_unknown_estimator():
    from fusion_ui.stats import pdf

    with pytest.raises(ValueError):
        pdf.compute(
            _trace(np.zeros(100)),
            pdf.PdfParams(bins=8, estimator="parzen", standardise=False,
                          log_y=False),
        )


def test_every_spec_renders_its_items():
    import plotly.graph_objects as go

    from fusion_ui.stats import acf, ccf, pdf, psd

    sine, shifted = _trace(_sine()), _trace(np.roll(_sine(), 5))
    cases = [
        ("pdf", pdf, (sine,), pdf.PdfParams()),
        ("psd", psd, (sine,), psd.PsdParams(fit=False)),
        ("acf", acf, (sine,), acf.AcfParams(fit=False)),
        ("ccf", ccf, (shifted, sine), ccf.CcfParams()),
    ]
    for key, module, args, params in cases:
        spec = statistics.get(key)
        if spec.pairwise:
            (trace, reference) = args
            result = spec.compute(trace, reference, params)
            items = [(trace, result)]
        else:
            (trace,) = args
            items = [(trace, spec.compute(trace, params))]
        figure = spec.render(items, params)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == len(items)
