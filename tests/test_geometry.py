"""Magnetic-coordinate labels: the separatrix distance and the probe rho."""

import numpy as np
import xarray as xr

from fusion_ui.core import geometry, loader


def _boundary_dataset(r_values, z_values=None, efit_times=(1.0, 1.01, 1.02)):
    """A dataset with a known analytic boundary: R constant or R(z) linear."""
    n = len(r_values)
    z_values = np.linspace(-8.0, 1.0, n) if z_values is None else z_values
    efit_times = np.asarray(list(efit_times), dtype=float)
    n_efit = len(efit_times)
    return xr.Dataset(
        {
            "rlcfs": (["xlcfs", "efit_time"], np.tile(r_values, (n_efit, 1)).T),
            "zlcfs": (["ylcfs", "efit_time"], np.tile(z_values, (n_efit, 1)).T),
            "R": (["y", "x"], np.array([[80.0, 90.0]])),
            "Z": (["y", "x"], np.array([[-4.0, -4.0]])),
        },
        coords={"efit_time": ("efit_time", efit_times)},
    )


def test_separatrix_radius_recovers_a_flat_boundary():
    ds = _boundary_dataset(np.full(5, 87.0))
    assert abs(geometry.separatrix_radius(ds, 1.005, -4.0) - 87.0) < 1e-6


def test_separatrix_radius_recovers_a_sloped_boundary():
    # R = 90 + 0.1 z, all of it above the r >= 86 outboard mask: the cubic
    # spline through collinear points is the line.
    z_values = np.linspace(-8.0, 1.0, 9)
    ds = _boundary_dataset(90.0 + 0.1 * z_values, z_values)
    assert abs(geometry.separatrix_radius(ds, 1.02, -3.0) - 89.7) < 0.05


def test_separatrix_radius_picks_the_nearest_efit_slice():
    before = _boundary_dataset(np.full(5, 87.0), efit_times=(1.0, 1.01, 1.02))
    assert geometry.separatrix_radius(before, 1.019, -4.0) is not None


def test_separatrix_radius_is_none_without_a_boundary(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path)).drop_vars(
        ["rlcfs", "zlcfs", "efit_time"]
    )
    assert geometry.separatrix_radius(ds, 1.01, -2.0) is None


def test_separatrix_radius_is_none_outside_the_spline_range(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    assert geometry.separatrix_radius(ds, 1.01, 5.0) is None


def test_pixel_dr_sep_sign_convention(apd_dataset_path):
    """Outboard of the R = 87 cm boundary is positive, inboard negative."""
    ds = loader.open_dataset(str(apd_dataset_path))
    # R is linspace(80, 90, 5): x = 4 sits at 90 cm, x = 0 at 80 cm.
    assert geometry.pixel_dr_sep(ds, 4, 0, 1.01) > 0
    assert geometry.pixel_dr_sep(ds, 0, 0, 1.01) < 0


def test_pixel_dr_sep_is_none_without_a_boundary(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path)).drop_vars(
        ["rlcfs", "zlcfs", "efit_time"]
    )
    assert geometry.pixel_dr_sep(ds, 4, 0, 1.01) is None


def test_probe_rho_is_the_window_mean(asp_dataset_path):
    ds = loader.open_dataset(str(asp_dataset_path))
    rho = geometry.probe_rho(ds, "ne", 0, 0.5, 0.6)
    assert rho is not None
    assert -0.5 <= rho <= 1.0
    # A window missing the rho record entirely gives up rather than guessing.
    assert geometry.probe_rho(ds, "ne", 0, 5.0, 6.0) is None
    assert geometry.probe_rho(ds, "nope", 0, 0.5, 0.6) is None
