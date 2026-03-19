from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from hfpytrace.model.rt2d import RT2D, RT2DProfile
from hfpytrace.utils import load_config_2D


def test_rt2dprofile_manual_validate_and_set_density():
    alt = np.array([60.0, 80.0, 100.0])
    lats = np.array([40.0, 40.5, 41.0, 41.5])
    lons = np.array([-75.0, -74.5, -74.0, -73.5])
    t = dt.datetime(2017, 5, 27, 16, 0, 0)

    p = RT2DProfile(alt_km=alt, lats=lats, lons=lons, time=t)
    assert p.x_km.shape == lats.shape
    assert p.ne_m3 is None

    ne = np.full((alt.size, lats.size), 1.0e11)
    p.set_electron_density(ne_m3=ne, source="test")
    assert p.source == "test"
    assert p.ne_m3.shape == (alt.size, lats.size)
    assert np.allclose(p.ne_cm3, ne * 1e-6)


def test_rt2dprofile_force_zero_density_below():
    alt = np.array([0.0, 50.0, 100.0, 150.0])
    lats = np.array([40.0, 40.5])
    lons = np.array([-75.0, -74.5])
    t = dt.datetime(2017, 5, 27, 16, 0, 0)
    ne = np.full((alt.size, lats.size), 2.5e11)

    p = RT2DProfile(alt_km=alt, lats=lats, lons=lons, time=t, ne_m3=ne)
    n_rows = p.force_zero_density_below(100.0)
    assert n_rows == 2
    assert np.allclose(p.ne_m3[:2, :], 0.0)
    assert np.allclose(p.ne_cm3[:2, :], 0.0)
    assert np.allclose(p.ne_m3[2:, :], 2.5e11)


def test_rt2dprofile_from_cfg_route_and_iri(monkeypatch):
    cfg = load_config_2D(None)
    t = dt.datetime(2017, 5, 27, 16, 0, 0)

    def _fake_fetch_dataset(self, time, lats, lons, alts, workers=1, to_file=None):
        ne_cm3 = np.full((alts.size, lats.size), 2.0e5, dtype=float)
        return ne_cm3, alts

    from hfpytrace.model import rt2d as rt2d_mod

    monkeypatch.setattr(rt2d_mod.IRI2d, "fetch_dataset", _fake_fetch_dataset)
    monkeypatch.setattr(
        rt2d_mod,
        "build_route_from_cfg",
        lambda cfg, n_range: (
            np.linspace(40.0, 41.0, int(n_range)),
            np.linspace(-75.0, -73.0, int(n_range)),
            90.0,
            250.0,
        ),
    )

    p = RT2DProfile.from_cfg(
        cfg=cfg,
        time=t,
        fetch_iri=True,
        fetch_msise=False,
        fetch_geomag=False,
        workers=1,
    )
    assert p.ne_m3 is not None
    assert p.ne_m3.shape == (p.alt_km.size, p.lats.size)
    assert np.allclose(p.ne_m3, 2.0e11)
    assert np.all(np.diff(p.x_km) > 0)


def test_rt2d_init_profile_and_legacy_paths():
    alt = np.array([60.0, 80.0, 100.0])
    lats = np.array([40.0, 40.5, 41.0])
    lons = np.array([-75.0, -74.5, -74.0])
    t = dt.datetime(2017, 5, 27, 16, 0, 0)
    ne = np.full((alt.size, lats.size), 5.0e10)

    p = RT2DProfile(alt_km=alt, lats=lats, lons=lons, time=t, ne_m3=ne)
    m1 = RT2D(profile=p)
    assert m1.ne_m3.shape == ne.shape
    assert np.allclose(m1.x_km, p.x_km)
    assert np.allclose(m1.z_km, alt)

    x = np.array([0.0, 20.0, 40.0])
    m2 = RT2D(x_km=x, z_km=alt, ne_m3=ne)
    assert np.allclose(m2.x_km, x)
    assert np.allclose(m2.z_km, alt)
    assert m2.ne_m3.shape == ne.shape


def test_rt2d_oblique_trace_cartesian_and_spherical():
    alt = np.linspace(60.0, 260.0, 81)
    x = np.linspace(0.0, 800.0, 101)
    zz, xx = np.meshgrid(alt, x, indexing="ij")
    ne = (
        1.2e11
        * np.exp(-(((zz - 170.0) / 70.0) ** 2))
        * (1.0 + 0.15 * np.cos(2.0 * np.pi * xx / x.max()))
    )
    model = RT2D(x_km=x, z_km=alt, ne_m3=ne)

    c = model.oblique_trace(
        freq_hz=5.0e6,
        elevation_deg=25.0,
        coordinate_system="cartesian",
        x0_km=0.0,
        z0_km=alt[0],
        s_max_km=1200.0,
    )
    assert c.coordinate_system == "cartesian"
    assert c.x_km.size > 1
    assert c.z_km.size == c.x_km.size
    assert np.isfinite(c.group_path_km)

    s = model.oblique_trace(
        freq_hz=5.0e6,
        elevation_deg=25.0,
        coordinate_system="spherical",
        x0_km=0.0,
        z0_km=alt[0],
        s_max_km=1200.0,
    )
    assert s.coordinate_system == "spherical"
    assert s.x_km.size > 1
    assert s.z_km.size == s.x_km.size
    assert np.isfinite(s.group_path_km)


def test_rt2d_oblique_trace_invalid_coordinate_system():
    alt = np.array([60.0, 120.0, 180.0])
    x = np.array([0.0, 100.0, 200.0])
    ne = np.full((alt.size, x.size), 1.0e11)
    model = RT2D(x_km=x, z_km=alt, ne_m3=ne)
    try:
        model.oblique_trace(
            freq_hz=4.0e6,
            elevation_deg=20.0,
            coordinate_system="bad",
        )
    except ValueError as exc:
        assert "coordinate_system" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid coordinate_system")


def _make_2d_profile_with_msise():
    """Return a small RT2DProfile with fake MSIS and Ne attached."""
    import datetime as dt
    from types import SimpleNamespace

    alt = np.array([100.0, 150.0, 200.0, 250.0])
    lats = np.array([40.0, 40.5, 41.0])
    lons = np.array([-75.0, -74.5, -74.0])
    t = dt.datetime(2017, 5, 27, 16, 0, 0)
    nz, nx = alt.size, lats.size
    ne = np.full((nz, nx), 1.5e11, dtype=float)

    p = RT2DProfile(alt_km=alt, lats=lats, lons=lons, time=t, ne_m3=ne)
    n_fake = np.full((nz, nx), 1e8, dtype=float)
    p.msise = SimpleNamespace(
        N2=0.78 * n_fake,
        O2=0.21 * n_fake,
        O=0.01 * n_fake,
        H=np.full((nz, nx), 1e4),
        He=np.full((nz, nx), 5e5),
        Tn=np.full((nz, nx), 800.0),
        t_nn=n_fake,
    )
    return p


def test_rt2dprofile_compute_collision_defaults():
    p = _make_2d_profile_with_msise()
    cc = p.compute_collision()
    assert p.collision is cc
    assert cc.collision.nu_ft.shape == (p.alt_km.size, p.lats.size)
    assert cc.collision.nu_sn.total.shape == (p.alt_km.size, p.lats.size)
    assert np.all(np.isfinite(cc.collision.nu_ft))


def test_rt2dprofile_compute_collision_requires_msise():
    alt = np.array([100.0, 150.0, 200.0])
    lats = np.array([40.0, 41.0])
    lons = np.array([-75.0, -74.0])
    p = RT2DProfile(
        alt_km=alt,
        lats=lats,
        lons=lons,
        time=dt.datetime(2017, 5, 27),
        ne_m3=np.ones((3, 2)) * 1e11,
    )
    with pytest.raises(ValueError, match="MSIS"):
        p.compute_collision()


def test_rt2d_fetch_collision_and_collision_type():
    p = _make_2d_profile_with_msise()
    rt = RT2D(
        x_km=np.linspace(0, 100, p.lats.size),
        z_km=p.alt_km,
        ne_m3=p.ne_m3,
    )
    rt.profile = p
    rt.fetch_collision()
    assert p.collision is not None

    # Verify _extract_collision_hz for each type
    for key in ("FT", "FT_CC", "FT_MB", "SN_EN", "SN_EI", "SN", "ATM"):
        nu = RT2D._extract_collision_hz(p.collision, key)
        assert nu.shape == p.ne_m3.shape, f"Shape mismatch for {key}"
        assert np.all(np.isfinite(nu)), f"Non-finite values for {key}"

    # Mutex guard
    with pytest.raises(ValueError, match="at most one"):
        rt.build_refractive_index_interpolators(
            freq_hz=10.5e6,
            collision_hz=np.ones_like(p.ne_m3),
            collision_type="FT",
        )

    # Unknown type guard
    with pytest.raises(ValueError, match="Unknown collision_type"):
        RT2D._extract_collision_hz(p.collision, "UNKNOWN")
