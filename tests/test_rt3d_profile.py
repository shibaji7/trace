from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from trace.model.rt3d import RT3D, RT3DProfile


def _cfg3d():
    return SimpleNamespace(
        event="2017-05-27T16:00:00",
        iono_grid=SimpleNamespace(
            lat_start=40.0,
            lat_step=0.5,
            num_lats=3,
            lon_start=-75.0,
            lon_step=0.5,
            num_lons=4,
            height_start_km=100.0,
            height_step_km=20.0,
            num_heights=5,
        ),
        geomag_grid=SimpleNamespace(coord_input="GEO", coeff_dir=""),
        iri_param=SimpleNamespace(
            f107=150.0, foF2_coeff="CCIR", hmF2_model="SHU2015", coord="GEO"
        ),
    )


def test_rt3dprofile_validate_set_density_and_zero_floor():
    lats = np.array([40.0, 40.5, 41.0])
    lons = np.array([-75.0, -74.5, -74.0, -73.5])
    alts = np.array([0.0, 50.0, 100.0, 150.0])
    t = dt.datetime(2017, 5, 27, 16, 0, 0)
    p = RT3DProfile(lats=lats, lons=lons, alts_km=alts, time=t)

    ne = np.full((lats.size, lons.size, alts.size), 2.5e11, dtype=float)
    p.set_electron_density(ne_m3=ne, source="test")
    assert p.source == "test"
    assert p.ne_m3.shape == ne.shape
    assert np.allclose(p.ne_cm3, ne * 1e-6)

    n_rows = p.force_zero_density_below(100.0)
    assert n_rows == 2
    assert np.allclose(p.ne_m3[:, :, :2], 0.0)
    assert np.allclose(p.ne_m3[:, :, 2:], 2.5e11)


def test_rt3dprofile_from_cfg_with_mocked_fetches(monkeypatch):
    from trace.model import rt3d as rt3d_mod

    cfg = _cfg3d()
    t = dt.datetime(2017, 5, 27, 16, 0, 0)

    class _FakeIRI3d:
        def __init__(self, cfg, event_time):
            self.cfg = cfg
            self.event_time = event_time

        def fetch_dataset(self, time, lats, lons, alts, workers=1):
            ne_cm3 = np.full((lats.size, lons.size, alts.size), 2.0e5, dtype=float)
            return ne_cm3, alts

    class _FakeMSISE3D:
        def __init__(self, **kwargs):
            nlat = kwargs["lats"].size
            nlon = kwargs["lons"].size
            nh = kwargs["heights_km"].size
            shape = (nlat, nlon, nh)
            self.msise = {
                "N2": np.ones(shape) * 1e12,
                "O2": np.ones(shape) * 2e12,
                "O": np.ones(shape) * 3e12,
                "H": np.ones(shape) * 4e11,
                "He": np.ones(shape) * 5e10,
                "Tn": np.ones(shape) * 900.0,
                "t_nn": np.zeros(shape),
            }

    def _fake_geomag_grid(**kwargs):
        nlat = kwargs["lats"].size
        nlon = kwargs["lons"].size
        nh = kwargs["alts_km"].size
        shape = (nlat, nlon, nh)
        return SimpleNamespace(
            Bx=np.ones(shape) * 2.0e-5,
            By=np.ones(shape) * 1.0e-5,
            Bz=np.ones(shape) * -4.0e-5,
            bmag_t=np.ones(shape) * 5.0e-5,
            inc_deg=np.ones(shape) * 60.0,
            dec_deg=np.ones(shape) * 5.0,
            psi_deg=np.ones(shape) * 30.0,
            lat_geo=np.zeros(shape),
            lon_geo=np.zeros(shape),
            qd=None,
            apex=None,
        )

    monkeypatch.setattr(rt3d_mod, "IRI3d", _FakeIRI3d)
    monkeypatch.setattr(rt3d_mod, "NRLMSISE3D", _FakeMSISE3D)
    monkeypatch.setattr(rt3d_mod, "build_geomag_grid", _fake_geomag_grid)

    p = RT3DProfile.from_cfg(
        cfg=cfg,
        time=t,
        fetch_iri=True,
        fetch_msise=True,
        fetch_geomag=True,
        workers=2,
    )
    assert p.ne_m3 is not None
    assert p.msise is not None
    assert p.geomag is not None
    assert p.ne_m3.shape == (p.lats.size, p.lons.size, p.alts_km.size)
    assert np.allclose(p.ne_m3, 2.0e11)


def test_rt3d_interpolators_and_short_cartesian_trace():
    lats = np.array([40.0, 40.5, 41.0], dtype=float)
    lons = np.array([-75.0, -74.5, -74.0, -73.5], dtype=float)
    alts = np.linspace(0.0, 300.0, 31, dtype=float)
    ll, oo, zz = np.meshgrid(lats, lons, alts, indexing="ij")
    ne = 1.1e11 * np.exp(-(((zz - 170.0) / 80.0) ** 2)) * (
        1.0 + 0.08 * np.cos(np.deg2rad((ll - lats.mean()) * 10.0))
    )

    p = RT3DProfile(
        lats=lats,
        lons=lons,
        alts_km=alts,
        time=dt.datetime(2017, 5, 27, 16, 0, 0),
        ne_m3=ne,
    )
    m = RT3D(profile=p)

    interp = m.build_refractive_index_interpolators(
        freq_hz=5.0e6,
        mode="O",
        formulation="appleton-hartree",
    )
    assert interp.n.shape == ne.shape
    assert interp.mup.shape == ne.shape

    c = m.oblique_trace(
        freq_hz=5.0e6,
        elevation_deg=25.0,
        coordinate_system="cartesian",
        x0_km=0.0,
        y0_km=0.0,
        z0_km=0.0,
        s_max_km=120.0,
        max_step_km=2.0,
    )
    assert c.coordinate_system == "cartesian"
    assert c.x_km.size > 1
    assert c.z_km.size == c.x_km.size
    assert np.isfinite(c.group_path_km)


def test_rt3d_invalid_coordinate_system_and_model_name():
    lats = np.array([40.0, 40.5, 41.0])
    lons = np.array([-75.0, -74.5, -74.0])
    alts = np.array([0.0, 100.0, 200.0])
    ne = np.full((lats.size, lons.size, alts.size), 1e11, dtype=float)
    m = RT3D(
        profile=RT3DProfile(
            lats=lats,
            lons=lons,
            alts_km=alts,
            time=dt.datetime(2017, 5, 27, 16, 0, 0),
            ne_m3=ne,
        )
    )
    with pytest.raises(ValueError):
        m.oblique_trace(freq_hz=5e6, elevation_deg=20.0, coordinate_system="bad")
    with pytest.raises(ValueError):
        m._resolve_dispersion_model_name("invalid-model")


def test_rt3d_hamiltonian_cartesian_trace_with_collision():
    lats = np.array([40.0, 40.5, 41.0], dtype=float)
    lons = np.array([-75.0, -74.5, -74.0], dtype=float)
    alts = np.linspace(0.0, 250.0, 26, dtype=float)
    ll, oo, zz = np.meshgrid(lats, lons, alts, indexing="ij")
    ne = 1.0e11 * np.exp(-(((zz - 160.0) / 70.0) ** 2))
    nu = np.full_like(ne, 50.0, dtype=float)

    m = RT3D(
        profile=RT3DProfile(
            lats=lats,
            lons=lons,
            alts_km=alts,
            time=dt.datetime(2017, 5, 27, 16, 0, 0),
            ne_m3=ne,
        )
    )
    out = m.oblique_trace(
        freq_hz=5.0e6,
        elevation_deg=25.0,
        coordinate_system="cartesian",
        solver="hamiltonian",
        x0_km=0.0,
        y0_km=0.0,
        z0_km=0.0,
        collision_hz=nu,
        s_max_km=120.0,
    )
    assert out.coordinate_system == "cartesian"
    assert out.solver == "hamiltonian"
    assert out.x_km.size > 1
    assert np.isfinite(out.group_path_km)


def test_rt3d_interpolator_sanitizes_nan_density_outputs():
    lats = np.array([40.0, 40.5, 41.0], dtype=float)
    lons = np.array([-75.0, -74.5, -74.0], dtype=float)
    alts = np.linspace(0.0, 250.0, 26, dtype=float)
    ne = np.full((lats.size, lons.size, alts.size), 1.0e11, dtype=float)
    ne[1, 1, 10] = np.nan
    ne[0, 2, 7] = np.inf

    m = RT3D(
        profile=RT3DProfile(
            lats=lats,
            lons=lons,
            alts_km=alts,
            time=dt.datetime(2017, 5, 27, 16, 0, 0),
            ne_m3=ne,
        )
    )
    out = m.build_refractive_index_interpolators(
        freq_hz=5.0e6,
        mode="O",
        formulation="appleton-hartree",
    )
    assert np.all(np.isfinite(out.n))
    assert np.all(out.n > 0.0)
    assert np.all(np.isfinite(out.mup))


def test_rt3d_cart_eval_clips_out_of_domain_queries():
    lats = np.array([40.0, 40.5, 41.0], dtype=float)
    lons = np.array([-75.0, -74.5, -74.0], dtype=float)
    alts = np.linspace(0.0, 250.0, 26, dtype=float)
    ne = np.full((lats.size, lons.size, alts.size), 1.0e11, dtype=float)

    m = RT3D(
        profile=RT3DProfile(
            lats=lats,
            lons=lons,
            alts_km=alts,
            time=dt.datetime(2017, 5, 27, 16, 0, 0),
            ne_m3=ne,
        )
    )
    _ = m.build_refractive_index_interpolators(
        freq_hz=6.0e6,
        mode="O",
        formulation="appleton-hartree",
    )

    n, dnx, dny, dnz = m._eval_n_grad_cart(
        x_km=np.array([-1e6, 0.0, 1e6]),
        y_km=np.array([-1e6, 0.0, 1e6]),
        z_km=np.array([-1e6, 50.0, 1e6]),
    )
    assert np.all(np.isfinite(n))
    assert np.all(np.isfinite(dnx))
    assert np.all(np.isfinite(dny))
    assert np.all(np.isfinite(dnz))
