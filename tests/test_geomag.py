import datetime as dt
import importlib
import sys
import types

import numpy as np


def _install_fake_pyiri(monkeypatch):
    fake_pyiri = types.ModuleType("PyIRI")
    fake_main = types.ModuleType("PyIRI.main_library")
    fake_igrf = types.ModuleType("PyIRI.igrf_library")
    fake_sh = types.ModuleType("PyIRI.sh_library")

    fake_main.decimal_year = lambda ts: 2024.5

    def _inclination(_coeff, _dec_year, lon, lat, _alt_km, only_inc=False):
        lon = np.asarray(lon, dtype=float).ravel()
        n = lon.size
        inc = np.full(n, 60.0, dtype=float)
        dec = np.full(n, 10.0, dtype=float)
        mag = np.full(n, 50000.0, dtype=float)  # nT
        return inc, dec, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), mag

    fake_igrf.inclination = _inclination
    fake_sh.Apex_geo_qd = lambda lat, lon, _ts, _mode: (
        np.asarray(lat, dtype=float),
        np.asarray(lon, dtype=float),
    )
    fake_sh.Apex = lambda lat, lon, _ts: (
        np.asarray(lat, dtype=float) + 1.0,
        np.asarray(lon, dtype=float) + 1.0,
    )

    fake_pyiri.main_library = fake_main
    fake_pyiri.igrf_library = fake_igrf
    fake_pyiri.sh_library = fake_sh
    fake_pyiri.coeff_dir = "/tmp/fake_coeff"

    monkeypatch.setitem(sys.modules, "PyIRI", fake_pyiri)
    monkeypatch.setitem(sys.modules, "PyIRI.main_library", fake_main)
    monkeypatch.setitem(sys.modules, "PyIRI.igrf_library", fake_igrf)
    monkeypatch.setitem(sys.modules, "PyIRI.sh_library", fake_sh)


def test_build_geomag_grid_geo(monkeypatch):
    _install_fake_pyiri(monkeypatch)
    m = importlib.import_module("trace.geomag")
    importlib.reload(m)

    out = m.build_geomag_grid(
        lats=np.array([30.0, 31.0]),
        lons=np.array([-90.0, -89.0, -88.0]),
        alts_km=np.array([100.0, 120.0]),
        time=dt.datetime(2024, 1, 1),
        coord_input="GEO",
    )
    assert out.Bx.shape == (2, 3, 2)
    assert out.By.shape == (2, 3, 2)
    assert out.Bz.shape == (2, 3, 2)
    assert np.all(out.bmag_t > 0.0)
    assert out.qd is not None
    assert out.apex is not None


def test_build_geomag_grid_qd_input(monkeypatch):
    _install_fake_pyiri(monkeypatch)
    m = importlib.import_module("trace.geomag")
    importlib.reload(m)

    out = m.build_geomag_grid(
        lats=np.array([10.0]),
        lons=np.array([20.0]),
        alts_km=np.array([300.0]),
        time=dt.datetime(2024, 1, 1),
        coord_input="QD",
    )
    assert out.Bx.shape == (1, 1, 1)
    assert np.isfinite(out.Bx).all()
