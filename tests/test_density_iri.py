import datetime as dt
import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np


def _install_fake_pyiri(monkeypatch, fn):
    fake_pyiri = types.ModuleType("PyIRI")
    fake_sh = types.ModuleType("PyIRI.sh_library")
    fake_sh.IRI_density_1day = fn
    fake_pyiri.sh_library = fake_sh
    monkeypatch.setitem(sys.modules, "PyIRI", fake_pyiri)
    monkeypatch.setitem(sys.modules, "PyIRI.sh_library", fake_sh)


def test_iri_fetch_and_load(monkeypatch, tmp_path):
    def _iri_density_1day(*args, **kwargs):
        alts = np.asarray(args[6], dtype=float)
        den = np.linspace(1e11, 3e11, alts.size, dtype=float)
        return None, None, None, None, None, den

    _install_fake_pyiri(monkeypatch, _iri_density_1day)

    m = importlib.import_module("trace.density.iri")
    importlib.reload(m)

    cfg = SimpleNamespace(iri_param=SimpleNamespace(f107=120.0))
    obj = m.IRI2d(cfg, dt.datetime(2024, 1, 1))
    ne, _ = obj.fetch_dataset(
        dt.datetime(2024, 1, 1),
        lats=np.array([40.0, 41.0]),
        lons=np.array([-75.0, -74.0]),
        alts=np.array([100.0, 110.0, 120.0]),
        workers=2,
    )
    assert ne.shape == (3, 2)

    matfile = tmp_path / "ne.mat"
    import scipy.io as sio

    sio.savemat(matfile, {"ne": np.ones((3, 2))})
    loaded = obj.load_from_file(str(matfile))
    assert loaded.shape == (3, 2)


def test_iri_fetch_large_altitudes(monkeypatch):
    calls = {"n": 0}

    def _iri_density_1day(*args, **kwargs):
        calls["n"] += 1
        alts = np.asarray(args[6], dtype=float)
        den = np.full(alts.size, 1e11, dtype=float)
        return None, None, None, None, None, den

    _install_fake_pyiri(monkeypatch, _iri_density_1day)

    m = importlib.import_module("trace.density.iri")
    importlib.reload(m)
    cfg = SimpleNamespace(iri_param=SimpleNamespace(f107=150.0))
    obj = m.IRI2d(cfg, dt.datetime(2024, 1, 1))
    alts = np.arange(100.0, 1301.0, 1.0)
    ne, _ = obj.fetch_dataset(
        dt.datetime(2024, 1, 1),
        lats=np.array([40.0]),
        lons=np.array([-75.0]),
        alts=alts,
        workers=1,
    )
    assert ne.shape == (alts.size, 1)
    assert calls["n"] >= 1


def test_iri3d_fetch_shape(monkeypatch):
    calls = {"n": 0}

    def _iri_density_1day(*args, **kwargs):
        calls["n"] += 1
        lons = np.asarray(args[4], dtype=float).ravel()
        alts = np.asarray(args[6], dtype=float).ravel()
        den = np.ones((lons.size, alts.size), dtype=float) * 1e11
        return None, None, None, None, None, den

    _install_fake_pyiri(monkeypatch, _iri_density_1day)
    m = importlib.import_module("trace.density.iri")
    importlib.reload(m)

    cfg = SimpleNamespace(iri_param=SimpleNamespace(f107=140.0))
    obj = m.IRI3d(cfg, dt.datetime(2024, 1, 1))
    lats = np.array([30.0, 31.0, 32.0], dtype=float)
    lons = np.array([-90.0, -89.5], dtype=float)
    alts = np.array([100.0, 120.0, 140.0, 160.0], dtype=float)
    ne, out_alts = obj.fetch_dataset(
        dt.datetime(2024, 1, 1),
        lats=lats,
        lons=lons,
        alts=alts,
        workers=8,
    )
    assert ne.shape == (lats.size, lons.size, alts.size)
    assert np.allclose(out_alts, alts)
    assert calls["n"] == 1


def test_iri3d_workers_ignored_warning(monkeypatch):
    def _iri_density_1day(*args, **kwargs):
        lons = np.asarray(args[4], dtype=float).ravel()
        alts = np.asarray(args[6], dtype=float).ravel()
        den = np.ones((lons.size, alts.size), dtype=float) * 1e11
        return None, None, None, None, None, den

    _install_fake_pyiri(monkeypatch, _iri_density_1day)
    m = importlib.import_module("trace.density.iri")
    importlib.reload(m)

    cfg = SimpleNamespace(iri_param=SimpleNamespace())
    obj = m.IRI3d(cfg, dt.datetime(2024, 1, 1))
    _ = obj.fetch_dataset(
        dt.datetime(2024, 1, 1),
        lats=np.array([30.0], dtype=float),
        lons=np.array([-90.0], dtype=float),
        alts=np.array([100.0, 120.0], dtype=float),
        workers=4,
    )
