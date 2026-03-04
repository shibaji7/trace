import datetime as dt
import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np


def _cfg():
    return SimpleNamespace(
        density_file_location="/tmp/",
        density_file_name="x.nc",
        grid_coordinate_file="grid.h5",
        scale="linear",
        kind="linear",
        density_simulated_datetime="2024-01-01T00:00:00",
        wam_paramteres=SimpleNamespace(
            coordinates=SimpleNamespace(nmp=2, nlp=2, iDIM=2, napex=3, grid_name="g"),
            dataset_name="d",
            density_params=["a", "b"],
        ),
    )


def _stub_density_deps(monkeypatch):
    # Avoid compiled dependency imports in this environment.
    fake_h5py = types.ModuleType("h5py")
    fake_h5py.File = object
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)

    fake_xr = types.ModuleType("xarray")
    fake_xr.open_dataset = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "xarray", fake_xr)

    # pandas import in gemini/sami modules is not needed for these unit tests.
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))


def test_gemini_fetch_dataset(monkeypatch):
    _stub_density_deps(monkeypatch)
    GEMINI2d = importlib.import_module("trace.density.gemini").GEMINI2d

    cfg = _cfg()
    monkeypatch.setattr(
        GEMINI2d, "load_grid", lambda self: setattr(self, "ccord_file", "/tmp/g.mat")
    )
    monkeypatch.setattr(
        GEMINI2d,
        "search_mat_files",
        lambda self: (
            setattr(self, "files", ["/tmp/20240101_0.mat"]),
            setattr(self, "dates", [dt.datetime(2024, 1, 1)]),
        ),
    )

    g = GEMINI2d(cfg, dt.datetime(2024, 1, 1))
    assert len(g.files) == 1
    assert len(g.dates) == 1


def test_gitm_fetch_dataset_from_store(monkeypatch):
    _stub_density_deps(monkeypatch)
    GITM2d = importlib.import_module("trace.density.gitm").GITM2d
    g = GITM2d.__new__(GITM2d)
    g.cfg = _cfg()
    g.store = {
        "time": [dt.datetime(2024, 1, 1)],
        "glat": np.array([40.0, 41.0]),
        "glon": np.array([285.0, 286.0]),
        "alt": np.array([100.0, 110.0, 120.0]),
        "eden": np.ones((1, 3, 2, 2)) * 1e11,
    }
    ne, _ = g.fetch_dataset(
        dt.datetime(2024, 1, 1), [40.0], [-75.0], np.array([100.0, 110.0, 120.0])
    )
    assert ne.shape == (3, 1)


def test_sami_find_time_index_and_fetch_interpolated_data(monkeypatch):
    _stub_density_deps(monkeypatch)
    SAMI3 = importlib.import_module("trace.density.sami").SAMI3
    s = SAMI3.__new__(SAMI3)
    s.cfg = _cfg()
    s.store = {
        "time": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1, 1)],
        "alt": np.array([100.0, 110.0, 120.0]),
        "glat": np.array([40.0, 41.0]),
        "glon": np.array([285.0, 286.0]),
        "eden": np.ones((2, 2, 3, 2)) * 1e5,
    }
    i, j = s.find_time_index(dt.datetime(2024, 1, 1, 0, 30))
    assert (i, j) == (0, 1)
    out, _ = s.fetch_interpolated_data(
        [40.0], [-75.0], np.array([100.0, 110.0, 120.0]), 0
    )
    assert out.shape == (3, 1)


def test_waccm_transform_and_fetch_interpolated_data(monkeypatch):
    _stub_density_deps(monkeypatch)
    WACCMX2d = importlib.import_module("trace.density.waccm").WACCMX2d
    w = WACCMX2d.__new__(WACCMX2d)
    w.cfg = _cfg()
    P = np.array([1.0, 2.0])
    T = np.ones((1, 2, 1, 1)) * 1000.0
    V = np.ones((1, 2, 1, 1)) * 1e-6
    den = w.transform_density(P, T, V)
    assert den.shape == V.shape

    w.store = {
        "glat": np.array([40.0]),
        "glon": np.array([285.0]),
        "alt": np.ones((1, 3, 1, 1)) * np.array([[[[100.0]], [[110.0]], [[120.0]]]]),
        "eden": np.ones((1, 3, 1, 1)) * 1e11,
    }
    out, _ = w.fetch_interpolated_data(
        [40.0], [-75.0], np.array([100.0, 110.0, 120.0]), 0
    )
    assert out.shape == (3, 1)


def test_wamipe_load_data(monkeypatch):
    _stub_density_deps(monkeypatch)
    WAMIPE2d = importlib.import_module("trace.density.wamipe").WAMIPE2d
    w = WAMIPE2d.__new__(WAMIPE2d)
    w.cfg = _cfg()
    vals = {"a": np.ones((2, 2, 2)), "b": 2 * np.ones((2, 2, 2))}
    monkeypatch.setattr(w, "_read_params_from_hdf_file_", lambda f, ds, p: vals[p])
    out = w.load_data("dummy")
    assert out.shape == (2, 2, 2)
    assert np.allclose(out, 3.0)
