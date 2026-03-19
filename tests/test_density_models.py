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
    GEMINI2d = importlib.import_module("hfpytrace.density.gemini").GEMINI2d

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

    class _DF:
        alt = np.array([1])

        def __getitem__(self, _key):
            return self

    monkeypatch.setattr(g, "load_data", lambda _fname: _DF())
    monkeypatch.setattr(
        g,
        "_fetch_profile_from_df",
        lambda *args, **kwargs: np.array([1e5, 1.1e5, 1.2e5], dtype=float),
    )
    ne3d, alts = g.fetch_dataset_3d(
        dt.datetime(2024, 1, 1),
        lats=np.array([40.0, 40.1]),
        lons=np.array([-75.0, -74.9]),
        alts=np.array([100.0, 110.0, 120.0]),
        workers=2,
    )
    assert ne3d.shape == (2, 2, 3)
    assert alts.shape == (3,)


def test_gitm_fetch_dataset_from_store(monkeypatch):
    _stub_density_deps(monkeypatch)
    GITM2d = importlib.import_module("hfpytrace.density.gitm").GITM2d
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
    ne3d, _ = g.fetch_dataset_3d(
        dt.datetime(2024, 1, 1),
        np.array([40.0, 41.0]),
        np.array([-75.0, -74.0]),
        np.array([100.0, 110.0, 120.0]),
        workers=2,
    )
    assert ne3d.shape == (2, 2, 3)


def _make_sami3(monkeypatch, eden=None):
    """Return a SAMI3 instance pre-populated with a minimal 2-time store.

    Grid: lat [40, 41], lon [285, 286], alt [100, 110, 120].
    eden shape: (n_time=2, n_lon=2, n_alt=3, n_lat=2).
    If *eden* is None all cells default to 1e5.
    """
    _stub_density_deps(monkeypatch)
    SAMI3 = importlib.import_module("hfpytrace.density.sami").SAMI3
    s = SAMI3.__new__(SAMI3)
    s.cfg = _cfg()
    if eden is None:
        eden = np.ones((2, 2, 3, 2)) * 1e5
    s.store = {
        "time": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1, 1)],
        "alt": np.array([100.0, 110.0, 120.0]),
        "glat": np.array([40.0, 41.0]),
        "glon": np.array([285.0, 286.0]),
        "eden": eden,
    }
    return s


def test_sami_find_time_index_and_fetch_interpolated_data(monkeypatch):
    s = _make_sami3(monkeypatch)
    i, j = s.find_time_index(dt.datetime(2024, 1, 1, 0, 30))
    assert (i, j) == (0, 1)
    out, _ = s.fetch_interpolated_data(
        [40.0], [-75.0], np.array([100.0, 110.0, 120.0]), 0
    )
    assert out.shape == (3, 1)
    ne3d, _ = s.fetch_dataset_3d(
        dt.datetime(2024, 1, 1),
        np.array([40.0, 41.0]),
        np.array([-75.0, -74.0]),
        np.array([100.0, 110.0, 120.0]),
        workers=2,
    )
    assert ne3d.shape == (2, 2, 3)


def test_sami_bilinear_interpolation(monkeypatch):
    """_bilinear_ne_profile blends the four surrounding cells correctly."""
    # Build eden with distinct corner values so we can verify the blend.
    # D shape per time: (n_lon=2, n_alt=3, n_lat=2)
    #   D[j=0, :, i=0] = 1.0  (lon=285, lat=40)
    #   D[j=1, :, i=0] = 2.0  (lon=286, lat=40)
    #   D[j=0, :, i=1] = 3.0  (lon=285, lat=41)
    #   D[j=1, :, i=1] = 4.0  (lon=286, lat=41)
    eden = np.zeros((2, 2, 3, 2))
    eden[:, 0, :, 0] = 1.0  # lon0, lat0
    eden[:, 1, :, 0] = 2.0  # lon1, lat0
    eden[:, 0, :, 1] = 3.0  # lon0, lat1
    eden[:, 1, :, 1] = 4.0  # lon1, lat1
    s = _make_sami3(monkeypatch, eden=eden)
    D = s.store["eden"][0]  # (n_lon=2, n_alt=3, n_lat=2)

    # At the exact grid corner (lat=40, lon=285) → should return p00 = 1.0
    p = s._bilinear_ne_profile(D, lat=40.0, lon=285.0)
    assert np.allclose(p, 1.0), f"corner mismatch: {p}"

    # At the centre (lat=40.5, lon=285.5) → average of all four = 2.5
    p = s._bilinear_ne_profile(D, lat=40.5, lon=285.5)
    assert np.allclose(p, 2.5), f"centre mismatch: {p}"

    # At mid-lat, left edge (lat=40.5, lon=285.0) → mean(p00, p01) = 2.0
    p = s._bilinear_ne_profile(D, lat=40.5, lon=285.0)
    assert np.allclose(p, 2.0), f"mid-lat edge mismatch: {p}"

    # Verify fetch_interpolated_data uses bilinear (lon=-75 → 285, lat=40.5)
    out, _ = s.fetch_interpolated_data(
        [40.5], [-75.0], np.array([100.0, 110.0, 120.0]), 0
    )
    assert out.shape == (3, 1)
    assert np.allclose(
        out[:, 0], 2.0
    ), f"fetch_interpolated_data bilinear mismatch: {out[:, 0]}"


def test_sami_time_interpolation_weights(monkeypatch):
    """fetch_dataset interpolates linearly between bracketing time frames."""
    # t=0: all Ne = 1.0;  t=1h: all Ne = 3.0
    # Query at t=15 min → alpha = 0.25 → expected = 0.75*1 + 0.25*3 = 1.5
    eden = np.ones((2, 2, 3, 2))
    eden[0] *= 1.0
    eden[1] *= 3.0
    s = _make_sami3(monkeypatch, eden=eden)

    t_query = dt.datetime(2024, 1, 1, 0, 15)  # 15 min into the bracket
    ne, _ = s.fetch_dataset(
        t_query,
        lats=np.array([40.0]),
        lons=np.array([-75.0]),
        alts=np.array([100.0, 110.0, 120.0]),
    )
    assert ne.shape == (3, 1)
    assert np.allclose(ne, 1.5, atol=1e-9), f"Expected 1.5 at t+15min, got {ne.ravel()}"

    # Query at t=45 min → alpha = 0.75 → expected = 0.25*1 + 0.75*3 = 2.5
    t_query2 = dt.datetime(2024, 1, 1, 0, 45)
    ne2, _ = s.fetch_dataset(
        t_query2,
        lats=np.array([40.0]),
        lons=np.array([-75.0]),
        alts=np.array([100.0, 110.0, 120.0]),
    )
    assert np.allclose(
        ne2, 2.5, atol=1e-9
    ), f"Expected 2.5 at t+45min, got {ne2.ravel()}"


def test_waccm_transform_and_fetch_interpolated_data(monkeypatch):
    _stub_density_deps(monkeypatch)
    WACCMX2d = importlib.import_module("hfpytrace.density.waccm").WACCMX2d
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
    w.store["time"] = [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1, 1)]
    ne3d, _ = w.fetch_dataset_3d(
        dt.datetime(2024, 1, 1),
        np.array([40.0, 40.5]),
        np.array([-75.0, -74.5]),
        np.array([100.0, 110.0, 120.0]),
        workers=2,
    )
    assert ne3d.shape == (2, 2, 3)


def test_wamipe_load_data(monkeypatch):
    _stub_density_deps(monkeypatch)
    WAMIPE2d = importlib.import_module("hfpytrace.density.wamipe").WAMIPE2d
    w = WAMIPE2d.__new__(WAMIPE2d)
    w.cfg = _cfg()
    vals = {"a": np.ones((2, 2, 2)), "b": 2 * np.ones((2, 2, 2))}
    monkeypatch.setattr(w, "_read_params_from_hdf_file_", lambda f, ds, p: vals[p])
    out = w.load_data("dummy")
    assert out.shape == (2, 2, 2)
    assert np.allclose(out, 3.0)
