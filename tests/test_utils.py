import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfpytrace import utils


def _cfg_with_end():
    return SimpleNamespace(
        route=SimpleNamespace(
            start=SimpleNamespace(lat=40.0, lon=-75.0),
            end=SimpleNamespace(lat=41.0, lon=-74.0),
            bearing=45.0,
        ),
        max_ground_range_km=500.0,
        start_height_km=100,
        end_height_km=110,
        height_incriment_km=2,
        start_elevation=10,
        end_elevation=12,
        elevation_inctiment=1,
        frequency=10.5,
    )


def _cfg_with_bearing_only():
    return SimpleNamespace(
        route=SimpleNamespace(start=SimpleNamespace(lat=40.0, lon=-75.0), bearing=70.0),
        max_ground_range_km=300.0,
        start_height_km=100,
        end_height_km=104,
        height_incriment_km=1,
        start_elevation=5,
        end_elevation=7,
        elevation_inctiment=1,
        frequency=9.0,
    )


def test_to_namespace_recursive():
    o = utils.to_namespace({"a": 1, "b": [{"c": 2}]})
    assert o.a == 1
    assert o.b[0].c == 2


def test_bearing_distance_helpers():
    b360, braw = utils.calculate_bearing(0, 0, 0, 1)
    assert 0 <= b360 <= 360
    d = utils.great_circle_distance(0, 0, 0, 1)
    assert d > 100


def test_build_route_from_cfg_with_end():
    fake_geopy = types.ModuleType("geopy")
    fake_dist = types.ModuleType("geopy.distance")

    class _GC:
        def __init__(self, kilometers=0.0):
            self.km = kilometers

        def destination(self, origin, bearing=0.0):
            lat, lon = origin
            ddeg = self.km / 111.0
            return SimpleNamespace(latitude=lat + ddeg, longitude=lon + ddeg)

    fake_dist.great_circle = _GC
    sys.modules["geopy"] = fake_geopy
    sys.modules["geopy.distance"] = fake_dist

    cfg = _cfg_with_end()
    lats, lons, b, total = utils.build_route_from_cfg(cfg, 5)
    assert lats.shape == (5,)
    assert lons.shape == (5,)
    assert isinstance(b, float)
    assert total > 0


def test_build_route_from_cfg_with_bearing_only():
    fake_geopy = types.ModuleType("geopy")
    fake_dist = types.ModuleType("geopy.distance")

    class _GC:
        def __init__(self, kilometers=0.0):
            self.km = kilometers

        def destination(self, origin, bearing=0.0):
            lat, lon = origin
            ddeg = self.km / 111.0
            return SimpleNamespace(latitude=lat + ddeg, longitude=lon + ddeg)

    fake_dist.great_circle = _GC
    sys.modules["geopy"] = fake_geopy
    sys.modules["geopy.distance"] = fake_dist

    cfg = _cfg_with_bearing_only()
    lats, lons, b, total = utils.build_route_from_cfg(cfg, 4)
    assert lats.shape == (4,)
    assert lons.shape == (4,)
    assert b == 70.0
    assert total == 300.0


def test_build_height_elev_freq_from_cfg():
    cfg = _cfg_with_end()
    h = utils.build_heights_from_cfg(cfg)
    e = utils.build_elevations_from_cfg(cfg)
    f = utils.build_freqs_from_cfg(cfg, e)
    assert np.allclose(h, np.array([100, 102, 104, 106, 108]))
    assert np.allclose(e, np.array([10, 11, 12]))
    assert np.allclose(f, np.array([10.5, 10.5, 10.5]))


def test_get_default_config_name():
    assert utils.get_default_config_name("2d") == "config2D.json"
    assert utils.get_default_config_name(2) == "config2D.json"
    assert utils.get_default_config_name("1d") == "config1D.json"
    assert utils.get_default_config_name("3d") == "config3D.json"
    with pytest.raises(ValueError):
        utils.get_default_config_name("4d")


def test_load_config_2d_default(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config2D.json"
    cfg_file.write_text('{"event":"2017-05-27T16:00:00Z","frequency":10.5}')

    monkeypatch.setattr(utils, "resolve_config_path", lambda *args, **kwargs: cfg_file)
    cfg = utils.load_config_2D()
    assert cfg.event == "2017-05-27T16:00:00Z"
    assert float(cfg.frequency) == 10.5
    cfg2 = utils.read_params()
    assert cfg2.event == "2017-05-27T16:00:00Z"


def test_resolve_config_path_user_and_package_fallback(monkeypatch, tmp_path):
    cfg = tmp_path / "my.json"
    cfg.write_text("{}")
    got = utils.resolve_config_path(cfg, "config2D.json")
    assert got == cfg.resolve()

    pkg_cfg = tmp_path / "config2D.json"
    pkg_cfg.write_text("{}")
    monkeypatch.setattr(utils, "get_installed_config_path", lambda name: pkg_cfg)
    got2 = utils.resolve_config_path(Path("missing.json"), "config2D.json")
    assert got2 == pkg_cfg.resolve()

    monkeypatch.setattr(
        utils,
        "get_installed_config_path",
        lambda name: tmp_path / "not_there.json",
    )
    with pytest.raises(FileNotFoundError):
        utils.resolve_config_path(Path("missing2.json"), "missing_default.json")


def test_load_config_1d_3d(monkeypatch, tmp_path):
    c1 = tmp_path / "config1D.json"
    c3 = tmp_path / "config3D.json"
    c1.write_text('{"event":"2017-01-01T00:00:00","frequency":5}')
    c3.write_text('{"event":"2017-01-01T00:00:00","frequency":10}')

    def _resolve(path, default_name):
        return c1 if default_name == "config1D.json" else c3

    monkeypatch.setattr(utils, "resolve_config_path", _resolve)
    p1 = utils.load_config_1D()
    p3 = utils.load_config_3D()
    assert float(p1.frequency) == 5
    assert float(p3.frequency) == 10


def test_extrap_interpolate_helpers():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 4.0])
    f = utils.extrap1d(x, y, kind="linear")
    out = f(np.array([-1.0, 0.5, 3.0]))
    assert out.shape == (3,)
    assert np.isfinite(out).all()

    h = np.array([100.0, 120.0, 140.0, 160.0])
    param = np.array([1e5, 2e5, 4e5, 8e5])
    hx = np.array([110.0, 150.0])
    p_lin = utils.interpolate_by_altitude(h, hx, param, scale="linear", method="intp")
    p_log = utils.interpolate_by_altitude(h, hx, param, scale="log", method="extp")
    assert p_lin.shape == (2,)
    assert p_log.shape == (2,)
    assert np.all(p_log > 0)


def test_smooth_and_clean(monkeypatch, tmp_path):
    x = np.linspace(0.0, 1.0, 31)
    y = utils.smooth(x, window_len=5, window="hanning")
    assert y.shape == x.shape
    y2 = utils.smooth(x, window_len=5, window="flat")
    assert y2.shape == x.shape
    with pytest.raises(ValueError):
        utils.smooth(np.ones((2, 2)))
    with pytest.raises(ValueError):
        utils.smooth(np.ones(3), window_len=7)
    with pytest.raises(ValueError):
        utils.smooth(np.ones(10), window_len=5, window="bad")

    f = tmp_path / "matlab_crash_dump.123"
    f.write_text("x")
    monkeypatch.setattr(utils.Path, "home", lambda: tmp_path)
    utils.clean()
    assert not f.exists()
