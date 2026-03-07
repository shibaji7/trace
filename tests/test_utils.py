import sys
import types
from trace import utils
from types import SimpleNamespace

import numpy as np
import pytest


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
    cfg2 = utils.read_params_2D()
    assert cfg2.event == "2017-05-27T16:00:00Z"
