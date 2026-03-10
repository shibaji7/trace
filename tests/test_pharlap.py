import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


class _WS(dict):
    def __init__(self):
        super().__init__()
        self.raise_struct_error = False

    def __getitem__(self, key):
        if self.raise_struct_error and key in {"ray_data", "ray_path_data"}:
            raise ValueError("only a scalar struct can be returned from MATLAB")
        return super().__getitem__(key)


class _FakeEngine:
    def __init__(self):
        self.workspace = _WS()

    def genpath(self, p):
        return str(p)

    def addpath(self, *_args, **_kwargs):
        return None

    def quit(self):
        return None

    def eval(self, s, nargout=0):
        if "raytrace_2d_sp" in s:
            self.workspace["ray_data"] = [{"frequency": 10.5}]
            self.workspace["ray_path_data"] = [
                {"ground_range": [0, 10], "height": [0, 100]}
            ]
        if "jsonencode" in s:
            self.workspace.raise_struct_error = False
            self.workspace["ray_data_json"] = json.dumps(self.workspace["ray_data"])
            self.workspace["ray_path_data_json"] = json.dumps(
                self.workspace["ray_path_data"]
            )
            self.workspace["ray_state_vec_json"] = json.dumps([{"s": [0, 1]}])
        if "raytrace_3d_sp" in s or "raytrace_3d(" in s:
            self.workspace["ray_data"] = [{"frequency": 10.5}]
            self.workspace["ray_path_data"] = [
                {"lat": [40.0, 40.1], "lon": [-75.0, -74.9], "height": [0, 100]}
            ]
            self.workspace["ray_state_vec"] = [{"s": [0, 1]}]


@pytest.fixture
def pharlap_module(monkeypatch, tmp_path):
    fake_matlab = types.ModuleType("matlab")
    fake_matlab.double = lambda x: x
    fake_engine_mod = types.ModuleType("matlab.engine")
    fake_engine_mod.start_matlab = lambda: _FakeEngine()
    fake_matlab.engine = fake_engine_mod

    monkeypatch.setitem(sys.modules, "matlab", fake_matlab)
    monkeypatch.setitem(sys.modules, "matlab.engine", fake_engine_mod)
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))

    import trace

    pharlap_root = tmp_path / "pharlap_lib" / "pharlap_4.5.3"
    pharlap_root.mkdir(parents=True)
    monkeypatch.setattr(trace, "PHARLAP_LIB_PATH", tmp_path / "pharlap_lib")

    m = importlib.import_module("trace.pharlap")
    importlib.reload(m)
    return m


def test_get_matlab_pharlap_lib(pharlap_module):
    p = pharlap_module.get_matlab_pharlap_lib(
        trace_spec=Path(pharlap_module.PHARLAP_LIB_PATH)
    )
    assert "pharlap_4.5.3" in p


def test_run_pharlap_struct_array_fallback(pharlap_module):
    eng = pharlap_module.Engine(lib_path="/tmp")
    eng.eng.workspace.raise_struct_error = True

    ray_data, ray_path_data = eng.run_pharlap(
        ne_grid=np.ones((3, 3)),
        collision_freq=np.zeros((3, 3)),
        elevs=np.array([10.0, 11.0]),
        rb=45.0,
        freqs=np.array([10.5, 10.5]),
        irreg=np.zeros((4, 3)),
    )
    assert ray_data[0].frequency == 10.5
    assert ray_path_data[0].ground_range == [0, 10]


def test_get_matlab_pharlap_lib_missing_raises(pharlap_module, tmp_path):
    with pytest.raises(FileNotFoundError):
        pharlap_module.get_matlab_pharlap_lib(trace_spec=tmp_path, version="0.0.0")


def test_as_matlab_double_helper(pharlap_module):
    eng = pharlap_module.Engine(lib_path="/tmp")
    assert eng._as_matlab_double(1.0) == 1.0
    row = eng._as_matlab_double(np.array([1.0, 2.0]), ensure_row=True)
    assert isinstance(row, list)
    assert row == [[1.0, 2.0]]


def test_fetch_struct_array_fallback_for_3_keys(pharlap_module):
    eng = pharlap_module.Engine(lib_path="/tmp")
    eng.eng.workspace["ray_data"] = [{"a": 1}]
    eng.eng.workspace["ray_path_data"] = [{"b": 2}]
    eng.eng.workspace["ray_state_vec"] = [{"c": 3}]
    eng.eng.workspace.raise_struct_error = True
    ray_data, ray_path_data, ray_state = eng._fetch_struct_array(
        ["ray_data", "ray_path_data", "ray_state_vec"]
    )
    assert ray_data[0].a == 1
    assert ray_path_data[0].b == 2
    assert ray_state[0].s == [0, 1]


def test_run_pharlap_3d_and_sp_smoke(pharlap_module):
    eng = pharlap_module.Engine(lib_path="/tmp")
    ne3 = np.ones((2, 2, 2))
    zeros = np.zeros((2, 2, 2))
    iono_grid_parms = np.array(
        [[40.0, -75.0, 100.0], [0.5, 0.5, 20.0], [2.0, 2.0, 2.0]]
    )
    gm_grid_parms = np.array(
        [[40.0, -75.0, 100.0], [0.5, 0.5, 20.0], [2.0, 2.0, 2.0]]
    )

    out = eng.run_pharlap_3d(
        origin_lat=40.0,
        origin_lon=-75.0,
        origin_ht=0.0,
        elevs=np.array([10.0, 20.0]),
        ray_bearings=np.array([270.0, 271.0]),
        freqs=np.array([10.5, 10.5]),
        iono_en_grid=ne3,
        iono_en_grid_5=ne3,
        collision_freq=zeros,
        iono_grid_parms=iono_grid_parms,
        Bx=zeros,
        By=zeros,
        Bz=zeros + 5e-5,
        geomag_grid_parms=gm_grid_parms,
    )
    assert len(out) == 3
    assert out[0][0].frequency == 10.5

    out2 = eng.run_pharlap_3d_sp(
        origin_lat=40.0,
        origin_lon=-75.0,
        origin_ht=0.0,
        elevs=np.array([10.0, 20.0]),
        ray_bearings=np.array([270.0, 271.0]),
        freqs=np.array([10.5, 10.5]),
        rad_earth_m=6371000.0,
        iono_en_grid=ne3,
        iono_en_grid_5=ne3,
        collision_freq=zeros,
        iono_grid_parms=iono_grid_parms,
        Bx=zeros,
        By=zeros,
        Bz=zeros + 5e-5,
        geomag_grid_parms=gm_grid_parms,
    )
    assert len(out2) == 3
