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
