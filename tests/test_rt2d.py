import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def rt2d_module(monkeypatch):
    # Stub pynasonde imports used by trace.rt2d.
    pmod = types.ModuleType("pynasonde")
    model = types.ModuleType("pynasonde.model")
    tmod = types.ModuleType("pynasonde.model.trace")
    iono = types.ModuleType("pynasonde.model.trace.ionosphere")
    plot = types.ModuleType("pynasonde.model.trace.plottrace")

    class _IonoModels:
        @staticmethod
        def create_chapman_ionosphere_bump(*args, **kwargs):
            x = args[0]
            hs = args[1]
            X, Z = np.meshgrid(x, hs)
            return X, Z, np.full_like(X, 1e11, dtype=float)

    iono.IRI = object
    iono.IonosphereModels = _IonoModels
    plot.PlotRays = object

    monkeypatch.setitem(sys.modules, "pynasonde", pmod)
    monkeypatch.setitem(sys.modules, "pynasonde.model", model)
    monkeypatch.setitem(sys.modules, "pynasonde.model.trace", tmod)
    monkeypatch.setitem(sys.modules, "pynasonde.model.trace.ionosphere", iono)
    monkeypatch.setitem(sys.modules, "pynasonde.model.trace.plottrace", plot)

    m = importlib.import_module("trace.rt2d")
    importlib.reload(m)
    return m


def test_rt2d_core(rt2d_module):
    m = rt2d_module
    x = np.linspace(0, 50, 21)
    y = np.linspace(0, 300, 31)
    Ne = np.full((y.size, x.size), 1e10)

    bi = m.Bilinear2D(x, y, Ne)
    assert bi(10, 100) > 0
    assert bi(-1, 100) == 0

    fp = m.plasma_freq_hz(1e10)
    assert fp > 0

    rt = m.RayTracer2D(x, y, Ne)
    cfg = m.RayConfig(f_MHz=8.0, el0_deg=30.0, s_max_km=20.0, ds_km=1.0, x_max_km=100)
    out = rt.trace(cfg)
    assert "reason" in out
    assert out["x_km"].size > 0
