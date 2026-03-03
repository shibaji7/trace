import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np


class _FakeAxes:
    transAxes = object()

    def plot(self, *args, **kwargs):
        return None

    def text(self, *args, **kwargs):
        return None

    def set_facecolor(self, *args, **kwargs):
        return None

    def fill_between(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_xlim(self, *args, **kwargs):
        return None

    def set_ylim(self, *args, **kwargs):
        return None

    def tick_params(self, *args, **kwargs):
        return None

    def set_yticks(self, *args, **kwargs):
        return None

    def pcolormesh(self, *args, **kwargs):
        return object()

    def get_position(self):
        return SimpleNamespace(x1=1.0, y0=0.0, height=1.0)


class _FakeColorbarAx:
    def tick_params(self, *args, **kwargs):
        return None


class _FakeColorbar:
    def __init__(self):
        self.ax = _FakeColorbarAx()

    def set_label(self, *args, **kwargs):
        return None


class _FakeFigure:
    def add_subplot(self, *args, **kwargs):
        return _FakeAxes()

    def add_axes(self, *args, **kwargs):
        return _FakeAxes()

    def colorbar(self, *args, **kwargs):
        return _FakeColorbar()

    def savefig(self, path, *args, **kwargs):
        with open(path, "wb") as fh:
            fh.write(b"ok")

    def clf(self):
        return None


def test_plotrays_density_and_rays_with_stubbed_matplotlib(monkeypatch, tmp_path):
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.rcParams = {}
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.rcParams = {}
    fake_pyplot.figure = lambda *args, **kwargs: _FakeFigure()
    fake_pyplot.close = lambda *args, **kwargs: None

    fake_colors = types.ModuleType("matplotlib.colors")
    fake_colors.LogNorm = lambda *args, **kwargs: object()
    fake_colors.Normalize = lambda *args, **kwargs: object()

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib.colors", fake_colors)
    monkeypatch.setitem(sys.modules, "scienceplots", types.ModuleType("scienceplots"))

    m = importlib.import_module("trace.plottrace")
    importlib.reload(m)

    rp = m.PlotRays(oth=False, xlim=[0, 10], ylim=[0, 100], figsize=(4, 3))
    rp.set_param_lims(edens_lim=(1e8, 1e12))

    X, Z = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 100, 6))
    Ne = np.full_like(X, 1e10)
    rp.set_density(X, Z, Ne)

    rays = [SimpleNamespace(x_km=np.array([0, 5, 10]), y_km=np.array([0, 40, 0]), el0_deg=30)]
    ax = rp.lay_rays(outputs=rays, kind="edens", add_cbar=False)
    assert ax is not None

    out = tmp_path / "ray.png"
    rp.save(out)
    rp.close()
    assert out.exists()
