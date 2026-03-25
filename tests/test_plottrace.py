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

    def grid(self, *args, **kwargs):
        return None

    def set_yticks(self, *args, **kwargs):
        return None

    def get_yticks(self, *args, **kwargs):
        return np.array([-200, 0, 200, 400, 600])

    def get_ylim(self, *args, **kwargs):
        return (-300.0, 600.0)

    def get_xlim(self, *args, **kwargs):
        return (-500.0, 2500.0)

    def set_xticks(self, *args, **kwargs):
        return None

    def set_xticklabels(self, *args, **kwargs):
        return None

    def pcolormesh(self, *args, **kwargs):
        return object()

    def set_title(self, *args, **kwargs):
        return None

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
    def __init__(self):
        self.subplots_adjust_calls = []
        self.set_size_inches_calls = []

    def add_subplot(self, *args, **kwargs):
        return _FakeAxes()

    def add_axes(self, *args, **kwargs):
        return _FakeAxes()

    def colorbar(self, *args, **kwargs):
        return _FakeColorbar()

    def subplots_adjust(self, *args, **kwargs):
        self.subplots_adjust_calls.append((args, kwargs))
        return None

    def set_size_inches(self, *args, **kwargs):
        self.set_size_inches_calls.append((args, kwargs))
        return None

    def savefig(self, path, *args, **kwargs):
        with open(path, "wb") as fh:
            fh.write(b"ok")

    def clf(self):
        return None


class _FakeMatlabWS(dict):
    pass


class _FakeMatlabEngine:
    def __init__(self):
        self.workspace = _FakeMatlabWS()
        self.closed = False

    def eval(self, s, nargout=0):
        if "has_geoplot3" in s:
            self.workspace["has_geoplot3"] = 1.0
            self.workspace["has_geoglobe"] = 1.0
            self.workspace["has_map_toolbox"] = 1.0
            self.workspace["has_display"] = 0.0
        if "exportgraphics" in s or "exportapp" in s:
            out = self.workspace.get("out_file")
            if out:
                with open(out, "wb") as fh:
                    fh.write(b"ok")
        return None

    def quit(self):
        self.closed = True
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

    m = importlib.import_module("hfpytrace.plottrace")
    importlib.reload(m)

    rp = m.PlotRays(
        oth=False,
        xlim=[0, 10],
        ylim=[0, 100],
        figsize=(4, 3),
        style_kwargs={"figure_dpi": 120, "savefig_dpi": 144},
        default_yticks=(0, 50, 100),
    )
    rp.set_param_lims(edens_lim=(1e8, 1e12))

    X, Z = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 100, 6))
    Ne = np.full_like(X, 1e10)
    rp.set_density(X, Z, Ne)

    rays = [
        SimpleNamespace(
            x_km=np.array([0, 5, 10]), y_km=np.array([0, 40, 0]), el0_deg=30
        )
    ]
    ax = rp.lay_rays(outputs=rays, kind="edens", add_cbar=False)
    assert ax is not None

    out = tmp_path / "ray.png"
    rp.save(out)
    rp.close()
    assert out.exists()


def test_plotrays3d_and_parameter_api_with_stubbed_matplotlib(monkeypatch, tmp_path):
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

    m = importlib.import_module("hfpytrace.plottrace")
    importlib.reload(m)

    p3 = m.PlotRays3D(
        oth=True,
        top_aspect="auto",
        style_kwargs={"figure_dpi": 110},
        axis_facecolor="0.97",
        ray_under_color="yellow",
        ray_over_color="blue",
    )
    p3.set_param_lims(edens_lim=(1e4, 1e6))
    ne_side = np.full((6, 5), 1e5, dtype=float)
    ne_front = np.full((6, 4), 1e5, dtype=float)
    p3.plot_faces(
        ne_side=ne_side,
        ne_front=ne_front,
        x_side=np.linspace(-10, 10, 5),
        x_front=np.linspace(30, 40, 4),
        heights=np.linspace(0, 500, 6),
        ray_side_x=[np.array([-5, 0, 5])],
        ray_front_x=[np.array([32, 35, 38])],
        ray_h=[np.array([0, 300, 0])],
        kind="edens",
        ylim=[-300, 600],
        panel_wspace=0.22,
    )
    out = tmp_path / "ray3d.png"
    p3.save(out)
    p3.close()
    assert out.exists()


def test_plotrays3d_route_faces_with_stubbed_matplotlib(monkeypatch, tmp_path):
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

    m = importlib.import_module("hfpytrace.plottrace")
    importlib.reload(m)

    p3 = m.PlotRays3DRouteFaces(
        oth=True,
        top_aspect="auto",
        style_kwargs={"figure_dpi": 110},
    )
    p3.set_param_lims(edens_lim=(1e4, 1e6))

    lats = np.linspace(40.5, 41.5, 5)
    lons = np.linspace(-75.0, -73.5, 6)
    heights = np.linspace(100.0, 500.0, 7)
    ne_grid = np.full((lats.size, lons.size, heights.size), 1e5, dtype=float)

    rays = [
        SimpleNamespace(
            lat=np.array([40.7, 40.85, 41.0], dtype=float),
            lon=np.array([-74.0, -73.8, -73.6], dtype=float),
            height=np.array([0.0, 180.0, 60.0], dtype=float),
        )
    ]
    p3.plot_route_faces(
        ne_grid=ne_grid,
        ray_path_data=rays,
        lats=lats,
        lons=lons,
        heights=heights,
        origin_lat=40.7,
        origin_lon=-74.0,
        bearing_deg=270.0,
        kind="edens",
    )
    assert p3.fig.subplots_adjust_calls

    out = tmp_path / "ray3d_route.png"
    p3.save(out)
    p3.close()
    assert out.exists()


def test_matlab_geoplot3d_headless_fallback(monkeypatch, tmp_path):
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

    fake_matlab = types.ModuleType("matlab")
    fake_matlab.double = lambda x: x
    fake_engine_mod = types.ModuleType("matlab.engine")
    fake_engine_mod.start_matlab = lambda: _FakeMatlabEngine()
    fake_matlab.engine = fake_engine_mod

    monkeypatch.setitem(sys.modules, "matlab", fake_matlab)
    monkeypatch.setitem(sys.modules, "matlab.engine", fake_engine_mod)

    m = importlib.import_module("hfpytrace.plottrace")
    importlib.reload(m)

    g = m.MatlabGeoPlot3D()
    assert g.available
    assert g.can_plot3
    assert not g.can_geoplot3

    rays = [
        SimpleNamespace(
            lat=np.array([30.0, 30.2, 30.5]),
            lon=np.array([-90.0, -89.8, -89.5]),
            height=np.array([0.0, 120.0, 40.0]),
        )
    ]
    out = tmp_path / "geo.png"
    g.plot_rays(rays, out_file=out, zoom_to_rays=True)
    g.close()
    assert out.exists()
