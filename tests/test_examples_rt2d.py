from __future__ import annotations

import datetime as dt
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_module(path: Path, module_name: str):
    # Avoid importing heavy matplotlib backend during example module load.
    prev_plottrace = sys.modules.get("trace.plottrace")
    stub = types.ModuleType("trace.plottrace")
    stub.PlotRays = object
    sys.modules["trace.plottrace"] = stub
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if prev_plottrace is None:
            sys.modules.pop("trace.plottrace", None)
        else:
            sys.modules["trace.plottrace"] = prev_plottrace


def test_cartesian_trace_fan_calls_cartesian_mode():
    root = Path(__file__).resolve().parents[1]
    mod = _load_module(
        root / "examples" / "run_rt2d_iri_cartesian.py",
        "ex_rt2d_cart",
    )

    calls = []

    class DummyModel:
        def oblique_trace(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                x_km=np.array([0.0, 50.0]), z_km=np.array([0.0, 80.0])
            )

    rays = mod._trace_fan_cartesian(
        model=DummyModel(),
        heights_km=np.array([0.0, 50.0, 100.0]),
        elevs_deg=np.array([10.0, 20.0]),
        freqs_mhz=np.array([5.0, 6.0]),
        mode="O",
        formulation="appleton-hartree",
    )
    assert len(rays) == 2
    assert all(c["coordinate_system"] == "cartesian" for c in calls)


def test_spherical_trace_fan_calls_spherical_mode():
    root = Path(__file__).resolve().parents[1]
    mod = _load_module(
        root / "examples" / "run_rt2d_iri_spherical.py",
        "ex_rt2d_sph",
    )

    calls = []

    class DummyModel:
        def oblique_trace(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                x_km=np.array([0.0, 80.0]), z_km=np.array([0.0, 120.0])
            )

    rays = mod._trace_fan_spherical(
        model=DummyModel(),
        heights_km=np.array([0.0, 50.0, 100.0]),
        elevs_deg=np.array([10.0, 20.0]),
        freqs_mhz=np.array([5.0, 6.0]),
        mode="O",
        formulation="appleton-hartree",
        r_earth_km=6371.0,
    )
    assert len(rays) == 2
    assert all(c["coordinate_system"] == "spherical" for c in calls)
    assert all("r_earth_km" in c for c in calls)


def test_cartesian_run_uses_profile_zeroing_and_outputs_figure(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    mod = _load_module(
        root / "examples" / "run_rt2d_iri_cartesian.py",
        "ex_rt2d_cart_run",
    )

    class DummyProfile:
        def __init__(self):
            self.alt_km = np.array([0.0, 50.0, 100.0, 150.0])
            self.ne_cm3 = np.ones((4, 3)) * 1e5
            self.x_km = np.array([0.0, 100.0, 200.0])
            self.called = None

        def force_zero_density_below(self, zmin):
            self.called = zmin
            return 2

    profile = DummyProfile()

    monkeypatch.setattr(
        mod.RT2DProfile,
        "from_cfg",
        classmethod(lambda cls, **kwargs: profile),
    )
    monkeypatch.setattr(mod, "RT2D", lambda profile: SimpleNamespace(profile=profile))
    monkeypatch.setattr(
        mod, "build_elevations_from_cfg", lambda cfg: np.array([10.0, 20.0])
    )
    monkeypatch.setattr(
        mod, "build_freqs_from_cfg", lambda cfg, elevs: np.array([5.0, 5.0])
    )
    monkeypatch.setattr(
        mod,
        "_trace_fan_cartesian",
        lambda **kwargs: [
            SimpleNamespace(
                x_km=np.array([0.0, 1.0]), y_km=np.array([0.0, 1.0]), el0_deg=10.0
            )
        ],
    )

    captured = {}

    def _fake_plot(profile, rays, out_file):
        captured["out_file"] = str(out_file)
        captured["nrays"] = len(rays)

    monkeypatch.setattr(mod, "_plot_density_and_rays", _fake_plot)

    cfg = SimpleNamespace(
        worker=1,
        end_height_km=200.0,
        height_incriment_km=50.0,
        start_height_km=100.0,
    )
    mod._run(
        cfg=cfg,
        event_time=dt.datetime(2017, 5, 27, 16, 0, 0),
        mode="O",
        formulation="appleton-hartree",
    )
    assert profile.called == 100.0
    assert captured["nrays"] == 1
    assert captured["out_file"].endswith(
        "docs/examples/figures/rt2d_iri_cartesian_ray_paths.png"
    )


def test_spherical_run_uses_profile_zeroing_and_outputs_figure(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    mod = _load_module(
        root / "examples" / "run_rt2d_iri_spherical.py",
        "ex_rt2d_sph_run",
    )

    class DummyProfile:
        def __init__(self):
            self.alt_km = np.array([0.0, 50.0, 100.0, 150.0])
            self.ne_cm3 = np.ones((4, 3)) * 1e5
            self.x_km = np.array([0.0, 100.0, 200.0])
            self.called = None

        def force_zero_density_below(self, zmin):
            self.called = zmin
            return 1

    profile = DummyProfile()

    monkeypatch.setattr(
        mod.RT2DProfile,
        "from_cfg",
        classmethod(lambda cls, **kwargs: profile),
    )
    monkeypatch.setattr(mod, "RT2D", lambda profile: SimpleNamespace(profile=profile))
    monkeypatch.setattr(
        mod, "build_elevations_from_cfg", lambda cfg: np.array([10.0, 20.0])
    )
    monkeypatch.setattr(
        mod, "build_freqs_from_cfg", lambda cfg, elevs: np.array([5.0, 5.0])
    )
    monkeypatch.setattr(
        mod,
        "_trace_fan_spherical",
        lambda **kwargs: [
            SimpleNamespace(
                x_km=np.array([0.0, 1.0]), y_km=np.array([0.0, 1.0]), el0_deg=10.0
            )
        ],
    )

    captured = {}

    def _fake_plot(profile, rays, out_file):
        captured["out_file"] = str(out_file)
        captured["nrays"] = len(rays)

    monkeypatch.setattr(mod, "_plot_density_and_rays", _fake_plot)

    cfg = SimpleNamespace(
        worker=1,
        end_height_km=200.0,
        height_incriment_km=50.0,
        start_height_km=100.0,
    )
    mod._run(
        cfg=cfg,
        event_time=dt.datetime(2017, 5, 27, 16, 0, 0),
        mode="O",
        formulation="appleton-hartree",
        r_earth_km=6371.0,
    )
    assert profile.called == 100.0
    assert captured["nrays"] == 1
    assert captured["out_file"].endswith(
        "docs/examples/figures/rt2d_iri_spherical_ray_paths.png"
    )
