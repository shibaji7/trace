import datetime as dt
import sys
import types
from trace.collision import ComputeCollision

import numpy as np
import pytest


def _arr(shape=(3, 4), v=1.0):
    return np.full(shape, v, dtype=float)


def test_compute_collision_outputs():
    cc = ComputeCollision(
        Te=_arr(v=1000),
        Ti=_arr(v=900),
        Tn=_arr(v=800),
        edens=_arr(v=1e5),
        O2p=_arr(v=2e4),
        Op=_arr(v=8e4),
        N2=_arr(v=1e10),
        O2=_arr(v=1e10),
        O=_arr(v=1e10),
        H=_arr(v=1e8),
        He=_arr(v=1e8),
        date=dt.datetime(2024, 1, 1),
    )
    assert cc.collision.nu_ft.shape == (3, 4)
    assert cc.collision.nu_sn.total.shape == (3, 4)
    assert np.all(cc.collision.nu_av_cc >= cc.collision.nu_ft)


def test_from_nrlmsise_with_mock(monkeypatch):
    class _BG:
        def __init__(self, **kwargs):
            self.msise = {
                "Tn": _arr(v=800),
                "N2": _arr(v=1e10),
                "O2": _arr(v=1e10),
                "O": _arr(v=1e10),
                "H": _arr(v=1e8),
                "He": _arr(v=1e8),
            }

    monkeypatch.setattr("trace.collision.NRLMSISE2D", _BG)
    cc = ComputeCollision.from_nrlmsise(
        date=dt.datetime(2024, 1, 1),
        lats=np.array([0, 1]),
        lons=np.array([0, 1]),
        heights_km=np.array([100, 110, 120]),
        Te=_arr(v=1000),
        Ti=_arr(v=900),
        edens=_arr(v=1e5),
        O2p=_arr(v=2e4),
        Op=_arr(v=8e4),
    )
    assert cc.collision.nu_ft.shape == (3, 4)


def test_from_nrlmsise_3d_with_mock(monkeypatch):
    class _BG3:
        def __init__(self, **kwargs):
            self.msise = {
                "Tn": np.full((2, 3, 4), 800.0),
                "N2": np.full((2, 3, 4), 1e10),
                "O2": np.full((2, 3, 4), 1e10),
                "O": np.full((2, 3, 4), 1e10),
                "H": np.full((2, 3, 4), 1e8),
                "He": np.full((2, 3, 4), 1e8),
            }

    monkeypatch.setattr("trace.collision.NRLMSISE3D", _BG3)
    shape = (2, 3, 4)
    cc = ComputeCollision.from_nrlmsise_3d(
        date=dt.datetime(2024, 1, 1),
        lats=np.array([30.0, 31.0]),
        lons=np.array([-90.0, -89.0, -88.0]),
        heights_km=np.array([100, 120, 140, 160]),
        Te=np.full(shape, 1000.0),
        Ti=np.full(shape, 900.0),
        edens=np.full(shape, 1e5),
        O2p=np.full(shape, 2e4),
        Op=np.full(shape, 8e4),
        workers=2,
    )
    assert cc.collision.nu_ft.shape == shape


def test_nrlmsise2d_workers_and_spaceweather_update(monkeypatch):
    from trace import collision as cm

    class _Var:
        def __init__(self, arr):
            self.values = arr

    class _DS:
        def __init__(self, nh):
            shp = (1, nh, 1, 1)
            self.variables = {
                "N2": _Var(np.ones(shp) * 1e10),
                "O2": _Var(np.ones(shp) * 2e10),
                "O": _Var(np.ones(shp) * 3e10),
                "H": _Var(np.ones(shp) * 4e8),
                "He": _Var(np.ones(shp) * 5e8),
                "Talt": _Var(np.ones(shp) * 800.0),
            }

    def _msise_4d(date, heights_km, lats, lons):
        return _DS(int(np.asarray(heights_km).size))

    fake_dataset = types.ModuleType("nrlmsise00.dataset")
    fake_dataset.msise_4d = _msise_4d
    fake_pkg = types.ModuleType("nrlmsise00")
    monkeypatch.setitem(sys.modules, "nrlmsise00", fake_pkg)
    monkeypatch.setitem(sys.modules, "nrlmsise00.dataset", fake_dataset)

    called = {"n": 0}

    class _SW:
        @staticmethod
        def update_data():
            called["n"] += 1

    fake_sw_mod = types.ModuleType("spaceweather")
    fake_sw_mod.sw = _SW()
    monkeypatch.setitem(sys.modules, "spaceweather", fake_sw_mod)

    class _FakePool:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def map(self, fn, iterable):
            return [fn(x) for x in iterable]

    monkeypatch.setattr(cm, "ProcessPoolExecutor", _FakePool)

    bg = cm.NRLMSISE2D(
        date=dt.datetime(2024, 1, 1),
        lats=np.array([0.0, 1.0, 2.0]),
        lons=np.array([0.0, 1.0, 2.0]),
        heights_km=np.array([100.0, 110.0]),
        workers=2,
        update_spaceweather=True,
    )
    assert called["n"] == 1
    assert bg.msise["N2"].shape == (2, 3)
    kw = bg.as_collision_kwargs()
    assert set(kw.keys()) == {"Tn", "N2", "O2", "O", "H", "He"}


def test_nrlmsise3d_import_error(monkeypatch):
    from trace import collision as cm

    monkeypatch.setitem(sys.modules, "nrlmsise00", types.ModuleType("nrlmsise00"))
    if "nrlmsise00.dataset" in sys.modules:
        monkeypatch.delitem(sys.modules, "nrlmsise00.dataset", raising=False)
    with pytest.raises(ImportError):
        cm.NRLMSISE3D(
            date=dt.datetime(2024, 1, 1),
            lats=np.array([0.0, 1.0]),
            lons=np.array([0.0, 1.0]),
            heights_km=np.array([100.0, 120.0]),
            workers=1,
        )


def test_collision_static_formula():
    nu = ComputeCollision.atmospheric_collision_frequency(
        ni=np.array([1e5, 2e5]),
        nn=np.array([1e10, 1e10]),
        T=np.array([500.0, 700.0]),
    )
    assert nu.shape == (2,)
    assert np.all(np.isfinite(nu))
    assert np.all(nu > 0)
