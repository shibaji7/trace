import datetime as dt
from trace.collision import ComputeCollision

import numpy as np


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
