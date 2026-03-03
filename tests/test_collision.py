import datetime as dt

import numpy as np

from trace.collision import ComputeCollision


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
