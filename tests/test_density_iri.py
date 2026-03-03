import datetime as dt
import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np


def test_iri_fetch_and_load(monkeypatch, tmp_path):
    fake_iricore = types.ModuleType("iricore")

    class _Out:
        def __init__(self):
            self.edens = np.array([1e11, 2e11, 3e11])

    fake_iricore.iri = lambda *args, **kwargs: _Out()
    monkeypatch.setitem(sys.modules, "iricore", fake_iricore)

    m = importlib.import_module("trace.density.iri")
    importlib.reload(m)

    cfg = SimpleNamespace(iri_param=SimpleNamespace(iri_version=20))
    obj = m.IRI2d(cfg, dt.datetime(2024, 1, 1))
    ne, alts = obj.fetch_dataset(
        dt.datetime(2024, 1, 1),
        lats=np.array([40.0, 41.0]),
        lons=np.array([-75.0, -74.0]),
        alts=np.array([100.0, 110.0, 120.0]),
    )
    assert ne.shape == (3, 2)

    matfile = tmp_path / "ne.mat"
    import scipy.io as sio

    sio.savemat(matfile, {"ne": np.ones((3, 2))})
    loaded = obj.load_from_file(str(matfile))
    assert loaded.shape == (3, 2)
