import importlib


def test_import_dispersion_and_homing():
    homing = importlib.import_module("hfpytrace.homing")
    assert homing is not None
