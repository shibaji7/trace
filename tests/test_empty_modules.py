import importlib


def test_import_dispersion_and_homing():
    dispersion = importlib.import_module("trace.dispersion")
    homing = importlib.import_module("trace.homing")
    assert dispersion is not None
    assert homing is not None
