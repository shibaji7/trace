import numpy as np
import pytest

from trace.model.dispersion import AppletonHartreeDispersion, SenWyllerDispersion


def test_appleton_scalar_modes():
    m = AppletonHartreeDispersion(
        frequency_hz=10e6,
        ne_m3=1.5e11,
        collision_hz=1e3,
        b_t=5e-5,
        theta_deg=30.0,
        te_k=1000.0,
        ti_k=900.0,
        tn_k=800.0,
    )
    for mode in ("N", "O", "X", "R", "L"):
        n = m.refractive_index(mode=mode)
        assert np.asarray(n).shape == ()
        ev = m.evaluate(mode=mode)
        assert np.isfinite(ev.phase_rad_per_km)
        assert np.isfinite(ev.absorption_db_per_km)
        assert ev.absorption_db_per_km >= 0.0


def test_appleton_broadcast_3d_inputs():
    ne = np.ones((2, 3, 4), dtype=float) * 2e11
    nu = np.ones((2, 3, 4), dtype=float) * 2e3
    b = np.ones((1, 3, 1), dtype=float) * 5e-5
    theta = np.array([0.0, 20.0, 40.0], dtype=float).reshape(1, 3, 1)
    m = AppletonHartreeDispersion(
        frequency_hz=12e6, ne_m3=ne, collision_hz=nu, b_t=b, theta_deg=theta
    )
    n = m.refractive_index(mode="X")
    assert n.shape == (2, 3, 4)
    ev = m.evaluate(mode="X")
    assert ev.absorption_db_per_km.shape == (2, 3, 4)


def test_sen_wyller_vector_and_modes():
    ne = np.array([1.0e11, 1.4e11, 2.0e11], dtype=float)
    nu = np.array([5.0e2, 8.0e2, 1.2e3], dtype=float)
    b = np.array([4.5e-5, 5.0e-5, 5.5e-5], dtype=float)
    sw = SenWyllerDispersion(
        frequency_hz=8e6, ne_m3=ne, collision_hz=nu, b_t=b, theta_deg=25.0
    )
    for mode in ("O", "X", "R", "L"):
        n = sw.refractive_index(mode=mode)
        assert n.shape == ne.shape
        ev = sw.evaluate(mode=mode)
        assert ev.phase_deg_per_km.shape == ne.shape
        assert np.all(np.isfinite(ev.absorption_db_per_km))
        assert np.all(ev.absorption_db_per_km >= 0.0)


def test_invalid_mode_and_frequency():
    with pytest.raises(ValueError):
        AppletonHartreeDispersion(
            frequency_hz=0.0,
            ne_m3=1e11,
            collision_hz=1e3,
        )
    ah = AppletonHartreeDispersion(10e6, 1e11, 1e3)
    sw = SenWyllerDispersion(10e6, 1e11, 1e3)
    with pytest.raises(ValueError):
        ah.refractive_index(mode="BAD")
    with pytest.raises(ValueError):
        sw.refractive_index(mode="BAD")
