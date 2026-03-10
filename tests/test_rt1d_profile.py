import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from hfpytrace.model.rt1d import RT1D, RT1DProfile


def _cfg():
    return SimpleNamespace(
        event="2017-05-27T16:00:00",
        origin=SimpleNamespace(lat=40.0, lon=-75.0),
        start_height_km=100.0,
        end_height_km=140.0,
        height_incriment_km=10.0,
        geomag_grid=SimpleNamespace(coord_input="GEO", coeff_dir=""),
        iri_param=SimpleNamespace(
            f107=150.0, foF2_coeff="CCIR", hmF2_model="SHU2015", coord="GEO"
        ),
    )


def test_validate_and_set_density():
    alt = np.array([100.0, 110.0, 120.0])
    p = RT1DProfile(alt_km=alt, lat=40.0, lon=-75.0, time=dt.datetime(2017, 5, 27))
    assert p.source == "iri"
    p.set_electron_density(ne_m3=np.array([1e11, 2e11, 3e11]))
    assert p.source == "iri"
    assert p.ne_cm3.shape == alt.shape
    assert np.allclose(p.ne_cm3, p.ne_m3 * 1e-6)


def test_validate_errors():
    with pytest.raises(ValueError):
        RT1DProfile(
            alt_km=np.array([120.0, 110.0]),
            lat=40.0,
            lon=-75.0,
            time=dt.datetime(2017, 5, 27),
        )
    p = RT1DProfile(
        alt_km=np.array([100.0, 110.0]),
        lat=40.0,
        lon=-75.0,
        time=dt.datetime(2017, 5, 27),
    )
    with pytest.raises(ValueError):
        p.set_electron_density(ne_m3=np.array([-1.0, 2.0]))


def test_freq_density_roundtrip():
    ne = np.array([1e10, 5e10, 1e11], dtype=float)
    fp = RT1D.den_to_plasma_freq_hz(ne)
    ne2 = RT1D.plasma_freq_to_den(fp)
    assert np.allclose(ne, ne2, rtol=1e-6, atol=0.0)
    with pytest.raises(ValueError):
        RT1D.plasma_freq_to_den(np.array([-1.0]))
    with pytest.raises(ValueError):
        RT1D.den_to_plasma_freq_hz(np.array([-1.0]))


def test_angles():
    inc = np.array([0.0, 30.0, -60.0])
    a = RT1DProfile.inclintation2vertical(inc)
    b = RT1DProfile.inclination_to_vertical_angle(inc)
    c = RT1DProfile.vertical_to_magnetic_angle(inc)
    assert np.allclose(a, np.array([90.0, 60.0, 30.0]))
    assert np.allclose(a, b)
    assert np.allclose(b, c)


def test_from_cfg_with_mocked_fetch(monkeypatch):
    cfg = _cfg()
    alt = np.array([100.0, 110.0, 120.0, 130.0], dtype=float)

    def _fake_iri(self, _cfg):
        self.ne_m3 = np.linspace(1e11, 4e11, self.alt_km.size)
        self.ne_cm3 = self.ne_m3 * 1e-6
        self.source = "iri"
        return self.ne_m3

    def _fake_msise(
        self, workers=1, update_spaceweather=False, suppress_spaceweather_warning=True
    ):
        self.msise = SimpleNamespace(
            N2=np.full(self.alt_km.shape, 1e12),
            O2=np.full(self.alt_km.shape, 2e12),
            O=np.full(self.alt_km.shape, 3e12),
            H=np.full(self.alt_km.shape, 4e11),
            He=np.full(self.alt_km.shape, 5e10),
            Tn=np.full(self.alt_km.shape, 900.0),
            t_nn=np.full(self.alt_km.shape, 0.0),
        )
        return self.msise

    def _fake_geomag(self, coord_input="GEO", coeff_dir=None):
        self.geomag = SimpleNamespace(
            Bx=np.full(self.alt_km.shape, 2.0e-5),
            By=np.full(self.alt_km.shape, 1.0e-5),
            Bz=np.full(self.alt_km.shape, -4.0e-5),
            bmag_t=np.full(self.alt_km.shape, 5.0e-5),
            inc_deg=np.full(self.alt_km.shape, 60.0),
            dec_deg=np.full(self.alt_km.shape, 5.0),
            psi_deg=np.full(self.alt_km.shape, 30.0),
        )
        return self.geomag

    monkeypatch.setattr(RT1DProfile, "fetch_iri", _fake_iri)
    monkeypatch.setattr(RT1DProfile, "fetch_msise", _fake_msise)
    monkeypatch.setattr(RT1DProfile, "fetch_geomag", _fake_geomag)

    p = RT1DProfile.from_cfg(
        cfg=cfg,
        time=dt.datetime(2017, 5, 27, 16, 0, 0),
        alt_km=alt,
        fetch_iri=True,
        fetch_msise=True,
        fetch_geomag=True,
        workers=2,
    )
    assert p.source == "iri"
    assert p.ne_m3.shape == alt.shape
    assert p.msise is not None
    assert p.geomag is not None


def test_init_from_iso_time_and_ne_cm3():
    p = RT1DProfile(
        alt_km=np.array([100.0, 110.0, 120.0]),
        lat=40.0,
        lon=-75.0,
        time="2017-05-27T16:00:00",
        ne_cm3=np.array([1e5, 2e5, 3e5]),
    )
    assert isinstance(p.time, dt.datetime)
    assert np.allclose(p.ne_m3, np.array([1e11, 2e11, 3e11]))


def test_set_density_requires_exactly_one_input():
    p = RT1DProfile(
        alt_km=np.array([100.0, 110.0]),
        lat=40.0,
        lon=-75.0,
        time=dt.datetime(2017, 5, 27),
    )
    with pytest.raises(ValueError):
        p.set_electron_density()
    with pytest.raises(ValueError):
        p.set_electron_density(ne_m3=np.array([1.0, 2.0]), ne_cm3=np.array([1.0, 2.0]))


def test_set_density_default_and_override_source():
    p = RT1DProfile(
        alt_km=np.array([100.0, 110.0]),
        lat=40.0,
        lon=-75.0,
        time=dt.datetime(2017, 5, 27),
    )
    p.set_electron_density(ne_m3=np.array([1e11, 2e11]))
    assert p.source == "iri"
    p.set_electron_density(ne_m3=np.array([1e11, 2e11]), source="user")
    assert p.source == "user"


def test_validate_msise_and_geomag_shape_and_fields():
    alt = np.array([100.0, 110.0, 120.0])
    p = RT1DProfile(
        alt_km=alt,
        lat=40.0,
        lon=-75.0,
        time=dt.datetime(2017, 5, 27),
    )
    p.msise = SimpleNamespace(N2=np.ones(3))  # missing fields
    with pytest.raises(ValueError):
        p.validate()

    p.msise = None
    p.geomag = SimpleNamespace(Bx=np.ones(3))  # missing fields
    with pytest.raises(ValueError):
        p.validate()

    p.geomag = None
    p.msise = SimpleNamespace(
        N2=np.ones(3),
        O2=np.ones(3),
        O=np.ones(3),
        H=np.ones(3),
        He=np.ones(3),
        Tn=np.ones(2),  # wrong shape
    )
    with pytest.raises(ValueError):
        p.validate()


def test_from_cfg_route_start_and_error_paths(monkeypatch):
    cfg_route = SimpleNamespace(
        event="2017-05-27T16:00:00",
        route=SimpleNamespace(start=SimpleNamespace(lat=10.0, lon=20.0)),
        start_height_km=100.0,
        end_height_km=130.0,
        height_incriment_km=10.0,
    )

    p = RT1DProfile.from_cfg(
        cfg_route,
        fetch_iri=False,
        fetch_msise=False,
        fetch_geomag=False,
    )
    assert p.lat == 10.0 and p.lon == 20.0
    assert np.allclose(p.alt_km, np.array([100.0, 110.0, 120.0]))

    cfg_bad = SimpleNamespace(
        event="2017-05-27T16:00:00",
        start_height_km=100.0,
        end_height_km=120.0,
        height_incriment_km=10.0,
    )
    with pytest.raises(ValueError):
        RT1DProfile.from_cfg(
            cfg_bad,
            fetch_iri=False,
            fetch_msise=False,
            fetch_geomag=False,
        )


def test_fetch_iri_msise_geomag_methods_with_stubs(monkeypatch):
    alt = np.array([100.0, 110.0, 120.0, 130.0])
    p = RT1DProfile(alt_km=alt, lat=40.0, lon=-75.0, time=dt.datetime(2017, 5, 27))

    class _FakeIRI2d:
        def __init__(self, cfg, event):
            self.cfg = cfg
            self.event = event

        def fetch_dataset(self, time, lats, lons, alts, workers=1):
            out = np.linspace(1e5, 4e5, alts.size).reshape(-1, 1)
            return out, alts

    class _FakeMSISE:
        def __init__(self, **kwargs):
            nh = kwargs["heights_km"].size
            self.msise = {
                "N2": np.ones((nh, 1)) * 1e12,
                "O2": np.ones((nh, 1)) * 2e12,
                "O": np.ones((nh, 1)) * 3e12,
                "H": np.ones((nh, 1)) * 4e11,
                "He": np.ones((nh, 1)) * 5e10,
                "Tn": np.ones((nh, 1)) * 900.0,
                "t_nn": np.ones((nh, 1)) * 0.0,
            }

    class _FakeGM:
        def __init__(self, nh):
            self.Bx = np.ones((1, 1, nh)) * 1e-5
            self.By = np.ones((1, 1, nh)) * 2e-5
            self.Bz = np.ones((1, 1, nh)) * -3e-5
            self.bmag_t = np.ones((1, 1, nh)) * 5e-5
            self.inc_deg = np.ones((1, 1, nh)) * 60.0
            self.dec_deg = np.ones((1, 1, nh)) * 5.0
            self.psi_deg = np.ones((1, 1, nh)) * 30.0

    monkeypatch.setattr("hfpytrace.model.rt1d.IRI2d", _FakeIRI2d)
    monkeypatch.setattr("hfpytrace.model.rt1d.NRLMSISE2D", _FakeMSISE)
    monkeypatch.setattr(
        "hfpytrace.model.rt1d.build_geomag_grid",
        lambda **kwargs: _FakeGM(kwargs["alts_km"].size),
    )

    cfg = _cfg()
    ne = p.fetch_iri(cfg)
    assert ne.shape == alt.shape
    assert p.source == "iri"

    ms = p.fetch_msise(
        workers=2,
        update_spaceweather=True,
        suppress_spaceweather_warning=False,
    )
    assert ms.N2.shape == alt.shape
    assert np.allclose(ms.Tn, 900.0)

    gm = p.fetch_geomag(coord_input="QD", coeff_dir=" ")
    assert gm.bmag_t.shape == alt.shape
    assert np.allclose(gm.psi_deg, 30.0)


def test_rt1d_init_default_source_iri():
    rt = RT1D(
        time=dt.datetime(2017, 5, 27),
        lat=40.0,
        lon=-75.0,
        alt_km=np.array([100.0, 110.0]),
        ne_m3=np.array([1e11, 2e11]),
    )
    assert rt.profile.source == "iri"


def test_rt1d_nvis_tracer():
    rt = RT1D(
        time=dt.datetime(2017, 5, 27),
        lat=40.0,
        lon=-75.0,
        alt_km=np.linspace(100.0, 300.0, 21),
        ne_m3=np.linspace(1e10, 2e11, 21),
    )
    out = rt.NVIS_tracer(
        freq_mhz=np.array([6.0, 8.0, 10.0]),
        mode="O",
        formulation="appleton",
    )
    assert out.freq_mhz.shape == (3,)
    assert out.vh_km.shape == (3,)
    assert out.turning_height_km.shape == (3,)
    assert out.n_profile.shape == (3, 21)
    assert out.reason.shape == (3,)
    assert np.all(np.isfinite(out.vh_km) | np.isnan(out.vh_km))

    # Non-O modes should be accepted if supported by dispersion backend.
    out_r = rt.NVIS_tracer(
        freq_mhz=np.array([8.0, 10.0]),
        mode="R",
        formulation="appleton",
    )
    assert out_r.vh_km.shape == (2,)

    out_l = rt.NVIS_tracer(
        freq_mhz=np.array([8.0, 10.0]),
        mode="L",
        formulation="senwyller",
    )
    assert out_l.vh_km.shape == (2,)


def test_rt1d_refractive_profile_formulation_errors():
    rt = RT1D(
        time=dt.datetime(2017, 5, 27),
        lat=40.0,
        lon=-75.0,
        alt_km=np.array([100.0, 110.0, 120.0]),
        ne_m3=np.array([1e11, 1.2e11, 1.4e11]),
    )
    n = rt._refractive_index_profile(
        frequency_hz=10e6, mode="O", formulation="senwyller"
    )
    assert n.shape == rt.profile.alt_km.shape
    with pytest.raises(ValueError):
        rt._refractive_index_profile(frequency_hz=10e6, mode="O", formulation="unknown")


def test_rt1d_nvis_tracer_invalid_freq():
    rt = RT1D(
        time=dt.datetime(2017, 5, 27),
        lat=40.0,
        lon=-75.0,
        alt_km=np.array([100.0, 110.0, 120.0]),
        ne_m3=np.array([1e11, 1.2e11, 1.4e11]),
    )
    with pytest.raises(ValueError):
        rt.NVIS_tracer(freq_mhz=np.array([0.0, 5.0]))
