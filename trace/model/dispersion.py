"""
Dispersion relations for magneto-ionic refractive index and absorption.

This module implements two formulations:

1. Appleton-Hartree
2. Sen-Wyller

Both formulations are designed for fully broadcasted NumPy workflows, so all
input fields may be scalar, 1D, 2D, or 3D (or any broadcast-compatible shape).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy import constants
from scipy.integrate import quad


def _as_broadcast(*arrs):
    """Convert all inputs to float arrays and NumPy-broadcast them."""
    return np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in arrs])


def _safe_sqrt_complex(z: np.ndarray) -> np.ndarray:
    """Complex square root with finite-value guard."""
    out = np.sqrt(z.astype(np.complex128))
    out = np.where(np.isfinite(out.real) & np.isfinite(out.imag), out, np.nan + 1j * np.nan)
    return out


def _gamma_factorial(x: float) -> float:
    """Gamma/Factorial hybrid used by the Sen-Wyller C(p,y) function."""
    n = int(float(x))
    frac = float(x) - n
    if frac > 0.0:
        return math.factorial(n) * math.gamma(frac)
    return float(math.factorial(n))


def _C_sw(p: float, y: float) -> float:
    """
    Sen-Wyller helper integral C(p, y).

    Parameters
    ----------
    p : float
        Exponent term used by the formulation (typically 1.5 or 2.5).
    y : float
        Dimensionless frequency ratio.
    """
    if not np.isfinite(y):
        return np.nan

    def _f(t):
        return (t**p) * np.exp(-t) / (t * t + y * y)

    val, _ = quad(_f, 0.0, np.inf, limit=200)
    return val / _gamma_factorial(p)


_V_C15 = np.vectorize(lambda y: _C_sw(1.5, float(y)), otypes=[float])
_V_C25 = np.vectorize(lambda y: _C_sw(2.5, float(y)), otypes=[float])


@dataclass
class DispersionResult:
    """
    Container for derived propagation metrics.

    Attributes
    ----------
    refractive_index : np.ndarray
        Complex refractive index (same shape as broadcasted input fields).
    absorption_db_per_km : np.ndarray
        Absorption coefficient in dB/km.
    phase_rad_per_km : np.ndarray
        Phase constant in rad/km.
    phase_deg_per_km : np.ndarray
        Phase constant in deg/km.
    """

    refractive_index: np.ndarray
    absorption_db_per_km: np.ndarray
    phase_rad_per_km: np.ndarray
    phase_deg_per_km: np.ndarray


class DispersionBase:
    """
    Base utilities shared by dispersion formulations.

    Parameters
    ----------
    frequency_hz : float
        Wave frequency in Hz.
    ne_m3 : array_like
        Electron density in m^-3.
    collision_hz : array_like
        Effective collision frequency in Hz.
    b_t : array_like, optional
        Magnetic field magnitude in Tesla.
    theta_deg : array_like, optional
        Angle between propagation direction and magnetic field (deg).
    te_k, ti_k, tn_k : array_like, optional
        Electron, ion, and neutral temperatures (K). These are accepted as
        metadata inputs for downstream/extended models.

    Notes
    -----
    All array-like parameters can be scalars or N-D arrays. They are broadcast
    internally to a common shape before evaluation.
    """

    def __init__(
        self,
        frequency_hz: float,
        ne_m3,
        collision_hz,
        b_t=0.0,
        theta_deg=0.0,
        te_k=None,
        ti_k=None,
        tn_k=None,
    ):
        self.frequency_hz = float(frequency_hz)
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be > 0")

        self.ne_m3 = np.asarray(ne_m3, dtype=float)
        self.collision_hz = np.asarray(collision_hz, dtype=float)
        self.b_t = np.asarray(b_t, dtype=float)
        self.theta_deg = np.asarray(theta_deg, dtype=float)
        self.te_k = None if te_k is None else np.asarray(te_k, dtype=float)
        self.ti_k = None if ti_k is None else np.asarray(ti_k, dtype=float)
        self.tn_k = None if tn_k is None else np.asarray(tn_k, dtype=float)
        logger.debug(
            "DispersionBase initialized: f={} Hz, ne_shape={}, nu_shape={}",
            self.frequency_hz,
            self.ne_m3.shape,
            self.collision_hz.shape,
        )

    @property
    def omega(self) -> float:
        """Angular frequency omega = 2*pi*f [rad/s]."""
        return 2.0 * np.pi * self.frequency_hz

    @property
    def k0(self) -> float:
        """Vacuum wavenumber k0 = omega/c [rad/m]."""
        return self.omega / constants.c  # rad/m

    def _params(self):
        """
        Build broadcasted dimensionless plasma parameters.

        Returns
        -------
        tuple[np.ndarray, ...]
            (X, Z, Y, YL, YT) where:
            X  = (omega_p/omega)^2
            Z  = nu/omega
            Y  = omega_H/omega
            YL = Y*cos(theta)
            YT = Y*sin(theta)
        """
        ne, nu, b, th = _as_broadcast(
            np.clip(self.ne_m3, 0.0, None),
            np.clip(self.collision_hz, 0.0, None),
            np.clip(np.abs(self.b_t), 0.0, None),
            self.theta_deg,
        )
        wp2 = ne * constants.e**2 / (constants.epsilon_0 * constants.m_e)
        X = wp2 / (self.omega**2)
        Z = nu / self.omega
        Y = (constants.e * b / constants.m_e) / self.omega
        th_r = np.deg2rad(th)
        YL = Y * np.cos(th_r)
        YT = Y * np.sin(th_r)
        return X, Z, Y, YL, YT

    def refractive_index(self, mode: str = "O") -> np.ndarray:
        """Return complex refractive index for the selected mode."""
        raise NotImplementedError

    def evaluate(self, mode: str = "O") -> DispersionResult:
        """
        Evaluate refractive index and derived absorption/phase products.

        Parameters
        ----------
        mode : str, optional
            Propagation mode selector. Supported values depend on subclass.

        Returns
        -------
        DispersionResult
            Complex refractive index plus absorption and phase metrics.
        """
        n = self.refractive_index(mode=mode)
        # 8.686 converts Np to dB; k0*n imag part gives attenuation constant.
        abs_db_km = np.abs(8.686 * self.k0 * np.imag(n) * 1e3)
        ph_rad_km = np.real(self.k0 * n) * 1e3
        ph_deg_km = np.rad2deg(ph_rad_km)
        logger.debug("Dispersion evaluated for mode={}", mode)
        return DispersionResult(
            refractive_index=n,
            absorption_db_per_km=abs_db_km,
            phase_rad_per_km=ph_rad_km,
            phase_deg_per_km=ph_deg_km,
        )


class AppletonHartreeDispersion(DispersionBase):
    """
    Appleton-Hartree style magneto-ionic formulation.

    Supported modes
    ---------------
    - "N"/"NO"/"ISO": isotropic-like branch
    - "O": ordinary
    - "X": extraordinary
    - "R": right-hand circular
    - "L": left-hand circular
    """

    def refractive_index(self, mode: str = "O") -> np.ndarray:
        """
        Compute Appleton-Hartree refractive index for the selected mode.
        """
        mode_u = str(mode).upper()
        X, Z, _, YL, YT = self._params()
        jz = 1j * Z

        if mode_u in {"N", "NO", "ISO"}:
            return _safe_sqrt_complex(1.0 - (X / (1.0 - jz)))
        if mode_u == "O":
            return _safe_sqrt_complex(1.0 - (X / (1.0 - jz)))
        if mode_u == "X":
            num = 2.0 * X * (1.0 - X - jz)
            den = (2.0 * (1.0 - X - jz) * (1.0 - jz)) - (2.0 * YT**2)
            return _safe_sqrt_complex(1.0 - (num / den))
        if mode_u == "R":
            return _safe_sqrt_complex(1.0 - (X / ((1.0 - jz) - YL)))
        if mode_u == "L":
            return _safe_sqrt_complex(1.0 - (X / ((1.0 - jz) + YL)))
        raise ValueError("mode must be one of: N/NO/ISO, O, X, R, L")


class SenWyllerDispersion(DispersionBase):
    """
    Sen-Wyller generalized formulation.

    Notes
    -----
    This implementation follows the structure used in the existing project
    absorption workflow and evaluates C(p,y) numerically with `scipy.integrate.quad`.
    """

    def refractive_index(self, mode: str = "O") -> np.ndarray:
        """
        Compute Sen-Wyller refractive index for the selected mode.

        Supported modes: "N"/"NO"/"ISO", "O", "X", "R", "L".
        """
        mode_u = str(mode).upper()
        X, _, Y, YL, _ = self._params()
        nu_sw = np.clip(self.collision_hz, 1e-12, None)
        w = self.omega
        ne, nu_sw, Y, YL = _as_broadcast(
            np.clip(self.ne_m3, 0.0, None),
            nu_sw,
            Y,
            YL,
        )

        wp2 = ne * constants.e**2 / (constants.epsilon_0 * constants.m_e)
        y = w / nu_sw
        yx = (w - YL * w) / nu_sw
        yo = (w + YL * w) / nu_sw

        C15_y = _V_C15(y)
        C25_y = _V_C25(y)
        C15_yx = _V_C15(yx)
        C15_yo = _V_C15(yo)
        C25_yx = _V_C25(yx)
        C25_yo = _V_C25(yo)

        # R/L branches from Sen-Wyller generalization.
        pref = wp2 / (2.0 * w * nu_sw)
        nL = 1.0 - pref * (yo * C15_yo + 1j * 2.5 * C25_yo)
        nR = 1.0 - pref * (yx * C15_yx + 1j * 2.5 * C25_yx)

        if mode_u == "L":
            return nL.astype(np.complex128)
        if mode_u == "R":
            return nR.astype(np.complex128)

        # O/X branches through generalized dielectric terms.
        ajb = (wp2 / (w * nu_sw)) * ((y * C15_y) + 1j * (2.5 * C25_y))
        c = (wp2 / (w * nu_sw)) * yx * C15_yx
        d = 2.5 * (wp2 / (w * nu_sw)) * C25_yx
        e = (wp2 / (w * nu_sw)) * yo * C15_yo
        f = 2.5 * (wp2 / (w * nu_sw)) * C25_yo

        eI = 1.0 - ajb
        eII = 0.5 * ((f - d) + 1j * (c - e))
        eIII = ajb - 0.5 * ((c + e) + 1j * (d + f))

        Aa = 2.0 * eI * (eI + eIII)
        Bb = (eIII * (eI + eII)) + eII**2
        Dd = 2.0 * eI
        Ee = 2.0 * eIII

        nO = _safe_sqrt_complex(Aa / (Dd + Ee))
        nX = _safe_sqrt_complex((Aa + Bb) / (Dd + Ee))

        if mode_u in {"N", "NO", "ISO", "O"}:
            return nO
        if mode_u == "X":
            return nX
        raise ValueError("mode must be one of: N/NO/ISO, O, X, R, L")
