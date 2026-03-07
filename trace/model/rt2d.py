"""2D HF ray utilities in class form."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from loguru import logger
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from .rt1d import RT1D, RT1DProfile


@dataclass
class RT2DConfig:
    """Integration controls for :class:`RT2D`."""

    x0_km: float = 0.0
    z0_km: float = 0.0
    ds_km: float = 1.0
    max_steps: int = 8000
    x_min_km: float | None = None
    x_max_km: float | None = None
    z_min_km: float = 0.0
    z_max_km: float | None = None
    n2_floor: float = 1e-5
    keep_every: int = 1


class RT2D:
    """
    2D ray tracing toolkit:
    - fast RK stepping (`trace`) for lightweight runs
    - ODE-based gradient tracing (`trace_cartesian_gradient`)
    - stratified Snell wrappers (`trace_cartesian_snell`, `trace_spherical_snell`)
    """

    def __init__(self, x_km: np.ndarray, z_km: np.ndarray, ne_m3: np.ndarray):
        self.x_km = np.asarray(x_km, dtype=float)
        self.z_km = np.asarray(z_km, dtype=float)
        self.ne_m3 = np.asarray(ne_m3, dtype=float)

        if self.x_km.ndim != 1 or self.z_km.ndim != 1:
            raise ValueError("x_km and z_km must be 1D arrays")
        if self.ne_m3.shape != (self.z_km.size, self.x_km.size):
            raise ValueError("ne_m3 shape must be (len(z_km), len(x_km))")
        if not np.all(np.diff(self.x_km) > 0) or not np.all(np.diff(self.z_km) > 0):
            raise ValueError("x_km and z_km must be strictly increasing")

        self._ne_interp = RegularGridInterpolator(
            (self.z_km, self.x_km),
            self.ne_m3,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        self._dx = float(np.mean(np.diff(self.x_km)))
        self._dz = float(np.mean(np.diff(self.z_km)))
        self._n_interp = None
        self._dn_dx_interp = None
        self._dn_dz_interp = None
        self._mup_interp = None
        logger.info(
            "RT2D initialized: nx={}, nz={}, ne_shape={}",
            self.x_km.size,
            self.z_km.size,
            self.ne_m3.shape,
        )

    def _inside(self, x: float, z: float, cfg: RT2DConfig) -> bool:
        x_min = self.x_km[0] if cfg.x_min_km is None else cfg.x_min_km
        x_max = self.x_km[-1] if cfg.x_max_km is None else cfg.x_max_km
        z_max = self.z_km[-1] if cfg.z_max_km is None else cfg.z_max_km
        return (x_min <= x <= x_max) and (cfg.z_min_km <= z <= z_max)

    def _sample_ne(self, x: float, z: float) -> float:
        return float(self._ne_interp(np.array([[z, x]], dtype=float))[0])

    def _n_and_grad_fd(
        self, freq_hz: float, x: float, z: float
    ) -> tuple[float, float, float]:
        ne = self._sample_ne(x, z)
        fp = RT1D.den_to_plasma_freq_hz(np.maximum(ne, 0.0))
        n2 = max(1.0 - (fp / float(freq_hz)) ** 2, 1e-12)
        n = float(np.sqrt(n2))

        hx = max(self._dx, 1e-3)
        hz = max(self._dz, 1e-3)
        ne_xp = self._sample_ne(x + hx, z)
        ne_xm = self._sample_ne(x - hx, z)
        ne_zp = self._sample_ne(x, z + hz)
        ne_zm = self._sample_ne(x, z - hz)

        n2_xp = max(
            1.0 - (RT1D.den_to_plasma_freq_hz(np.maximum(ne_xp, 0.0)) / freq_hz) ** 2,
            1e-12,
        )
        n2_xm = max(
            1.0 - (RT1D.den_to_plasma_freq_hz(np.maximum(ne_xm, 0.0)) / freq_hz) ** 2,
            1e-12,
        )
        n2_zp = max(
            1.0 - (RT1D.den_to_plasma_freq_hz(np.maximum(ne_zp, 0.0)) / freq_hz) ** 2,
            1e-12,
        )
        n2_zm = max(
            1.0 - (RT1D.den_to_plasma_freq_hz(np.maximum(ne_zm, 0.0)) / freq_hz) ** 2,
            1e-12,
        )

        if n <= 0:
            return n, 0.0, 0.0
        dn_dx = 0.25 * (n2_xp - n2_xm) / (hx * n)
        dn_dz = 0.25 * (n2_zp - n2_zm) / (hz * n)
        return n, float(dn_dx), float(dn_dz)

    def build_refractive_index_interpolators(
        self,
        freq_hz: float,
        b_abs_t: np.ndarray | float | None = None,
        b_psi_deg: np.ndarray | float | None = None,
        mode: str = "O",
    ) -> SimpleNamespace:
        """
        Build n, dn/dx, dn/dz, mup interpolators used by gradient tracing.
        """
        logger.info(
            "Building RT2D refractive index interpolators: freq={} Hz, mode={}",
            float(freq_hz),
            mode,
        )
        if b_abs_t is None:
            b_abs_t_arr = np.zeros_like(self.ne_m3)
        else:
            b_abs_t_arr = np.asarray(b_abs_t, dtype=float)
            if b_abs_t_arr.ndim == 0:
                b_abs_t_arr = np.full_like(self.ne_m3, float(b_abs_t_arr))
        if b_psi_deg is None:
            b_psi_arr = np.zeros_like(self.ne_m3)
        else:
            b_psi_arr = np.asarray(b_psi_deg, dtype=float)
            if b_psi_arr.ndim == 0:
                b_psi_arr = np.full_like(self.ne_m3, float(b_psi_arr))

        mu, mup = RT1D.refractive_indices(
            freq_hz=freq_hz,
            ne_m3=self.ne_m3,
            b_abs_t=b_abs_t_arr,
            b_psi_deg=b_psi_arr,
            mode=mode,
        )
        n = np.where(np.isfinite(mu), mu, np.nan)
        dn_dz, dn_dx = np.gradient(n, self.z_km, self.x_km)

        self._n_interp = RegularGridInterpolator(
            (self.z_km, self.x_km),
            n,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        self._dn_dx_interp = RegularGridInterpolator(
            (self.z_km, self.x_km),
            dn_dx,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        self._dn_dz_interp = RegularGridInterpolator(
            (self.z_km, self.x_km),
            dn_dz,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        self._mup_interp = RegularGridInterpolator(
            (self.z_km, self.x_km),
            mup,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        logger.debug("RT2D refractive index interpolators ready")
        return SimpleNamespace(n=n, mup=mup, dn_dx=dn_dx, dn_dz=dn_dz)

    def _eval_n_grad(
        self, x: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._n_interp is None:
            raise RuntimeError("Call build_refractive_index_interpolators() first")
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        x_arr, z_arr = np.broadcast_arrays(x_arr, z_arr)
        pts = np.column_stack([z_arr.ravel(), x_arr.ravel()])
        n = self._n_interp(pts).reshape(x_arr.shape)
        dnx = self._dn_dx_interp(pts).reshape(x_arr.shape)
        dnz = self._dn_dz_interp(pts).reshape(x_arr.shape)
        return n, dnx, dnz

    def _eval_mup(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        if self._mup_interp is None:
            raise RuntimeError("Call build_refractive_index_interpolators() first")
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        x_arr, z_arr = np.broadcast_arrays(x_arr, z_arr)
        pts = np.column_stack([z_arr.ravel(), x_arr.ravel()])
        return self._mup_interp(pts).reshape(x_arr.shape)

    @staticmethod
    def _ray_rhs(_s: float, y: np.ndarray, n_grad_fn) -> np.ndarray:
        x, z, vx, vz = y
        n, dnx, dnz = n_grad_fn(np.array([x]), np.array([z]))
        n = float(n[0])
        dnx = float(dnx[0])
        dnz = float(dnz[0])
        if not np.isfinite(n) or n <= 0:
            return np.zeros(4)
        dot = dnx * vx + dnz * vz
        dvx = (dnx - dot * vx) / n
        dvz = (dnz - dot * vz) / n
        return np.array([vx, vz, dvx, dvz], dtype=float)

    def trace_cartesian_gradient(
        self,
        freq_hz: float,
        elevation_deg: float,
        x0_km: float = 0.0,
        z0_km: float = 0.0,
        s_max_km: float = 5000.0,
        mode: str = "O",
        b_abs_t: np.ndarray | float | None = None,
        b_psi_deg: np.ndarray | float | None = None,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_step_km: float | None = None,
    ) -> SimpleNamespace:
        logger.info(
            "RT2D gradient trace start: freq={} Hz, elev={} deg, mode={}",
            float(freq_hz),
            float(elevation_deg),
            mode,
        )
        self.build_refractive_index_interpolators(
            freq_hz=freq_hz, b_abs_t=b_abs_t, b_psi_deg=b_psi_deg, mode=mode
        )

        elev = np.deg2rad(float(elevation_deg))
        y0 = np.array(
            [float(x0_km), float(z0_km), np.cos(elev), np.sin(elev)], dtype=float
        )
        y0[2:] /= max(np.linalg.norm(y0[2:]), 1e-12)

        z_ground = float(self.z_km[0])
        z_top = float(self.z_km[-1])
        x_left = float(self.x_km[0])
        x_right = float(self.x_km[-1])

        def ev_ground(_s, y):
            return y[1] - z_ground - 1e-3

        def ev_top(_s, y):
            return z_top - y[1]

        def ev_left(_s, y):
            return y[0] - x_left

        def ev_right(_s, y):
            return x_right - y[0]

        for ev in (ev_ground, ev_top, ev_left, ev_right):
            ev.terminal = True
            ev.direction = -1.0

        sol = solve_ivp(
            lambda s, y: self._ray_rhs(s, y, self._eval_n_grad),
            (0.0, float(s_max_km)),
            y0,
            method="RK45",
            rtol=float(rtol),
            atol=float(atol),
            max_step=max_step_km,
            events=[ev_ground, ev_top, ev_left, ev_right],
        )
        x = sol.y[0, :]
        z = sol.y[1, :]
        dx = np.diff(x)
        dz = np.diff(z)
        ds = np.hypot(dx, dz)
        group_path_km = float(np.nansum(ds))
        x_mid = 0.5 * (x[:-1] + x[1:])
        z_mid = 0.5 * (z[:-1] + z[1:])
        mup_mid = np.asarray(self._eval_mup(x_mid, z_mid), dtype=float)
        valid = np.isfinite(mup_mid)
        group_delay_sec = float(np.nansum((mup_mid[valid] / RT1D.C_KM_S) * ds[valid]))
        status = "length"
        if sol.status == 1:
            status = "ground" if len(sol.t_events[0]) > 0 else "domain"
        elif sol.status == -1:
            status = "failure"
        out = SimpleNamespace(
            x_km=x,
            z_km=z,
            vx=sol.y[2, :],
            vz=sol.y[3, :],
            t=sol.t,
            status=status,
            reason=status,
            group_path_km=group_path_km,
            group_delay_sec=group_delay_sec,
            ground_range_km=float(x[-1]) if status == "ground" else np.nan,
            x_apex_km=float(x[np.nanargmax(z)]) if z.size else np.nan,
            z_apex_km=float(np.nanmax(z)) if z.size else np.nan,
            freq_hz=float(freq_hz),
            elevation_deg=float(elevation_deg),
            mode=mode,
        )
        logger.info("RT2D gradient trace complete: status={}", out.status)
        return out

    def trace_cartesian_snell(
        self,
        freq_hz: float,
        elevation_deg: float,
        x_col_index: int = 0,
        mode: str = "O",
        b_abs_t: np.ndarray | None = None,
        b_psi_deg: np.ndarray | None = None,
    ) -> SimpleNamespace:
        """
        Run layered Snell trace using a vertical profile extracted at one x-column.
        """
        i = int(np.clip(x_col_index, 0, self.x_km.size - 1))
        prof = RT1DProfile(
            alt_km=self.z_km,
            ne_m3=self.ne_m3[:, i],
            b_abs_t=None if b_abs_t is None else np.asarray(b_abs_t)[:, i],
            b_psi_deg=None if b_psi_deg is None else np.asarray(b_psi_deg)[:, i],
        )
        return RT1D().trace_cartesian_snell(
            profile=prof, freq_hz=freq_hz, elevation_deg=elevation_deg, mode=mode
        )

    def trace_spherical_snell(
        self,
        freq_hz: float,
        elevation_deg: float,
        x_col_index: int = 0,
        mode: str = "O",
        b_abs_t: np.ndarray | None = None,
        b_psi_deg: np.ndarray | None = None,
        r_earth_km: float | None = None,
    ) -> SimpleNamespace:
        i = int(np.clip(x_col_index, 0, self.x_km.size - 1))
        prof = RT1DProfile(
            alt_km=self.z_km,
            ne_m3=self.ne_m3[:, i],
            b_abs_t=None if b_abs_t is None else np.asarray(b_abs_t)[:, i],
            b_psi_deg=None if b_psi_deg is None else np.asarray(b_psi_deg)[:, i],
        )
        return RT1D().trace_spherical_snell(
            profile=prof,
            freq_hz=freq_hz,
            elevation_deg=elevation_deg,
            mode=mode,
            r_earth_km=r_earth_km,
        )

    def trace(
        self,
        freq_hz: float,
        elevation_deg: float,
        cfg: RT2DConfig | None = None,
    ) -> SimpleNamespace:
        """
        Lightweight finite-difference tracer (kept for compatibility).
        """
        logger.info(
            "RT2D FD trace start: freq={} Hz, elev={} deg",
            float(freq_hz),
            float(elevation_deg),
        )
        cfg = cfg or RT2DConfig()
        r = np.array([float(cfg.x0_km), float(cfg.z0_km)], dtype=float)
        el = np.deg2rad(float(elevation_deg))
        t = np.array([np.cos(el), np.sin(el)], dtype=float)

        xs: list[float] = []
        zs: list[float] = []
        ns: list[float] = []
        reason = "max_steps"

        for i in range(int(cfg.max_steps)):
            if not self._inside(r[0], r[1], cfg):
                reason = "out_of_bounds"
                break

            n, gx, gz = self._n_and_grad_fd(float(freq_hz), r[0], r[1])
            if not np.isfinite(n):
                reason = "nan_field"
                break
            if n * n < float(cfg.n2_floor):
                reason = "evanescent"
                break

            if i % max(1, int(cfg.keep_every)) == 0:
                xs.append(float(r[0]))
                zs.append(float(r[1]))
                ns.append(float(n))

            g = np.array([gx, gz], dtype=float)
            a = (g - np.dot(g, t) * t) / max(n, 1e-9)
            t = t + a * float(cfg.ds_km)
            t_norm = np.linalg.norm(t)
            if t_norm <= 0:
                reason = "invalid_direction"
                break
            t /= t_norm
            r = r + t * float(cfg.ds_km)

        out = SimpleNamespace(
            x_km=np.asarray(xs, dtype=float),
            z_km=np.asarray(zs, dtype=float),
            n=np.asarray(ns, dtype=float),
            freq_hz=float(freq_hz),
            elevation_deg=float(elevation_deg),
            reason=reason,
            status=reason,
        )
        logger.info("RT2D FD trace complete: status={}", out.status)
        return out

    def trace_fan(
        self,
        freqs_hz: np.ndarray,
        elevations_deg: np.ndarray,
        cfg: RT2DConfig | None = None,
    ) -> list[SimpleNamespace]:
        out: list[SimpleNamespace] = []
        for f in np.asarray(freqs_hz, dtype=float):
            for el in np.asarray(elevations_deg, dtype=float):
                out.append(
                    self.trace(freq_hz=float(f), elevation_deg=float(el), cfg=cfg)
                )
        return out
