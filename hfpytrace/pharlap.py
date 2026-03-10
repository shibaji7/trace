import json
from pathlib import Path

import matlab
import matlab.engine
import numpy as np
from loguru import logger

from hfpytrace import PHARLAP_LIB_PATH
from hfpytrace.utils import to_namespace


def get_matlab_pharlap_lib(trace_spec: Path | None = None, version: str = "4.5.3"):
    base_path = PHARLAP_LIB_PATH if trace_spec is None else trace_spec
    lib_path = (
        base_path / f"pharlap_{version}"
        if base_path.name == "pharlap_lib"
        else base_path / "pharlap_lib" / f"pharlap_{version}"
    )
    if lib_path.exists():
        logger.info(f"Matlab library path found: {lib_path}")
        return str(lib_path)
    raise FileNotFoundError(f"Matlab library path not found: {lib_path}")


class Engine:
    def __init__(self, lib_path: str = None):
        self.eng = matlab.engine.start_matlab()
        env_path = get_matlab_pharlap_lib() if lib_path is None else lib_path
        self.eng.addpath(self.eng.genpath(env_path), nargout=0)
        logger.info("Matlab engine started and library path added.")
        return

    def close(self):
        logger.info("Closing Matlab engine.")
        self.eng.quit()
        return

    def run_pharlap(
        self,
        ne_grid: np.ndarray,
        collision_freq: np.ndarray,
        elevs: np.ndarray,
        rb: float,
        freqs: np.ndarray,
        irreg: np.ndarray,
        nhops: int = 1,
        tol: float = 1e-7,
        radius_earth: float = 6371,
        irregs_flag: int = 0,
        start_height: int = 50,
        height_inc: int = 1,
        range_inc: int = 1,
    ):
        logger.info("Running Pharlap...")
        self.eng.eval("close all; clear all; clc;", nargout=0)

        self.eng.workspace["ne_grid"] = matlab.double(ne_grid.tolist())
        self.eng.workspace["elevs"] = matlab.double(elevs.tolist())
        self.eng.workspace["rb"] = rb
        self.eng.workspace["collision_freq"] = matlab.double(collision_freq.tolist())
        self.eng.workspace["freqs"] = matlab.double(freqs.tolist())
        self.eng.workspace["nhops"] = nhops
        self.eng.workspace["tol"] = tol
        self.eng.workspace["radius_earth"] = radius_earth
        self.eng.workspace["irregs_flag"] = irregs_flag
        self.eng.workspace["start_height"] = start_height
        self.eng.workspace["height_inc"] = height_inc
        self.eng.workspace["range_inc"] = range_inc
        self.eng.workspace["irreg"] = matlab.double(irreg.tolist())

        self.eng.eval(
            """
            [ray_data, ray_path_data] = ...
                raytrace_2d_sp(elevs, rb, freqs, nhops, tol, ...
                radius_earth, irregs_flag, ne_grid, ne_grid, ...
                collision_freq, start_height, height_inc, range_inc, irreg ...
                );
            """,
            nargout=0,
        )
        try:
            ray_data, ray_path_data = (
                self.eng.workspace["ray_data"],
                self.eng.workspace["ray_path_data"],
            )
        except ValueError:
            # MATLAB Engine cannot directly return non-scalar struct arrays.
            self.eng.eval(
                "ray_data_json = jsonencode(ray_data); "
                "ray_path_data_json = jsonencode(ray_path_data);",
                nargout=0,
            )
            ray_data = json.loads(self.eng.workspace["ray_data_json"])
            ray_path_data = json.loads(self.eng.workspace["ray_path_data_json"])
        logger.info("Pharlap run completed.")
        return to_namespace(ray_data), to_namespace(ray_path_data)

    @staticmethod
    def _as_matlab_double(arr, ensure_row: bool = False):
        a = np.asarray(arr, dtype=float)
        if a.ndim == 0:
            return float(a)
        if ensure_row and a.ndim == 1:
            a = a.reshape(1, -1)
        return matlab.double(a.tolist())

    def _fetch_struct_array(self, keys: list[str]):
        try:
            out = [self.eng.workspace[k] for k in keys]
        except ValueError:
            # MATLAB Engine cannot directly return non-scalar struct arrays.
            enc = "; ".join([f"{k}_json = jsonencode({k})" for k in keys]) + ";"
            self.eng.eval(enc, nargout=0)
            out = [json.loads(self.eng.workspace[f"{k}_json"]) for k in keys]
        return [to_namespace(v) for v in out]

    def run_pharlap_3d(
        self,
        origin_lat: float,
        origin_lon: float,
        origin_ht: float,
        elevs: np.ndarray,
        ray_bearings: np.ndarray,
        freqs: np.ndarray,
        iono_en_grid: np.ndarray,
        iono_en_grid_5: np.ndarray,
        collision_freq: np.ndarray,
        iono_grid_parms: np.ndarray,
        Bx: np.ndarray,
        By: np.ndarray,
        Bz: np.ndarray,
        geomag_grid_parms: np.ndarray,
        OX_mode: int = 1,
        nhops: int = 1,
        tol: float | np.ndarray = 1e-7,
        ray_state_vec_in=None,
    ):
        """
        Wrapper for PHaRLAP `raytrace_3d` (WGS84 formulation).
        """
        logger.info("Running PHaRLAP raytrace_3d...")
        self.eng.eval("close all; clear all; clc;", nargout=0)

        self.eng.workspace["origin_lat"] = float(origin_lat)
        self.eng.workspace["origin_lon"] = float(origin_lon)
        self.eng.workspace["origin_ht"] = float(origin_ht)
        self.eng.workspace["elevs"] = self._as_matlab_double(elevs, ensure_row=True)
        self.eng.workspace["ray_bearings"] = self._as_matlab_double(
            ray_bearings, ensure_row=True
        )
        self.eng.workspace["freqs"] = self._as_matlab_double(freqs, ensure_row=True)
        self.eng.workspace["OX_mode"] = int(OX_mode)
        self.eng.workspace["nhops"] = int(nhops)
        self.eng.workspace["tol"] = self._as_matlab_double(tol, ensure_row=True)

        self.eng.workspace["iono_en_grid"] = self._as_matlab_double(iono_en_grid)
        self.eng.workspace["iono_en_grid_5"] = self._as_matlab_double(iono_en_grid_5)
        self.eng.workspace["collision_freq"] = self._as_matlab_double(collision_freq)
        self.eng.workspace["iono_grid_parms"] = self._as_matlab_double(
            iono_grid_parms, ensure_row=False
        )

        self.eng.workspace["Bx"] = self._as_matlab_double(Bx)
        self.eng.workspace["By"] = self._as_matlab_double(By)
        self.eng.workspace["Bz"] = self._as_matlab_double(Bz)
        self.eng.workspace["geomag_grid_parms"] = self._as_matlab_double(
            geomag_grid_parms, ensure_row=False
        )

        if ray_state_vec_in is None:
            self.eng.eval(
                """
                [ray_data, ray_path_data, ray_state_vec] = raytrace_3d( ...
                    origin_lat, origin_lon, origin_ht, elevs, ray_bearings, ...
                    freqs, OX_mode, nhops, tol, iono_en_grid, iono_en_grid_5, ...
                    collision_freq, iono_grid_parms, Bx, By, Bz, geomag_grid_parms);
                """,
                nargout=0,
            )
        else:
            self.eng.workspace["ray_state_vec_in"] = ray_state_vec_in
            self.eng.eval(
                """
                [ray_data, ray_path_data, ray_state_vec] = raytrace_3d( ...
                    origin_lat, origin_lon, origin_ht, elevs, ray_bearings, ...
                    freqs, OX_mode, nhops, tol, iono_en_grid, iono_en_grid_5, ...
                    collision_freq, iono_grid_parms, Bx, By, Bz, ...
                    geomag_grid_parms, ray_state_vec_in);
                """,
                nargout=0,
            )

        ray_data, ray_path_data, ray_state_vec = self._fetch_struct_array(
            ["ray_data", "ray_path_data", "ray_state_vec"]
        )
        logger.info("PHaRLAP raytrace_3d run completed.")
        return ray_data, ray_path_data, ray_state_vec

    def run_pharlap_3d_sp(
        self,
        origin_lat: float,
        origin_lon: float,
        origin_ht: float,
        elevs: np.ndarray,
        ray_bearings: np.ndarray,
        freqs: np.ndarray,
        rad_earth_m: float,
        iono_en_grid: np.ndarray,
        iono_en_grid_5: np.ndarray,
        collision_freq: np.ndarray,
        iono_grid_parms: np.ndarray,
        Bx: np.ndarray,
        By: np.ndarray,
        Bz: np.ndarray,
        geomag_grid_parms: np.ndarray,
        OX_mode: int = 1,
        nhops: int = 1,
        tol: float | np.ndarray = 1e-7,
        ray_state_vec_in=None,
    ):
        """
        Wrapper for PHaRLAP `raytrace_3d_sp` (spherical Earth formulation).
        """
        logger.info("Running PHaRLAP raytrace_3d_sp...")
        self.eng.eval("close all; clear all; clc;", nargout=0)

        self.eng.workspace["origin_lat"] = float(origin_lat)
        self.eng.workspace["origin_lon"] = float(origin_lon)
        self.eng.workspace["origin_ht"] = float(origin_ht)
        self.eng.workspace["elevs"] = self._as_matlab_double(elevs, ensure_row=True)
        self.eng.workspace["ray_bearings"] = self._as_matlab_double(
            ray_bearings, ensure_row=True
        )
        self.eng.workspace["freqs"] = self._as_matlab_double(freqs, ensure_row=True)
        self.eng.workspace["OX_mode"] = int(OX_mode)
        self.eng.workspace["nhops"] = int(nhops)
        self.eng.workspace["tol"] = self._as_matlab_double(tol, ensure_row=True)
        self.eng.workspace["rad_earth_m"] = float(rad_earth_m)

        self.eng.workspace["iono_en_grid"] = self._as_matlab_double(iono_en_grid)
        self.eng.workspace["iono_en_grid_5"] = self._as_matlab_double(iono_en_grid_5)
        self.eng.workspace["collision_freq"] = self._as_matlab_double(collision_freq)
        self.eng.workspace["iono_grid_parms"] = self._as_matlab_double(
            iono_grid_parms, ensure_row=False
        )

        self.eng.workspace["Bx"] = self._as_matlab_double(Bx)
        self.eng.workspace["By"] = self._as_matlab_double(By)
        self.eng.workspace["Bz"] = self._as_matlab_double(Bz)
        self.eng.workspace["geomag_grid_parms"] = self._as_matlab_double(
            geomag_grid_parms, ensure_row=False
        )

        if ray_state_vec_in is None:
            self.eng.eval(
                """
                [ray_data, ray_path_data, ray_state_vec] = raytrace_3d_sp( ...
                    origin_lat, origin_lon, origin_ht, elevs, ray_bearings, ...
                    freqs, OX_mode, nhops, tol, rad_earth_m, iono_en_grid, ...
                    iono_en_grid_5, collision_freq, iono_grid_parms, ...
                    Bx, By, Bz, geomag_grid_parms);
                """,
                nargout=0,
            )
        else:
            self.eng.workspace["ray_state_vec_in"] = ray_state_vec_in
            self.eng.eval(
                """
                [ray_data, ray_path_data, ray_state_vec] = raytrace_3d_sp( ...
                    origin_lat, origin_lon, origin_ht, elevs, ray_bearings, ...
                    freqs, OX_mode, nhops, tol, rad_earth_m, iono_en_grid, ...
                    iono_en_grid_5, collision_freq, iono_grid_parms, ...
                    Bx, By, Bz, geomag_grid_parms, ray_state_vec_in);
                """,
                nargout=0,
            )

        ray_data, ray_path_data, ray_state_vec = self._fetch_struct_array(
            ["ray_data", "ray_path_data", "ray_state_vec"]
        )
        logger.info("PHaRLAP raytrace_3d_sp run completed.")
        return ray_data, ray_path_data, ray_state_vec
