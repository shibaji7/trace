from pathlib import Path

import matlab
import matlab.engine
import numpy as np
import pandas as pd
from loguru import logger

from trace import PHARLAP_LIB_PATH


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
        ne_grid,
        elevs,
        rb,
        freqs,
        nhops,
        tol,
        radius_earth,
        irregs_flag,
        collision_freq,
        start_height,
        height_inc,
        range_inc,
        irreg,
    ):
        logger.info("Running Pharlap...")
        self.eng.eval("close all; clear all; clc;", nargout=0)

        self.eng.workspace["ne_grid"] = matlab.double(ne_grid.tolist())
        self.eng.workspace["elevs"] = matlab.double(elevs.tolist())
        self.eng.workspace["rb"] = rb
        self.eng.workspace["freqs"] = matlab.double(freqs.tolist())
        self.eng.workspace["nhops"] = nhops
        self.eng.workspace["tol"] = tol
        self.eng.workspace["radius_earth"] = radius_earth
        self.eng.workspace["irregs_flag"] = irregs_flag
        self.eng.workspace["collision_freq"] = collision_freq
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
        ray_data, ray_path_data = (
            self.eng.workspace["ray_data"],
            self.eng.workspace["ray_path_data"],
        )
        logger.info("Pharlap run completed.")
        return ray_data, ray_path_data
