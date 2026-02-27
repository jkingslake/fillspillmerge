from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _ensure_module_loaded(build_dir: Optional[Path] = None):
    try:
        return importlib.import_module("_fsm")
    except ModuleNotFoundError:
        pass

    if build_dir is None:
        repo_root = Path(__file__).resolve().parents[1]
        build_dir = repo_root / "build"

    if not build_dir.exists():
        raise ModuleNotFoundError(
            f"Could not import _fsm and build directory does not exist: {build_dir}"
        )

    sys.path.insert(0, str(build_dir))
    return importlib.import_module("_fsm")


def _build_hierarchy(
    topography: np.ndarray,
    ocean_level: float,
    nodata: Optional[float],
    build_dir: Optional[Path],
):
    topo_arr = np.asarray(topography, dtype=np.float64)
    if topo_arr.ndim != 2:
        raise ValueError("topography must be a 2D array")

    _fsm = _ensure_module_loaded(build_dir=build_dir)
    return _fsm.Hierarchy(topo_arr, float(ocean_level), nodata)


def fill_spill_merge(
    wtd: Optional[np.ndarray] = None,
    topography: Optional[np.ndarray] = None,
    ocean_level: Optional[float] = None,
    nodata: Optional[float] = None,
    hierarchy=None,
    build_dir: Optional[Path] = None,
    return_hierarchy: bool = False,
):
    if hierarchy is None:
        if topography is None:
            raise ValueError("topography is required when hierarchy is None")
        if ocean_level is None:
            raise ValueError("ocean_level is required when hierarchy is None")

        hierarchy = _build_hierarchy(
            topography=topography,
            ocean_level=float(ocean_level),
            nodata=nodata,
            build_dir=build_dir,
        )

    if wtd is None:
        wtd_arr = np.zeros((hierarchy.height, hierarchy.width), dtype=np.float64)
    else:
        wtd_arr = np.asarray(wtd, dtype=np.float64)
        if wtd_arr.ndim != 2:
            raise ValueError("wtd must be a 2D array")

    if wtd_arr.shape != (hierarchy.height, hierarchy.width):
        raise ValueError("wtd shape must match hierarchy shape")

    out = hierarchy.run(wtd_arr, nodata)
    if return_hierarchy:
        return out, hierarchy
    return out
