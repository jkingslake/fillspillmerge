import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Import package from repository root without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fillspillmerge import fill_spill_merge


# Stabilize native threading behavior in notebook-like environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")


def _demo_topography() -> np.ndarray:
    return np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 3.0, 3.0, 3.0, 10.0],
            [10.0, 3.0, 1.0, 3.0, 10.0],
            [10.0, 3.0, 3.0, 3.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float64,
    )


def test_build_and_reuse_hierarchy() -> None:
    topo = _demo_topography()

    wtd1 = np.zeros_like(topo)
    wtd1[2, 2] = 6.0

    out1, hierarchy = fill_spill_merge(
        wtd=wtd1,
        topography=topo,
        ocean_level=10.0,
        return_hierarchy=True,
    )

    assert out1.shape == topo.shape
    assert np.isclose(out1.sum(), wtd1.sum())
    assert hierarchy.height == topo.shape[0]
    assert hierarchy.width == topo.shape[1]

    wtd2 = np.zeros_like(topo)
    wtd2[1, 1] = 2.5
    out2 = fill_spill_merge(wtd=wtd2, hierarchy=hierarchy)

    assert out2.shape == topo.shape
    assert np.isclose(out2.sum(), wtd2.sum())


def test_default_zero_wtd() -> None:
    topo = _demo_topography()

    out, hierarchy = fill_spill_merge(
        topography=topo,
        ocean_level=10.0,
        return_hierarchy=True,
    )

    assert out.shape == topo.shape
    assert np.isclose(out.sum(), 0.0)

    out_reuse = fill_spill_merge(hierarchy=hierarchy)
    assert out_reuse.shape == topo.shape
    assert np.isclose(out_reuse.sum(), 0.0)


def test_requires_topography_when_no_hierarchy() -> None:
    with pytest.raises(ValueError, match="topography is required"):
        fill_spill_merge(ocean_level=10.0)


def test_wtd_shape_must_match_hierarchy() -> None:
    topo = _demo_topography()
    _, hierarchy = fill_spill_merge(topography=topo, ocean_level=10.0, return_hierarchy=True)

    bad_wtd = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="wtd shape must match"):
        fill_spill_merge(wtd=bad_wtd, hierarchy=hierarchy)


def test_xarray_topography_returns_xarray() -> None:
    xr = pytest.importorskip("xarray")

    topo_np = _demo_topography()
    topo_da = xr.DataArray(
        topo_np,
        dims=("y", "x"),
        coords={"y": np.arange(topo_np.shape[0]), "x": np.arange(topo_np.shape[1])},
        name="topography",
        attrs={"source": "unit-test"},
    )

    out_da, hierarchy = fill_spill_merge(
        topography=topo_da,
        ocean_level=10.0,
        return_hierarchy=True,
    )

    assert isinstance(out_da, xr.DataArray)
    assert out_da.dims == topo_da.dims
    assert np.array_equal(out_da.coords["x"], topo_da.coords["x"])
    assert np.array_equal(out_da.coords["y"], topo_da.coords["y"])
    assert np.isclose(out_da.values.sum(), 0.0)
    assert out_da.attrs["fillspillmerge_used_cached_hierarchy"] is False

    wtd_da = xr.DataArray(
        np.zeros_like(topo_np),
        dims=("y", "x"),
        coords=topo_da.coords,
        name="melt",
    )
    wtd_da.values[2, 2] = 2.0
    out_da_2 = fill_spill_merge(wtd=wtd_da, hierarchy=hierarchy)

    assert isinstance(out_da_2, xr.DataArray)
    assert out_da_2.dims == wtd_da.dims
    assert np.isclose(out_da_2.values.sum(), wtd_da.values.sum())
    assert out_da_2.attrs["fillspillmerge_used_cached_hierarchy"] is True
