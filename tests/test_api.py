import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Import package from repository root without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from py_fill_spill_merge import fill_spill_merge


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
