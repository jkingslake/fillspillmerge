from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "examples" / "data"

sys.path.insert(0, str(REPO_ROOT))
from py_fill_spill_merge import fill_spill_merge  # noqa: E402


def main() -> int:
    topography = np.loadtxt(DATA_DIR / "topography.csv", delimiter=",", dtype=np.float64)
    wtd_initial = np.loadtxt(DATA_DIR / "wtd_initial.csv", delimiter=",", dtype=np.float64)

    wtd_out_1, hierarchy = fill_spill_merge(
        wtd=wtd_initial,
        topography=topography,
        ocean_level=10.0,
        nodata=None,
        return_hierarchy=True,
    )

    wtd_second_input = np.zeros_like(topography)
    wtd_second_input[2, 2] = 4.0
    wtd_out_2 = fill_spill_merge(wtd=wtd_second_input, hierarchy=hierarchy, nodata=None)

    np.save(DATA_DIR / "wtd_output_run1.npy", wtd_out_1)
    np.savetxt(DATA_DIR / "wtd_output_run1.csv", wtd_out_1, delimiter=",", fmt="%.6f")

    np.save(DATA_DIR / "wtd_output_run2.npy", wtd_out_2)
    np.savetxt(DATA_DIR / "wtd_output_run2.csv", wtd_out_2, delimiter=",", fmt="%.6f")

    print("topography shape:", topography.shape)
    print("run1 initial water volume:", float(wtd_initial.sum()))
    print("run1 final water volume:", float(wtd_out_1.sum()))
    print("run2 initial water volume:", float(wtd_second_input.sum()))
    print("run2 final water volume:", float(wtd_out_2.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
