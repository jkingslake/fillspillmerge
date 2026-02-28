# fillspillmerge

Python wrapper around Fill-Spill-Merge, using Kerry Callaghan's fork of
Barnes2020-FillSpillMerge (as a submodule):
[https://github.com/KCallaghan/Barnes2020-FillSpillMerge](https://github.com/KCallaghan/Barnes2020-FillSpillMerge)

This repo keeps the upstream C++ algorithm in a submodule and keeps the Python
wrapper code in this repo.

## Repository layout

- `Barnes2020-FillSpillMerge/`: upstream algorithm code (submodule)
- `src/pybind_module.cpp`: pybind11 C++ extension source (`_fsm`)
- `fillspillmerge/`: Python API package
- `examples/`: example arrays, script, and notebooks

## Setup

```bash
cd /Users/jkingslake/Documents/science/meltwater_routing/fsm_pywrapper/fillspillmerge
git submodule update --init --recursive
uv sync
```

Optional xarray support:

```bash
uv sync --extra xarray
```

## Build the extension

The wrapper needs the compiled `_fsm` extension.

```bash
cmake -S . -B build -DPython3_EXECUTABLE=.venv/bin/python
cmake --build build -j --target _fsm
```

When to rebuild:

- first setup on a machine
- after C++/CMake changes
- after deleting `build/`
- after changing Python environment/interpreter

## API

```python
from fillspillmerge import fill_spill_merge
```

Main function:

```python
fill_spill_merge(
    wtd=None,
    topography=None,
    ocean_level=None,
    nodata=None,
    hierarchy=None,
    build_dir=None,
    return_hierarchy=False,
)
```

Behavior:

- If `hierarchy` is provided, it is reused and `topography`/`ocean_level` are ignored.
- If `hierarchy` is not provided, `topography` and `ocean_level` are required.
- If `wtd` is omitted, it defaults to a zero raster.
- If xarray is installed and `topography` or `wtd` is a `DataArray`, output is a `DataArray`.

## Usage (NumPy)

### 1. Single run (compute hierarchy internally)

```python
import numpy as np
from fillspillmerge import fill_spill_merge

topography = np.load("topography.npy")
wtd = np.load("wtd.npy")

wtd_out = fill_spill_merge(
    wtd=wtd,
    topography=topography,
    ocean_level=10.0,
    nodata=np.nan,
)
```

### 2. Reuse hierarchy across multiple runs

```python
import numpy as np
from fillspillmerge import fill_spill_merge

# First call: build and return hierarchy
wtd_out_1, hier = fill_spill_merge(
    wtd=wtd_1,
    topography=topography,
    ocean_level=10.0,
    nodata=np.nan,
    return_hierarchy=True,
)

# Next calls: reuse hierarchy (faster)
wtd_out_2 = fill_spill_merge(wtd=wtd_2, hierarchy=hier, nodata=np.nan)
wtd_out_3 = fill_spill_merge(wtd=wtd_3, hierarchy=hier, nodata=np.nan)
```

## Usage (xarray)

If you pass an `xarray.DataArray`, output will also be a `DataArray`.

```python
import xarray as xr
from fillspillmerge import fill_spill_merge

topo_da = xr.open_dataarray("topography.nc")
wtd_da = xr.zeros_like(topo_da)

out_da, hier = fill_spill_merge(
    wtd=wtd_da,
    topography=topo_da,
    ocean_level=10.0,
    nodata=np.nan,
    return_hierarchy=True,
)

# Reuse hierarchy with another DataArray
out_da2 = fill_spill_merge(wtd=wtd_da + 1.0, hierarchy=hier, nodata=np.nan)
```

Output DataArray keeps dims/coords from the xarray input used as template.

## Example run

```bash
PYTHONPATH=build .venv/bin/python examples/run_fsm_example.py
```

## Notebook notes

Set OpenMP environment variables before importing/running if kernels are unstable:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
```

## Common issues

- `No OCEAN cells found...`: `ocean_level` does not classify any edge-connected ocean cells.
- Kernel/process crash with NaNs: pass `nodata=np.nan` consistently.
- `TypeError: cannot pickle '_fsm.Hierarchy' object`: do not store hierarchy in xarray attrs; keep it in a Python variable/cache.
