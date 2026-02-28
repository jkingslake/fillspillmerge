# fillspillmerge

Python wrapper around the Fill-Spill-Merge algorithm, specifically Kerry Callaghan's fork which includes the capability to save and reuse the depression hierarchy ([repo](https://github.com/KCallaghan/Barnes2020-FillSpillMerge)).

This repo keeps `Barnes2020-FillSpillMerge` as a submodule, and keeps the
Python wrapper implementation in this repo (not inside the submodule).

## Layout

- `Barnes2020-FillSpillMerge/`: upstream algorithm code (submodule)
- `src/pybind_module.cpp`: C++ pybind11 wrapper module source (`_fsm`)
- `fillspillmerge/`: Python API package (`fill_spill_merge`)
- `examples/`: example arrays, script, and notebooks

## Setup

```bash
cd /Users/jkingslake/Documents/science/meltwater_routing/fsm_pywrapper/fillspillmerge
uv sync
```

Optional xarray support:

```bash
uv sync --extra xarray
```

## Build extension

```bash
cmake -S . -B build -DPython3_EXECUTABLE=.venv/bin/python
cmake --build build -j --target _fsm
```

## Use in Python

```python
from fillspillmerge import fill_spill_merge

# Build hierarchy on first call
wtd_out, dh = fill_spill_merge(
    wtd=wtd,
    topography=topography,
    ocean_level=ocean_level,
    nodata=np.nan,
    return_hierarchy=True,
)

# Reuse hierarchy on later calls
wtd_out2 = fill_spill_merge(wtd=wtd2, hierarchy=dh, nodata=np.nan)
```

If `wtd` is omitted, it defaults to a zero raster of the hierarchy shape.

If you pass an `xarray.DataArray` as `topography` or `wtd` (and `xarray` is installed), `fill_spill_merge` returns an `xarray.DataArray` with matching dims/coords.

## Example run

```bash
PYTHONPATH=build .venv/bin/python examples/run_fsm_example.py
```
