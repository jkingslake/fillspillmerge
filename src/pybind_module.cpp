#include <fsm/fill_spill_merge.hpp>

#include <richdem/common/Array2D.hpp>
#include <richdem/misc/misc_methods.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;
namespace rd = richdem;
namespace dh = richdem::dephier;

namespace {

rd::Array2D<double> array2d_from_numpy(
  const py::array_t<double, py::array::c_style | py::array::forcecast> &arr,
  const py::object &nodata_obj
) {
  auto buf = arr.request();
  if (buf.ndim != 2) {
    throw std::invalid_argument("Input array must be 2D");
  }

  const auto rows = static_cast<int>(buf.shape[0]);
  const auto cols = static_cast<int>(buf.shape[1]);
  rd::Array2D<double> out(cols, rows, 0.0);

  const bool use_nodata = !nodata_obj.is_none();
  double nodata = 0.0;
  if (use_nodata) {
    nodata = nodata_obj.cast<double>();
    out.setNoData(nodata);
  }

  auto *ptr = static_cast<const double *>(buf.ptr);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const double v = ptr[static_cast<size_t>(y) * static_cast<size_t>(cols) + static_cast<size_t>(x)];
      if (use_nodata && ((std::isnan(nodata) && std::isnan(v)) || (!std::isnan(nodata) && v == nodata))) {
        out(x, y) = nodata;
      } else {
        out(x, y) = v;
      }
    }
  }

  return out;
}

py::array_t<double> numpy_from_array2d(const rd::Array2D<double> &arr) {
  py::array_t<double> out({arr.height(), arr.width()});
  auto buf = out.request();
  auto *ptr = static_cast<double *>(buf.ptr);

  for (int y = 0; y < arr.height(); ++y) {
    for (int x = 0; x < arr.width(); ++x) {
      ptr[static_cast<size_t>(y) * static_cast<size_t>(arr.width()) + static_cast<size_t>(x)] = arr(x, y);
    }
  }

  return out;
}

class CachedHierarchy {
 public:
  CachedHierarchy(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &topography,
    const double ocean_level,
    const py::object &nodata
  ) : topo_(array2d_from_numpy(topography, nodata)),
      label_(topo_.width(), topo_.height(), dh::NO_DEP),
      flowdirs_(topo_.width(), topo_.height(), rd::NO_FLOW),
      ocean_level_(ocean_level),
      has_nodata_(!nodata.is_none()),
      nodata_(has_nodata_ ? nodata.cast<double>() : 0.0) {

    rd::BucketFillFromEdges<rd::Topology::D8>(topo_, label_, ocean_level_, dh::OCEAN);

    unsigned int ocean_cell_count = 0;
    #pragma omp parallel for reduction(+:ocean_cell_count)
    for (unsigned int i = 0; i < label_.size(); ++i) {
      if (topo_.isNoData(i) || label_(i) == dh::OCEAN) {
        label_(i) = dh::OCEAN;
        ocean_cell_count += 1;
      }
    }

    if (ocean_cell_count == 0) {
      throw std::runtime_error("No OCEAN cells found, could not make a DepressionHierarchy!");
    }

    deps_ = dh::GetDepressionHierarchy<double, rd::Topology::D8>(topo_, label_, flowdirs_);
  }

  py::array_t<double> run(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &wtd,
    const py::object &nodata
  ) {
    py::object effective_nodata = nodata;
    if (effective_nodata.is_none() && has_nodata_) {
      effective_nodata = py::float_(nodata_);
    }

    auto wtd_arr = array2d_from_numpy(wtd, effective_nodata);

    if (wtd_arr.width() != topo_.width() || wtd_arr.height() != topo_.height()) {
      throw std::invalid_argument("wtd shape must match the shape used to build the hierarchy");
    }

    if (!effective_nodata.is_none()) {
      wtd_arr.setNoData(effective_nodata.cast<double>());
    }

    #pragma omp parallel for
    for (unsigned int i = 0; i < label_.size(); ++i) {
      if (topo_.isNoData(i) || label_(i) == dh::OCEAN) {
        wtd_arr(i) = 0;
      }
    }

    dh::FillSpillMerge(topo_, label_, flowdirs_, deps_, wtd_arr);
    return numpy_from_array2d(wtd_arr);
  }

  int width() const { return topo_.width(); }
  int height() const { return topo_.height(); }
  double ocean_level() const { return ocean_level_; }

 private:
  rd::Array2D<double> topo_;
  rd::Array2D<dh::dh_label_t> label_;
  rd::Array2D<rd::flowdir_t> flowdirs_;
  dh::DepressionHierarchy<double> deps_;
  double ocean_level_;
  bool has_nodata_;
  double nodata_;
};

}  // namespace

PYBIND11_MODULE(_fsm, m) {
  m.doc() = "Python bindings for Fill-Spill-Merge";

  py::class_<CachedHierarchy>(m, "Hierarchy")
    .def(
      py::init<const py::array_t<double, py::array::c_style | py::array::forcecast> &, double, const py::object &>(),
      py::arg("topography"),
      py::arg("ocean_level"),
      py::arg("nodata") = py::none(),
      "Build and cache the depression hierarchy for a fixed topography."
    )
    .def(
      "run",
      &CachedHierarchy::run,
      py::arg("wtd"),
      py::arg("nodata") = py::none(),
      "Run FSM using cached hierarchy (fast path for repeated runs)."
    )
    .def_property_readonly("width", &CachedHierarchy::width)
    .def_property_readonly("height", &CachedHierarchy::height)
    .def_property_readonly("ocean_level", &CachedHierarchy::ocean_level);

  m.def(
    "build_hierarchy",
    [](const py::array_t<double, py::array::c_style | py::array::forcecast> &topography,
       const double ocean_level,
       const py::object &nodata) {
      return CachedHierarchy(topography, ocean_level, nodata);
    },
    py::arg("topography"),
    py::arg("ocean_level"),
    py::arg("nodata") = py::none(),
    "Build and return a reusable Hierarchy object."
  );

  m.def(
    "run",
    [](const py::array_t<double, py::array::c_style | py::array::forcecast> &topography,
       const py::array_t<double, py::array::c_style | py::array::forcecast> &wtd,
       const double ocean_level,
       const py::object &nodata) {
      CachedHierarchy cached(topography, ocean_level, nodata);
      return cached.run(wtd, nodata);
    },
    py::arg("topography"),
    py::arg("wtd"),
    py::arg("ocean_level"),
    py::arg("nodata") = py::none(),
    R"pbdoc(
Run Fill-Spill-Merge entirely in memory.

Args:
    topography: 2D numpy array of elevations.
    wtd: 2D numpy array of initial water table depth / standing water.
    ocean_level: Elevation threshold for edge-connected ocean cells.
    nodata: Optional nodata scalar; use None to disable nodata handling.

Returns:
    2D numpy array containing updated WTD after FSM.
)pbdoc"
  );
}
