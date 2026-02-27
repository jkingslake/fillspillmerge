cmake_minimum_required(VERSION 3.16)
project(py_fill_spill_merge LANGUAGES C CXX)

set(RICHDEM_NO_PROGRESS ON CACHE BOOL "Whether or not to show RichDEM progress bars")

find_package(Boost COMPONENTS serialization system REQUIRED)

add_subdirectory(Barnes2020-FillSpillMerge/submodules/dephier EXCLUDE_FROM_ALL)

add_library(fsm_headers INTERFACE)
target_include_directories(fsm_headers INTERFACE Barnes2020-FillSpillMerge/include)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT
    OUTPUT_VARIABLE PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(PYBIND11_CMAKEDIR_RESULT EQUAL 0)
    find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKEDIR}" NO_DEFAULT_PATH)
  else()
    message(FATAL_ERROR "pybind11 not found. Install it with 'uv add pybind11'.")
  endif()
endif()

pybind11_add_module(_fsm MODULE src/pybind_module.cpp)
target_link_libraries(_fsm PRIVATE fsm_headers dephier richdem Boost::serialization)
target_compile_features(_fsm PRIVATE cxx_std_17)
