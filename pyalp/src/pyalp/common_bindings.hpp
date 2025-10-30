// Common pybind11 bindings shared by CMake targets and setuptools builds.
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <graphblas.hpp>

#include "utils.hpp"
#include "matrix_wrappers.hpp"
#include "vector_wrappers.hpp"
#include "conjugate_gradient.hpp"

namespace py = pybind11;

// Register all pyalp bindings. Module-local registration can be enabled by
// instantiating with ModuleLocal = true. When ModuleLocal==true the
// py::module_local() policy is applied to class bindings to avoid symbol
// collisions when multiple compiled variants are imported in the same
// interpreter.
template <bool ModuleLocal>
void register_pyalp(py::module_ &m) {
    // Common bindings for all backends
    m.def("backend_name", [](){ return "backend"; });

    if constexpr (ModuleLocal) {
        py::class_<grb::Matrix< ScalarType >>(m, "Matrix", py::module_local())
        .def(py::init([](size_t m_, size_t n_,
                py::array data1,
                py::array data2,
                py::array_t<ScalarType> data3) {
            return matrix_factory<ScalarType>(m_, n_, data1, data2, data3);
        }),
         py::arg("m"), py::arg("n"),
         py::arg("i_array"), py::arg("j_array"), py::arg("k_array"));

        py::class_<grb::Vector< ScalarType >>(m, "Vector", py::module_local())
        .def(py::init<size_t>())
        .def(py::init([](size_t m,
                             py::array_t<ScalarType> data3) {
                grb::Vector< ScalarType > vec(m); // call the basic constructor
                buildVector(vec, data3); // initialize with data
                return vec;
            }),
         py::arg("m"),
         py::arg("k_array")
        )
        .def("to_numpy", &to_numpy, "Convert to numpy array");
    } else {
        py::class_<grb::Matrix< ScalarType >>(m, "Matrix")
        .def(py::init([](size_t m_, size_t n_,
                py::array data1,
                py::array data2,
                py::array_t<ScalarType> data3) {
            return matrix_factory<ScalarType>(m_, n_, data1, data2, data3);
        }),
         py::arg("m"), py::arg("n"),
         py::arg("i_array"), py::arg("j_array"), py::arg("k_array"));

        py::class_<grb::Vector< ScalarType >>(m, "Vector")
        .def(py::init<size_t>())
        .def(py::init([](size_t m,
                             py::array_t<ScalarType> data3) {
                grb::Vector< ScalarType > vec(m); // call the basic constructor
                buildVector(vec, data3); // initialize with data
                return vec;
            }),
         py::arg("m"),
         py::arg("k_array")
        )
        .def("to_numpy", &to_numpy, "Convert to numpy array");
    }

    m.def("buildVector", &buildVector, "Fill Vector from 1 NumPy array");
    m.def("print_my_numpy_array", &print_my_numpy_array, "Print a numpy array as a flattened std::vector");
    m.def("conjugate_gradient", &conjugate_gradient, "Pass alp data to alp CG solver",
      py::arg("L"),
      py::arg("x"),
      py::arg("b"),
      py::arg("r"),
      py::arg("u"),
      py::arg("temp"),
      py::arg("solver_iterations") = 1000,
      py::arg("verbose") = 0
    );
}
