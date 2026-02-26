// Common pybind11 bindings shared by CMake targets and setuptools builds.
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <graphblas.hpp>

#include "utils.hpp"
#include "matrix_wrappers.hpp"
#include "vector_wrappers.hpp"
#include "conjugate_gradient.hpp"
#include "simulated_annealing_replica_exchange.hpp"

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

    if (ModuleLocal) {
        py::class_<grb::Matrix< ScalarType >>(m, "Matrix", py::module_local())
        .def(py::init([](size_t m_, size_t n_,
                py::array data1,
                py::array data2,
                py::array_t<ScalarType> data3) {
            return matrix_factory<ScalarType>(m_, n_, data1, data2, data3);
        }),
         py::arg("m"), py::arg("n"),
         py::arg("i_array"), py::arg("j_array"), py::arg("k_array"));

    // Expose a COO serializer so Matrix instances can be moved between
    // modules/processes without depending on pybind11 cross-module
    // type registration. Returns (i_array, j_array, values_array, nrows, ncols).
    m.def("matrix_to_coo", &matrix_to_coo<ScalarType>, "Serialize Matrix to COO arrays");

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
        .def("to_numpy", &to_numpy< ScalarType >, "Convert to numpy array");

        py::class_<grb::Vector< StateType >>(m, "State", py::module_local())
        .def(py::init<size_t>())
        .def(py::init([](size_t m,
                             py::array_t<StateType> data3) {
                grb::Vector< StateType > vec(m); // call the basic constructor
                buildVector(vec, data3); // initialize with data
                return vec;
            }),
         py::arg("m"),
         py::arg("k_array")
        )
        .def("to_numpy", &to_numpy< StateType >, "Convert to numpy array");
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

        // Expose the matrix_to_coo helper in the non-module_local case as well.
        m.def("matrix_to_coo", &matrix_to_coo<ScalarType>, "Serialize Matrix to COO arrays");

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
        .def("to_numpy", &to_numpy< ScalarType >, "Convert to numpy array");

        py::class_<grb::Vector< StateType >>(m, "State")
        .def(py::init<size_t>())
        .def(py::init([](size_t m,
                             py::array_t<StateType> data3) {
                grb::Vector< StateType > vec(m); // call the basic constructor
                buildVector(vec, data3); // initialize with data
                return vec;
            }),
         py::arg("m"),
         py::arg("k_array")
        )
        .def("to_numpy", &to_numpy< StateType >, "Convert to numpy array");

    }
	
	py::class_< std::vector<grb::Vector<StateType>> >(m, "stdVectorStates")
        .def(py::init<size_t>())
		.def(py::init([](
						size_t m,
						py::array_t< StateType > arr ) {
				(void) m;

				const py::buffer_info buf = arr.request();
				
				if (buf.ndim != 2) {
					throw std::runtime_error("Input array must be 2-dimensional");
				}
				const size_t sz = buf.shape[0];
                std::vector< grb::Vector<StateType> > vec (sz);
				assert( static_cast<size_t>( buf.shape[1] ) <= m );
				auto ptr = static_cast<StateType*>(buf.ptr);

    			grb::RC io_rc = grb::SUCCESS;
				for (size_t i = 0; i < sz; i++) {
					io_rc = io_rc ? io_rc :
						grb::buildVector( vec[i], ptr, ptr + buf.shape[1], grb::SEQUENTIAL );
				}
                return vec;
            }),
         py::arg("m"),
         py::arg("k_array")
		 );
        // .def("get_vec", & );

    m.def("buildVector", &buildVector<ScalarType>, "Fill Vector from 1 NumPy array");
    m.def("buildVectorInt8", &buildVector<StateType>, "Fill Vector from 1 NumPy array");
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
    m.def("SARE_QUBO", &SARE_QUBO, "Pass alp data to alp SARE_QUBO solver",
      py::arg("Q"),
      py::arg("states"),
      py::arg("energies"),
      py::arg("betas"),
      py::arg("best_state"),
      py::arg("solver_iterations") = 100,
      py::arg("seed") = 0,
      py::arg("verbose") = 0
    );

    m.def("SARE_Ising", &SARE_Ising, "Pass alp data to alp SARE_Ising solver",
      py::arg("Q"),
      py::arg("h"),
      py::arg("states"),
      py::arg("energies"),
      py::arg("betas"),
      py::arg("best_state"),
      py::arg("solver_iterations") = 100,
      py::arg("seed") = 0,
      py::arg("verbose") = 0
    );

}
