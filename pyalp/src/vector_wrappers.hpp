#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <exception>

#include <inttypes.h>

#include <graphblas.hpp>

#include <graphblas/nonzeroStorage.hpp>
#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>

#include <graphblas/utils/iterators/nonzeroIterator.hpp>


namespace py = pybind11;


using BaseScalarType = double;
#ifdef _CG_COMPLEX
 using ScalarType = std::complex< BaseScalarType >;
#else
 using ScalarType = BaseScalarType;
#endif

void buildVector(grb::Vector< ScalarType >& V, py::array_t<ScalarType> arrv) {

    // Check array is 1D
    py::buffer_info info_v = arrv.request();
    if (info_v.ndim != 1) throw std::runtime_error("Array must be 1D");
    ScalarType* data_ptr_v = static_cast<ScalarType*>(info_v.ptr);

    grb::RC io_rc;
    (void)io_rc;
    io_rc = grb::buildVector( V, data_ptr_v, data_ptr_v + info_v.size, grb::SEQUENTIAL );
    assert( io_rc == grb::SUCCESS );
}

py::array_t<ScalarType>
to_numpy(grb::Vector< ScalarType >& x) {
    grb::PinnedVector< ScalarType > pinnedVector;
    pinnedVector = grb::PinnedVector< ScalarType >( x, grb::SEQUENTIAL );

    std::cout << "create numpy array from grb::vector\n";

    ScalarType* data = new ScalarType[grb::size(x)];
    for( size_t k = 0; k < grb::size(x); ++k ) {
	const auto &value = pinnedVector.getNonzeroValue( k );
	data[k]=value;
    }

    // Capsule to manage memory (will delete[] when array is destroyed in Python)
    py::capsule free_when_done(data, [](void *f) {
	delete[] reinterpret_cast<ScalarType*>(f);
    });

    // Create NumPy array that shares memory with C++
    py::array_t<ScalarType> arr({grb::size(x)}, {sizeof(ScalarType)}, data, free_when_done);
    return arr;

}
