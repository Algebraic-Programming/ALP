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


template< typename T >
void buildVector(grb::Vector< T >& V, py::array_t<T> arrv) {

    // Check array is 1D
    py::buffer_info info_v = arrv.request();
    if (info_v.ndim != 1) throw std::runtime_error("Array must be 1D");
    T* data_ptr_v = static_cast<T*>(info_v.ptr);

    grb::RC io_rc;
    (void)io_rc;
    io_rc = grb::buildVector( V, data_ptr_v, data_ptr_v + info_v.size, grb::SEQUENTIAL );
    assert( io_rc == grb::SUCCESS );
}

template< typename T >
py::array_t< T > to_numpy( const grb::Vector< T >& x ) {
    grb::PinnedVector< T > pinnedVector;
    pinnedVector = grb::PinnedVector< T >( x, grb::SEQUENTIAL );
	const size_t sz = pinnedVector.size();

    std::cout << "create numpy array from grb::vector\n";

	auto result = py::array_t<T>(sz);
	py::buffer_info buf = result.request();
	T* ptr = static_cast< T* >( buf.ptr );

    for( size_t k = 0; k < sz; ++k ) {
		ptr[ k ] = 0;
	}
    for( size_t k = 0; k < pinnedVector.nonzeroes(); ++k ) {
		const auto &value = pinnedVector.getNonzeroValue( k );
		ptr[pinnedVector.getNonzeroIndex( k )] = value;
    }

    return result;
}
