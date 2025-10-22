#include <stdexcept>
#include <vector>
#include <exception>

#include <graphblas.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>
#include <graphblas/utils/iterators/nonzeroIterator.hpp>

namespace py = pybind11;

// using BaseScalarType = double;
// #ifdef _CG_COMPLEX
//  using ScalarType = std::complex< BaseScalarType >;
// #else
//  using ScalarType = BaseScalarType;
// #endif

// /** Parser type */
// typedef grb::utils::MatrixFileReader<
// 	ScalarType,
// 	std::conditional<
// 		(sizeof(grb::config::RowIndexType) > sizeof(grb::config::ColIndexType)),
// 		grb::config::RowIndexType,
// 		grb::config::ColIndexType
// 	>::type
// > Parser;

// /** Nonzero type */
// typedef grb::internal::NonzeroStorage<
// 	grb::config::RowIndexType,
// 	grb::config::ColIndexType,
// 	ScalarType
// > NonzeroT;

// /** In-memory storage type */
// typedef grb::utils::Singleton<
// 	std::pair<
// 		// stores n and nz (according to parser)
// 		std::pair< size_t, size_t >,
// 		// stores the actual nonzeroes
// 		std::vector< NonzeroT >
// 	>
// > Storage;

template<
    typename IntType
    , typename ScalarType
    >
void buildMatrix(
    grb::Matrix< ScalarType >& M,
    py::array_t<IntType> arri,
    py::array_t<IntType> arrj,
    py::array_t<ScalarType> arrv
    ) {
    // Check array is 1D
    py::buffer_info info_i = arri.request();
    if (info_i.ndim != 1) throw std::runtime_error("Array must be 1D");
    IntType* data_ptr_i = static_cast<IntType*>(info_i.ptr);
    auto nnz = info_i.size;

    // Check array is 1D
    py::buffer_info info_j = arrj.request();
    if (info_j.ndim != 1) throw std::runtime_error("Array must be 1D");
    IntType* data_ptr_j = static_cast<IntType*>(info_j.ptr);
    assert( nnz == info_j.size );

    // Check array is 1D
    py::buffer_info info_v = arrv.request();
    if (info_v.ndim != 1) throw std::runtime_error("Array must be 1D");
    ScalarType* data_ptr_v = static_cast<ScalarType*>(info_v.ptr);
    assert( nnz == info_v.size );

    grb::RC io_rc;
    (void)io_rc;
    io_rc = grb::buildMatrixUnique( M, data_ptr_i, data_ptr_j , data_ptr_v, nnz, grb::SEQUENTIAL );
    assert( io_rc == grb::SUCCESS );
}

// helper for template specialisation
template <typename ScalarType>
grb::Matrix<ScalarType> matrix_factory(
    size_t m, size_t n,
    py::array data1,
    py::array data2,
    py::array_t<ScalarType> data3)
{
    grb::Matrix<ScalarType> mat(m, n);

    // Helper for dispatch
    bool handled = false;
    auto try_type = [&](auto dummy) {
        using IntType = decltype(dummy);
        if (py::dtype::of<IntType>().is(data1.dtype()) && py::dtype::of<IntType>().is(data2.dtype())) {
            buildMatrix<IntType, ScalarType>(
                mat,
                data1.cast<py::array_t<IntType>>(),
                data2.cast<py::array_t<IntType>>(),
                data3
            );
            handled = true;
        }
    };

    // List of supported integer types
    (try_type(int8_t{}), try_type(int16_t{}), try_type(int32_t{}), try_type(int64_t{}),
     try_type(uint8_t{}), try_type(uint16_t{}), try_type(uint32_t{}), try_type(uint64_t{}));

    if (!handled)
        throw std::runtime_error("Unsupported integer dtype for data1/data2 or nonmatching types of data1 and data2 ");

    return mat;
}
