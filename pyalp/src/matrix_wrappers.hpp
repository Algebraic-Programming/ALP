#include <stdexcept>
#include <vector>
#include <exception>

#include <graphblas.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>
#include <graphblas/utils/iterators/nonzeroIterator.hpp>

namespace py = pybind11;

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
    io_rc = grb::buildMatrixUnique( M, data_ptr_i, data_ptr_i + nnz, data_ptr_j, data_ptr_j + nnz, data_ptr_v, data_ptr_v + nnz, grb::SEQUENTIAL );
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
    auto try_type = [&](const auto dummy) {
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

// Convert a GraphBLAS matrix to COO (i, j, values) numpy arrays and return
// a tuple: (i_array, j_array, values_array, nrows, ncols)
template <typename ScalarType>
py::tuple matrix_to_coo(grb::Matrix<ScalarType> &M) {
    // Iterate using the matrix const iterators directly. Using the
    // nonzeroIterator adapter here triggered instantiation issues due to
    // incomplete iterator types in some compilation units. Iterating via the
    // matrix's own const_iterator works across backends and avoids the
    // incomplete-type problem.
    std::vector<size_t> rows;
    std::vector<size_t> cols;
    std::vector<ScalarType> vals;

    for (auto it = M.cbegin(); it != M.cend(); ++it) {
        // Dereferenced iterator is expected to be a pair where the first
        // element contains a pair (i,j) and the second element is the value.
        // This matches the ALP/GraphBLAS iterator contract used by backends.
        auto entry = *it;
        rows.push_back( static_cast<size_t>( entry.first.first ) );
        cols.push_back( static_cast<size_t>( entry.first.second ) );
        vals.push_back( static_cast<ScalarType>( entry.second ) );
    }

    // Create numpy arrays (copies are fine for interoperability)
    py::array_t<size_t> i_arr(rows.size());
    py::buffer_info i_info = i_arr.request();
    size_t *i_ptr = static_cast<size_t*>(i_info.ptr);
    for (size_t k = 0; k < rows.size(); ++k) i_ptr[k] = rows[k];

    py::array_t<size_t> j_arr(cols.size());
    py::buffer_info j_info = j_arr.request();
    size_t *j_ptr = static_cast<size_t*>(j_info.ptr);
    for (size_t k = 0; k < cols.size(); ++k) j_ptr[k] = cols[k];

    py::array_t<ScalarType> v_arr(vals.size());
    py::buffer_info v_info = v_arr.request();
    ScalarType *v_ptr = static_cast<ScalarType*>(v_info.ptr);
    for (size_t k = 0; k < vals.size(); ++k) v_ptr[k] = vals[k];

    size_t nrows = grb::nrows(M);
    size_t ncols = grb::ncols(M);

    return py::make_tuple(i_arr, j_arr, v_arr, nrows, ncols);
}
