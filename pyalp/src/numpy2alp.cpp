#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// Print a NumPy array as a std::vector (flattened)
void print_numpy_array(py::array_t<double> input) {
    py::buffer_info buf = input.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<double> vec(ptr, ptr + buf.size);

    std::cout << "Vector contents (flattened): ";
    for (double v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

// Add two NumPy arrays (supports multi-dimensional, as long as shapes match)
py::array_t<double> add_numpy_arrays(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info buf_a = a.request();
    py::buffer_info buf_b = b.request();

    // Check that shapes match
    if (buf_a.ndim != buf_b.ndim)
        throw std::runtime_error("Input arrays must have the same number of dimensions");
    for (ssize_t i = 0; i < buf_a.ndim; ++i) {
        if (buf_a.shape[i] != buf_b.shape[i])
            throw std::runtime_error("Input array shapes must match");
    }

    // Prepare output array with the same shape
    auto result = py::array_t<double>(buf_a.size);
    py::buffer_info buf_result = result.request();

    double* ptr_a = static_cast<double*>(buf_a.ptr);
    double* ptr_b = static_cast<double*>(buf_b.ptr);
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    // Element-wise addition (flat)
    for (ssize_t i = 0; i < buf_a.size; ++i) {
        ptr_result[i] = ptr_a[i] + ptr_b[i];
    }

    // Reshape result to match input shape
    result.resize(buf_a.shape);

    return result;
}

PYBIND11_MODULE(numpy2alp, m) {
    m.def("print_numpy_array", &print_numpy_array, "Print a numpy array as a flattened std::vector");
    m.def("add_numpy_arrays", &add_numpy_arrays, "Add two numpy arrays element-wise (supports multi-dimensional arrays)");
}
