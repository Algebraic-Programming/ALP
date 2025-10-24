#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <exception>
#include <iostream>
#include <vector>

namespace py = pybind11;

// Print a NumPy array as a std::vector (flattened)
void print_my_numpy_array(py::array_t<double> input) {
    py::buffer_info buf = input.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<double> vec(ptr, ptr + buf.size);

    std::cout << "Vector contents (flattened): ";
    for (double v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}
