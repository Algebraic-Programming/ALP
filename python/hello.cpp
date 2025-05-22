#include <pybind11/pybind11.h>

std::string say_hello() {
    return "Hello, world from C++!";
}

PYBIND11_MODULE(hello, m) {
    m.def("say_hello", &say_hello, "A function that returns a greeting");
}
