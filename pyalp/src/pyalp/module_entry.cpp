#include <pybind11/pybind11.h>
#include "common_bindings.hpp"

#ifndef PYALP_MODULE_LOCAL
#define PYALP_MODULE_LOCAL 1
#endif

PYBIND11_MODULE(PYALP_MODULE_NAME, m) {
#if PYALP_MODULE_LOCAL
    register_pyalp<true>(m);
#else
    register_pyalp<false>(m);
#endif
}
