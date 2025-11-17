cmake_minimum_required(VERSION 3.13)

# Eigen is header-only; we assume user cloned it into extern/eigen
# If not present, emit a message and skip target creation gracefully.
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/eigen/Eigen")
    message(WARNING "Eigen not found at extern/eigen. Run scripts/clone_eigen.sh first.")
endif()

add_library(eigen_dummy INTERFACE) # placeholder to attach include path
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/eigen/Eigen")
    target_include_directories(eigen_dummy INTERFACE ${CMAKE_SOURCE_DIR}/extern/eigen)
endif()

find_package(OpenMP REQUIRED)

add_executable(cg_eigen
    cg_eigen.cpp
    mm_loader.cpp
)

target_include_directories(cg_eigen PRIVATE ${CMAKE_SOURCE_DIR}/bench/eigen_cg)

# Eigen (recent versions) use C++14/17 features. Request C++17 for this target to avoid
# template/constexpr compile errors on modern Eigen commits.
if (NOT TARGET_COMPILE_FEATURES)
    # CMake < 3.1 fallback: set property directly
    set_property(TARGET cg_eigen PROPERTY CXX_STANDARD 17)
    set_property(TARGET cg_eigen PROPERTY CXX_STANDARD_REQUIRED ON)
else()
    target_compile_features(cg_eigen PUBLIC cxx_std_17)
endif()

# Link only OpenMP (Eigen header-only)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cg_eigen PRIVATE OpenMP::OpenMP_CXX)
endif()

# If we created the eigen_dummy interface (and it has include dirs), link it so Eigen headers are visible
if(TARGET eigen_dummy)
    target_link_libraries(cg_eigen PRIVATE eigen_dummy)
endif()
# Allow large files and optimization; user can override via CMake cache
target_compile_definitions(cg_eigen PRIVATE EIGEN_NO_DEBUG)

install(TARGETS cg_eigen RUNTIME DESTINATION bin)
