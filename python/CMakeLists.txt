cmake_minimum_required(VERSION 3.14)
project(python_module LANGUAGES CXX)

add_subdirectory(extern/pybind11)

pybind11_add_module(python2alp python2alp.cpp)
pybind11_add_module(numpy2alp numpy2alp.cpp)
