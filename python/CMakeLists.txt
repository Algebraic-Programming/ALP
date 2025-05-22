cmake_minimum_required(VERSION 3.14)
project(python_module LANGUAGES CXX)

add_subdirectory(extern/pybind11)

pybind11_add_module(hello hello.cpp)
