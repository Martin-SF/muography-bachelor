cmake_minimum_required(VERSION 3.4)
project(pybinding_EcoMug)
add_subdirectory(pybind11)
pybind11_add_module(EcoMug pybinding_EcoMug.cpp)