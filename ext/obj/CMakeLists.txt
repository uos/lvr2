cmake_minimum_required(VERSION 2.8)

PROJECT(obj)

ADD_LIBRARY(obj objLoader.cpp list.cpp obj_parser.cpp string_extra.cpp)
set_target_properties(obj PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

