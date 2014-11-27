cmake_minimum_required(VERSION 2.8)

PROJECT(rply)


ADD_LIBRARY(lvrrply_static STATIC rply.c)
#set_target_properties(rply_local_static PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

ADD_LIBRARY(lvrrply SHARED rply.c)
#set_target_properties(rply_local PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
