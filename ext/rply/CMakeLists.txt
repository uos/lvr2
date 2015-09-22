cmake_minimum_required(VERSION 2.8)

PROJECT(rply)

find_path(RPLY_INCLUDE_DIR rply.h PATH .)

ADD_LIBRARY(lvrrply_static STATIC rply.c)

ADD_LIBRARY(lvrrply SHARED rply.c)
