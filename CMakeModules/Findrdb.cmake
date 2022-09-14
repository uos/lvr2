#*
#*******************************************************************************
#*
#*  Copyright 2022 RIEGL Laser Measurement Systems
#*
#*  Licensed under the Apache License, Version 2.0 (the "License");
#*  you may not use this file except in compliance with the License.
#*  You may obtain a copy of the License at
#*
#*      http://www.apache.org/licenses/LICENSE-2.0
#*
#*  Unless required by applicable law or agreed to in writing, software
#*  distributed under the License is distributed on an "AS IS" BASIS,
#*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#*  See the License for the specific language governing permissions and
#*  limitations under the License.
#*
#*  SPDX-License-Identifier: Apache-2.0
#*
#*******************************************************************************
#*
#*
#*******************************************************************************
#*
#* \file    rdb-config.cmake
#* \author  RIEGL LMS GmbH, Austria
#* \brief   CMake config file for RDB library C/C++ interface
#* \version 2015-12-04/RS: Initial version
#* \version 2016-06-29/GC: Compatible with cmake < 3.2
#* \version 2016-09-12/GC+NW: Improved handling of include directories
#* \version 2016-09-14/GC+NW: Check if RDB targets are already defined
#* \version 2017-01-10/GC: Define include and lib dir once
#* \version 2017-07-05/GC: MINGW support
#* \version 2018-04-11/AW: macOS support
#* \version 2018-04-11/GC: Fixed C++11 requirements
#* \version 2018-04-12/NW: Fix CMake version detection, check for GNU extensions
#* \version 2020-04-14/AW: Add target rdbcpp-rtl for explicit run-time linking
#* \version 2021-09-09/GC+NW: Mark includes as SYSTEM includes (#3991)
#*
#*  In the project's CMakeLists.txt file write:
#*
#*      find_package(rdb)
#*
#*  which will define two libraries: "rdbc" and "rdbcpp" that
#*  can be used in the target_link_libraries command like so:
#*
#*      target_link_libraries( my_program rdbcpp )
#*
#*  Alternative approach with explicit run-time linking:
#*
#*      target_link_libraries( my_program rdbcpp-rtl )
#*
#*  In this case, the library is not automatically loaded on application startup
#*  and must be manually loaded by calling the function riegl::rdb::library::load().
#*
#*******************************************************************************
#*

message(STATUS "rdb: providing targets rdbc, rdbcpp and rdbcpp-rtl")

# get root directory of the rdb package
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(RDB_ROOT_DIR ${_IMPORT_PREFIX}/ DIRECTORY)
get_filename_component(RDB_ROOT_DIR ${RDB_ROOT_DIR}/ DIRECTORY)

# define include and library dir
set (RDB_INCLUDE_DIR     "${RDB_ROOT_DIR}/interface")
set (RDB_INCLUDE_C_DIR   "${RDB_INCLUDE_DIR}/c")
set (RDB_INCLUDE_CPP_DIR "${RDB_INCLUDE_DIR}/cpp")
set (RDB_LIB_DIR         "${RDB_ROOT_DIR}/library")

# create imported target rdbc
if(TARGET rdbc)
    get_target_property(_old_include_dir rdbc INTERFACE_INCLUDE_DIRECTORIES)
    string(COMPARE EQUAL "${_old_include_dir}" "${RDB_INCLUDE_C_DIR}" _matches)
    if(NOT _matches)
        message(FATAL_ERROR "Target rdbc was previously defined with a different INCLUDE_DIRECTORY
			old: ${_old_include_dir}
			new: ${RDB_INCLUDE_C_DIR}")
    endif()
else()
    add_library(rdbc SHARED IMPORTED GLOBAL)
    if(WIN32)
        set_property(TARGET rdbc PROPERTY IMPORTED_LOCATION "${RDB_LIB_DIR}/rdblib.dll")
        set_property(TARGET rdbc PROPERTY IMPORTED_IMPLIB   "${RDB_LIB_DIR}/rdblib.lib")
    elseif(APPLE)
        set_property(TARGET rdbc PROPERTY IMPORTED_LOCATION "${RDB_LIB_DIR}/librdb.dylib")
        set_property(TARGET rdbc PROPERTY IMPORTED_IMPLIB   "${RDB_LIB_DIR}/librdb.dylib")
    elseif(UNIX)
        set_property(TARGET rdbc PROPERTY IMPORTED_LOCATION "${RDB_LIB_DIR}/librdb.so")
        set_property(TARGET rdbc PROPERTY IMPORTED_IMPLIB   "${RDB_LIB_DIR}/librdb.so")
    else()
        message(FATAL_ERROR "Platform not supported.")
    endif()
    # add libraries for rdbc
    set_property(TARGET rdbc PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES
            "${RDB_INCLUDE_C_DIR}"
            )
endif()

# create compiled target rdbcpp
if(TARGET rdbcpp)
    get_target_property(_old_include_dir rdbcpp INTERFACE_INCLUDE_DIRECTORIES)
    string(COMPARE EQUAL "${_old_include_dir}" "${RDB_INCLUDE_CPP_DIR}" _matches)
    if(NOT _matches)
        message(FATAL_ERROR "Target rdbcpp was previously defined with a different INCLUDE_DIRECTORY
			old: ${_old_include_dir}
			new: ${RDB_INCLUDE_CPP_DIR}")
    endif()
else()
    add_library(rdbcpp STATIC
            "../../rdb_test/rdblib-2.4.0-x86_64-linux/interface/cpp/riegl/rdb.cpp"
            )
    # add libraries for rdbcpp
    target_include_directories(rdbcpp SYSTEM PUBLIC
            "${RDB_INCLUDE_CPP_DIR}"
            )
    target_link_libraries(rdbcpp PUBLIC rdbc)
    if(NOT MSVC AND NOT MINGW)
        set_target_properties(rdbcpp PROPERTIES COMPILE_FLAGS "-fPIC")
    endif()
endif()

# create compiled target rdbcpp-rtl (for explicit run-time linking)
if(TARGET rdbcpp-rtl)
    get_target_property(_old_include_dir rdbcpp-rtl INTERFACE_INCLUDE_DIRECTORIES)
    string(COMPARE EQUAL "${_old_include_dir}" "${RDB_INCLUDE_C_DIR};${RDB_INCLUDE_CPP_DIR}" _matches)
    if(NOT _matches)
        message(FATAL_ERROR "Target rdbcpp-rtl was previously defined with a different INCLUDE_DIRECTORY
			old: ${_old_include_dir}
			new: ${RDB_INCLUDE_C_DIR};${RDB_INCLUDE_CPP_DIR}")
    endif()
else()
    add_library(rdbcpp-rtl STATIC
            "${RDB_INCLUDE_CPP_DIR}/riegl/rdb.cpp"
            )
    target_include_directories(rdbcpp-rtl SYSTEM PUBLIC
            "${RDB_INCLUDE_C_DIR}"
            "${RDB_INCLUDE_CPP_DIR}"
            )
    target_compile_definitions(rdbcpp-rtl PUBLIC
            RDB_RUNTIME_LINKING
            )
    if(NOT MSVC AND NOT MINGW)
        set_target_properties(rdbcpp-rtl PROPERTIES COMPILE_FLAGS "-fPIC")
        target_link_libraries(rdbcpp-rtl PRIVATE -ldl)
    endif()
endif()

set(_IMPORT_PREFIX)

if(CMAKE_VERSION VERSION_EQUAL 3.8 OR CMAKE_VERSION VERSION_GREATER 3.8)
    # require C++11, which is supported only in cmake >= 3.8
    target_compile_features(rdbcpp PUBLIC cxx_std_11)
    target_compile_features(rdbcpp-rtl PUBLIC cxx_std_11)
elseif(CMAKE_VERSION VERSION_EQUAL 3.2 OR CMAKE_VERSION VERSION_GREATER 3.2)
    # require C++11, which is supported only in cmake >= 3.2
    # require C++11 by requiring rvalue reference (introduced in C++11)
    target_compile_features(rdbcpp PUBLIC cxx_rvalue_references)
    target_compile_features(rdbcpp-rtl PUBLIC cxx_rvalue_references)
elseif(NOT MSVC)
    include(CheckCXXCompilerFlag)
    if(CMAKE_CXX_EXTENSIONS) # should gnu* extensions be used?
        CHECK_CXX_COMPILER_FLAG("-std=gnu++11" COMPILER_SUPPORTS_GNUXX11)
        CHECK_CXX_COMPILER_FLAG("-std=gnu++0x" COMPILER_SUPPORTS_GNUXX0X)
    endif()
    CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
    CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

    if(CMAKE_CXX_EXTENSIONS AND COMPILER_SUPPORTS_GNUXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    elseif(CMAKE_CXX_EXTENSIONS AND COMPILER_SUPPORTS_GNUXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
    elseif(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
    endif()
endif()
