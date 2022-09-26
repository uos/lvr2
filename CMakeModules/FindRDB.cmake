unset(RDB_FOUND)
unset(RDB_INCLUDE_DIRS)
unset(RDB_LIBRARY_DIRS)
unset(RDB_LIBRARIES)

mark_as_advanced(RDB_FOUND)
mark_as_advanced(RDB_INCLUDE_DIRS)
mark_as_advanced(RDB_LIBRARY_DIRS)
mark_as_advanced(RDB_LIBRARIES)

get_filename_component(RDB_ROOT_DIR ${_IMPORT_PREFIX}/ DIRECTORY)
get_filename_component(RDB_ROOT_DIR ${RDB_ROOT_DIR}/ DIRECTORY)

# Set draco_INCLUDE_DIRS
find_path(RDB_INCLUDE_DIRS NAMES "riegl/rdb.hpp" HINTS "/usr/include/ /usr/local/incude/")

find_library(RDB_LIBRARY_IMPORT NAMES rdb.dll librdb.dylib librdb.so)
get_filename_component(RDB_LIBRARY_DIRS ${RDB_LIBRARIES} DIRECTORY)

# create imported target rdbc
if(TARGET rdbc)
else()
    add_library(rdbc SHARED IMPORTED GLOBAL)

    set_property(TARGET rdbc PROPERTY IMPORTED_LOCATION "${RDB_LIBRARY_IMPORT}")
    set_property(TARGET rdbc PROPERTY IMPORTED_IMPLIB   "${RDB_LIBRARY_IMPORT}")

    # add libraries for rdbc
    set_property(TARGET rdbc PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES
            "${RDB_INCLUDE_DIRS}"
            )
endif()

# create compiled target rdbcpp
if(TARGET rdbcpp)

else()
    add_library(rdbcpp STATIC
            "${RDB_INCLUDE_DIRS}/riegl/rdb.cpp"
            )
    # add libraries for rdbcpp
    target_include_directories(rdbcpp SYSTEM PUBLIC
            "${RDB_INCLUDE_DIRS}"
            )
    target_link_libraries(rdbcpp PUBLIC ${RDB_LIBRARIES})
    if(NOT MSVC AND NOT MINGW)
        set_target_properties(rdbcpp PROPERTIES COMPILE_FLAGS "-fPIC")
    endif()
endif()

if(CMAKE_VERSION VERSION_EQUAL 3.8 OR CMAKE_VERSION VERSION_GREATER 3.8)
    # require C++11, which is supported only in cmake >= 3.8
    target_compile_features(rdbcpp PUBLIC cxx_std_11)
elseif(CMAKE_VERSION VERSION_EQUAL 3.2 OR CMAKE_VERSION VERSION_GREATER 3.2)
    # require C++11, which is supported only in cmake >= 3.2
    # require C++11 by requiring rvalue reference (introduced in C++11)
    target_compile_features(rdbcpp PUBLIC cxx_rvalue_references)
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


set(RDB_LIBRARIES rdbc rdbcpp)

if(RDB_INCLUDE_DIRS
        AND RDB_LIBRARY_DIRS
        AND RDB_LIBRARIES)
    set(RDB_FOUND YES)
endif()

