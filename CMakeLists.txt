cmake_minimum_required(VERSION 3.15)

#
# Project details
#

project(
  "spdmon"
  VERSION 1.0.0
  LANGUAGES CXX
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CXX_STANDARD 17)

#
# Set project options
#

include(cmake/StandardSettings.cmake)
include(cmake/Utils.cmake)
include(cmake/cpplint.cmake)
message(STATUS "Started CMake for ${PROJECT_NAME} v${PROJECT_VERSION}...\n")


# add every sub directory of the current source dir if it contains a CMakeLists.txt
function(template_add_subdirectories source_dir)
    message(STATUS "Find all submodules in project\n")
    file(GLOB subDirectories ${source_dir}/*)
    foreach(dir ${subDirectories})
        if(IS_DIRECTORY "${dir}")
            if(EXISTS "${dir}/CMakeLists.txt")
                message(STATUS "Add ${dir} submodule\n")
                add_subdirectory("${dir}")
            endif()
        endif()
    endforeach(dir)
endfunction(template_add_subdirectories)

#
# Setup alternative names
#

if(${PROJECT_NAME}_USE_ALT_NAMES)
	string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)
	string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UPPERCASE)
else()
	set(PROJECT_NAME_LOWERCASE ${PROJECT_NAME})
	set(PROJECT_NAME_UPPERCASE ${PROJECT_NAME})
endif()

#
# Prevent building in the source directory
#

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(STATUS "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n")
endif()

#
# Enable package managers
#

include(cmake/Conan.cmake)
include(cmake/Vcpkg.cmake)
include (CTest)

#
# Include source files
#

include(cmake/SourcesAndHeaders.cmake)

#
# Add version header
#

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/version.hpp.in
    ${CMAKE_CURRENT_LIST_DIR}/include/${PROJECT_NAME_LOWERCASE}/version.hpp
    @ONLY
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME_LOWERCASE}/version.hpp
  DESTINATION
    include/${PROJECT_NAME_LOWERCASE}
)

#
# Create library, setup header and source files
#

# Find all headers and implementation files

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/spdlog)

template_add_subdirectories(${CMAKE_CURRENT_LIST_DIR}/src)


if(${PROJECT_NAME}_BUILD_EXECUTABLE)
  add_executable(${PROJECT_NAME} ${exe_sources})
  if(${sources})
    add_library(${PROJECT_NAME}_LIB ${headers} ${sources})
      target_link_libraries(${PROJECT_NAME}_LIB
          spdlog::spdlog
          ${linking_dep}
          )
  endif()
  target_link_libraries(${PROJECT_NAME}
    spdlog::spdlog
    ${linking_dep}
  )
elseif(${PROJECT_NAME}_BUILD_HEADERS_ONLY)
  add_library(${PROJECT_NAME} INTERFACE)
  target_link_libraries(${PROJECT_NAME}
    ${linking_dep}
  )
else()
  add_library(
    ${PROJECT_NAME}
    ${headers}
    ${sources}
  )
  target_link_libraries(${PROJECT_NAME}
    ${linking_dep}
  )
endif()

function (ListToString result delim)
    list(GET ARGV 2 temp)
    math(EXPR N "${ARGC}-1")
    foreach(IDX RANGE 3 ${N})
        list(GET ARGV ${IDX} STR)
        set(temp "${temp}${delim}${STR}")
    endforeach()
    set(${result} "${temp}" PARENT_SCOPE)
endfunction(ListToString)

message(STATUS "Added all header and implementation files.\n")

#
# Set the project standard and warnings
#

if(${PROJECT_NAME}_BUILD_HEADERS_ONLY)
  target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
else()
  target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
endif()
include(cmake/CompilerWarnings.cmake)
set_project_warnings(${PROJECT_NAME})

verbose_message("Applied compiler warnings. Using standard ${CXX_STANDARD}.\n")

#
# Enable Doxygen
#

include(cmake/Doxygen.cmake)

#
# Set the build/user include directories
#

# Allow usage of header files in the `src` directory, but only for utilities
if(${PROJECT_NAME}_BUILD_HEADERS_ONLY)
  target_include_directories(
    ${PROJECT_NAME}
    INTERFACE 
      $<INSTALL_INTERFACE:include>    
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
else()
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC 
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  if(${PROJECT_NAME}_BUILD_EXECUTABLE)
  if(${PROJECT_NAME}_LIB)
		target_include_directories(
			${PROJECT_NAME}_LIB
			PUBLIC 
				$<INSTALL_INTERFACE:include>    
				$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
			PRIVATE
				${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    endif()
	endif()
endif()

message(STATUS "Finished setting up include directories.")

#
# Model project dependencies 
#

# Identify and link with the specific "packages" the project uses
#find_package(package_name package_version REQUIRED package_type [other_options])
#target_link_libraries(
#  ${PROJECT_NAME}
#  PUBLIC
#    dependency1 ...
#  PRIVATE
#    dependency2 ...
#    ${PROJECT_NAME}_PROJECT_OPTIONS        
#    ${PROJECT_NAME}_PROJECT_WARNINGS        
#)

verbose_message("Successfully added all dependencies and linked against them.")

#
# Provide alias to library for 
#

if(${PROJECT_NAME}_BUILD_EXECUTABLE)
  add_executable(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
else()
  add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
endif()

verbose_message("Project is now aliased as ${PROJECT_NAME}::${PROJECT_NAME}.\n")

#
# Format the project using the `clang-format` target (i.e: cmake --build build --target clang-format)
#

add_clang_format_target()

#
# Format the project using the `cpplint` target (i.e: cmake --build build --target cpplint-target)
#

add_cpplint_style_check_target()

#
# Install library for easy downstream inclusion
#

include(GNUInstallDirs)
install(
  TARGETS
    ${PROJECT_NAME}
  EXPORT
    ${PROJECT_NAME}Targets
  LIBRARY DESTINATION
    ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION
    ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION
    ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION
    include
  PUBLIC_HEADER DESTINATION
    include
)

install(
  EXPORT 
    ${PROJECT_NAME}Targets
  FILE
    ${PROJECT_NAME}Targets.cmake
  NAMESPACE
    ${PROJECT_NAME}::
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

#
# Install the `include` directory
#

install(
  DIRECTORY
    include/${PROJECT_NAME_LOWERCASE}
  DESTINATION
    include
)

verbose_message("Install targets successfully build. Install with `cmake --build <build_directory> --target install --config <build_config>`.")

#
# Quick `ConfigVersion.cmake` creation
#

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${PROJECT_NAME}ConfigVersion.cmake
  VERSION
    ${PROJECT_VERSION}
  COMPATIBILITY
    SameMajorVersion
)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION 
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

#
# Generate export header if specified
#

if(${PROJECT_NAME}_GENERATE_EXPORT_HEADER)
  include(GenerateExportHeader)
  generate_export_header(${PROJECT_NAME})
  install(
    FILES
      ${PROJECT_BINARY_DIR}/${PROJECT_NAME_LOWERCASE}_export.h 
    DESTINATION
      include
  )

message(STATUS "Generated the export header `${PROJECT_NAME_LOWERCASE}_export.h` and installed it.")
endif()

message(STATUS "Finished building requirements for installing the package.\n")

#
# Unit testing setup
#

if(${PROJECT_NAME}_ENABLE_UNIT_TESTING)
  enable_testing()
	message(STATUS "Build unit tests for the project. Tests should always be found in the test folder\n")
  add_subdirectory(test)
endif()

if(${PROJECT_NAME}_ENABLE_BENCHMARKING)
  enable_testing()
	message(STATUS "Build benchmark tests for the project. Tests should always be found in the benchmark folder\n")
  add_subdirectory(benchmark)
endif()
