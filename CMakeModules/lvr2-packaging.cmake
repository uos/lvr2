############
# AUTO PACKAGING USING CPACK
############

# Only enable packaging when lvr2 is the top-level project
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    return()
endif()

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH "aarch64")
else()
    set(ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Detect OS
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(OS_NAME "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(OS_NAME "macos")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(OS_NAME "windows")
else()
    set(OS_NAME "${CMAKE_SYSTEM_NAME}")
endif()

# Package name: lvr2-version-arch-os
set(CPACK_PACKAGE_FILE_NAME
    "lvr2-${PROJECT_VERSION}-${ARCH}-${OS_NAME}"
)

# Generators
set(CPACK_GENERATOR "TGZ;DEB")

# Debian metadata
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "jubraun@uos.de")

# Let dpkg-shlibdeps detect shared-lib deps automatically
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

set(_LVR2_DEPS
    libeigen3-dev
    libflann-dev
    libgdal-dev
    libglut-dev
    libgsl-dev
    libhdf5-dev
    liblz4-dev
    # We need the full opencv dependency since the FindOpenCV CMake makro is only part of the libopencv-dev package
    libopencv-dev
    libtbb-dev
    libtiff-dev
    libxi-dev
    libxmu-dev
    libyaml-cpp-dev
    ocl-icd-opencl-dev
    openmpi-bin
)

# Depend on MPI if it was found during build
if(MPI_FOUND)
    list(APPEND _LVR2_DEPS "libopenmpi-dev")
endif(MPI_FOUND)

# Depend on the boost components
foreach(_BOOST_COMPONENT ${Boost_COMPONENTS})
    string(REPLACE "_" "-" _BOOST_COMPONENT ${_BOOST_COMPONENT})
    list(APPEND _LVR2_DEPS "libboost-${_BOOST_COMPONENT}-dev")
endforeach()

# Depend on embree if it was used during build
if(embree_FOUND)
    list(APPEND _LVR2_DEPS "libembree-dev")
endif()

# Convert list → comma-separated string
string(REPLACE ";" ", " _LVR2_DEPS_STR "${_LVR2_DEPS}")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "${_LVR2_DEPS_STR}")

include(CPack)
