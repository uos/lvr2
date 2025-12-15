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
    libboost-program-options-dev
    libboost-filesystem-dev
    libboost-thread-dev
    libboost-serialization-dev
    libboost-timer-dev
    libboost-iostreams-dev
    libboost-date-time-dev
    libflann-dev
    libgsl-dev
    libeigen3-dev
    libopenmpi-dev
    libopencv-core-dev
    libopencv-imgproc-dev
    libopencv-imgcodecs-dev
    libopencv-features2d-dev
    libopencv-calib3d-dev
    openmpi-bin
    ocl-icd-opencl-dev
    libhdf5-dev
    liblz4-dev
    libyaml-cpp-dev
)

# Convert list → comma-separated string
string(REPLACE ";" ", " _LVR2_DEPS_STR "${_LVR2_DEPS}")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "${_LVR2_DEPS_STR}")

include(CPack)
