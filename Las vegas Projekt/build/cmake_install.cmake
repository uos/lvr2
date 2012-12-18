# Install script for directory: /home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/ext/rply/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/ext/laslib/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/ext/obj/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/ext/libfreenect/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/ext/slam6d/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/lib/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/reconstruct/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/colorize/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/scanreduce/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/convert/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/asciiconverter/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/meshoptimizer/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/cgal_reconstruction/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/scanfilter/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/kinectgrabber/cmake_install.cmake")
  INCLUDE("/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/src/qglviewer/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

IF(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
ELSE(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
ENDIF(CMAKE_INSTALL_COMPONENT)

FILE(WRITE "/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/${CMAKE_INSTALL_MANIFEST}" "")
FOREACH(file ${CMAKE_INSTALL_MANIFEST_FILES})
  FILE(APPEND "/home/student/d/dofeldsc/Arbeitsfläche/meshing.dofeldsc/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
ENDFOREACH(file)
