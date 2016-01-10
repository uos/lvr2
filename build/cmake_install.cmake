# Install script for directory: /home/student/i/imitschke/meshing.largescale

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/student/i/imitschke/meshing.largescale/include/lvr")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/lvr/lvrTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/lvr/lvrTargets.cmake"
         "/home/student/i/imitschke/meshing.largescale/build/CMakeFiles/Export/lib64/cmake/lvr/lvrTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/lvr/lvrTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/lvr/lvrTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/lvr" TYPE FILE FILES "/home/student/i/imitschke/meshing.largescale/build/CMakeFiles/Export/lib64/cmake/lvr/lvrTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/lvr" TYPE FILE FILES "/home/student/i/imitschke/meshing.largescale/build/CMakeFiles/Export/lib64/cmake/lvr/lvrTargets-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/lvr" TYPE FILE FILES "/home/student/i/imitschke/meshing.largescale/build/lvrConfig.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/student/i/imitschke/meshing.largescale/build/ext/nanoflann/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/ext/psimpl/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/ext/rply/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/ext/laslib/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/ext/slam6d/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/liblvr/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/tools/reconstruct/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/tools/classifier/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/tools/scanreduce/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/tools/convert/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/tools/asciiconverter/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/tools/texman/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/tools/tiogen/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/tools/meshoptimizer/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/tools/transform/cmake_install.cmake")
  include("/home/student/i/imitschke/meshing.largescale/build/src/tools/mpi/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/student/i/imitschke/meshing.largescale/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
