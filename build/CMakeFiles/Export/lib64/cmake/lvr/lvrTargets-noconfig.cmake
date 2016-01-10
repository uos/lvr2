#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lvr::lvrrply_static" for configuration ""
set_property(TARGET lvr::lvrrply_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lvr::lvrrply_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/liblvrrply_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS lvr::lvrrply_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_lvr::lvrrply_static "${_IMPORT_PREFIX}/lib64/liblvrrply_static.a" )

# Import target "lvr::lvrrply" for configuration ""
set_property(TARGET lvr::lvrrply APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lvr::lvrrply PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/liblvrrply.so"
  IMPORTED_SONAME_NOCONFIG "liblvrrply.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS lvr::lvrrply )
list(APPEND _IMPORT_CHECK_FILES_FOR_lvr::lvrrply "${_IMPORT_PREFIX}/lib64/liblvrrply.so" )

# Import target "lvr::lvrlas_static" for configuration ""
set_property(TARGET lvr::lvrlas_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lvr::lvrlas_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/liblvrlas_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS lvr::lvrlas_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_lvr::lvrlas_static "${_IMPORT_PREFIX}/lib64/liblvrlas_static.a" )

# Import target "lvr::lvrlas" for configuration ""
set_property(TARGET lvr::lvrlas APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lvr::lvrlas PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/liblvrlas.so"
  IMPORTED_SONAME_NOCONFIG "liblvrlas.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS lvr::lvrlas )
list(APPEND _IMPORT_CHECK_FILES_FOR_lvr::lvrlas "${_IMPORT_PREFIX}/lib64/liblvrlas.so" )

# Import target "lvr::lvrslam6d_static" for configuration ""
set_property(TARGET lvr::lvrslam6d_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lvr::lvrslam6d_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/liblvrslam6d_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS lvr::lvrslam6d_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_lvr::lvrslam6d_static "${_IMPORT_PREFIX}/lib64/liblvrslam6d_static.a" )

# Import target "lvr::lvrslam6d" for configuration ""
set_property(TARGET lvr::lvrslam6d APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lvr::lvrslam6d PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/liblvrslam6d.so"
  IMPORTED_SONAME_NOCONFIG "liblvrslam6d.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS lvr::lvrslam6d )
list(APPEND _IMPORT_CHECK_FILES_FOR_lvr::lvrslam6d "${_IMPORT_PREFIX}/lib64/liblvrslam6d.so" )

# Import target "lvr::lvr_static" for configuration ""
set_property(TARGET lvr::lvr_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lvr::lvr_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "lvr::lvrrply_static;lvr::lvrlas_static;lvr::lvrslam6d_static;/home/student/i/imitschke/local/lib/libboost_program_options.so;/home/student/i/imitschke/local/lib/libboost_system.so;/home/student/i/imitschke/local/lib/libboost_thread.so;/home/student/i/imitschke/local/lib/libboost_filesystem.so;/usr/lib64/libGLU.so;/usr/lib64/libGL.so;/usr/lib64/libglut.so;/usr/lib64/libXmu.so;/usr/lib64/libXi.so;opencv_videostab;opencv_video;opencv_superres;opencv_stitching;opencv_photo;opencv_ocl;opencv_objdetect;opencv_nonfree;opencv_ml;opencv_legacy;opencv_imgproc;opencv_highgui;opencv_gpu;opencv_flann;opencv_features2d;opencv_core;opencv_contrib;opencv_calib3d;pthread"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/liblvr_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS lvr::lvr_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_lvr::lvr_static "${_IMPORT_PREFIX}/lib64/liblvr_static.a" )

# Import target "lvr::lvr" for configuration ""
set_property(TARGET lvr::lvr APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lvr::lvr PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "lvr::lvrrply;lvr::lvrlas;lvr::lvrslam6d;/home/student/i/imitschke/local/lib/libboost_program_options.so;/home/student/i/imitschke/local/lib/libboost_system.so;/home/student/i/imitschke/local/lib/libboost_thread.so;/home/student/i/imitschke/local/lib/libboost_filesystem.so;/usr/lib64/libGLU.so;/usr/lib64/libGL.so;/usr/lib64/libglut.so;/usr/lib64/libXmu.so;/usr/lib64/libXi.so;opencv_videostab;opencv_video;opencv_superres;opencv_stitching;opencv_photo;opencv_ocl;opencv_objdetect;opencv_nonfree;opencv_ml;opencv_legacy;opencv_imgproc;opencv_highgui;opencv_gpu;opencv_flann;opencv_features2d;opencv_core;opencv_contrib;opencv_calib3d;pthread"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/liblvr.so"
  IMPORTED_SONAME_NOCONFIG "liblvr.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS lvr::lvr )
list(APPEND _IMPORT_CHECK_FILES_FOR_lvr::lvr "${_IMPORT_PREFIX}/lib64/liblvr.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
