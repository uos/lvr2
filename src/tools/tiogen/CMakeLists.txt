
add_executable( lvr_tiogen tiogen.cpp )
FIND_PACKAGE( OpenCV REQUIRED )
TARGET_LINK_LIBRARIES( lvr_tiogen ${OpenCV_LIBS} )
target_link_libraries( lvr_tiogen staticlvr )
target_link_libraries( lvr_tiogen ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})