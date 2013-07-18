
message( STATUS "Creating target texman…" )
add_executable( tiogen tiogen.cpp )
FIND_PACKAGE( OpenCV REQUIRED )
TARGET_LINK_LIBRARIES( tiogen ${OpenCV_LIBS} )
target_link_libraries( tiogen staticlvr )
target_link_libraries( tiogen ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})