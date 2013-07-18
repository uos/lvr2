
message( STATUS "Creating target texman…" )
add_executable( texman texman.cpp )
FIND_PACKAGE( OpenCV REQUIRED )
TARGET_LINK_LIBRARIES( texman ${OpenCV_LIBS} )
target_link_libraries( texman staticlvr )
