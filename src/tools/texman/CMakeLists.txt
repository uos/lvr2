
message( STATUS "Creating target texman…" )
add_executable( lvr_texman texman.cpp )
FIND_PACKAGE( OpenCV REQUIRED )
TARGET_LINK_LIBRARIES( lvr_texman ${OpenCV_LIBS} )
target_link_libraries( lvr_texman staticlvr )
