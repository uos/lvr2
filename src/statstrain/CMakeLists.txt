
message( STATUS "Creating target statstrain…" )
add_executable( statstrain statstrain.cpp )
#FIND_PACKAGE( OpenCV REQUIRED )
#TARGET_LINK_LIBRARIES( statstrain ${OpenCV_LIBS} )
target_link_libraries( statstrain staticlssr )
