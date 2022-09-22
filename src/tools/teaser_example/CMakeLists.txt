#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/home/praktikum/open3d_install/")

# The options need to be the same as Open3D's default
# If Open3D is configured and built with custom options, you'll also need to
# specify the same custom options.


find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(teaserpp REQUIRED)

cmake_policy(SET CMP0074 NEW)
set(Open3D_ROOT "/home/praktikum/open3d_install/")
find_package(Open3D REQUIRED)

# Change this line to include your own executable file
add_executable(teaser_example Main.cpp util/pointtovertex.cpp util/MyMatching.cpp)

# Link to teaserpp & Eigen3
#add_dependencies(Eigen3)
target_link_libraries(teaser_example PUBLIC Eigen3::Eigen teaserpp::teaser_registration teaserpp::teaser_io)
target_link_libraries(teaser_example PRIVATE Open3D::Open3D)