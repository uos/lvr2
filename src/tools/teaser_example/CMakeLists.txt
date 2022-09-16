
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(teaserpp REQUIRED)

# Change this line to include your own executable file
add_executable(teaser_example Main.cpp util/pointtovertex.cpp util/pointtovertex.h)

# Link to teaserpp & Eigen3
#add_dependencies(Eigen3)
target_link_libraries(teaser_example Eigen3::Eigen teaserpp::teaser_registration teaserpp::teaser_io)
