## The options need to be the same as Open3D's default
## If Open3D is configured and built with custom options, you'll also need to
## specify the same custom options.
#option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
#if(STATIC_WINDOWS_RUNTIME)
#    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
#else()
#    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
#endif()


#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(teaserpp REQUIRED)
##find_package(lvr2::ProgressBar)
#
#cmake_policy(SET CMP0074 NEW)
#set(Open3D_ROOT "/home/praktikum/open3d_install/")
#find_package(Open3D REQUIRED)

# Change this line to include your own executable file


#add_library(lvr2::ProgressBar)

# Link to teaserpp & Eigen3
#add_dependencies(Eigen3)
#
#
#set(teaser_example_DEPENDENCIES
#        lvr2_static
#        lvr2las_static
#        lvr2rply_static
#        ${teaser_example_DEPENDENCIES}
#        )
#
##add_executable( teaser_example ${teaser_example_SOURCES} )
#add_executable(teaser_example Main.cpp util/pointtovertex.cpp util/MyMatching.cpp ${teaser_example_SOURCES} )
#target_link_libraries( teaser_example PUBLIC Eigen3::Eigen teaserpp::teaser_registration teaserpp::teaser_io PRIVATE Open3D::Open3D ${teaser_example_DEPENDENCIES} )
