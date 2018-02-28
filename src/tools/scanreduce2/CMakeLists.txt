#####################################################################################
# Set source files
#####################################################################################

set(LVR_REDUCE2_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_REDUCE2_DEPENDENCIES
    lvr_static
  )

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr_pointcloud_reduce2 ${LVR_REDUCE2_SOURCES})
target_link_libraries(lvr_pointcloud_reduce2 ${LVR_REDUCE2_DEPENDENCIES})

install(TARGETS lvr_pointcloud_reduce2
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
