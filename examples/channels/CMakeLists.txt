#####################################################################################
# Set source files
#####################################################################################

include_directories(
  include
)

set(LVR2_EXAMPLE_CHANNELS_SRCS
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_EXAMPLE_CHANNELS_DEPENDENCIES
    lvr2_static
)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_examples_channels
    Main.cpp
    ${LVR2_EXAMPLE_CHANNELS_SRCS}
)

target_link_libraries(lvr2_examples_channels ${LVR2_EXAMPLE_CHANNELS_DEPENDENCIES})
