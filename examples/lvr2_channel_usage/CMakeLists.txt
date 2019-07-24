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

add_executable(lvr2_example_channel_usage
    ChannelUsage.cpp
    ${LVR2_IOPATCH_SOURCES}
)

target_link_libraries(lvr2_example_channel_usage ${LVR2_EXAMPLE_CHANNELS_DEPENDENCIES})
