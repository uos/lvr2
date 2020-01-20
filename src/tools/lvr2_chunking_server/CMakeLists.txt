#####################################################################################
# Set source files
#####################################################################################

set(LVR2_CHUNKING_SERVER_SOURCES
    Main.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_CHUNKING_SERVER_DEPENDENCIES
    lvr2_static
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_chunking_server ${LVR2_CHUNKING_SERVER_SOURCES})
target_link_libraries(lvr2_chunking_server ${LVR2_CHUNKING_SERVER_DEPENDENCIES})

install(TARGETS lvr2_chunking_server
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
