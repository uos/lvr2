#####################################################################################
# Set source files
#####################################################################################

set(LVR2_CHUNKING_SOURCES
    Main.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_CHUNKING_DEPENDENCIES
    lvr2_static
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_chunking ${LVR2_CHUNKING_SOURCES} )
target_link_libraries( lvr2_chunking ${LVR2_CHUNKING_DEPENDENCIES} )

install(TARGETS lvr2_chunking
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
