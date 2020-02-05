#####################################################################################
# Set source files
#####################################################################################

set(LVR2_REGISTRATION_SOURCES
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_REGISTRATION_DEPENDENCIES
    lvr2_static
    ${LVR2_LIB_DEPENDENCIES}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_registration ${LVR2_REGISTRATION_SOURCES})
target_link_libraries(lvr2_registration ${LVR2_REGISTRATION_DEPENDENCIES})

install(TARGETS lvr2_registration RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
