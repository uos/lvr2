#####################################################################################
# Set source files
#####################################################################################
set(LVR2_ASCII_VIEWER_SRC
    Main.cpp
    Options.cpp
    src/ascii_viewer/AsciiRenderer.cpp
    src/ascii_viewer/Braille.cpp
)

include_directories(include)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_ASCII_VIEWER_DEPS
    lvr2_static
    lvr2las_static
    lvr2rply_static
)

#####################################################################################
# Add executable
#####################################################################################

###### ADD YOUR CODE HERE #######

add_executable(lvr2_ascii_viewer ${LVR2_ASCII_VIEWER_SRC})
target_link_libraries(lvr2_ascii_viewer 
    ${LVR2_ASCII_VIEWER_DEPS}
    ncursesw
)
