# Objects that use Qt's signals and slots
set(VIEWER_MOCS 
EventHandler.h  
ObjectHandler.h  
RenderFrame.h  
TouchPad.h)

# Dynamically generated GUI object's from
# QtDesigner 
set(VIEWER_UIS
viewer.ui
objectdialog.ui
movedock.ui
matrixdialog.ui
objectdialog.ui)

# Main Sources
set(VIEWER_SOURCES
EventHandler.cpp
ObjectHandler.cpp
RenderFrame.cpp
TouchPad.cpp
ViewerWindow.cpp
Viewport.cpp)

# Generate mocs and header files for qt objects
# and forms
qt_wrap_cpp(viewer VIEWER_MOC_SRC ${VIEWER_MOCS})
qt_wrap_ui(viewer VIEWER_UI_HDRS VIEWER_UI_SRC ${VIEWER_UIS})

add_definitions(${QT_DEFINITIONS})

# Add current binary dir to includes to find
# the generated header files
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Create a new targer
add_executable(viewer Viewer.cpp ${VIEWER_SOURCES} ${VIEWER_MOC_SRC} ${VIEWER_UI_HDRS})

# Add needed libraries
target_link_libraries(viewer ${QT_QTCORE_LIBRARIES})
target_link_libraries(viewer ${QT_QTGUI_LIBRARIES})
target_link_libraries(viewer ${QT_QTOPENGL_LIBRARIES})
target_link_libraries(viewer model3d)
target_link_libraries(viewer ${OPENGL_LIBRARY})
target_link_libraries(viewer ${GLUT_LIBRARIES})