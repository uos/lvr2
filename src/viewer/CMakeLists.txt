set(VIEWER_MOCS 
EventHandler.h  
ObjectHandler.h  
RenderFrame.h  
TouchPad.h)

set(VIEWER_SOURCES
EventHandler.cpp
MatrixDialog.cpp
MoveDock.cpp 
ObjectDialog.cpp
ObjectHandler.cpp
RenderFrame.cpp
TouchPad.cpp
ViewerWindow.cpp
Viewport.cpp)

qt_wrap_cpp(viewer VIEWER_MOC_SRC ${VIEWER_MOCS})
add_definitions(${QT_DEFINITIONS})

add_executable(viewer Viewer.cpp ${VIEWER_SOURCES} ${VIEWER_MOC_SRC})
target_link_libraries(viewer ${QT_QTCORE_LIBRARIES})
target_link_libraries(viewer ${QT_QTGUI_LIBRARIES})
target_link_libraries(viewer ${QT_QTOPENGL_LIBRARIES})
target_link_libraries(viewer lib3d.a)
target_link_libraries(viewer ${OPENGL_LIBRARY})
target_link_libraries(viewer ${GLUT_LIBRARIES})