set(VIEWER_MOCS 
EventHandlerMoc.h  
ObjectHandlerMoc.h  
RenderFrameMoc.h  
TouchPadMoc.h)

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

qt_wrap_cpp(VIEWER_MOC_SRC ${VIEWER_MOCS})
message(STATUS "MOCS: ${VIEWER_MOC_SRC}")

add_executable(viewer Viewer.cpp ${VIEWER_SOURCES})

target_link_libraries(viewer ${QT_QTCORE_LIBRARIES})
target_link_libraries(viewer ${QT_QTGUI_LIBRARIES})
target_link_libraries(viewer ${QT_QTOPENGL_LIBRARIES})
target_link_libraries(viewer lib3d.a)
target_link_libraries(viewer ${OPENGL_LIBRARY})
target_link_libraries(viewer ${GLUT_LIBRARIES})