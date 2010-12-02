set(VIEWER_MOCS 
EventHandlerMoc.h  
ObjectHandlerMoc.h  
RenderFrameMoc.h  
TouchPadMoc.h)

add_executable(viewer Viewer.cpp ${VIEWER_SOURCES})


qt_wrap_cpp(VIEWER_MOC_SRC ${VIEWER_MOCS})