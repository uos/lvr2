set(ANN_SOURCES 
ANN.cpp brute.cpp 
kd_tree.cpp 
kd_util.cpp
kd_split.cpp 
kd_search.cpp 
kd_pr_search.cpp
kd_fix_rad_search.cpp 
kd_dump.cpp 
bd_tree.cpp
bd_search.cpp 
bd_pr_search.cpp
bd_fix_rad_search.cpp 
perf.cpp)

add_library(ann SHARED ${ANN_SOURCES})
set_target_properties(ann PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)