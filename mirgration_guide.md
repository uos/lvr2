# Migration Guide

## 25.1.0 -> 25.2.0

We switched to more modern CMake. Which is why you'll need to change the old style CMake:

```cmake
find_package(LVR2 REQUIRED)
# ...
target_link_libraries(my_app ${LVR2_LIBRARIES})
```

to 


```cmake
find_package(lvr2 REQUIRED)
# ...
target_link_libraries(my_app lvr2::lvr2)
```
