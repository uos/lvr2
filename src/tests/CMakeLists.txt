# Use google test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

###############################################################################
# ADD TESTS HERE
###############################################################################
list(APPEND LVR2_TEST_SOURCES
    test_bvh.cpp
    test_basevector.cpp
)

if (embree_FOUND)
  list(APPEND LVR2_TEST_SOURCES test_embree.cpp)
endif()

add_executable(
  lvr2_test
  ${LVR2_TEST_SOURCES}
)
target_link_libraries(
  lvr2_test
  GTest::gtest_main
  # Prefer static linking in tests
  lvr2_static
)

include(GoogleTest)
gtest_discover_tests(lvr2_test)
