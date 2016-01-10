file(REMOVE_RECURSE
  "../../lib/liblvr.pdb"
  "../../lib/liblvr.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/lvr.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
