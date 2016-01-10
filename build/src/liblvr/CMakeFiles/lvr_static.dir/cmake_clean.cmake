file(REMOVE_RECURSE
  "../../lib/liblvr_static.pdb"
  "../../lib/liblvr_static.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/lvr_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
