# CUDA GCC Compatibility
# See here: https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version

# CUDA version                max GCC version
# 11.4.1+, 11.5	                     11
# 11.1, 11.2, 11.3, 11.4.0	         10
# 11	                                9
# 10.1, 10.2	                        8
# 9.2, 10.0	                          7
# 9.0, 9.1	                          6
# 8	                                  5.3
# 7	                                  4.9
# 5.5, 6	                            4.8
# 4.2, 5	                            4.6
# 4.1	                                4.5
# 4.0	                                4.4

# Usage: 
# max_cuda_gcc_version(CUDA_VERSION MAX_CUDA_GCC_VERSION)
# message(STATUS "Maximum allowed gcc version for cuda is: ${MAX_CUDA_GCC_VERSION}")
function(max_cuda_gcc_version _CUDA_VERSION _MAX_GCC_VERSION)

if(${_CUDA_VERSION} VERSION_GREATER_EQUAL 12.0.0)
  set(${_MAX_GCC_VERSION} 12 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 11.4.0)   # 11.4.1+, 11.5  -> 11
  set(${_MAX_GCC_VERSION} 11 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 11.0) # 11.1, 11.2, 11.3, 11.4.0 -> 10
  set(${_MAX_GCC_VERSION} 10 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 10.2) # 11	       -> 9
  set(${_MAX_GCC_VERSION} 9 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 10.0) # 10.1 10.2 -> 8
  set(${_MAX_GCC_VERSION} 8 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 9.1)  # 9.2 10.0 -> 7
  set(${_MAX_GCC_VERSION} 7 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 8)    # 9.0 9.1 -> 6
  set(${_MAX_GCC_VERSION} 6 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 7)    # 8 -> 5.3
  set(${_MAX_GCC_VERSION} 5.3 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 6)    # 7 -> 4.9
  set(${_MAX_GCC_VERSION} 4.9 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 5.4)  # 5.5  6 -> 4.8
  set(${_MAX_GCC_VERSION} 4.8 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 4.1)  # 4.2 5 -> 4.6
  set(${_MAX_GCC_VERSION} 4.6 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 4.0)  # 4.1 -> 4.5
  set(${_MAX_GCC_VERSION} 4.5 PARENT_SCOPE)
elseif(${_CUDA_VERSION} VERSION_GREATER 3)    # 4.0 -> 4.4
  set(${_MAX_GCC_VERSION} 4.4 PARENT_SCOPE)
endif()

endfunction()