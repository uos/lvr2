#
# Print a message only if the `VERBOSE_OUTPUT` option is on
#
function(verbose_message content)
    if(${PROJECT_NAME}_VERBOSE_OUTPUT)
			message(STATUS ${content})
    endif()
endfunction()

#
# Add a target for formating the project using `clang-format` (i.e: cmake --build build --target clang-format)
#

function(add_clang_format_target)
    if(NOT ${PROJECT_NAME}_CLANG_FORMAT_BINARY)
			find_program(${PROJECT_NAME}_CLANG_FORMAT_BINARY clang-format)
	endif()

	if(${PROJECT_NAME}_CLANG_FORMAT_BINARY)
			file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        		${PROJECT_SOURCE_DIR}/src/*.[ch]pp
    		)
			if(${PROJECT_NAME}_BUILD_EXECUTABLE)
				add_custom_target(clang-format
						COMMAND ${${PROJECT_NAME}_CLANG_FORMAT_BINARY}
						-i  ${ALL_CXX_SOURCE_FILES})
			elseif(${PROJECT_NAME}_BUILD_HEADERS_ONLY)
				add_custom_target(clang-format
						COMMAND ${${PROJECT_NAME}_CLANG_FORMAT_BINARY}
						-i  ${ALL_CXX_SOURCE_FILES})
			else()
				add_custom_target(clang-format
						COMMAND ${${PROJECT_NAME}_CLANG_FORMAT_BINARY}
						-i  ${ALL_CXX_SOURCE_FILES})
			endif()

			# message(STATUS "Format the project using the `clang-format` target (i.e: cmake --build build --target clang-format).\n")
    endif()
endfunction(add_clang_format_target)


