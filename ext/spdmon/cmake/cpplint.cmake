# Copyright (C) 2013 Daniel Scharrer
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the author(s) be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.


# Copyright (C) 2014 Greg Horn
# In order to comply with the above copyright, I am noting that I took
# Daniel's script and hacked it a bit, mostly changing paths and filters

find_package(PythonInterp)

set(STYLE_FILTER)

# disable unwanted filters
set(STYLE_FILTER ${STYLE_FILTER}-whitespace/braces,)
set(STYLE_FILTER ${STYLE_FILTER}-whitespace/semicolon,)
set(STYLE_FILTER ${STYLE_FILTER}-whitespace/blank_line,)
set(STYLE_FILTER ${STYLE_FILTER}-whitespace/comma,)
set(STYLE_FILTER ${STYLE_FILTER}-whitespace/operators,)
set(STYLE_FILTER ${STYLE_FILTER}-whitespace/parens,)
set(STYLE_FILTER ${STYLE_FILTER}-whitespace/indent,)
set(STYLE_FILTER ${STYLE_FILTER}-whitespace/comments,)
set(STYLE_FILTER ${STYLE_FILTER}-whitespace/newline,)
set(STYLE_FILTER ${STYLE_FILTER}-whitespace/tab,)

set(STYLE_FILTER ${STYLE_FILTER}-build/include_order,)
set(STYLE_FILTER ${STYLE_FILTER}-build/namespaces,)
set(STYLE_FILTER ${STYLE_FILTER}-build/include_what_you_use,)

set(STYLE_FILTER ${STYLE_FILTER}-readability/streams,)
set(STYLE_FILTER ${STYLE_FILTER}-readability/todo,)

set(STYLE_FILTER ${STYLE_FILTER}-runtime/references,)
set(STYLE_FILTER ${STYLE_FILTER}-runtime/int,)
set(STYLE_FILTER ${STYLE_FILTER}-runtime/explicit,)
set(STYLE_FILTER ${STYLE_FILTER}-runtime/printf,)

# Add a target that runs cpplint.py
#
function(add_cpplint_style_check_target)
    if(NOT PYTHONINTERP_FOUND)
        return()
    endif()

    list(REMOVE_DUPLICATES SOURCES_LIST)
    list(SORT SOURCES_LIST)

    if(NOT ${PROJECT_NAME}_CPPLINT_BINARY)
			find_program(${PROJECT_NAME}_CPPLINT_BINARY clang-format)
	endif()

	if(${PROJECT_NAME}_CPPLINT_BINARY)
			file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        		${PROJECT_SOURCE_DIR}/src/*.[ch]pp
            )
            
			add_custom_target(cpplint-format
            COMMAND "${CMAKE_COMMAND}" -E chdir
                    "${CMAKE_CURRENT_SOURCE_DIR}"
                    "cpplint"
                    "--filter=${STYLE_FILTER}"
                    "--counting=detailed"
                    "--extensions=cpp,hpp,h"
                    "--linelength=250"
                    ${ALL_CXX_SOURCE_FILES}
            DEPENDS ${ALL_CXX_SOURCE_FILES}
            COMMENT "Linting ${TARGET_NAME}"
            VERBATIM)


			# message(STATUS "Format the project using the `cpplint` target (i.e: cmake --build build --target cpplint-format).\n")
    endif()
endfunction(add_cpplint_style_check_target)