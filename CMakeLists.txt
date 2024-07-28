cmake_minimum_required (VERSION 3.5)
project (
    gomoku
    VERSION 1.2.2
)
set (CMAKE_VERBOSE_MAKEFILE OFF)
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (CMAKE_C_STANDARD 99)
add_compile_options (-Wall -Wextra -Wshadow)
add_definitions (-DVERSION="${PROJECT_VERSION}")
aux_source_directory (src SRC_LIST)
include_directories (include)
add_executable (gomoku main.c ${SRC_LIST})
target_link_libraries (gomoku m) # linking to math library
