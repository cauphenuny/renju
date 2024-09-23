cmake_minimum_required (VERSION 3.5)
project (
    gomoku
    VERSION 1.6.1
    LANGUAGES C
)
set (CMAKE_VERBOSE_MAKEFILE OFF)
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set (CMAKE_C_STANDARD 99)
set (CMAKE_EXPORT_COMPILE_COMMANDS True)
add_compile_options (-Wall -Wextra -Wshadow) 
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release CACHE STRING "No build type selected, default to Release" FORCE)
endif()
string (ASCII 27 ESC)
set (COLOR_GREEN "${ESC}[1;32m")
set (COLOR_RED "${ESC}[1;31m")
set (COLOR_RESET "${ESC}[0m")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message ("-- ${COLOR_RED}Debug mode${COLOR_RESET}")
    add_compile_options (-DDEBUG -g -fno-omit-frame-pointer -fsanitize=undefined,address,leak,null,bounds)
    add_link_options (-fno-omit-frame-pointer -fsanitize=undefined,address,leak,null,bounds)
    set (VERSION_STRING "${PROJECT_VERSION} (debug)")
else()
    message ("-- ${COLOR_GREEN}Release mode${COLOR_RESET}")
    add_compile_options (-DRELEASE)
    set (VERSION_STRING "${PROJECT_VERSION}")
endif()
add_definitions (-DVERSION="${VERSION_STRING}")
aux_source_directory (src SRC_LIST)
include_directories (include)
file(GLOB_RECURSE INCLUDE_LIST "*.h")

add_library (gomoku SHARED ${SRC_LIST})
target_compile_definitions (gomoku PRIVATE GAME_TIME_LIMIT=1000)
target_compile_definitions (gomoku PRIVATE INSTANT_LOG) # instantly echo log
target_link_libraries(gomoku m)

add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/lib/gomoku.py
    COMMAND ctypesgen -lgomoku ./include/* -o lib/gomoku.py
    DEPENDS ${INCLUDE_LIST}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running ctypesgen to generate Python bindings"
)
add_custom_target(generate_bindings ALL DEPENDS ${PROJECT_SOURCE_DIR}/lib/gomoku.py)

add_executable (main gomoku.c)
set_source_files_properties (gomoku.c PROPERTIES COMPILE_DEFINITIONS "P1=MCTS;P2=MANUAL")
# target_compile_definitions (main PRIVATE GAME_TIME_LIMIT=1000)
target_compile_definitions (main PRIVATE INSTANT_LOG)
target_link_libraries (main gomoku) # linking to math library

add_executable (botzone botzone.c ${SRC_LIST})
set_source_files_properties (botzone.c PROPERTIES COMPILE_DEFINITIONS "ALGORITHM=MCTS")
# target_compile_definitions (botzone PRIVATE INSTANT_LOG)
target_link_libraries (botzone m)

# add_executable (trainer trainer.c ${SRC_LIST})
# target_compile_definitions (trainer PRIVATE GAME_TIME_LIMIT=500)
# target_link_libraries (trainer m)
