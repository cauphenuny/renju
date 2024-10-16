cmake_minimum_required (VERSION 3.5)
project (
    gomoku
    VERSION 1.11.1
    LANGUAGES C
)
set (CMAKE_VERBOSE_MAKEFILE OFF)
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set (CMAKE_C_STANDARD 99)
set (CMAKE_EXPORT_COMPILE_COMMANDS True)
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release CACHE STRING "No build type selected, default to Release" FORCE)
endif()
string (ASCII 27 ESC)
set (COLOR_GREEN "${ESC}[1;32m")
set (COLOR_RED "${ESC}[1;31m")
set (COLOR_RESET "${ESC}[0m")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options (-Wall -Wextra -Wshadow) 
    message ("-- ${COLOR_RED}Debug mode${COLOR_RESET}")
    add_compile_options (-DDEBUG -g -fno-omit-frame-pointer -fsanitize=undefined,address,leak,null,bounds)
    add_link_options (-fno-omit-frame-pointer -fsanitize=undefined,address,leak,null,bounds)
    set (VERSION_STRING "${PROJECT_VERSION} (debug)")
else()
    message ("-- ${COLOR_GREEN}Release mode${COLOR_RESET}")
    if (MSVC)
        add_compile_options (/O2 /DRELEASE)
    else()
        add_compile_options (-DRELEASE -O3)
    endif()
    set (VERSION_STRING "${PROJECT_VERSION}")
endif()
add_definitions (-DVERSION="${VERSION_STRING}")
aux_source_directory (src SRC_LIST)
list(REMOVE_ITEM SRC_LIST "src/util.c") # exclude util.c for controlling log
include_directories (include)
file(GLOB_RECURSE INCLUDE_LIST "*.h")

add_library(gomoku_objects OBJECT ${SRC_LIST})
target_link_libraries(gomoku_objects m)

add_library(libgomoku_shared SHARED $<TARGET_OBJECTS:gomoku_objects> src/util.c)
# disable log
target_compile_definitions(libgomoku_shared PRIVATE DISABLE_LOG)
set_target_properties(libgomoku_shared PROPERTIES OUTPUT_NAME "gomoku")
set_target_properties(libgomoku_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_library(libgomoku STATIC $<TARGET_OBJECTS:gomoku_objects> src/util.c)
# instant log
target_compile_definitions(libgomoku PRIVATE INSTANT_LOG)
set_target_properties(libgomoku PROPERTIES OUTPUT_NAME "gomoku")

add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/lib/gomoku.py
    COMMAND ctypesgen -Llib -lgomoku ./include/* -o lib/gomoku.py
    DEPENDS ${INCLUDE_LIST} libgomoku_shared
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running ctypesgen to generate Python bindings"
)
add_custom_target(generate_bindings ALL DEPENDS lib/gomoku.py)
set_target_properties(generate_bindings PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_executable (main gomoku.c)
target_link_libraries(main libgomoku)

add_executable (test tests/main.c)
target_link_libraries(test libgomoku)
set_target_properties(test PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_executable (botzone botzone.c ${SRC_LIST} src/util.c)
target_compile_definitions(botzone PRIVATE NOCOLOR)
target_link_libraries (botzone m)
set_target_properties(botzone PROPERTIES EXCLUDE_FROM_ALL TRUE)