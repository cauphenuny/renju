cmake_minimum_required (VERSION 3.5)
project (
    gomoku
    VERSION 1.12.0
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
    message ("-- ${COLOR_RED}debug${COLOR_RESET}")
    add_compile_options (-Wall -Wextra -Wshadow) 
    add_compile_options (-DDEBUG -O0 -g -fno-omit-frame-pointer -fsanitize=undefined,address,leak,null,bounds)
    add_link_options (-fno-omit-frame-pointer -fsanitize=undefined,address,leak,null,bounds)
    set (VERSION_STRING "${PROJECT_VERSION} (debug)")
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message ("-- ${COLOR_GREEN}release with debug info${COLOR_RESET}")
    add_compile_options (-Wall -Wextra -Wshadow) 
    add_compile_options(-DRELEASE -O3 -g)
    set (VERSION_STRING "${PROJECT_VERSION} (with debug info)")
else()
    message ("-- ${COLOR_GREEN}release${COLOR_RESET}")
    if (MSVC)
        add_compile_options (/O2 /DRELEASE)
    else()
        add_compile_options (-DRELEASE -O3)
    endif()
    set (VERSION_STRING "${PROJECT_VERSION}")
endif()

add_definitions (-DVERSION="${VERSION_STRING}")

aux_source_directory (src SRC_LIST)
include_directories (include)
include_directories (${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE INCLUDE_LIST "*.h")

add_library(gomoku_objects OBJECT ${SRC_LIST})
target_link_libraries(gomoku_objects m)

add_library(libgomoku SHARED $<TARGET_OBJECTS:gomoku_objects>)
set_target_properties(libgomoku PROPERTIES OUTPUT_NAME "gomoku")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(libgomoku PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif ()

add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/lib/libgomoku.py
    COMMAND ctypesgen -Llib -lgomoku ./include/* -o lib/libgomoku.py
    DEPENDS ${INCLUDE_LIST}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running ctypesgen to generate Python bindings"
)
add_custom_target(generate_bindings ALL DEPENDS lib/libgomoku.py)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(generate_bindings PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif ()

add_executable (main gomoku.c $<TARGET_OBJECTS:gomoku_objects>)

add_executable (test tests/main.c ${SRC_LIST})
target_compile_definitions(test PRIVATE TEST)
set_target_properties(test PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_executable (botzone botzone.c ${SRC_LIST})
target_compile_definitions(botzone PRIVATE NOCOLOR)
target_link_libraries (botzone m)
set_target_properties(botzone PROPERTIES EXCLUDE_FROM_ALL TRUE)
