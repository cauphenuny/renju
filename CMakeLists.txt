cmake_minimum_required (VERSION 3.5)
project (
    gomoku
    VERSION 2.5.0
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
find_package(OpenMP)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message (STATUS "${COLOR_RED}debug${COLOR_RESET}")
    add_compile_options (-Wall -Wextra -Wshadow) 
    add_compile_options (-DDEBUG_LEVEL=2 -O0 -g -fno-omit-frame-pointer -fsanitize=undefined,address,leak,null,bounds)
    add_link_options (-fno-omit-frame-pointer -fsanitize=undefined,address,leak,null,bounds)
    set (VERSION_STRING "${PROJECT_VERSION} (debug)")
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message (STATUS "${COLOR_GREEN}release with debug info${COLOR_RESET}")
    add_compile_options (-Wall -Wextra -Wshadow) 
    add_compile_options(-DDEBUG_LEVEL=1 -O3 -g)
    set (VERSION_STRING "${PROJECT_VERSION} (with debug info)")
else()
    message (STATUS "${COLOR_GREEN}release${COLOR_RESET}")
    if (MSVC)
        add_compile_options (/DRELEASE /O2)
        if(OpenMP_FOUND)
            add_compile_options (/openmp)
        endif()
    else()
        add_compile_options (-DRELEASE -O3)
        if(OpenMP_FOUND)
            add_compile_options (-fopenmp)
        endif()
    endif()
    set (VERSION_STRING "${PROJECT_VERSION}")
endif()

add_definitions (-DVERSION="${VERSION_STRING}")

file(GLOB_RECURSE SRC_LIST "${PROJECT_SOURCE_DIR}/src/*/*.c")
list(FILTER SRC_LIST EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/src/_tests/*")
include_directories (include)
include_directories (${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE INCLUDE_LIST "*.h")

add_library(gomoku_objects OBJECT ${SRC_LIST})
target_link_libraries(gomoku_objects m)
# target_compile_definitions(gomoku_objects PRIVATE NO_FORBID)

add_library(libgomoku SHARED $<TARGET_OBJECTS:gomoku_objects>)
set_target_properties(libgomoku PROPERTIES OUTPUT_NAME "gomoku")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(libgomoku PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif ()

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/train)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/train/lib)

add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/train/lib/libgomoku.py
    COMMAND ctypesgen -L${PROJECT_SOURCE_DIR}/lib -lgomoku ${PROJECT_SOURCE_DIR}/include/* -o ${PROJECT_SOURCE_DIR}/train/lib/libgomoku.py
    DEPENDS ${INCLUDE_LIST}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/train
    COMMENT "Running ctypesgen to generate Python bindings"
)
add_custom_target(generate_bindings ALL DEPENDS train/lib/libgomoku.py)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(generate_bindings PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif ()

add_executable (main src/gomoku.c $<TARGET_OBJECTS:gomoku_objects>)
# target_compile_definitions(main PRIVATE NO_INTERACTIVE)

file(GLOB TEST_SRC_LIST "${PROJECT_SOURCE_DIR}/src/_tests/*.c")
add_executable (test ${TEST_SRC_LIST} ${SRC_LIST})
target_compile_definitions(test PRIVATE TEST)
set_target_properties(test PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_executable (botzone src/botzone.c ${SRC_LIST})
target_compile_definitions(botzone PRIVATE NOCOLOR)
target_link_libraries (botzone m)
set_target_properties(botzone PROPERTIES EXCLUDE_FROM_ALL TRUE)

if(OpenMP_FOUND)
    target_link_libraries(main OpenMP::OpenMP_C)
    target_link_libraries(libgomoku OpenMP::OpenMP_C)
    target_link_libraries(botzone OpenMP::OpenMP_C)
    target_link_libraries(test OpenMP::OpenMP_C)
endif()