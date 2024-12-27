cmake_minimum_required (VERSION 3.5)
project (
    renju
    VERSION 2.16.0
    LANGUAGES C
)

set (CMAKE_VERBOSE_MAKEFILE OFF)
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (CMAKE_C_STANDARD 99)
set (CMAKE_EXPORT_COMPILE_COMMANDS True)
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

string (ASCII 27 ESC)
set (COLOR_GREEN "${ESC}[1;32m")
set (COLOR_YELLOW "${ESC}[1;33m")
set (COLOR_RED "${ESC}[1;31m")
set (COLOR_RESET "${ESC}[0m")

find_package(OpenMP)
find_package(BLAS)
if(NOT OpenMP_FOUND)
    message(STATUS "${COLOR_YELLOW}OpenMP not found.${COLOR_RESET}")
endif()
if(BLAS_FOUND)
    find_path(BLAS_INCLUDE_DIRS cblas.h
        PATHS
        /usr/include
        /usr/local/include
        $ENV{BLAS_HOME}/include
        PATH_SUFFIXES include)
    message(STATUS "BLAS include directories: ${BLAS_INCLUDE_DIRS}")
    include_directories(${BLAS_INCLUDE_DIRS})
    add_compile_definitions(HAVE_BLAS)
else()
    message(STATUS "${COLOR_YELLOW}BLAS not found.${COLOR_RESET}")
endif(BLAS_FOUND)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message (STATUS "${COLOR_RED}debug${COLOR_RESET}")
    add_compile_options (-Wall -Wextra -Wshadow -DDEBUG_LEVEL=2 -O0 -g -fno-omit-frame-pointer)
    add_compile_options (-fsanitize=undefined,null,bounds,leak,address)
    add_link_options (-fsanitize=undefined,address,leak,null,bounds)
    # add_compile_options (-fsanitize=undefined,null,bounds,thread)
    # add_link_options (-fsanitize=undefined,null,bounds,thread)
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

add_library(renju_objects OBJECT ${SRC_LIST})

add_library(librenju SHARED $<TARGET_OBJECTS:renju_objects>)

set_target_properties(librenju PROPERTIES OUTPUT_NAME "renju")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(librenju PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif ()

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/train)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/train/lib)

add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/train/lib/librenju.py
    COMMAND ctypesgen -L${PROJECT_SOURCE_DIR}/bin -lrenju ${PROJECT_SOURCE_DIR}/include/* -o ${PROJECT_SOURCE_DIR}/train/lib/librenju.py
    DEPENDS ${INCLUDE_LIST}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/train
    COMMENT "Running ctypesgen to generate Python bindings"
)
add_custom_target(generate_bindings ALL DEPENDS train/lib/librenju.py librenju)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(generate_bindings PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif ()

add_executable (main src/main.c $<TARGET_OBJECTS:renju_objects>)
# target_compile_definitions(main PRIVATE NO_INTERACTIVE)

file(GLOB TEST_SRC_LIST "${PROJECT_SOURCE_DIR}/src/_tests/*.c")
add_executable (test ${TEST_SRC_LIST} ${SRC_LIST})
target_compile_definitions(test PRIVATE TEST)
# target_compile_definitions(test PRIVATE NO_COLOR)

add_executable (botzone src/botzone.c ${SRC_LIST})
target_compile_definitions(botzone PRIVATE NO_COLOR)

if (UNIX AND NOT APPLE)
    target_link_libraries (main m)
    target_link_libraries (librenju m)
    target_link_libraries (test m)
    target_link_libraries (botzone m)
endif()

if(OpenMP_FOUND)
    target_link_libraries(main OpenMP::OpenMP_C)
    target_link_libraries(librenju OpenMP::OpenMP_C)
    target_link_libraries(test OpenMP::OpenMP_C)
    target_link_libraries(botzone OpenMP::OpenMP_C)
endif(OpenMP_FOUND)

if(BLAS_FOUND)
    target_link_libraries(main ${BLAS_LIBRARIES})
    target_link_libraries(librenju ${BLAS_LIBRARIES})
    target_link_libraries(test ${BLAS_LIBRARIES})
    target_link_libraries(botzone ${BLAS_LIBRARIES})
endif(BLAS_FOUND)
