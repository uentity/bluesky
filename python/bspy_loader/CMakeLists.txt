cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(bs)
# target
set(T ${PROJECT_NAME})
# sources
file(GLOB_RECURSE SOURCES src/*.cpp)
# define target
add_library(${T} SHARED ${SOURCES})
set(CMAKE_SHARED_LIBRARY_PREFIX "")
if(WIN32)
    set(CMAKE_SHARED_LIBRARY_SUFFIX .pyd)
endif()
# set output dir
set_outdir(${T} ${BS_OUTPUT_DIR})
# include dirs
target_include_directories(${T} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../kernel/include)
# dependencies
use_bs(${T})
use_python(${T})
use_pybind11(${T})
