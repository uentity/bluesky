cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(blue-sky-re)
# set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
# set bitness
include(bitness)
include(buildtype)
# third party dir
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)
# output dir
if(NOT BS_OUTPUT_DIR)
    set(BS_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/exe/${CMAKE_BUILD_TYPE}${SYSTEM_BITNESS}-${CMAKE_SYSTEM_NAME})
endif()
set(PLUGIN_OUTPUT_DIR ${BS_OUTPUT_DIR}/plugins)
link_directories(${BS_OUTPUT_DIR})
# includes
include(definitions)
include(all)
# subprojects
if(NOT PY)
    set(PY 0)
endif()
if(UNIX)
    add_subdirectory(${THIRD_PARTY_DIR}/actor-framework)
endif()
add_subdirectory(kernel)
if(PY EQUAL 1)
    add_subdirectory(python/bspy_loader)
endif()

if(NOT AUTO_FIND_PLUGINS)
    set(AUTO_FIND_PLUGINS 0)
endif()

#set(AUTO_FIND_PLUGINS 1)
init_plugins(${AUTO_FIND_PLUGINS})
