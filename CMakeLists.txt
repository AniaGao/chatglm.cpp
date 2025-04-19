cmake_minimum_required(VERSION 3.15)
project(chatglm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add GGML as a submodule.  Assuming it's in a directory called 'ggml'
add_subdirectory(ggml)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml)

# Source files
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "python/*.cpp"
)

add_library(chatglm SHARED ${SOURCES})

# Link against GGML
target_link_libraries(chatglm ggml)

# Python bindings
find_package(pybind11 REQUIRED)

add_library(chatglm_module MODULE python/chatglm.cpp)
target_link_libraries(chatglm_module PRIVATE pybind11::module chatglm)
target_compile_features(chatglm_module PRIVATE cxx_attribute)

set_target_properties(chatglm_module PROPERTIES PREFIX "" SUFFIX ".so")