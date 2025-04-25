cmake_minimum_required(VERSION 3.15)
project(chatglm)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/python)


add_library(chatglm_core STATIC
    src/chatglm.cpp
    src/tokenizer.cpp
    src/sentencepiece.cpp
    src/utils.cpp
    src/ggml_wrapper.cpp
)

# Add GGML as a submodule
add_subdirectory(ggml)
target_link_libraries(chatglm_core PRIVATE ggml)


# Python Bindings
find_package(Python COMPONENTS Interpreter Development REQUIRED)

add_library(chatglm_python SHARED
    python/chatglm.cpp
)

target_link_libraries(chatglm_python PRIVATE chatglm_core)
target_link_libraries(chatglm_python PRIVATE ggml)

set_target_properties(chatglm_python PROPERTIES PREFIX "" SUFFIX ".so")


# Install Python Module
install(TARGETS chatglm_python DESTINATION ${CMAKE_INSTALL_PREFIX}/${Python_SITE_PACKAGES})

install(DIRECTORY data DESTINATION ${CMAKE_INSTALL_PREFIX})


# Benchmarks
add_executable(benchmark benchmarks/benchmark.py)


# Tests
enable_testing()
add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core chatglm_core)
add_test(NAME test_core_execution TARGET test_core)

