cmake_minimum_required(VERSION 3.15)
project(chatglmcpp)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add GGML as a submodule (or use find_package if you have it installed)
# add_subdirectory(ggml ggml)

# pybind11
find_package(pybind11 REQUIRED)

add_library(chatglm_core STATIC
    src/chatglm.cpp
    src/tokenizer.cpp
    src/utils.cpp
)

# Link against GGML.  If you're using it as a submodule:
# target_link_libraries(chatglm_core ggml)

add_library(chatglm_cpp MODULE
    python/chatglm.cpp
)
# Link against the core C++ library
target_link_libraries(chatglm_cpp chatglm_core ${PYBIND11_LIBRARIES})

set_target_properties(chatglm_cpp PROPERTIES PREFIX ""
                SUFFIX ".so")

