cmake_minimum_required(VERSION 3.15)
project(chatglm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add pybind11 as a submodule
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1 # Use a specific tag for stability
)
FetchContent_MakeAvailable(pybind11)

include_directories(${pybind11_INCLUDE_DIRS})

# Source files
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "python/*.cpp"
)

add_library(chatglm SHARED ${SOURCES})

target_link_libraries(chatglm PRIVATE pybind11::headers)

# Create a target alias for direct use in add_executable/add_library
add_library(chatglm_alias INTERFACE)
target_link_libraries(chatglm_alias INTERFACE chatglm)

# Install
install(TARGETS chatglm DESTINATION lib)