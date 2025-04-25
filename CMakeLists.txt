cmake_minimum_required(VERSION 3.15)
project(chatglm_cpp)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/src)

# Find Python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})


# SentencePiece
include(FetchContent)
FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG        v0.1.99 # Specify the desired version here
)
FetchContent_MakeAvailable(sentencepiece)
include_directories(${sentencepiece_SOURCE_DIR}/src) # Ensure the sentencepiece headers are available

add_library(chatglm_cpp src/chatglm.cpp src/tokenizer.cpp src/utils.cpp src/sentencepiece.cpp)

target_link_libraries(chatglm_cpp sentencepiece::sentencepiece)


# Create Python module
add_library(chatglm_python SHARED python/chatglm.cpp)
target_link_libraries(chatglm_python chatglm_cpp ${PYTHON_LIBRARIES})
set_target_properties(chatglm_python PROPERTIES PREFIX "" SUFFIX ".so")
