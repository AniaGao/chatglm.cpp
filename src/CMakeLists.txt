cmake_minimum_required(VERSION 3.10)

project(chatglm_src)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR})

# Define source files
set(SOURCES
    chatglm.cpp
    ggml_wrapper.cpp
    sentencepiece.cpp
    tokenizer.cpp
    utils.cpp
)

# Create a library
add_library(chatglm_src SHARED ${SOURCES})

# Add GoogleTest if testing is enabled
enable_testing()
if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})

  # Create a test executable
  add_executable(chatglm_tests
      test/test_chatglm.cpp
      test/test_tokenizer.cpp
      test/test_ggml_wrapper.cpp
  )

  # Link the test executable with the library and GTest
  target_link_libraries(chatglm_tests chatglm_src GTest::GTest GTest::Main)

  # Add a test case to run the test executable
  add_test(NAME chatglm_tests COMMAND chatglm_tests)
endif()
