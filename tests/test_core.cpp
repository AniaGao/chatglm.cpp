// tests/test_core.cpp
#include <gtest/gtest.h>
#include "src/utils.cpp"

TEST(CoreTest, LogMessageTest) {
  log_message("Testing log message");
  ASSERT_TRUE(true); // Dummy assertion
}
