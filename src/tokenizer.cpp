#include "tokenizer.h"
#include <sstream>

std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
  std::vector<std::string> tokens;
  std::stringstream ss(text);
  std::string token;
  while (ss >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

std::string Tokenizer::detokenize(const std::vector<std::string>& tokens) {
  std::string text;
  for (size_t i = 0; i < tokens.size(); ++i) {
    text += tokens[i];
    if (i < tokens.size() - 1) {
      text += " ";
    }
  }
  return text;
}