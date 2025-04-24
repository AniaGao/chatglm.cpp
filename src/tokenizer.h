#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>

class Tokenizer {
public:
  Tokenizer() {}
  ~Tokenizer() {}

  std::vector<std::string> tokenize(const std::string& text);
  std::string detokenize(const std::vector<std::string>& tokens);
};

#endif