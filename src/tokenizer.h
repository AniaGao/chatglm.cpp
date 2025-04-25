#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <stdexcept>

namespace tokenizer {

class Tokenizer {
public:
  Tokenizer(const std::string& model_path);
  std::vector<int> Encode(const std::string& text) const;
  std::string Decode(const std::vector<int>& tokens) const;

private:
  class TokenizerImpl;
  std::unique_ptr<TokenizerImpl> impl;
};

} // namespace tokenizer

#endif // TOKENIZER_H