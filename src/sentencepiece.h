#ifndef SENTENCEPIECE_H
#define SENTENCEPIECE_H

#include <string>
#include <vector>

namespace sentencepiece {\

class SentencePieceTokenizer {
public:
  SentencePieceTokenizer(const std::string& model_path);
  ~SentencePieceTokenizer();

  std::vector<int> Encode(const std::string& text) const;
  std::string Decode(const std::vector<int>& tokens) const;

private:
  void* tokenizer_;
};
} // namespace sentencepiece

#endif // SENTENCEPIECE_H