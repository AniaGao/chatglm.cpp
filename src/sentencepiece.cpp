#include "sentencepiece.h"
#include <sentencepiece_processor.h>

namespace sentencepiece {

SentencePieceTokenizer::SentencePieceTokenizer(const std::string& model_path) {
  tokenizer_ = new sentencepiece::SentencePieceProcessor();
  auto status = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(tokenizer_)->Load(model_path);
  if (!status.ok()) {
      throw std::runtime_error("Failed to load sentencepiece model: " + status.ToString());
  }
}

SentencePieceTokenizer::~SentencePieceTokenizer() {
  delete reinterpret_cast<sentencepiece::SentencePieceProcessor*>(tokenizer_);
}

std::vector<int> SentencePieceTokenizer::Encode(const std::string& text) const {
  std::vector<int> ids;
  auto status = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(tokenizer_)->Encode(text, &ids);
    if (!status.ok()) {
        throw std::runtime_error("SentencePiece encoding failed: " + status.ToString());
    }
  return ids;
}

std::string SentencePieceTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string text;
  auto status = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(tokenizer_)->Decode(tokens, &text);
    if (!status.ok()) {
        throw std::runtime_error("SentencePiece decoding failed: " + status.ToString());
    }
  return text;
}

} // namespace sentencepiece