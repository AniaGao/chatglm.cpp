#include "tokenizer.h"
#include "sentencepiece.h"
#include <memory>

namespace tokenizer {

class Tokenizer::TokenizerImpl {
public:
    TokenizerImpl(const std::string& model_path) : sp(model_path) {}
    std::vector<int> Encode(const std::string& text) const { return sp.Encode(text); }
    std::string Decode(const std::vector<int>& tokens) const { return sp.Decode(tokens); }
private:
    sentencepiece::SentencePieceTokenizer sp;
};

Tokenizer::Tokenizer(const std::string& model_path)
    : impl(std::make_unique<TokenizerImpl>(model_path)) {}

std::vector<int> Tokenizer::Encode(const std::string& text) const {
    return impl->Encode(text);
}

std::string Tokenizer::Decode(const std::vector<int>& tokens) const {
    return impl->Decode(tokens);
}

} // namespace tokenizer