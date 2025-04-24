#include "chatglm.h"
#include "tokenizer.h"
#include <iostream>

ChatGLM::ChatGLM() {
    tokenizer = new Tokenizer();
}

ChatGLM::~ChatGLM() {
    delete tokenizer;
}

std::string ChatGLM::generate(const std::string& prompt) {
  // Simple echo for now, utilizing tokenizer
  std::vector<std::string> tokens = tokenizer->tokenize(prompt);
  std::string detokenized_prompt = tokenizer->detokenize(tokens);
  std::cout << "Tokenized prompt: " << prompt << std::endl;
  return "Response: " + detokenized_prompt; // Echo the processed prompt
}
