/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * This is a simple tool to instantiate a tokenizer and run it over some text.
 * It can be used to evaluate the tokenization done by a given tokenizer model
 * relative to its native python library.
 */

// Standard
#include <iostream>
#include <memory>
#include <sstream>

// Local
#include "tokenizer.h"

std::string help(char* argv[]) {
    std::stringstream ss;
    ss << "Usage: " << argv[0] << " <type> <model> <input to tokenize...>" << std::endl << std::endl;
    ss << "Types:\n" << std::endl;
    ss << "* sentencepiece: SPTokenizer" << std::endl;
    ss << "* tiktoken: Tiktoken" << std::endl;
    ss << "* tokenizers: TokenizersTokenizer" << std::endl;
    return ss.str();
}

int main(int argc, char* argv[]) {

  // Check for the right number of CLI args
  if (argc < 4) {
    std::cerr << help(argv) << std::endl;
    return 1;
  }

  // Parse CLI args
  const std::string tokenizer_type(argv[1]);
  const std::string model_path(argv[2]);
  std::stringstream prompt_ss;
  for (auto i = 3; i < argc; ++i) {
    if (i > 3) {
      prompt_ss << " ";
    }
    prompt_ss << argv[i];
  }
  const std::string prompt = prompt_ss.str();

  // Instantiate the tokenizer
  std::unique_ptr<Tokenizer> tok_ptr;
  if (tokenizer_type == "sentencepiece") {
    tok_ptr.reset(new SPTokenizer());
  } else if (tokenizer_type == "tiktoken") {
    tok_ptr.reset(new Tiktoken());
  } else if (tokenizer_type == "tokenizers") {
    tok_ptr.reset(new TokenizersTokenizer());
  } else {
    std::stringstream ss;
    ss << "ERROR: Invalid tokenizer type: " << tokenizer_type << std::endl << std::endl;
    ss << help(argv);
    std::cerr << ss.str() << std::endl;
    return 1;
  }

  // Load from the path
  tok_ptr->load(model_path);

  // Log out the IDs for the BOS/EOS tokens
  std::cout << "Vocab Size: " << tok_ptr->vocab_size() << std::endl;
  std::cout << "BOS: " << tok_ptr->bos_tok() << std::endl;
  std::cout << "EOS: " << tok_ptr->eos_tok() << std::endl << std::endl;

  // Encode
  std::cout << "PROMPT:" << std::endl << prompt << std::endl << std::endl;
  std::cout << "Encoding..." << std::endl;
  const auto encoded = tok_ptr->encode(prompt, 1, 1);
  std::cout << "[";
  for (const auto tok_id : encoded) {
    std::cout << " " << tok_id;
  }
  std::cout << " ]" << std::endl << std::endl;

  // Decode
  std::cout << "Decoding..." << std::endl;
  for (auto i = 1; i < encoded.size() - 1; ++i) {
    std::cout << tok_ptr->decode(encoded[i-1], encoded[i]);
  }
  std::cout << std::endl;

  return 0;
}