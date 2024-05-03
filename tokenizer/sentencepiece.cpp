/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// sentencepiece tokenizer

#include <sentencepiece_processor.h>
#include <tokenizer.h>
#include <cinttypes>
#include <string>
#include "absl/strings/str_replace.h"

const char kSpaceSymbol[] = "\xe2\x96\x81";

SPTokenizer::SPTokenizer()
    : Tokenizer(),
      _processor(std::make_unique<sentencepiece::SentencePieceProcessor>()) {}

/**
 * @brief Load the tokenizer from a file. The tokenizer file contains the
 * vocabulary and scores. The format is: the first integer is the maximum
 * token length, followed by a list of (word_len, word) pairs. Here we
 * are reading all the vocabulary into memory and keep it sorted for fast
 * lookup.
 *
 * @param tokenizer_path The path to the tokenizer file.
 * @return void
 */
void SPTokenizer::load(const std::string& tokenizer_path) {
  if (initialized_) {
    fprintf(stderr, "Tokenizer already initialized.\n");
    return;
  }
  // read in the file
  const auto status = _processor->Load(tokenizer_path);
  if (!status.ok()) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path.c_str());
    exit(EXIT_FAILURE);
  }
  // load vocab_size, bos_tok, eos_tok
  vocab_size_ = _processor->GetPieceSize();
  bos_tok_ = _processor->bos_id();
  eos_tok_ = _processor->eos_id();
  initialized_ = true;
}

SPTokenizer::~SPTokenizer() {}

/**
 * @brief Decode a token into string.
 *
 * @param prev_token The previous token.
 * @param token The current token.
 * @return std::string A pointer to the string representation of the
 * token.
 */
std::string SPTokenizer::decode(uint64_t prev_token, uint64_t token) {
  if (!initialized_) {
    fprintf(stderr, "Tokenizer not initialized\n");
    exit(EXIT_FAILURE);
  }
  // get rid of the control ids <s> and </s>
  if (_processor->IsControl(token)) {
    // NB: returning empty string doesn't work for some reason. It causes
    // free(): invalid pointer error.
    return " ";
  }

  std::string result =
      absl::StrReplaceAll(_processor->IdToPiece(token), {{kSpaceSymbol, " "}});

  // following BOS token, sentencepiece decoder strips any leading
  // whitespace
  if (prev_token == bos_tok_ && result[0] == ' ') {
    result = result.substr(1);
  }

  // handle <0x0A>
  result = absl::StrReplaceAll(result, {{"<0x0A>", "\n"}});

  return result;
}

/**
 * @brief Encode a string into a sequence of tokens.
 *
 * @param text The string to be encoded.
 * @param bos The number of BOS to prepend to the token list.
 * @param eos The number of EOS to append to the token list.
 * @return std::vector<uint64_t>
 */
std::vector<uint64_t>
SPTokenizer::encode(const std::string& text, int8_t bos, int8_t eos) {
  if (!initialized_) {
    fprintf(stderr, "Tokenizer not initialized\n");
    exit(EXIT_FAILURE);
  }
  // workaround a weird issue that text doesn't have correct size()
  std::string input(text.c_str());
  // should we reserve memory?
  std::vector<int> res;
  auto status = _processor->Encode(input, &res);
  if (!status.ok()) {
    fprintf(stderr, "couldn't encode %s\n", text.c_str());
    exit(EXIT_FAILURE);
  }

  std::vector<uint64_t> tokens;
  for (auto i = 0; i < bos; ++i) {
    tokens.push_back(bos_tok_);
  }

  for (auto i = 0; i < res.size(); ++i) {
    tokens.push_back(res[i]);
  }

  for (auto i = 0; i < eos; ++i) {
    tokens.push_back(eos_tok_);
  }
  return tokens;
}
