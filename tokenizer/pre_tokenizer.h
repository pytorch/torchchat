/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

// Standard
#include <memory>
#include <string>
#include <vector>

// Third Party
#include <re2/re2.h>


// -- Bases --------------------------------------------------------------------

/**
 * Base class for all pre-tokenizers with a single virtual method to split the
 * input string piece
 */
class PreTokenizer {
 public:

  /** Split the input string piece into sub-pieces
   *
   * This pre-tokenization may result in sub-pieces that are not contained
   * within the original input, therefore the resulting pieces will be owned by
   * the caller.
   */
  virtual std::vector<std::string> pre_tokenize(re2::StringPiece& input) const = 0;
};  // end class PreTokenizer


/**
 * Base class for all decoders with a single virtual method to decode from the
 * token IDs back to a string
 */
class TokenDecoder {
 public:

  /** Common type for token IDs */
  typedef uint64_t TokenId;

  /** Decode the sequence of token IDs back to a string */
  virtual std::string decode(const std::vector<uint8_t>& input) const = 0;
};  // end class TokenDecoder

// -- Regex --------------------------------------------------------------------
// Used for general-purpose single-regex pre tokenization (e.g. tiktoken)
// CITE: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/split.rs

class RegexPreTokenizer : public PreTokenizer {
 public:

  typedef std::unique_ptr<re2::RE2> Re2UPtr;

  explicit RegexPreTokenizer(const std::string& pattern)
    : regex_(RegexPreTokenizer::create_regex_(pattern))
  {}

  /** Pre-tokenize with the stored regex */
  std::vector<std::string> pre_tokenize(re2::StringPiece& input) const;

 protected:
  static Re2UPtr create_regex_(const std::string& pattern);

 private:
  Re2UPtr regex_;

};  // end class RegexPreTokenizer

// -- Digits -------------------------------------------------------------------
// Used by tokenizers
// CITE: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/digits.rs

class DigitsPreTokenizer : public RegexPreTokenizer {
 public:
  explicit DigitsPreTokenizer(bool individual_digits)
    : RegexPreTokenizer(individual_digits ? R"([^\p{N}]+|\p{N})" : R"([^\p{N}]+|[\p{N}]+)")
  {}
};  // end class DigitsPreTokenizer

// -- ByteLevel ----------------------------------------------------------------
// Used by tokenizers
// CITE: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs