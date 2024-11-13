/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple Tokenizer interface.
#pragma once

#include <re2/re2.h>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>
#include "sentencepiece_processor.h"

class Tokenizer {
 public:
  explicit Tokenizer() {}
  virtual ~Tokenizer() {}

  virtual void load(const std::string& tokenizer_path) = 0;

  virtual std::vector<uint64_t>
  encode(const std::string& input, int8_t bos, int8_t eos) const = 0;

  virtual std::string decode(uint64_t prev_token, uint64_t token) const = 0;

  // getters
  int32_t vocab_size() const {
    return vocab_size_;
  }

  uint64_t bos_tok() const {
    return bos_tok_;
  }

  uint64_t eos_tok() const {
    return eos_tok_;
  }

 protected:
  bool initialized_ = false;
  int32_t vocab_size_;
  uint64_t bos_tok_, eos_tok_;
};

// ----------------------- SPTokenizer -----------------------
// Used by sentencepiece. Adapted from llama2.c.
struct TokenIndex {
  const char* str;
  int32_t id;
};

class SPTokenizer : public Tokenizer {
 public:
  explicit SPTokenizer();
  ~SPTokenizer() override;

  void load(const std::string& tokenizer_path) override;

  std::vector<uint64_t> encode(const std::string& input, int8_t bos, int8_t eos)
      const override;

  std::string decode(uint64_t prev_token, uint64_t token) const override;

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> _processor;
};

// ----------------------- Tiktoken -----------------------
// Used by OpenAI, adapted from https://github.com/sewenew/tokenizer
//
// The main changes from the upstream implementation are to split out the core
// of the BPE logic into a base class that both Tiktoken and HFTokenizer can
// inherit from.

using Encoder = std::unordered_map<std::string, uint64_t>;
using Decoder = std::unordered_map<uint64_t, std::string>;
using Re2UPtr = std::unique_ptr<re2::RE2>;

class BPETokenizerBase : public Tokenizer {
 protected:

  explicit BPETokenizerBase() {};
  virtual ~BPETokenizerBase() {};

  std::pair<std::optional<std::string>, re2::StringPiece>
  split_with_allowed_special_token_(
      re2::StringPiece& input,
      const Encoder& allowed_special) const;

  std::pair<std::vector<uint64_t>, uint64_t> encode_with_special_token_(
      const std::string& text,
      const Encoder& allowed_special) const;

  // Protected members that can be overloaded by other BPE tokenizers
  Re2UPtr special_token_regex_;
  Encoder encoder_;
  Encoder special_token_encoder_;
  Decoder decoder_;
  Decoder special_token_decoder_;

 private:
  virtual void _encode(
      re2::StringPiece& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len) const = 0;
};

class Tiktoken : public BPETokenizerBase {
 public:
  explicit Tiktoken();
  ~Tiktoken() override {};

  void load(const std::string& tokenizer_path) override;

  std::vector<uint64_t>
  encode(const std::string& input, int8_t bos, int8_t eos) const override;

  std::string decode(uint64_t prev_token, uint64_t token) const override;

 private:
  static inline const Encoder _get_special_tokens(ssize_t num_base_tokens) {
    Encoder special_tokens;
    special_tokens.emplace("<|begin_of_text|>", num_base_tokens++);
    special_tokens.emplace("<|end_of_text|>", num_base_tokens++);
    special_tokens.emplace("<|reserved_special_token_0|>", num_base_tokens++);
    special_tokens.emplace("<|reserved_special_token_1|>", num_base_tokens++);
    special_tokens.emplace("<|reserved_special_token_2|>", num_base_tokens++);
    special_tokens.emplace("<|reserved_special_token_3|>", num_base_tokens++);
    special_tokens.emplace("<|start_header_id|>", num_base_tokens++);
    special_tokens.emplace("<|end_header_id|>", num_base_tokens++);
    special_tokens.emplace("<|reserved_special_token_4|>", num_base_tokens++);
    special_tokens.emplace("<|eot_id|>", num_base_tokens++);
    for (auto i = 5; i < 251; ++i) {
      special_tokens.emplace(
          "<|reserved_special_token_" + std::to_string(i) + "|>",
          num_base_tokens++);
    }
    return special_tokens;
  }

  void _encode(
      re2::StringPiece& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len) const override;

  // Removed negative lookahead \s+(?!\S) since it's not supported by RE2.
  const std::string _pattern =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)";

  Re2UPtr _regex;
};


// ----------------------- HF Tokenizers -----------------------
// Used by many Huggingface models. Adapted from a combination of the original
// rust implementation (https://github.com/huggingface/tokenizers/tree/main)
// and the corresponding support in llama.cpp
// (https://github.com/ggerganov/llama.cpp)

class HFTokenizer : public Tiktoken {
 public:
  /*-- Public Interface --*/

  /**
   * Default initialize with no loaded data
   */
  explicit HFTokenizer();
  ~HFTokenizer() {};

  /**
   * Load the model data into the
   */
  void load(const std::string& tokenizer_path) override;
};
