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
  encode(const std::string& input, int8_t bos, int8_t eos) = 0;

  virtual std::string decode(uint64_t prev_token, uint64_t token) = 0;

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
      override;

  std::string decode(uint64_t prev_token, uint64_t token) override;

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> _processor;
};

// ----------------------- Tiktoken -----------------------
// Used by OpenAI, adapted from https://github.com/sewenew/tokenizer

using Encoder = std::unordered_map<std::string, uint64_t>;
using Decoder = std::unordered_map<uint64_t, std::string>;
using Re2UPtr = std::unique_ptr<re2::RE2>;

class Tiktoken : public Tokenizer {
 public:
  explicit Tiktoken();
  ~Tiktoken(){};

  void load(const std::string& tokenizer_path);

  std::vector<uint64_t>
  encode(const std::string& input, int8_t bos, int8_t eos);

  std::string decode(uint64_t prev_token, uint64_t token);

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

  template <typename T>
  std::pair<std::optional<std::string>, re2::StringPiece>
  _split_with_allowed_special_token(
      re2::StringPiece& input,
      const T& allowed_special);

  void _encode(
      re2::StringPiece& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len);

  template <typename T>
  std::pair<std::vector<uint64_t>, uint64_t> _encode_with_special_token(
      const std::string& text,
      const T& allowed_special);

  // Removed negative lookahead \s+(?!\S) since it's not supported by RE2.
  const std::string _pattern =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)";
  Encoder _encoder;
  Encoder _special_token_encoder;
  Decoder _decoder;
  Decoder _special_token_decoder;

  Re2UPtr _regex;
  Re2UPtr _special_token_regex;
};
