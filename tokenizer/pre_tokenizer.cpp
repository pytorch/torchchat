/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "pre_tokenizer.h"

// Standard
#include <utility>


RegexPreTokenizer::Re2UPtr
RegexPreTokenizer::create_regex_(const std::string& pattern) {
  assert(!pattern.empty());
  return std::make_unique<re2::RE2>("(" + pattern + ")");
}

std::vector<re2::StringPiece> RegexPreTokenizer::pre_tokenize(re2::StringPiece& input) const {
  std::vector<re2::StringPiece> result;
  re2::StringPiece piece;
  while (RE2::FindAndConsume(&input, *regex_, &piece)) {
    result.emplace_back(std::move(piece));
  }
  return result;
}
