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

// Local
#include "unicode.h"

// RegexPreTokenizer ///////////////////////////////////////////////////////////

RegexPreTokenizer::Re2UPtr
RegexPreTokenizer::create_regex_(const std::string& pattern) {
  assert(!pattern.empty());
  return std::make_unique<re2::RE2>("(" + pattern + ")");
}

std::vector<std::string> RegexPreTokenizer::pre_tokenize(re2::StringPiece input) const {
  std::vector<std::string> result;
  std::string piece;
  while (RE2::FindAndConsume(&input, *regex_, &piece)) {
    result.emplace_back(piece);
  }
  return result;
}

// ByteLevelPreTokenizer ///////////////////////////////////////////////////////

//////////////////
// Impl Details //
//////////////////
namespace
{

// Standard GPT2 regex
// https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
static const std::string GPT2_EXPR = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

} // end anon namespace

//////////////////
// Construction //
//////////////////

ByteLevelPreTokenizer::ByteLevelPreTokenizer(
  bool add_prefix_space, const std::string& pattern)
  : pattern_(pattern.empty() ? GPT2_EXPR : pattern),
    add_prefix_space_(add_prefix_space)
{}

std::vector<std::string>
ByteLevelPreTokenizer::pre_tokenize(re2::StringPiece input) const {

  // Add the prefix space if configured to do so
  std::string input_str(input);
  if (add_prefix_space_ && !input_str.empty() && input_str[0] != ' ') {
    input_str.insert(input_str.begin(), ' ');
  }

  return unicode_regex_split(input_str, {pattern_});
}
