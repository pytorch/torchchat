/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "pre_tokenizer.h"

// Standard
#include <algorithm>
#include <iterator>
#include <utility>

// Local
#include "unicode.h"

// PreTokenizerConfig //////////////////////////////////////////////////////////

PreTokenizerConfig::PreTokenizerConfig(std::string type)
  : type(std::move(type))
{}

PreTokenizer::Ptr PreTokenizerConfig::create() const {
  // NOTE: These types must line up with the type strings found in the
  //  tokenizers library
  //  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/mod.rs#L73
  if (type == "Regex") {
    if (!pattern) {
      throw std::runtime_error("Missing pattern for PreTokenizer of type Regex");
    }
    return PreTokenizer::Ptr(new RegexPreTokenizer(*pattern));
  }
  if (type == "Digits") {
    if (individual_digits) {
      return PreTokenizer::Ptr(new DigitsPreTokenizer(*individual_digits));
    }
    return PreTokenizer::Ptr(new DigitsPreTokenizer());
  }
  if (type == "ByteLevel") {
    if (add_prefix_space && pattern) {
      return PreTokenizer::Ptr(new ByteLevelPreTokenizer(*add_prefix_space, *pattern));
    }
    if (add_prefix_space) {
      return PreTokenizer::Ptr(new ByteLevelPreTokenizer(*add_prefix_space));
    }
    if (pattern) {
      return PreTokenizer::Ptr(new ByteLevelPreTokenizer(*pattern));
    }
    return PreTokenizer::Ptr(new ByteLevelPreTokenizer());
  }
  if (type == "Sequence") {
    if (!pretokenizers or pretokenizers->empty()) {
      throw std::runtime_error("Missing pretokenizers for PreTokenizer of type Sequence");
    }
    std::vector<PreTokenizer::Ptr> pretoks;
    std::transform(
      pretokenizers->begin(),
      pretokenizers->end(),
      std::back_inserter(pretoks),
      [](const PreTokenizerConfig& cfg) {
        return cfg.create();
      }
    );
    return PreTokenizer::Ptr(new SequencePreTokenizer(pretoks));
  }
  throw std::runtime_error("Unsupported PreTokenizer type: " + type);
}

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

// SequencePreTokenizer ////////////////////////////////////////////////////////

SequencePreTokenizer::SequencePreTokenizer(
  std::vector<PreTokenizer::Ptr> pre_tokenizers)
  : pre_tokenizers_(std::move(pre_tokenizers))
{}

std::vector<std::string> SequencePreTokenizer::pre_tokenize(re2::StringPiece input) const {
  std::vector<std::string> pieces{std::string(input)};
  for (const auto& pre_tokenizer : pre_tokenizers_) {
    std::vector<std::string> new_pieces;
    for (const auto& piece : pieces) {
      for (const auto& subpiece : pre_tokenizer->pre_tokenize(piece)) {
        new_pieces.push_back(subpiece);
      }
    }
    pieces = std::move(new_pieces);
  }
  return pieces;
}
