/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Third Party
#include <gtest/gtest.h>
#include <re2/re2.h>

// Local
#include "pre_tokenizer.h"


// Helpers /////////////////////////////////////////////////////////////////////


void assert_split_match(
  const PreTokenizer& ptok,
  const std::string& prompt,
  const std::vector<std::string>& expected
) {
  re2::StringPiece prompt_view(prompt);
  const auto& got = ptok.pre_tokenize(prompt_view);
  EXPECT_EQ(got.size(), expected.size());
  for (auto i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i], expected[i]);
  }
}


// RegexPreTokenizer ///////////////////////////////////////////////////////////
class RegexPreTokenizerTest : public ::testing::Test {};

// Test the basic construction
TEST_F(RegexPreTokenizerTest, Construct) {
  RegexPreTokenizer ptok("[0-9]+");
}

// Test basic splitting using the expression for Tiktoken
TEST_F(RegexPreTokenizerTest, TiktokenExpr) {
  RegexPreTokenizer ptok(R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)");
  assert_split_match(ptok, "How are you doing?", {"How", " are", " you", " doing", "?"});
}

// DigitsPreTokenizer //////////////////////////////////////////////////////////
class DigitsPreTokenizerTest : public ::testing::Test {};

// Test digit splitting with individual digits
TEST_F(DigitsPreTokenizerTest, IndividualDigits) {
  DigitsPreTokenizer ptok(true);
  assert_split_match(
    ptok,
    "The number 1 then 234 then 5.",
    {"The number ", "1", " then ", "2", "3", "4", " then ", "5", "."}
  );
}

// Test digit splitting with contiguous digits
TEST_F(DigitsPreTokenizerTest, ContiguousDigits) {
  DigitsPreTokenizer ptok(false);
  assert_split_match(
    ptok,
    "The number 1 then 234 then 5.",
    {"The number ", "1", " then ", "234", " then ", "5", "."}
  );
}

// ByteLevelPreTokenizer ///////////////////////////////////////////////////////
class ByteLevelPreTokenizerTest : public ::testing::Test {};

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeDefault) {
  ByteLevelPreTokenizer ptok;
  assert_split_match(
    ptok,
    "Hello World",
    {"ĠHello", "ĠWorld"}
  );
  assert_split_match(
    ptok,
    "The number 1 then 234 then 5.",
    {"ĠThe", "Ġnumber", "Ġ1", "Ġthen", "Ġ234", "Ġthen", "Ġ5", "."}
  );
}

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeNoPrefix) {
  ByteLevelPreTokenizer ptok(false);
  assert_split_match(
    ptok,
    "Hello World",
    {"Hello", "ĠWorld"}
  );
}
