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


// RegexPreTokenizer ///////////////////////////////////////////////////////////

class RegexPreTokenizerTest : public ::testing::Test {};

TEST_F(RegexPreTokenizerTest, Construct) {
    RegexPreTokenizer tok("[0-9]+");
}

TEST_F(RegexPreTokenizerTest, PatternSplit) {
    RegexPreTokenizer tok(R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)");
    std::string text = "How are you doing?";
    std::vector<std::string> expected = {"How", " are", " you", " doing", "?"};
    re2::StringPiece text_view(text);
    const auto res = tok.pre_tokenize(text_view);
    EXPECT_EQ(res.size(), 5);
    for (auto i = 0; i < res.size(); ++i) {
        EXPECT_EQ(res[i], expected[i]);
    }
}
