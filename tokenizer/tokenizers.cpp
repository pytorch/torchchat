/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "tokenizer.h"

// Standard
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

// Third Party
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// // ------------------------------Util start------------------------------------

// static uint64_t _max_size() {
//   return std::numeric_limits<uint64_t>::max();
// }

// static Re2UPtr _create_regex(const std::string& pattern) {
//   assert(!pattern.empty());

//   return std::make_unique<re2::RE2>("(" + pattern + ")");
// }

// static Re2UPtr _build_special_token_regex(const Encoder& special_encoder) {
//   std::string special_pattern;
//   for (const auto& ele : special_encoder) {
//     if (!special_pattern.empty()) {
//       special_pattern += "|";
//     }
//     special_pattern += re2::RE2::QuoteMeta(ele.first);
//   }

//   if (special_pattern.empty()) {
//     return nullptr;
//   }

//   return _create_regex(special_pattern);
// }

// static std::pair<std::string, uint64_t> _parse(const std::string& line) {
//   auto pos = line.find(" ");
//   if (pos == std::string::npos) {
//     throw std::invalid_argument("invalid encoder line: " + line);
//   }

//   auto token = base64::decode({line.data(), pos});
//   uint64_t rank = 0;
//   try {
//     rank = std::stoul(line.substr(pos + 1));
//   } catch (const std::exception&) {
//     throw std::invalid_argument("invalid encoder rank:  " + line);
//   }

//   return {std::move(token), rank};
// }

// static Encoder _load_encoder(const std::string& path) {
//   std::ifstream file(path);
//   if (!file) {
//     fprintf(stderr, "failed to open encoder file: %s\n", path.c_str());
//     exit(EXIT_FAILURE);
//   }

//   Encoder encoder;
//   std::string line;
//   while (std::getline(file, line)) {
//     auto [token, rank] = _parse(line);

//     if (!encoder.emplace(std::move(token), rank).second) {
//       fprintf(stderr, "duplicate item: %s\n", line.c_str());
//     }
//   }
//   return encoder;
// }

// static Decoder _build_decoder(const Encoder& encoder) {
//   Decoder decoder;
//   for (const auto& [k, v] : encoder) {
//     decoder.emplace(v, k);
//   }

//   if (encoder.size() != decoder.size()) {
//     fprintf(stderr, "duplicate items in encoder");
//     exit(EXIT_FAILURE);
//   }

//   return decoder;
// }

// static std::vector<uint64_t> _byte_pair_merge(
//     const std::string& piece,
//     const std::unordered_map<std::string, uint64_t>& ranks,
//     std::function<uint64_t(uint64_t, uint64_t)> func) {
//   // This is a vector of (start, rank).
//   // The rank is of the byte pair starting at position start.
//   // The rank of the last item in the vector is not a valid value.
//   std::vector<std::pair<uint64_t, uint64_t>> parts;
//   parts.reserve(piece.size() + 1);
//   for (auto idx = 0U; idx < piece.size() + 1; ++idx) {
//     parts.emplace_back(idx, _max_size());
//   }

//   auto get_rank = [&piece, &ranks](
//                       const std::vector<std::pair<uint64_t, uint64_t>>& parts,
//                       uint64_t start_idx,
//                       uint64_t skip) -> std::optional<uint64_t> {
//     if (start_idx + skip + 2 < parts.size()) {
//       auto s = parts[start_idx].first;
//       auto e = parts[start_idx + skip + 2].first;
//       auto key = piece.substr(s, e - s);
//       auto iter = ranks.find(key);
//       if (iter != ranks.end()) {
//         return iter->second;
//       }
//     }
//     return std::nullopt;
//   };

//   // We look up the ranks once in the beginning and iteratively update
//   // them during each merge, which reduces the number of rank lookups.
//   for (auto i = 0U; i < parts.size() - 2; ++i) {
//     auto rank = get_rank(parts, i, 0);
//     if (rank) {
//       // usize::MAX is a sentinel value and cannot be a valid rank
//       if (*rank == _max_size()) {
//         fprintf(stderr, "at %" PRIu32 " rank is too large\n", i);
//       }
//       parts[i].second = *rank;
//     }
//   }

//   // If you have n parts and m merges, this does O(mn) work.
//   // We could do something with a heap and do O(m log n) work.
//   // It is important to consider that n is often small (<100), and as such
//   // the cache-locality benefits outweigh the algorithmic complexity downsides
//   // of the `parts` vector data structure above.

//   // Note that we hash bytes, not token pairs. As long as we train BPE the way
//   // we currently do, this is equivalent. An easy way to break this would be
//   // to decouple merge priority from token index or to prevent specific token
//   // merges.
//   while (true) {
//     if (parts.size() == 1) {
//       break;
//     }

//     // usize::MAX is a sentinel rank value allowing us to
//     // take the min more quickly
//     auto min_rank = std::make_pair<uint64_t, uint64_t>(_max_size(), 0);
//     for (auto i = 0U; i < parts.size() - 1; ++i) {
//       auto rank = parts[i].second;
//       if (rank < min_rank.first) {
//         min_rank.first = rank;
//         min_rank.second = i;
//       }
//     }

//     if (min_rank.first != _max_size()) {
//       auto i = min_rank.second;

//       // NOTE: We are about to remove parts[i + 1]. We do not do it
//       // yet because there are cache-locality benefits to updating
//       // parts[i] and parts[i-1] before removing, which could thrash
//       // the cache. Thus, we update the rank calculation by skipping over
//       // parts[i + 1], by invoking `get_rank!` with `skip = 1`.
//       auto rank = get_rank(parts, i, 1);
//       if (rank) {
//         parts[i].second = *rank;
//       } else {
//         parts[i].second = _max_size();
//       }
//       if (i > 0) {
//         rank = get_rank(parts, i - 1, 1);
//         if (rank) {
//           parts[i - 1].second = *rank;
//         } else {
//           parts[i - 1].second = _max_size();
//         }
//       }

//       parts.erase(parts.begin() + (i + 1));
//     } else {
//       break;
//     }
//   }
//   std::vector<uint64_t> out;
//   out.reserve(parts.size() - 1);
//   for (auto i = 0U; i < parts.size() - 1; ++i) {
//     auto s = parts[i].first;
//     auto e = parts[i + 1].first;
//     out.push_back(func(s, e));
//   }
//   return out;
// }

// static std::vector<uint64_t> _byte_pair_encode(
//     const std::string& piece,
//     const Encoder& encoder) {
//   if (piece.size() == 1) {
//     auto iter = encoder.find(piece);
//     if (iter != encoder.end()) {
//       return std::vector<uint64_t>({iter->second});
//     } else {
//       // TODO: is it possible?
//       return {};
//     }
//   }

//   return _byte_pair_merge(
//       piece, encoder, [&piece, &encoder](uint64_t start, uint64_t stop) {
//         std::string key = piece.substr(start, stop - start);
//         auto iter = encoder.find(key);
//         if (iter != encoder.end()) {
//           return iter->second;
//         } else {
//           // TODO: what if key does not exist? Should we return `unknown`?
//           // assert(false); // ??
//           return uint64_t(0);
//         }
//       });
// }
// // ------------------------------Util end------------------------------------
// // -------------------------private method start-------------------------------

// template <typename T>
// std::pair<std::optional<std::string>, re2::StringPiece>
// Tiktoken::_split_with_allowed_special_token(
//     re2::StringPiece& input,
//     const T& allowed_special) {
//   if (!_special_token_regex) {
//     return std::make_pair(std::nullopt, input);
//   }

//   auto start = input.begin();
//   std::string special;
//   while (true) {
//     if (!re2::RE2::FindAndConsume(&input, *_special_token_regex, &special)) {
//       // No special token.
//       break;
//     }

//     if (allowed_special.count(special) == 1) {
//       // Found an allowed special token, split the text with it.
//       return std::make_pair(
//           special,
//           re2::StringPiece(start, input.begin() - start - special.size()));
//     } // else try to find the next special token
//   }

//   return std::make_pair(std::nullopt, input);
// }

// void Tiktoken::_encode(
//     re2::StringPiece& input,
//     std::vector<uint64_t>& ret,
//     uint64_t& last_piece_token_len) {
//   std::string piece;
//   assert(_regex);
//   while (re2::RE2::FindAndConsume(&input, *_regex, &piece)) {
//     auto iter = _encoder.find(piece);
//     if (iter != _encoder.end()) {
//       last_piece_token_len = 1;
//       ret.push_back(iter->second);
//       continue;
//     }
//     auto tokens = _byte_pair_encode(piece, _encoder);
//     last_piece_token_len = tokens.size();
//     ret.insert(ret.end(), tokens.begin(), tokens.end());
//   }
// }

// template <typename T>
// std::pair<std::vector<uint64_t>, uint64_t> Tiktoken::_encode_with_special_token(
//     const std::string& text,
//     const T& allowed_special) {
//   std::vector<uint64_t> tokens;
//   uint64_t last_piece_token_len = 0;
//   re2::StringPiece input(text);
//   while (true) {
//     auto [special, sub_input] =
//         _split_with_allowed_special_token(input, allowed_special);

//     _encode(sub_input, tokens, last_piece_token_len);

//     if (special) {
//       uint64_t token = 0;
//       try {
//         token = _special_token_encoder.at(*special);
//       } catch (const std::out_of_range&) {
//         // Should never go here, since special pattern includes all special
//         // chars.
//         fprintf(stderr, "unknown special token: %s\n", special->c_str());
//         exit(EXIT_FAILURE);
//       }

//       tokens.push_back(token);
//       last_piece_token_len = 0;
//     } else {
//       break;
//     }
//   }

//   // last_piece_token_len is how many tokens came from the last regex split.
//   // This is used for determining unstable tokens, since you can't merge
//   // across (stable) regex splits
//   return std::make_pair(tokens, last_piece_token_len);
// }


// -------------------------private method end-------------------------------
// -------------------------public method start-------------------------------

TokenizersTokenizer::TokenizersTokenizer() : Tiktoken() {}

void TokenizersTokenizer::load(const std::string& path) {

  // If this is a directory, look for tokenizer.json and tokenizer_config.json
  std::string model_json = path;
  std::string model_config_json = "";
  if (fs::is_directory(path)) {
    const fs::path root(path);
    model_json = root / "tokenizer.json";
    if (!fs::exists(model_json)) {
      fprintf(stderr, "no tokenizer.json found in %s\n", path.c_str());
      exit(EXIT_FAILURE);
    }
    const auto model_config_json_path = root / "tokenizer_config.json";
    if (fs::exists(model_config_json_path)) {
      model_config_json = model_config_json_path;
    }
  }

  // Load the tokenizer.json file
  std::ifstream file(model_json);
  if (!file) {
    fprintf(stderr, "failed to open encoder file: %s\n", path.c_str());
    exit(EXIT_FAILURE);
  }
  std::string contents(
    (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  json parsed_json;
  try {
    parsed_json = json::parse(contents);
  } catch (const json::exception& e) {
    std::cout << "Error parsing json file: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  // Parse the special tokens
  try {
    const auto& special_tokens = parsed_json.at("added_tokens");
    for (auto it = special_tokens.begin(); it != special_tokens.end(); ++it) {
      const std::string token = it->at("content");
      const uint64_t token_id = it->at("id");
      if (!special_token_encoder_.emplace(token, token_id).second) {
        fprintf(stderr, "duplicate special token: %s\n", token.c_str());
        exit(EXIT_FAILURE);
      }
      if (!special_token_decoder_.emplace(token_id, token).second) {
        fprintf(stderr, "duplicate special token id: %llu\n", token_id);
        exit(EXIT_FAILURE);
      }
    }
  } catch (const json::out_of_range& e) {
    fprintf(stderr, "Could not parse special tokens: %s\n", e.what());
    exit(EXIT_FAILURE);
  }

  // Parse the standard tokens
  try {
    const auto& vocab = parsed_json.at("/model/vocab"_json_pointer);
    for (const auto& entry : vocab.items()) {
      const std::string token = entry.key();
      const uint64_t token_id = entry.value();
      // Skip adding special tokens to the standard encoder/decoder
      if (special_token_decoder_.find(token_id) == special_token_decoder_.end()) {
        if (!encoder_.emplace(token, token_id).second) {
          fprintf(stderr, "duplicate token: %s\n", token.c_str());
          exit(EXIT_FAILURE);
        }
        if (!decoder_.emplace(token_id, token).second) {
          fprintf(stderr, "duplicate token id: %llu\n", token_id);
          exit(EXIT_FAILURE);
        }
      }
    }
  } catch (const json::out_of_range& e) {
    fprintf(stderr, "Could not parse tokens: %s\n", e.what());
    exit(EXIT_FAILURE);
  }

  // Set the vocab size to include special tokens
  vocab_size_ = encoder_.size() + special_token_encoder_.size();

  // Parse out the regexes for the pre-tokenizer
  try {
    const auto& pre_tokenizer = parsed_json.at("pre_tokenizer");
    const std::string pt_type = pre_tokenizer.at("type");
    if (pt_type == "Sequence") {
      for (const auto& entry : pre_tokenizer.at("pretokenizers")) {
        const std::string entry_type = entry.at("type");
        std::string regex;
        if (entry_type == "Split") {
          regex = entry.at("Regex");
        } else if (entry_type == "Digits") {
          regex = "\\p{N}";
        } else if (entry_type == "ByteLevel") {
          //HACK! This should be better.....
          regex =  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+";
        } else {
          fprintf(stderr, "Unknown pre-tokenizer type: %s\n", entry_type.c_str());
          exit(EXIT_FAILURE);
        }

        // Build the regex and add it to the list
        regexes_.push_back(_create_regex(regex));
      }
    } else {
      fprintf(stderr, "Unknown pre-tokenizer type: %s\n", pt_type.c_str());
      exit(EXIT_FAILURE);
    }
  } catch (const json::out_of_range& e) {
    fprintf(stderr, "Could not parse pre_tokenizer: %s\n", e.what());
    exit(EXIT_FAILURE);
  }

  // TODO: Do we need to parse the merges?

  // If a tokenizer config file is found, parse it to look up the eos/bos tokens
  if (!model_config_json.empty()) {

    // Load it and parse it as json
    std::ifstream file(model_config_json);
    if (!file) {
      fprintf(stderr, "failed to open encoder file: %s\n", path.c_str());
      exit(EXIT_FAILURE);
    }
    std::string contents(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    json parsed_json;
    try {
      parsed_json = json::parse(contents);
    } catch (const json::exception& e) {
      std::cout << "Error parsing model config json json file: " << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }

    // Pull out the token strings
    try {
      const std::string bos_token = parsed_json.at("bos_token");
      const std::string eos_token = parsed_json.at("eos_token");
      const auto& bos_it = special_token_encoder_.find(bos_token);
      const auto& eos_it = special_token_encoder_.find(eos_token);
      if (bos_it == special_token_encoder_.end()) {
        fprintf(stderr, "BOS token %s not in special tokens\n", bos_token.c_str());
        exit(EXIT_FAILURE);
      }
      if (eos_it == special_token_encoder_.end()) {
        fprintf(stderr, "EOS token %s not in special tokens\n", eos_token.c_str());
        exit(EXIT_FAILURE);
      }
      bos_tok_ = bos_it->second;
      eos_tok_ = eos_it->second;
    } catch (const json::out_of_range& e) {
      fprintf(stderr, "Could not eos/bos from tokenizer config: %s\n", e.what());
      exit(EXIT_FAILURE);
    }
  }

  // Otherwise, make an educated guess with the following logic:
  // 1. Look for special tokens with "bos"/"begin" or "eos"/"end" in them
  // 2. Sub-qualify with the word "text" if needed
  // 3. If EOS found, but BOS is not (or vice versa), assume they are the same
  else {
    std::vector<std::string> bos_candidates;
    std::vector<std::string> eos_candidates;
    for (const auto& token : special_token_encoder_) {
      if (
        token.first.find("bos") != std::string::npos ||
        token.first.find("begin") != std::string::npos
      ) {
        bos_candidates.push_back(token.first);
      }
      if (
        token.first.find("eos") != std::string::npos ||
        token.first.find("end") != std::string::npos
      ) {
        eos_candidates.push_back(token.first);
      }
    }
    if (bos_candidates.size() > 1) {
      const auto orig_candidates = bos_candidates;
      bos_candidates.clear();
      for (const auto& cand : orig_candidates) {
        if (cand.find("text") != std::string::npos) {
          bos_candidates.push_back(cand);
        }
      }
    }
    if (eos_candidates.size() > 1) {
      const auto orig_candidates = eos_candidates;
      eos_candidates.clear();
      for (const auto& cand : orig_candidates) {
        if (cand.find("text") != std::string::npos) {
          eos_candidates.push_back(cand);
        }
      }
    }

    // Use if a single candidate
    bool bos_found = false;
    bool eos_found = false;
    if (bos_candidates.size() == 1) {
      bos_found = true;
      bos_tok_ = special_token_encoder_[bos_candidates[0]];
    }
    if (eos_candidates.size() == 1) {
      eos_found = true;
      eos_tok_ = special_token_encoder_[eos_candidates[0]];
    }

    // Make them the same if only one found
    if (bos_found && ! eos_found) {
      eos_tok_ = bos_tok_;
    } else if (! bos_found && eos_found) {
      bos_tok_ = eos_tok_;
    }
  }

  // Mark initialized once everything is done
  initialized_ = true;
}

std::vector<uint64_t>
TokenizersTokenizer::encode(const std::string& text, int8_t bos, int8_t eos) const {
  // if (!initialized_) {
  //   exit(EXIT_FAILURE);
  // }
  // auto res = _encode_with_special_token(text, _special_token_encoder).first;
  // for (auto i = 0; i < bos; ++i) {
  //   res.insert(res.begin(), bos_tok_);
  // }
  // for (auto i = 0; i < eos; ++i) {
  //   res.push_back(eos_tok_);
  // }
  // return res;

  //DEBUG
  return {};
}

std::string TokenizersTokenizer::decode(uint64_t prev, uint64_t cur) const {
  // (void)prev;
  // if (!initialized_) {
  //   exit(EXIT_FAILURE);
  // }
  // std::string ret;

  // std::string token_bytes;
  // auto iter = _decoder.find(cur);
  // if (iter != _decoder.end()) {
  //   token_bytes = iter->second;
  // } else {
  //   iter = _special_token_decoder.find(cur);
  //   if (iter != _special_token_decoder.end()) {
  //     token_bytes = iter->second;
  //   } else {
  //     fprintf(stderr, "unknown token: %" PRIu64 "\n", cur);
  //     exit(EXIT_FAILURE);
  //   }
  // }
  // ret += token_bytes;

  // return ret;

  //DEBUG
  return "";
}
// -------------------------public method end-------------------------------
