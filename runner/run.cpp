/*
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

/* Inference for Llama-2 Transformer model in pure C++ */
#include "sentencepiece.h"
#include "tiktoken.h"
#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <ctype.h>
#include <iterator>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#ifdef DEBUG
#include <cassert>
#include <iostream>
#endif

#if defined(__AOTI_MODEL__) || (defined(__ET_MODEL__) && defined(USE_ATENLIB))
#include <torch/torch.h>
#endif

#ifdef __AOTI_MODEL__
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
torch::Device aoti_device(torch::kCPU);

#else // __ET_MODEL__
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#if defined(ET_USE_ADAPTIVE_THREADS)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

using exec_aten::ScalarType;
using executorch::extension::make_tensor_ptr;
using executorch::extension::TensorPtr;
using torch::executor::EValue;
using torch::executor::Module;
using torch::executor::Result;
#endif

using tokenizers::SPTokenizer;
using tokenizers::Tiktoken;
using tokenizers::Tokenizer;

#define UNWRAP(x)                                                              \
  ({                                                                           \
    if (!(x).ok()) {                                                           \
      fprintf(stderr, "Got error code % " PRIu32, x.error());                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    std::move(x.get());                                                        \
  })
// ----------------------------------------------------------------------------
// Transformer model

enum ModelType {
  UNKNOWN_MODEL = 0,
  LLAMA2_MODEL = 2,
  LLAMA3_MODEL = 3,
};

ModelType get_model_type(int model_int) {
  switch (model_int) {
  case 2:
    return LLAMA2_MODEL;
    break;
  case 3:
    return LLAMA3_MODEL;
    break;
  default:
    return UNKNOWN_MODEL;
  }
}

typedef struct {
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len;    // max sequence length
} Config;

typedef struct {
  float *logits; // output logits
  int64_t *toks; // tokens seen so far; no kv-cache :(
} RunState;

typedef struct {
  Config config;  // the hyperparameters of the architecture (the blueprint)
  RunState state; // buffers for the "wave" of activations in the forward pass
  std::unordered_map<std::string, std::string> metadata;

#ifdef __AOTI_MODEL__
  torch::inductor::AOTIModelPackageLoader *runner;
#else // __ET_MODEL__
  Module *runner;
#endif

} Transformer;

void malloc_run_state(RunState *s, Config *p) {
  // we calloc instead of malloc to keep valgrind happy
  s->logits = (float *)calloc(p->vocab_size, sizeof(float));
  s->toks = (int64_t *)calloc(p->seq_len, sizeof(int64_t));
  if (!s->logits || !s->toks) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState *s) {
  free(s->logits);
  free(s->toks);
}

void read_checkpoint(char *checkpoint, Config *config) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in the config header
  if (fread(config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
}

void build_transformer(Transformer *t, char *model_path) {
#ifdef __AOTI_MODEL__
  t->runner = new torch::inductor::AOTIModelPackageLoader(model_path);
#else //__ET_MODEL__
  t->runner = new Module(
      /* path to PTE model */ model_path,
      /* PTE mmap settings */ Module::LoadMode::MmapUseMlockIgnoreErrors);
#endif
}

void free_transformer(Transformer *t) {
  // free the RunState buffers
  free_run_state(&t->state);
  delete t->runner;
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void softmax(float *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

float *forward(Transformer *transformer, int token, int pos) {
  Config *p = &transformer->config;
  RunState *s = &transformer->state;
  s->toks[pos] = token;
  long token_buffer[1] = {token};
  long pos_buffer[1] = {pos};

#ifdef DEBUG
  std::cerr << "token: " << token << " pos: " << pos << "\n";
#endif

#ifdef __AOTI_MODEL__
  torch::Tensor token_tensor =
      torch::from_blob(token_buffer, {1, 1}, torch::kLong);
  torch::Tensor pos_tensor = torch::from_blob(pos_buffer, {1}, torch::kLong);
  std::vector<torch::Tensor> inputs{token_tensor.to(aoti_device),
                                    pos_tensor.to(aoti_device)};

  torch::Tensor result = transformer->runner->run(inputs)[0]
                             .to(torch::dtype(torch::kFloat32))
                             .to(torch::kCPU);
  auto logits = result[0].data_ptr();
  memcpy(s->logits, logits, p->vocab_size * sizeof(float));
#else // __ET_MODEL__
  TensorPtr pos_managed = make_tensor_ptr({1}, pos_buffer, ScalarType::Long);
  TensorPtr tokens_managed =
      make_tensor_ptr({1, 1}, token_buffer, ScalarType::Long);
  std::vector<EValue> inputs;
  auto tmp1 = EValue(tokens_managed);
  auto tmp2 = EValue(pos_managed);

  inputs.push_back(tmp1);
  inputs.push_back(tmp2);
  Result<std::vector<EValue>> outputs_res =
      transformer->runner->forward(inputs);
  if (!outputs_res.ok()) {
    fprintf(stderr, "Executorch forward() failed.");
    exit(EXIT_FAILURE);
  }
  std::vector<EValue> result = outputs_res.get();
  // HACK: the rest of this runner assumes that logits must be float,
  // so we simply convert them rather than plumbing
  // templating/switch-on-type through the rest of this file.
  const auto &result_tensor = result[0].toTensor();
  ET_SWITCH_REALHBBF16_TYPES(
      result_tensor.scalar_type(), unused, "forward", CTYPE, [&]() {
        const CTYPE *logits = result_tensor.const_data_ptr<CTYPE>();
        std::transform(logits, logits + p->vocab_size, s->logits,
                       [](auto x) { return static_cast<float>(x); });
      });
#endif

  return s->logits;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex,
                float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature,
                   float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex =
      (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp,
                         sampler->probindex, coin);
    }
  }
  return next;
}

Tokenizer *build_tokenizer(const char *tokenizer_path, ModelType model_type) {
  Tokenizer *tokenizer = NULL;
  switch (model_type) {
  case LLAMA2_MODEL:
    tokenizer = new SPTokenizer();
    tokenizer->load(tokenizer_path);
    break;
  case LLAMA3_MODEL:
    tokenizer = new Tiktoken();
    tokenizer->load(tokenizer_path);
    break;
  default:
    fprintf(stderr, "No tokenizer defined for model type %d.\n", model_type);
    exit(EXIT_FAILURE);
  }
  return tokenizer;
}

void free_tokenizer(Tokenizer *tokenizer) { delete tokenizer; }

// ----------------------------------------------------------------------------
// utilities: time

void safe_printf(const char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars
  // or whitespace because some of the other bytes can be various control codes,
  // backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

// Prints decoded tokens generated from the transformer.
// The first token is not printed and is assumed to be a BOS or other similar
// token
unsigned generate_from_prompt_tokens(Transformer *transformer,
                                     Tokenizer *tokenizer, Sampler *sampler,
                                     const std::vector<uint64_t> &prompt_tokens,
                                     unsigned pos,
                                     const std::vector<uint64_t> &stop_tokens,
                                     int stop_pos, bool print_prompt,
                                     bool print_tok_per_sec) {
  if (prompt_tokens.size() == 0) {
    return pos;
  }

  uint64_t next;  // will store the next token in the sequence
  uint64_t token; // stores the current token to feed into the transformer
  bool done_with_prompt; // whether we are done processing prompt

  bool found_stop_token = false; // whether we've found the stop_token after
                                 // processing prompt_tokens

  unsigned pos_in_prompt = 0; // position relative to start of prompt

  long start = 0; // timer start (initialized after first token)

  // If stop_pos == -1, we go until we find stop_token
  // If stop_pos >= 0, we go until we find stop_token or pos <= stop_pos.
  while (!found_stop_token && (stop_pos == -1 || pos <= stop_pos)) {
    // Get token and next
    if (pos_in_prompt < prompt_tokens.size()) {
      // Token comes from prompt
      token = prompt_tokens[pos_in_prompt++];
      float *logits = forward(transformer, token, pos);

      // Next token is either from prompt or if on last
      // prompt token, next is sampled
      if (pos_in_prompt < prompt_tokens.size()) {
        next = prompt_tokens[pos_in_prompt];
      } else {
        next = sample(sampler, logits);
      }
    } else {
      // Token comes from next sampled from previous round.
      token = next;
      float *logits = forward(transformer, token, pos);
      next = sample(sampler, logits);
    }
    done_with_prompt = (pos_in_prompt >= prompt_tokens.size());

    // we terminate on finding the stop_token if we are done processing the
    // prompt (stop_tokens in the prompt do not terminate the loop)
    if (done_with_prompt && (std::find(stop_tokens.begin(), stop_tokens.end(),
                                       token) != stop_tokens.end())) {
      found_stop_token = true;
    }

    // We print next in each iteration of the loop, not token
    if (!found_stop_token && (print_prompt || done_with_prompt)) {
      // The stop_token is printed as newline
      bool next_is_stop = std::find(stop_tokens.begin(), stop_tokens.end(),
                                    next) != stop_tokens.end();
      if (next_is_stop) {
        printf("\n");
      } else {
        std::string piece = UNWRAP(tokenizer->decode(token, next));
        safe_printf(piece.c_str()); // same as printf("%s", piece), but skips
                                    // "unsafe" bytes
        fflush(stdout);
      }
    }

    // init the timer here because the first iteration can be slower
    if (pos == 0) {
      start = time_in_ms();
    }
    pos++;
  }

  // report achieved tok/s (pos-1 because the timer starts after first
  // iteration)
  if (print_tok_per_sec && pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "\n\nachieved tok/s: %f\n",
            (pos - 1) / (double)(end - start) * 1000);
  }

  return pos;
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
              const char *prompt, int steps, ModelType model_type) {
  const char *default_prompt = "Once upon a time";
  if (prompt == NULL) {
    prompt = default_prompt;
  }

  if (steps == 0) {
    return;
  }

  std::vector<uint64_t> prompt_tokens;
  std::vector<uint64_t> stop_tokens;
  switch (model_type) {
  case LLAMA2_MODEL:
    prompt_tokens = UNWRAP(tokenizer->encode(prompt, 1, 0));
    stop_tokens.push_back(tokenizer->eos_tok());
    break;
  case LLAMA3_MODEL:
    prompt_tokens = UNWRAP(tokenizer->encode(prompt, 1, 0));
    stop_tokens.push_back(
        UNWRAP(tokenizer->encode("<|end_of_text|>", 0, 0))[0]);
    stop_tokens.push_back(UNWRAP(tokenizer->encode("<|eot_id|>", 0, 0))[0]);
    break;
  default:
    fprintf(stderr, "Generate does not support model type %d.\n", model_type);
    exit(EXIT_FAILURE);
  }

  generate_from_prompt_tokens(transformer, tokenizer, sampler, prompt_tokens,
                              /*pos=*/0,
                              /*stop_tokens=*/stop_tokens,
                              /*stop_pos=*/steps - 1,
                              /*print_prompt=*/true,
                              /*print_tok_per_sec=*/true);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

std::vector<uint64_t> get_initial_prompt_tokens(const char *cli_system_prompt,
                                                const char *cli_user_prompt,
                                                Tokenizer *tokenizer,
                                                ModelType model_type) {
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[512 * 2 + 200]; // the prompt template is ~170
                                       // characters.  We use 200 to be safe.

  if (cli_system_prompt != NULL) {
    strcpy(system_prompt, cli_system_prompt);
  } else {
    read_stdin("Enter system prompt (optional): ", system_prompt,
               sizeof(system_prompt));
  }

  if (cli_user_prompt != NULL) {
    strcpy(user_prompt, cli_user_prompt);
  } else {
    read_stdin("User: ", user_prompt, sizeof(user_prompt));
  }

  std::vector<uint64_t> tokens;

  switch (model_type) {
  case LLAMA2_MODEL:
    if (system_prompt[0] != '\0') {
      snprintf(rendered_prompt, sizeof(rendered_prompt) - 1,
               "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]", system_prompt,
               user_prompt);
    } else {
      snprintf(rendered_prompt, sizeof(rendered_prompt) - 1,
               "[INST] %s [/INST]", user_prompt);
    }

    // We need to add BOS token here and not in template because llama2
    // tokenizer does not pattern match special tokens
    tokens = UNWRAP(tokenizer->encode(rendered_prompt, 1, 0));
    break;

  case LLAMA3_MODEL:
    if (system_prompt[0] != '\0') {
      snprintf(rendered_prompt, sizeof(rendered_prompt) - 1,
               "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
               "\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<"
               "|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
               system_prompt, user_prompt);
    } else {
      snprintf(rendered_prompt, sizeof(rendered_prompt) - 1,
               "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%"
               "s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
               user_prompt);
    }
    tokens = UNWRAP(tokenizer->encode(rendered_prompt, 0, 0));
    break;

  default:
    fprintf(stderr, "Chat does not support model type %d.\n", model_type);
    exit(EXIT_FAILURE);
  }

#ifdef DEBUG
  std::cerr << "Start of rendered prompt:" << std::endl;
  std::cerr << rendered_prompt;
  std::cerr << "End of rendered prompt:" << std::endl;
  std::cerr << "Encoded prompt: ";
  for (int i = 0; i < tokens.size(); i++) {
    std::cerr << tokens[i] << ", ";
  }
  std::cerr << std::endl << std::flush;
#endif

  return tokens;
}

std::vector<uint64_t> get_next_user_prompt_tokens(Tokenizer *tokenizer,
                                                  ModelType model_type) {
  char user_prompt[512];
  char rendered_prompt[512 + 150]; // the prompt template is ~100 characters. We
                                   // use 150 to be safe.

  read_stdin("User: ", user_prompt, sizeof(user_prompt));
  std::vector<uint64_t> tokens;

  switch (model_type) {
  case LLAMA2_MODEL:
    snprintf(rendered_prompt, sizeof(rendered_prompt) - 1, "[INST] %s [/INST]",
             user_prompt);

    // We need to add BOS token here and not in template because llama2
    // tokenizer does not pattern match special tokens
    tokens = UNWRAP(tokenizer->encode(rendered_prompt, /*bos*/ 1, /*eos*/ 0));
    break;

  case LLAMA3_MODEL:
    snprintf(rendered_prompt, sizeof(rendered_prompt) - 1,
             "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_"
             "header_id|>assistant<|end_header_id|>\n\n",
             user_prompt);
    tokens = UNWRAP(tokenizer->encode(rendered_prompt, 0, 0));
    break;

  default:
    fprintf(stderr, "Chat does not support model type %d.\n", model_type);
    exit(EXIT_FAILURE);
  }

#ifdef DEBUG
  std::cerr << "Start of rendered prompt:" << std::endl;
  std::cerr << rendered_prompt;
  std::cerr << "End of rendered prompt:" << std::endl;
  std::cerr << "Encoded prompt: ";
  for (int i = 0; i < tokens.size(); i++) {
    std::cerr << tokens[i] << ", ";
  }
  std::cerr << std::endl << std::flush;
#endif

  return tokens;
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          const char *cli_user_prompt, const char *cli_system_prompt,
          unsigned steps, ModelType model_type) {
  if (steps == 0) {
    return;
  }

  uint64_t eot_token;
  std::vector<uint64_t> prompt_tokens;
  switch (model_type) {
  case LLAMA2_MODEL:
    // llama2 uses EOS as EOT token
    eot_token = tokenizer->eos_tok();
    break;
  case LLAMA3_MODEL:
    eot_token = UNWRAP(tokenizer->encode("<|eot_id|>", 0, 0))[0];
    break;
  default:
    fprintf(stderr, "Chat does not support model type %d.\n", model_type);
    exit(EXIT_FAILURE);
  }

  std::vector<uint64_t> stop_tokens{eot_token};
  unsigned pos = 0;
  while (pos < steps) {
    if (pos == 0) {
      prompt_tokens = get_initial_prompt_tokens(
          cli_system_prompt, cli_user_prompt, tokenizer, model_type);
    } else {
      prompt_tokens = get_next_user_prompt_tokens(tokenizer, model_type);
    }
    printf("Assistant: ");
    pos = generate_from_prompt_tokens(
        transformer, tokenizer, sampler, prompt_tokens, pos,
        /*stop_tokens=*/stop_tokens,
        /*stop_pos=*/steps - 1, // We could pass in -1 here if we do not want
                                // the model to stop mid-reply
        /*print_prompt=*/false,
        /*print_tok_per_sec=*/false);
  }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <model_path> [options]\n");
  fprintf(stderr,
          "Example: run model.{so,pte} -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], "
                  "default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = "
                  "max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> path to tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(stderr,
          "  -v <int>    (optional) vocab size, default is model-specific.\n");
  fprintf(stderr,
          "  -l <int>    (optional) llama version (2 or 3), default 2.\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  // default parameters
  char *model_path = NULL;
  char *tokenizer_path = NULL;
  float temperature =
      1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f; // top-p in nucleus sampling. 1.0 = off. 0.9 works well,
                     // but slower

  int steps = 128;                 // number of steps to run for
  const char *prompt = NULL;       // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  const char *mode = "generate";   // generate|chat
  char *system_prompt =
      NULL; // the (optional) system prompt to use in chat mode

  int vocab_size = -1;
  int llama_ver = 2;

#if defined(ET_USE_ADAPTIVE_THREADS)
  uint32_t num_performant_cores =
      torch::executorch::cpuinfo::get_num_performant_cores();
  if (num_performant_cores > 0) {
    torch::executorch::threadpool::get_threadpool()->_unsafe_reset_threadpool(
        num_performant_cores);
  }
#endif
  // poor man's C argparse so we can override the defaults above from the
  // command line
  if (argc >= 2) {
    model_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 1) {
    // do some basic validation
    char *parm = argv[i+1];
    // uniarg means the arg comes right after the letter in accordance with posix
    int uniarg = strlen(argv[i]) > 2; 

    if (argv[i][0] != '-') {
      error_usage();
    } // must start with dash

    if (strlen(argv[i]) < 2) {
      error_usage();
    } // must have at least dash '-' and option letter
    
    if (uniarg) {
      parm=&argv[i][2];
    } else if (i + 1 >= argc) {
      error_usage();
    } // must have arg after option if flag is not contiguous to option
    
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(parm);
    } else if (argv[i][1] == 'p') {
      topp = atof(parm);
    } else if (argv[i][1] == 's') {
      rng_seed = atoi(parm);
    } else if (argv[i][1] == 'n') {
      steps = atoi(parm);
    } else if (argv[i][1] == 'v') {
      vocab_size = atoi(parm);
    } else if (argv[i][1] == 'i') {
      prompt = parm;
    } else if (argv[i][1] == 'z') {
      tokenizer_path = parm;
    } else if (argv[i][1] == 'm') {
      mode = parm;
    } else if (argv[i][1] == 'y') {
      system_prompt = parm;
    } else if (argv[i][1] == 'l') {
      llama_ver = atoi(parm);
    } else {
      error_usage();
    }

    // account for parameter
    i += (uniarg)?0:1;
  }

  if (model_path == NULL) {
    fprintf(stderr, "No model_path provided.");
    error_usage();
  }

  Transformer transformer;
  build_transformer(&transformer, model_path);

#ifdef __AOTI_MODEL__
  auto aoti_metadata = transformer.runner->get_metadata();
  aoti_device = aoti_metadata["AOTI_DEVICE_KEY"] == "cpu"
                    ? torch::Device(torch::kCPU)
                    : torch::Device(torch::kCUDA);
  ModelType model_type = get_model_type(std::stoi(aoti_metadata["tokenizer_type"]));
#else // __ET_MODEL__
  ModelType model_type = get_model_type(llama_ver);
#endif

  if (model_type == UNKNOWN_MODEL) {
    fprintf(stderr, "Unknown model type passed by -l argument.  Received l=%d.",
            llama_ver);
    error_usage();
  }

  if (tokenizer_path == NULL) {
    fprintf(stderr, "No tokenizer_path provided.");
    error_usage();
  }

  // parameter validation/overrides
  if (rng_seed <= 0)
    rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0)
    temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp)
    topp = 0.9;
  if (steps < 0)
    steps = 0;

  Tokenizer *tokenizer = build_tokenizer(tokenizer_path, model_type);

  // If no tokenizer path provided, get default for model_type
  if (vocab_size == -1) {
    vocab_size = tokenizer->vocab_size();
  }

  // read in the Config and the Weights from the model
  // read_checkpoint(model_path, &t->config);
  // allocate the RunState buffers
  transformer.config.vocab_size = vocab_size;
  transformer.config.seq_len = steps;
  malloc_run_state(&transformer.state, &transformer.config);

  Sampler sampler;
  build_sampler(&sampler, vocab_size, temperature, topp, rng_seed);

  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, tokenizer, &sampler, prompt, steps, model_type);
  } else if (strcmp(mode, "chat") == 0) {
    chat(&transformer, tokenizer, &sampler, prompt, system_prompt, steps,
         model_type);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(tokenizer);
  free_transformer(&transformer);
  return 0;
}
#endif
