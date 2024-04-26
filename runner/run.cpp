/* Inference for Llama-2 Transformer model in pure C++ */
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tokenizer.h>

#ifdef DEBUG
#include <cassert>
#include <iostream>
#endif

#if defined(__AOTI_MODEL__) || (defined(__ET_MODEL__) && defined(USE_ATENLIB))
#include <torch/torch.h>
#endif

#ifdef __AOTI_MODEL__
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#else // __ET_MODEL__
#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/managed_tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#if defined(ET_USE_ADPATIVE_THREADS)
#include <executorch/backends/xnnpack/threadpool/cpuinfo_utils.h>
#include <executorch/backends/xnnpack/threadpool/threadpool.h>
#endif

using exec_aten::ScalarType;
using torch::executor::EValue;
using torch::executor::ManagedTensor;
using torch::executor::Module;
using torch::executor::Result;
#endif

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len; // max sequence length
} Config;

typedef struct {
  float* logits; // output logits
  int64_t* toks; // tokens seen so far; no kv-cache :(
} RunState;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  RunState state; // buffers for the "wave" of activations in the forward pass

#ifdef __AOTI_MODEL__
  torch::inductor::AOTIModelContainerRunnerCpu* runner;
#else // __ET_MODEL__
  Module* runner;
#endif

} Transformer;

void malloc_run_state(RunState* s, Config* p) {
  // we calloc instead of malloc to keep valgrind happy
  s->logits = (float*)calloc(p->vocab_size, sizeof(float));
  s->toks = (int64_t*)calloc(p->seq_len, sizeof(int64_t));
  if (!s->logits || !s->toks) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState* s) {
  free(s->logits);
  free(s->toks);
}

void read_checkpoint(char* checkpoint, Config* config) {
  FILE* file = fopen(checkpoint, "rb");
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

void build_transformer(
    Transformer* t,
    char* checkpoint_path,
    int vocab_size,
    int seq_len) {
  // read in the Config and the Weights from the checkpoint
  // read_checkpoint(checkpoint_path, &t->config);
  // allocate the RunState buffers
  t->config.vocab_size = vocab_size;
  t->config.seq_len = seq_len;
  malloc_run_state(&t->state, &t->config);

#ifdef __AOTI_MODEL__
  t->runner = new torch::inductor::AOTIModelContainerRunnerCpu(
      /* path to model DSO */ checkpoint_path,
      /* thread pool size  */ 1);
#else //__ET_MODEL__
  t->runner = new Module(
      /* path to PTE model */ checkpoint_path,
      /* PTE mmap settings */ Module::MlockConfig::UseMlockIgnoreErrors);
#endif
}

void free_transformer(Transformer* t) {
  // free the RunState buffers
  free_run_state(&t->state);
  delete t->runner;
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void softmax(float* x, int size) {
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

float* forward(Transformer* transformer, int token, int pos) {
  Config* p = &transformer->config;
  RunState* s = &transformer->state;
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
  std::vector<torch::Tensor> inputs{token_tensor, pos_tensor};

  torch::Tensor result = transformer->runner->run(inputs)[0].to(torch::dtype(torch::kFloat32));
  auto logits = result[0].data_ptr();

#else // __ET_MODEL__
  ManagedTensor pos_managed(pos_buffer, sizeof(int64_t), {1}, ScalarType::Long);
#ifndef __KV_CACHE__
  // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
  ManagedTensor tokens_managed(
      &(s->toks[pos]),
      /*ignored*/ sizeof(int64_t) * (pos + 1),
      {1, 1},
      ScalarType::Long);
#else // __KV_CACHE__
  ManagedTensor tokens_managed(
      token_buffer, sizeof(int64_t), {1, 1}, ScalarType::Long);
#endif
  std::vector<EValue> inputs;
  auto tmp1 = EValue(tokens_managed.get_aliasing_tensor());
  auto tmp2 = EValue(pos_managed.get_aliasing_tensor());

  inputs.push_back(tmp1);
  inputs.push_back(tmp2);
  Result<std::vector<EValue>> outputs_res =
      transformer->runner->forward(inputs);
  if (!outputs_res.ok()) {
    fprintf(stderr, "Executorch forward() failed.");
    exit(EXIT_FAILURE);
  }
  std::vector<EValue> result = outputs_res.get();
  auto logits = result[0].toTensor().const_data_ptr();
#endif

  memcpy(s->logits, logits, p->vocab_size * sizeof(float));
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
  ProbIndex* probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
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

int sample_mult(float* probabilities, int n, float coin) {
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

int compare(const void* a, const void* b) {
  ProbIndex* a_ = (ProbIndex*)a;
  ProbIndex* b_ = (ProbIndex*)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(
    float* probabilities,
    int n,
    float topp,
    ProbIndex* probindex,
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

void build_sampler(
    Sampler* sampler,
    int vocab_size,
    float temperature,
    float topp,
    unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex =
      (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
  free(sampler->probindex);
}

unsigned int random_u32(unsigned long long* state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
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
      next = sample_topp(
          logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

void safe_printf(const char* piece) {
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

void generate(
    Transformer* transformer,
    Tokenizer* tokenizer,
    Sampler* sampler,
    const char* prompt,
    int steps) {
  const char* default_prompt = "Once upon a time";
  if (prompt == NULL) {
    prompt = default_prompt;
  }

  // encode the (string) prompt into tokens sequence
  std::string prompt_str = prompt;
  std::vector<uint64_t> prompt_tokens = tokenizer->encode(prompt_str, 1, 0);
  int num_prompt_tokens = prompt_tokens.size();
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

#ifdef DEBUG
  std::cerr << "# " << num_prompt_tokens << "\n";
  for (int i = 0; i < num_prompt_tokens; i++)
    std::cerr << "[" << i << "] " << prompt_tokens[i];
  std::cerr << "\n";
#endif

  // start the main loop
  long start =
      0; // used to time our code, only initialized after first iteration
  int next; // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0; // position in the sequence
  while (pos < steps) {
    // forward the transformer to get logits for the next token
    float* logits = forward(transformer, token, pos);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits
    // sequences
    if (next == 1) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    std::string res = tokenizer->decode(token, next);
    safe_printf(
        res.c_str()); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = time_in_ms();
    }
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first
  // iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(
        stderr,
        "achieved tok/s: %f\n",
        (pos - 1) / (double)(end - start) * 1000);
  }
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
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

void chat(
    Transformer* transformer,
    Tokenizer* tokenizer,
    Sampler* sampler,
    const char* cli_user_prompt,
    const char* cli_system_prompt,
    int steps) {
  // special tokens
  const int SOS_TOKEN = tokenizer->bos_tok(); // token starts the assistant turn
  const int EOS_TOKEN = tokenizer->eos_tok(); // token ends the assistant turn
  const int SYSTEM_PROMPT_SIZE = 512;
  const int USER_PROMPT_SIZE = 512;
  const int RENDERED_PROMPT_SIZE = SYSTEM_PROMPT_SIZE + USER_PROMPT_SIZE + 128; // This is big enough to hold the expanded template



  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are soomewhat haphazardly and unsafely set atm
  char system_prompt[SYSTEM_PROMPT_SIZE];
  char user_prompt[USER_PROMPT_SIZE];
  char rendered_prompt[RENDERED_PROMPT_SIZE];
  int num_prompt_tokens = 0;
  std::vector<uint64_t> prompt_tokens;
  int user_idx;

  // start the main loop
  int8_t user_turn = 1; // user starts
  int next; // will store the next token in the sequence
  int token; // stores the current token to feed into the transformer
  int prev_token;
  int pos = 0; // position in the sequence
  while (pos < steps) {

    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        if (cli_system_prompt == NULL) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin(
              "Enter system prompt (optional): ",
              system_prompt,
              sizeof(system_prompt));
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
      }
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User: ", user_prompt, sizeof(user_prompt));
      }
      // render user/system prompts into the Llama 2 Chat schema
      if (pos == 0 && system_prompt[0] != '\0') {
        // We do not add <s> because that is added by tokenizer->encode(x, 1, 0)
        const char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
        snprintf(
            rendered_prompt, RENDERED_PROMPT_SIZE-1, system_template, system_prompt, user_prompt);
      } else {
        // Assistant should produce </s>, so we do not include it in template
        // We do not add <s> because that is added by tokenizer->encode(x, 1, 0)
        const char user_template[] = "[INST] %s [/INST]";
        snprintf(rendered_prompt, RENDERED_PROMPT_SIZE-1, user_template, user_prompt);
      }

      // encode the rendered prompt into tokens
      prompt_tokens = tokenizer->encode(rendered_prompt, 1, 0);
      num_prompt_tokens = prompt_tokens.size();

      user_idx = 0; // reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the transformer next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt
      // token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }

    // forward the transformer to get logits for the next token
    float* logits = forward(transformer, token, pos);
    next = sample(sampler, logits);


    if (token == EOS_TOKEN) {
      user_turn = 1;
    }

    if (user_idx >= num_prompt_tokens && token != EOS_TOKEN && next != EOS_TOKEN) {
      std::string piece = tokenizer->decode(token, next);
      safe_printf(piece.c_str()); // same as printf("%s", piece), but skips
                                  // "unsafe" bytes
      fflush(stdout);
    }

    if (next == EOS_TOKEN) {
      printf("\n");
    }
    pos++;

  }
  printf("\n");
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(
      stderr,
      "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(
      stderr,
      "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
  // default parameters
  char* checkpoint_path = NULL; // e.g. out/model.bin
  const char* tokenizer_path = "tokenizer.bin";
  float temperature =
      1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp =
      0.9f; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int vocab_size = 32000;
  int steps = 256; // number of steps to run for
  const char* prompt = NULL; // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  const char* mode = "generate"; // generate|chat
  char* system_prompt =
      NULL; // the (optional) system prompt to use in chat mode

#if defined(ET_USE_ADPATIVE_THREADS)
  uint32_t num_performant_cores = torch::executorch::cpuinfo::get_num_performant_cores();
  if (num_performant_cores > 0) {
    torch::executorch::threadpool::get_threadpool()->_unsafe_reset_threadpool(
        num_performant_cores);
  }
#endif
  // poor man's C argparse so we can override the defaults above from the
  // command line
  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) {
      error_usage();
    } // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage();
    } // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage();
    } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(argv[i + 1]);
    } else if (argv[i][1] == 'p') {
      topp = atof(argv[i + 1]);
    } else if (argv[i][1] == 's') {
      rng_seed = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'n') {
      steps = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'v') {
      vocab_size = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'i') {
      prompt = argv[i + 1];
    } else if (argv[i][1] == 'z') {
      tokenizer_path = argv[i + 1];
    } else if (argv[i][1] == 'm') {
      mode = argv[i + 1];
    } else if (argv[i][1] == 'y') {
      system_prompt = argv[i + 1];
    } else {
      error_usage();
    }
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

  // build the Transformer via the model .bin file
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path, vocab_size, steps);

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer* tokenizer = nullptr;

  // Try to load using Tiktoken, if exception then switch to another tokenizer
  try {
    tokenizer =
        new Tiktoken(transformer.config.vocab_size, /*bos*/ 1, /*eos*/ 2);
    tokenizer->load(tokenizer_path);
  } catch (const std::invalid_argument&) {
    fprintf(
        stderr,
        "Failed to load %s into a Tiktoken tokenizer. Trying sentencepiece tokenizer..\n",
        tokenizer_path);
    delete tokenizer;
    tokenizer =
        new BPETokenizer(transformer.config.vocab_size, /*bos*/ 1, /*eos*/ 2);
    tokenizer->load(tokenizer_path);
  }

  // build the Sampler
  Sampler sampler;
  build_sampler(
      &sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // run!
  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, tokenizer, &sampler, prompt, steps);
  } else if (strcmp(mode, "chat") == 0) {
    chat(&transformer, tokenizer, &sampler, prompt, system_prompt, steps);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  delete tokenizer;
  free_transformer(&transformer);
  return 0;
}
#endif
