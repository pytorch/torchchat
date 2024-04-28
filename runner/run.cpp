/* Inference for Llama-2 Transformer model in pure C++ */
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tokenizer.h>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <string>

#ifdef DEBUG
#include <cassert>
#include <iostream>
#endif

#if defined(__AOTI_MODEL__) || (defined(__ET_MODEL__) && defined(USE_ATENLIB))
#include <torch/torch.h>
#endif

#ifdef __AOTI_MODEL__
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
torch::Device cpu_device(torch::kCPU);

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

enum class ModelType {
  unknown = 0,
  llama2 = 2,
  llama3 = 3,
};

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
    char* model_path,
    int vocab_size,
    int seq_len) {
  // read in the Config and the Weights from the model
  // read_checkpoint(model_path, &t->config);
  // allocate the RunState buffers
  t->config.vocab_size = vocab_size;
  t->config.seq_len = seq_len;
  malloc_run_state(&t->state, &t->config);

#ifdef __AOTI_MODEL__
  t->runner = new torch::inductor::AOTIModelContainerRunnerCpu(
      /* path to model DSO */ model_path,
      /* thread pool size  */ 1);
#else //__ET_MODEL__
  t->runner = new Module(
      /* path to PTE model */ model_path,
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

  torch::Tensor result =
      transformer->runner->run(inputs)[0]
          .to(torch::dtype(torch::kFloat32))
          .to(cpu_device);
  auto logits = result[0].data_ptr();

#else // __ET_MODEL__
  ManagedTensor pos_managed(pos_buffer, sizeof(int64_t), {1}, ScalarType::Long);
  ManagedTensor tokens_managed(
      token_buffer, sizeof(int64_t), {1, 1}, ScalarType::Long);
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

Tokenizer* build_tokenizer(
    const char* tokenizer_path,
    ModelType model_type,
    int vocab_size) {
  Tokenizer* tokenizer = nullptr;
  switch (model_type) {
    case ModelType::llama2:
      tokenizer = new BPETokenizer(vocab_size, /*bos*/ 1, /*eos*/ 2);
      tokenizer->load(tokenizer_path);
      break;
    case ModelType::llama3:
      tokenizer = new Tiktoken(vocab_size, /*bos*/ 1, /*eos*/ 2);
      tokenizer->load(tokenizer_path);
      break;
    default:
      throw std::runtime_error(
          "No tokenizer defined for model type " +
          std::to_string(static_cast<int>(model_type)));
  }
  return tokenizer;
}

void free_tokenizer(Tokenizer* tokenizer) {
  delete tokenizer;
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
    int steps,
    ModelType model_type) {
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

ModelType get_model_type(int model_int) {
  switch (model_int) {
    case 2:
      return ModelType::llama2;
      break;
    case 3:
      return ModelType::llama3;
      break;
    default:
      return ModelType::unknown;
  }
}

uint64_t get_eot_token(Tokenizer* tokenizer, ModelType model_type) {
  if (model_type == ModelType::llama2) {
    // llama2 uses EOS as EOT token
    return tokenizer->eos_tok();
  }

  if (model_type == ModelType::llama3) {
    auto tokens = tokenizer->encode("<|eot_id|>", 0, 0);
    return tokens[0];
  }

  fprintf(
      stderr, "No chat template implemnation for model type %d", model_type);
  exit(EXIT_FAILURE);
}

std::vector<uint64_t> get_initial_prompt_tokens(
    const char* cli_system_prompt,
    const char* cli_user_prompt,
    Tokenizer* tokenizer,
    ModelType model_type) {
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[512 * 2 + 200]; // the prompt template is ~170
                                       // characters.  We use 200 to be safe.

  if (cli_system_prompt != NULL) {
    strcpy(system_prompt, cli_system_prompt);
  } else {
    read_stdin(
        "Enter system prompt (optional): ",
        system_prompt,
        sizeof(system_prompt));
  }

  if (cli_user_prompt != NULL) {
    strcpy(user_prompt, cli_user_prompt);
  } else {
    read_stdin("User: ", user_prompt, sizeof(user_prompt));
  }

  std::vector<uint64_t> tokens;

  switch (model_type) {
    case ModelType::llama2:
      if (system_prompt[0] != '\0') {
        snprintf(
            rendered_prompt,
            sizeof(rendered_prompt) - 1,
            "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]",
            system_prompt,
            user_prompt);
      } else {
        snprintf(
            rendered_prompt,
            sizeof(rendered_prompt) - 1,
            "[INST] %s [/INST]",
            user_prompt);
      }

      // We need to add BOS token here and not in template because llama2
      // tokenizer does not pattern match special tokens
      tokens = tokenizer->encode(rendered_prompt, 1, 0);
      break;

    case ModelType::llama3:
      if (system_prompt[0] != '\0') {
        snprintf(
            rendered_prompt,
            sizeof(rendered_prompt) - 1,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            system_prompt,
            user_prompt);
      } else {
        snprintf(
            rendered_prompt,
            sizeof(rendered_prompt) - 1,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            user_prompt);
      }
      tokens = tokenizer->encode(rendered_prompt, 0, 0);
      break;

    default:
      fprintf(
          stderr,
          "No chat template implemnation for model type %d",
          model_type);
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

std::vector<uint64_t> get_next_user_prompt_tokens(
    Tokenizer* tokenizer,
    ModelType model_type) {
  char user_prompt[512];
  char rendered_prompt[512 + 150]; // the prompt template is ~100 characters. We
                                   // use 150 to be safe.

  read_stdin("User: ", user_prompt, sizeof(user_prompt));
  std::vector<uint64_t> tokens;

  switch (model_type) {
    case ModelType::llama2:
      snprintf(
          rendered_prompt,
          sizeof(rendered_prompt) - 1,
          "[INST] %s [/INST]",
          user_prompt);

      // We need to add BOS token here and not in template because llama2
      // tokenizer does not pattern match special tokens
      tokens = tokenizer->encode(rendered_prompt, /*bos*/ 1, /*eos*/ 0);
      break;

    case ModelType::llama3:
      snprintf(
          rendered_prompt,
          sizeof(rendered_prompt) - 1,
          "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
          user_prompt);
      tokens = tokenizer->encode(rendered_prompt, 0, 0);
      break;

    default:
      fprintf(
          stderr,
          "No chat template implemnation for model type %d",
          model_type);
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

unsigned generate_from_prompt_tokens(
    Transformer* transformer,
    Tokenizer* tokenizer,
    Sampler* sampler,
    const std::vector<uint64_t>& prompt_tokens,
    unsigned pos,
    uint64_t stop_token,
    int stop_pos) {
  uint64_t next; // will store the next token in the sequence
  uint64_t token; // stores the current token to feed into the transformer
  bool done_with_prompt; // whether we are done processing prompt

  bool found_stop_token = false; // whether we've found the stop_token after
                                 // processing prompt_tokens
  unsigned pos_in_prompt = 0; // position relative to start of prompt

  // If stop_pos == -1, we go until we find stop_token
  // If stop_pos >= 0, we go until we find stop_token or pos <= stop_pos.
  while (!found_stop_token && (stop_pos == -1 || pos <= stop_pos)) {
    if (pos_in_prompt < prompt_tokens.size()) {
      // force prompt token
      token = prompt_tokens[pos_in_prompt++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    done_with_prompt = (pos_in_prompt >= prompt_tokens.size());

    // forward the transformer to get logits for the next token
    float* logits = forward(transformer, token, pos);
    next = sample(sampler, logits);

    // std::cout << "\npos: " << pos << " token: " << token << " next: " << next
    // << " done_with_prompt: " << done_with_prompt << " decoded: " <<
    // tokenizer->decode(token, next) << std::endl << std::flush;

    if (done_with_prompt) {
      // we terminate on finding the stop_token if we are done processing the
      // prompt (stop_tokens in the prompt do not terminate the loop)
      if (token == stop_token) {
        found_stop_token = true;
      }

      // We print next in each iteration of the loop, not token
      // We do not print the predicted token after stop_token
      // (i.e., we do not print next when token is stop_token)
      // The stop_token is printed as newline
      if (next == stop_token) {
        printf("\n");
      } else {
        // Do not print next if prev token was stop_token
        if (token != stop_token) {
          std::string piece = tokenizer->decode(token, next);
          safe_printf(piece.c_str()); // same as printf("%s", piece), but skips
                                      // "unsafe" bytes
          fflush(stdout);
        }
      }
    }

    pos++;
  }

  return pos;
}

void chat(
    Transformer* transformer,
    Tokenizer* tokenizer,
    Sampler* sampler,
    const char* cli_user_prompt,
    const char* cli_system_prompt,
    unsigned steps,
    ModelType model_type) {
  if (steps == 0) {
    return;
  }

  std::vector<uint64_t> prompt_tokens;
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
        transformer,
        tokenizer,
        sampler,
        prompt_tokens,
        pos,
        /*stop_token=*/get_eot_token(tokenizer, model_type),
        /*stop_pos=*/steps - 1); // We could pass in -1 here if we do not want
                                 // the model to stop mid-reply
  }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <model_path> [options]\n");
  fprintf(
      stderr, "Example: run model.{so,pte} -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(
      stderr,
      "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(
      stderr,
      "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> path to tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(
      stderr,
      "  -v <int>    (optional) vocab size, default is model-specific.\n");
  fprintf(
      stderr, "  -l <int>    (optional) llama version (2 or 3), default 2.\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
  // default parameters
  char* model_path = NULL;
  char* tokenizer_path = NULL;
  float temperature =
      1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f; // top-p in nucleus sampling. 1.0 = off. 0.9 works well,
                     // but slower

  int steps = 256; // number of steps to run for
  const char* prompt = NULL; // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  const char* mode = "generate"; // generate|chat
  char* system_prompt =
      NULL; // the (optional) system prompt to use in chat mode

  int vocab_size = -1;
  int llama_ver = 2;

#if defined(ET_USE_ADPATIVE_THREADS)
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
    } else if (argv[i][1] == 'l') {
      llama_ver = atoi(argv[i + 1]);
    } else {
      error_usage();
    }
  }

  ModelType model_type = get_model_type(llama_ver);
  if (model_type == ModelType::unknown) {
    fprintf(
        stderr,
        "Unknown model type passed by -l argument.  Received l=%d.",
        llama_ver);
    error_usage();
  }

  if (model_path == NULL) {
    fprintf(stderr, "No model_path provided.");
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

  // If no tokenizer path provided, get default for model_type
  if (vocab_size == -1) {
    switch (model_type) {
      case ModelType::llama2:
        vocab_size = 32000;
        break;
      case ModelType::llama3:
        vocab_size = 128256;
        break;
      default:
        fprintf(
            stderr,
            "No vocab_size was provided with -v argument, and there is no default vocab_size for model_type ModelType::%d.",
            model_type);
        error_usage();
    }
  }

  Transformer transformer;
  build_transformer(&transformer, model_path, vocab_size, steps);

  Tokenizer* tokenizer =
      build_tokenizer(tokenizer_path, model_type, vocab_size);

  Sampler sampler;
  build_sampler(&sampler, vocab_size, temperature, topp, rng_seed);

  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, tokenizer, &sampler, prompt, steps, model_type);
  } else if (strcmp(mode, "chat") == 0) {
    chat(
        &transformer,
        tokenizer,
        &sampler,
        prompt,
        system_prompt,
        steps,
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
