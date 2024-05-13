# pthreadpool

[![BSD (2 clause) License](https://img.shields.io/badge/License-BSD%202--Clause%20%22Simplified%22%20License-blue.svg)](https://github.com/Maratyszcza/pthreadpool/blob/master/LICENSE)
[![Build Status](https://img.shields.io/travis/Maratyszcza/pthreadpool.svg)](https://travis-ci.org/Maratyszcza/pthreadpool)

**pthreadpool** is a portable and efficient thread pool implementation.
It provides similar functionality to `#pragma omp parallel for`, but with additional features.

## Features:

* C interface (C++-compatible).
* 1D-6D loops with step parameters.
* Run on user-specified or auto-detected number of threads.
* Work-stealing scheduling for efficient work balancing.
* Wait-free synchronization of work items.
* Compatible with Linux (including Android), macOS, iOS, Windows, Emscripten environments.
* 100% unit tests coverage.
* Throughput and latency microbenchmarks.

## Example

  The following example demonstates using the thread pool for parallel addition of two arrays:

```c
static void add_arrays(struct array_addition_context* context, size_t i) {
  context->sum[i] = context->augend[i] + context->addend[i];
}

#define ARRAY_SIZE 4

int main() {
  double augend[ARRAY_SIZE] = { 1.0, 2.0, 4.0, -5.0 };
  double addend[ARRAY_SIZE] = { 0.25, -1.75, 0.0, 0.5 };
  double sum[ARRAY_SIZE];

  pthreadpool_t threadpool = pthreadpool_create(0);
  assert(threadpool != NULL);

  const size_t threads_count = pthreadpool_get_threads_count(threadpool);
  printf("Created thread pool with %zu threads\n", threads_count);

  struct array_addition_context context = { augend, addend, sum };
  pthreadpool_parallelize_1d(threadpool,
    (pthreadpool_task_1d_t) add_arrays,
    (void*) &context,
    ARRAY_SIZE,
    PTHREADPOOL_FLAG_DISABLE_DENORMALS /* flags */);

  pthreadpool_destroy(threadpool);
  threadpool = NULL;

  printf("%8s\t%.2lf\t%.2lf\t%.2lf\t%.2lf\n", "Augend",
    augend[0], augend[1], augend[2], augend[3]);
  printf("%8s\t%.2lf\t%.2lf\t%.2lf\t%.2lf\n", "Addend",
    addend[0], addend[1], addend[2], addend[3]);
  printf("%8s\t%.2lf\t%.2lf\t%.2lf\t%.2lf\n", "Sum",
    sum[0], sum[1], sum[2], sum[3]);

  return 0;
}
```
