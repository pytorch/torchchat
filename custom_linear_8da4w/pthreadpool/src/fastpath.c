/* Standard C headers */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if PTHREADPOOL_USE_CPUINFO
	#include <cpuinfo.h>
#endif

/* Dependencies */
#include <fxdiv.h>

/* Public library header */
#include <pthreadpool.h>

/* Internal library headers */
#include "threadpool-atomics.h"
#include "threadpool-common.h"
#include "threadpool-object.h"
#include "threadpool-utils.h"


PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_1d_t task = (pthreadpool_task_1d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, range_start++);
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			task(argument, index);
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_1d_with_thread_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_1d_with_thread_t task = (pthreadpool_task_1d_with_thread_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t thread_number = thread->thread_number;
	size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, thread_number, range_start++);
	}

	/* There still may be other threads with work */
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			task(argument, thread_number, index);
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_1d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_1d_with_id_t task = (pthreadpool_task_1d_with_id_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const uint32_t default_uarch_index = threadpool->params.parallelize_1d_with_uarch.default_uarch_index;
	uint32_t uarch_index = default_uarch_index;
	#if PTHREADPOOL_USE_CPUINFO
		uarch_index = cpuinfo_get_current_uarch_index_with_default(default_uarch_index);
		if (uarch_index > threadpool->params.parallelize_1d_with_uarch.max_uarch_index) {
			uarch_index = default_uarch_index;
		}
	#endif

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, uarch_index, range_start++);
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			task(argument, uarch_index, index);
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_1d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_1d_tile_1d_t task = (pthreadpool_task_1d_tile_1d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const size_t tile = threadpool->params.parallelize_1d_tile_1d.tile;
	size_t tile_start = range_start * tile;

	const size_t range = threadpool->params.parallelize_1d_tile_1d.range;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, tile_start, min(range - tile_start, tile));
		tile_start += tile;
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t tile_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const size_t tile_start = tile_index * tile;
			task(argument, tile_start, min(range - tile_start, tile));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_2d_t task = (pthreadpool_task_2d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_2d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(range_start, range_j);
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;

	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j);
		if (++j == range_j.value) {
			j = 0;
			i += 1;
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(linear_index, range_j);
			task(argument, index_i_j.quotient, index_i_j.remainder);
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_with_thread_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_2d_with_thread_t task = (pthreadpool_task_2d_with_thread_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_2d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(range_start, range_j);
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;

	const size_t thread_number = thread->thread_number;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, thread_number, i, j);
		if (++j == range_j.value) {
			j = 0;
			i += 1;
		}
	}

	/* There still may be other threads with work */
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(linear_index, range_j);
			task(argument, thread_number, index_i_j.quotient, index_i_j.remainder);
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_2d_tile_1d_t task = (pthreadpool_task_2d_tile_1d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_j = threadpool->params.parallelize_2d_tile_1d.tile_range_j;
	const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(range_start, tile_range_j);
	const size_t tile_j = threadpool->params.parallelize_2d_tile_1d.tile_j;
	size_t i = tile_index_i_j.quotient;
	size_t start_j = tile_index_i_j.remainder * tile_j;

	const size_t range_j = threadpool->params.parallelize_2d_tile_1d.range_j;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, start_j, min(range_j - start_j, tile_j));
		start_j += tile_j;
		if (start_j >= range_j) {
			start_j = 0;
			i += 1;
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(linear_index, tile_range_j);
			const size_t start_j = tile_index_i_j.remainder * tile_j;
			task(argument, tile_index_i_j.quotient, start_j, min(range_j - start_j, tile_j));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_tile_1d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_2d_tile_1d_with_id_t task = (pthreadpool_task_2d_tile_1d_with_id_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const uint32_t default_uarch_index = threadpool->params.parallelize_2d_tile_1d_with_uarch.default_uarch_index;
	uint32_t uarch_index = default_uarch_index;
	#if PTHREADPOOL_USE_CPUINFO
		uarch_index = cpuinfo_get_current_uarch_index_with_default(default_uarch_index);
		if (uarch_index > threadpool->params.parallelize_2d_tile_1d_with_uarch.max_uarch_index) {
			uarch_index = default_uarch_index;
		}
	#endif

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_j = threadpool->params.parallelize_2d_tile_1d_with_uarch.tile_range_j;
	const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(range_start, tile_range_j);
	const size_t tile_j = threadpool->params.parallelize_2d_tile_1d_with_uarch.tile_j;
	size_t i = tile_index_i_j.quotient;
	size_t start_j = tile_index_i_j.remainder * tile_j;

	const size_t range_j = threadpool->params.parallelize_2d_tile_1d_with_uarch.range_j;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, uarch_index, i, start_j, min(range_j - start_j, tile_j));
		start_j += tile_j;
		if (start_j >= range_j) {
			start_j = 0;
			i += 1;
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(linear_index, tile_range_j);
			const size_t start_j = tile_index_i_j.remainder * tile_j;
			task(argument, uarch_index, tile_index_i_j.quotient, start_j, min(range_j - start_j, tile_j));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_tile_1d_with_uarch_with_thread_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_2d_tile_1d_with_id_with_thread_t task =
		(pthreadpool_task_2d_tile_1d_with_id_with_thread_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const uint32_t default_uarch_index = threadpool->params.parallelize_2d_tile_1d_with_uarch.default_uarch_index;
	uint32_t uarch_index = default_uarch_index;
	#if PTHREADPOOL_USE_CPUINFO
		uarch_index = cpuinfo_get_current_uarch_index_with_default(default_uarch_index);
		if (uarch_index > threadpool->params.parallelize_2d_tile_1d_with_uarch.max_uarch_index) {
			uarch_index = default_uarch_index;
		}
	#endif

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_j = threadpool->params.parallelize_2d_tile_1d_with_uarch.tile_range_j;
	const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(range_start, tile_range_j);
	const size_t tile_j = threadpool->params.parallelize_2d_tile_1d_with_uarch.tile_j;
	size_t i = tile_index_i_j.quotient;
	size_t start_j = tile_index_i_j.remainder * tile_j;

	const size_t range_j = threadpool->params.parallelize_2d_tile_1d_with_uarch.range_j;
	const size_t thread_number = thread->thread_number;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, uarch_index, thread_number, i, start_j, min(range_j - start_j, tile_j));
		start_j += tile_j;
		if (start_j >= range_j) {
			start_j = 0;
			i += 1;
		}
	}

	/* There still may be other threads with work */
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(linear_index, tile_range_j);
			const size_t start_j = tile_index_i_j.remainder * tile_j;
			task(argument, uarch_index, thread_number, tile_index_i_j.quotient, start_j, min(range_j - start_j, tile_j));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_tile_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_2d_tile_2d_t task = (pthreadpool_task_2d_tile_2d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_j = threadpool->params.parallelize_2d_tile_2d.tile_range_j;
	const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(range_start, tile_range_j);
	const size_t tile_i = threadpool->params.parallelize_2d_tile_2d.tile_i;
	const size_t tile_j = threadpool->params.parallelize_2d_tile_2d.tile_j;
	size_t start_i = tile_index_i_j.quotient * tile_i;
	size_t start_j = tile_index_i_j.remainder * tile_j;

	const size_t range_i = threadpool->params.parallelize_2d_tile_2d.range_i;
	const size_t range_j = threadpool->params.parallelize_2d_tile_2d.range_j;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, start_i, start_j, min(range_i - start_i, tile_i), min(range_j - start_j, tile_j));
		start_j += tile_j;
		if (start_j >= range_j) {
			start_j = 0;
			start_i += tile_i;
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(linear_index, tile_range_j);
			const size_t start_i = tile_index_i_j.quotient * tile_i;
			const size_t start_j = tile_index_i_j.remainder * tile_j;
			task(argument, start_i, start_j, min(range_i - start_i, tile_i), min(range_j - start_j, tile_j));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_tile_2d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_2d_tile_2d_with_id_t task = (pthreadpool_task_2d_tile_2d_with_id_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const uint32_t default_uarch_index = threadpool->params.parallelize_2d_tile_2d_with_uarch.default_uarch_index;
	uint32_t uarch_index = default_uarch_index;
	#if PTHREADPOOL_USE_CPUINFO
		uarch_index = cpuinfo_get_current_uarch_index_with_default(default_uarch_index);
		if (uarch_index > threadpool->params.parallelize_2d_tile_2d_with_uarch.max_uarch_index) {
			uarch_index = default_uarch_index;
		}
	#endif

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const struct fxdiv_divisor_size_t tile_range_j = threadpool->params.parallelize_2d_tile_2d_with_uarch.tile_range_j;
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_result_size_t index = fxdiv_divide_size_t(range_start, tile_range_j);
	const size_t range_i = threadpool->params.parallelize_2d_tile_2d_with_uarch.range_i;
	const size_t tile_i = threadpool->params.parallelize_2d_tile_2d_with_uarch.tile_i;
	const size_t range_j = threadpool->params.parallelize_2d_tile_2d_with_uarch.range_j;
	const size_t tile_j = threadpool->params.parallelize_2d_tile_2d_with_uarch.tile_j;
	size_t start_i = index.quotient * tile_i;
	size_t start_j = index.remainder * tile_j;

	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, uarch_index, start_i, start_j, min(range_i - start_i, tile_i), min(range_j - start_j, tile_j));
		start_j += tile_j;
		if (start_j >= range_j) {
			start_j = 0;
			start_i += tile_i;
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(linear_index, tile_range_j);
			const size_t start_i = tile_index_i_j.quotient * tile_i;
			const size_t start_j = tile_index_i_j.remainder * tile_j;
			task(argument, uarch_index, start_i, start_j, min(range_i - start_i, tile_i), min(range_j - start_j, tile_j));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_3d_t task = (pthreadpool_task_3d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t range_k = threadpool->params.parallelize_3d.range_k;
	const struct fxdiv_result_size_t index_ij_k = fxdiv_divide_size_t(range_start, range_k);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_3d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_k.quotient, range_j);
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t k = index_ij_k.remainder;

	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, k);
		if (++k == range_k.value) {
			k = 0;
			if (++j == range_j.value) {
				j = 0;
				i += 1;
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t index_ij_k = fxdiv_divide_size_t(linear_index, range_k);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_k.quotient, range_j);
			task(argument, index_i_j.quotient, index_i_j.remainder, index_ij_k.remainder);
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_3d_tile_1d_t task = (pthreadpool_task_3d_tile_1d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_k = threadpool->params.parallelize_3d_tile_1d.tile_range_k;
	const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(range_start, tile_range_k);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_3d_tile_1d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, range_j);
	const size_t tile_k = threadpool->params.parallelize_3d_tile_1d.tile_k;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t start_k = tile_index_ij_k.remainder * tile_k;

	const size_t range_k = threadpool->params.parallelize_3d_tile_1d.range_k;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, start_k, min(range_k - start_k, tile_k));
		start_k += tile_k;
		if (start_k >= range_k) {
			start_k = 0;
			if (++j == range_j.value) {
				j = 0;
				i += 1;
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(linear_index, tile_range_k);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, range_j);
			const size_t start_k = tile_index_ij_k.remainder * tile_k;
			task(argument, index_i_j.quotient, index_i_j.remainder, start_k, min(range_k - start_k, tile_k));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_1d_with_thread_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_3d_tile_1d_with_thread_t task = (pthreadpool_task_3d_tile_1d_with_thread_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_k = threadpool->params.parallelize_3d_tile_1d.tile_range_k;
	const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(range_start, tile_range_k);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_3d_tile_1d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, range_j);
	const size_t tile_k = threadpool->params.parallelize_3d_tile_1d.tile_k;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t start_k = tile_index_ij_k.remainder * tile_k;

	const size_t range_k = threadpool->params.parallelize_3d_tile_1d.range_k;
	const size_t thread_number = thread->thread_number;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, thread_number, i, j, start_k, min(range_k - start_k, tile_k));
		start_k += tile_k;
		if (start_k >= range_k) {
			start_k = 0;
			if (++j == range_j.value) {
				j = 0;
				i += 1;
			}
		}
	}

	/* There still may be other threads with work */
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(linear_index, tile_range_k);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, range_j);
			const size_t start_k = tile_index_ij_k.remainder * tile_k;
			task(argument, thread_number, index_i_j.quotient, index_i_j.remainder, start_k, min(range_k - start_k, tile_k));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_1d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_3d_tile_1d_with_id_t task = (pthreadpool_task_3d_tile_1d_with_id_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const uint32_t default_uarch_index = threadpool->params.parallelize_3d_tile_1d_with_uarch.default_uarch_index;
	uint32_t uarch_index = default_uarch_index;
	#if PTHREADPOOL_USE_CPUINFO
		uarch_index = cpuinfo_get_current_uarch_index_with_default(default_uarch_index);
		if (uarch_index > threadpool->params.parallelize_3d_tile_1d_with_uarch.max_uarch_index) {
			uarch_index = default_uarch_index;
		}
	#endif

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_k = threadpool->params.parallelize_3d_tile_1d_with_uarch.tile_range_k;
	const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(range_start, tile_range_k);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_3d_tile_1d_with_uarch.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, range_j);
	const size_t tile_k = threadpool->params.parallelize_3d_tile_1d_with_uarch.tile_k;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t start_k = tile_index_ij_k.remainder * tile_k;

	const size_t range_k = threadpool->params.parallelize_3d_tile_1d_with_uarch.range_k;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, uarch_index, i, j, start_k, min(range_k - start_k, tile_k));
		start_k += tile_k;
		if (start_k >= range_k) {
			start_k = 0;
			if (++j == range_j.value) {
				j = 0;
				i += 1;
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(linear_index, tile_range_k);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, range_j);
			const size_t start_k = tile_index_ij_k.remainder * tile_k;
			task(argument, uarch_index, index_i_j.quotient, index_i_j.remainder, start_k, min(range_k - start_k, tile_k));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_1d_with_uarch_with_thread_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_3d_tile_1d_with_id_with_thread_t task =
		(pthreadpool_task_3d_tile_1d_with_id_with_thread_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const uint32_t default_uarch_index = threadpool->params.parallelize_3d_tile_1d_with_uarch.default_uarch_index;
	uint32_t uarch_index = default_uarch_index;
	#if PTHREADPOOL_USE_CPUINFO
		uarch_index = cpuinfo_get_current_uarch_index_with_default(default_uarch_index);
		if (uarch_index > threadpool->params.parallelize_3d_tile_1d_with_uarch.max_uarch_index) {
			uarch_index = default_uarch_index;
		}
	#endif

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_k = threadpool->params.parallelize_3d_tile_1d_with_uarch.tile_range_k;
	const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(range_start, tile_range_k);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_3d_tile_1d_with_uarch.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, range_j);
	const size_t tile_k = threadpool->params.parallelize_3d_tile_1d_with_uarch.tile_k;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t start_k = tile_index_ij_k.remainder * tile_k;

	const size_t range_k = threadpool->params.parallelize_3d_tile_1d_with_uarch.range_k;
	const size_t thread_number = thread->thread_number;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, uarch_index, thread_number, i, j, start_k, min(range_k - start_k, tile_k));
		start_k += tile_k;
		if (start_k >= range_k) {
			start_k = 0;
			if (++j == range_j.value) {
				j = 0;
				i += 1;
			}
		}
	}

	/* There still may be other threads with work */
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(linear_index, tile_range_k);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, range_j);
			const size_t start_k = tile_index_ij_k.remainder * tile_k;
			task(argument, uarch_index, thread_number, index_i_j.quotient, index_i_j.remainder, start_k, min(range_k - start_k, tile_k));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_3d_tile_2d_t task = (pthreadpool_task_3d_tile_2d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_k = threadpool->params.parallelize_3d_tile_2d.tile_range_k;
	const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(range_start, tile_range_k);
	const struct fxdiv_divisor_size_t tile_range_j = threadpool->params.parallelize_3d_tile_2d.tile_range_j;
	const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, tile_range_j);
	const size_t tile_j = threadpool->params.parallelize_3d_tile_2d.tile_j;
	const size_t tile_k = threadpool->params.parallelize_3d_tile_2d.tile_k;
	size_t i = tile_index_i_j.quotient;
	size_t start_j = tile_index_i_j.remainder * tile_j;
	size_t start_k = tile_index_ij_k.remainder * tile_k;

	const size_t range_k = threadpool->params.parallelize_3d_tile_2d.range_k;
	const size_t range_j = threadpool->params.parallelize_3d_tile_2d.range_j;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, start_j, start_k, min(range_j - start_j, tile_j), min(range_k - start_k, tile_k));
		start_k += tile_k;
		if (start_k >= range_k) {
			start_k = 0;
			start_j += tile_j;
			if (start_j >= range_j) {
				start_j = 0;
				i += 1;
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(linear_index, tile_range_k);
			const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, tile_range_j);
			const size_t start_j = tile_index_i_j.remainder * tile_j;
			const size_t start_k = tile_index_ij_k.remainder * tile_k;
			task(argument, tile_index_i_j.quotient, start_j, start_k, min(range_j - start_j, tile_j), min(range_k - start_k, tile_k));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_2d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_3d_tile_2d_with_id_t task = (pthreadpool_task_3d_tile_2d_with_id_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const uint32_t default_uarch_index = threadpool->params.parallelize_3d_tile_2d_with_uarch.default_uarch_index;
	uint32_t uarch_index = default_uarch_index;
	#if PTHREADPOOL_USE_CPUINFO
		uarch_index = cpuinfo_get_current_uarch_index_with_default(default_uarch_index);
		if (uarch_index > threadpool->params.parallelize_3d_tile_2d_with_uarch.max_uarch_index) {
			uarch_index = default_uarch_index;
		}
	#endif

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_k = threadpool->params.parallelize_3d_tile_2d_with_uarch.tile_range_k;
	const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(range_start, tile_range_k);
	const struct fxdiv_divisor_size_t tile_range_j = threadpool->params.parallelize_3d_tile_2d_with_uarch.tile_range_j;
	const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, tile_range_j);
	const size_t tile_j = threadpool->params.parallelize_3d_tile_2d_with_uarch.tile_j;
	const size_t tile_k = threadpool->params.parallelize_3d_tile_2d_with_uarch.tile_k;
	size_t i = tile_index_i_j.quotient;
	size_t start_j = tile_index_i_j.remainder * tile_j;
	size_t start_k = tile_index_ij_k.remainder * tile_k;

	const size_t range_k = threadpool->params.parallelize_3d_tile_2d_with_uarch.range_k;
	const size_t range_j = threadpool->params.parallelize_3d_tile_2d_with_uarch.range_j;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, uarch_index, i, start_j, start_k, min(range_j - start_j, tile_j), min(range_k - start_k, tile_k));
		start_k += tile_k;
		if (start_k >= range_k) {
			start_k = 0;
			start_j += tile_j;
			if (start_j >= range_j) {
				start_j = 0;
				i += 1;
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(linear_index, tile_range_k);
			const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, tile_range_j);
			const size_t start_j = tile_index_i_j.remainder * tile_j;
			const size_t start_k = tile_index_ij_k.remainder * tile_k;
			task(argument, uarch_index, tile_index_i_j.quotient, start_j, start_k, min(range_j - start_j, tile_j), min(range_k - start_k, tile_k));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_4d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_4d_t task = (pthreadpool_task_4d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t range_kl = threadpool->params.parallelize_4d.range_kl;
	const struct fxdiv_result_size_t index_ij_kl = fxdiv_divide_size_t(range_start, range_kl);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_4d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_kl.quotient, range_j);
	const struct fxdiv_divisor_size_t range_l = threadpool->params.parallelize_4d.range_l;
	const struct fxdiv_result_size_t index_k_l = fxdiv_divide_size_t(index_ij_kl.remainder, range_l);
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t k = index_k_l.quotient;
	size_t l = index_k_l.remainder;

	const size_t range_k = threadpool->params.parallelize_4d.range_k;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, k, l);
		if (++l == range_l.value) {
			l = 0;
			if (++k == range_k) {
				k = 0;
				if (++j == range_j.value) {
					j = 0;
					i += 1;
				}
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t index_ij_kl = fxdiv_divide_size_t(linear_index, range_kl);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_kl.quotient, range_j);
			const struct fxdiv_result_size_t index_k_l = fxdiv_divide_size_t(index_ij_kl.remainder, range_l);
			task(argument, index_i_j.quotient, index_i_j.remainder, index_k_l.quotient, index_k_l.remainder);
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_4d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_4d_tile_1d_t task = (pthreadpool_task_4d_tile_1d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_kl = threadpool->params.parallelize_4d_tile_1d.tile_range_kl;
	const struct fxdiv_result_size_t tile_index_ij_kl = fxdiv_divide_size_t(range_start, tile_range_kl);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_4d_tile_1d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_kl.quotient, range_j);
	const struct fxdiv_divisor_size_t tile_range_l = threadpool->params.parallelize_4d_tile_1d.tile_range_l;
	const struct fxdiv_result_size_t tile_index_k_l = fxdiv_divide_size_t(tile_index_ij_kl.remainder, tile_range_l);
	const size_t tile_l = threadpool->params.parallelize_4d_tile_1d.tile_l;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t k = tile_index_k_l.quotient;
	size_t start_l = tile_index_k_l.remainder * tile_l;

	const size_t range_l = threadpool->params.parallelize_4d_tile_1d.range_l;
	const size_t range_k = threadpool->params.parallelize_4d_tile_1d.range_k;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, k, start_l, min(range_l - start_l, tile_l));
		start_l += tile_l;
		if (start_l >= range_l) {
			start_l = 0;
			if (++k == range_k) {
				k = 0;
				if (++j == range_j.value) {
					j = 0;
					i += 1;
				}
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ij_kl = fxdiv_divide_size_t(linear_index, tile_range_kl);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_kl.quotient, range_j);
			const struct fxdiv_result_size_t tile_index_k_l = fxdiv_divide_size_t(tile_index_ij_kl.remainder, tile_range_l);
			const size_t start_l = tile_index_k_l.remainder * tile_l;
			task(argument, index_i_j.quotient, index_i_j.remainder, tile_index_k_l.quotient, start_l, min(range_l - start_l, tile_l));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_4d_tile_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_4d_tile_2d_t task = (pthreadpool_task_4d_tile_2d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_kl = threadpool->params.parallelize_4d_tile_2d.tile_range_kl;
	const struct fxdiv_result_size_t tile_index_ij_kl = fxdiv_divide_size_t(range_start, tile_range_kl);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_4d_tile_2d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_kl.quotient, range_j);
	const struct fxdiv_divisor_size_t tile_range_l = threadpool->params.parallelize_4d_tile_2d.tile_range_l;
	const struct fxdiv_result_size_t tile_index_k_l = fxdiv_divide_size_t(tile_index_ij_kl.remainder, tile_range_l);
	const size_t tile_k = threadpool->params.parallelize_4d_tile_2d.tile_k;
	const size_t tile_l = threadpool->params.parallelize_4d_tile_2d.tile_l;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t start_k = tile_index_k_l.quotient * tile_k;
	size_t start_l = tile_index_k_l.remainder * tile_l;

	const size_t range_l = threadpool->params.parallelize_4d_tile_2d.range_l;
	const size_t range_k = threadpool->params.parallelize_4d_tile_2d.range_k;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, start_k, start_l, min(range_k - start_k, tile_k), min(range_l - start_l, tile_l));
		start_l += tile_l;
		if (start_l >= range_l) {
			start_l = 0;
			start_k += tile_k;
			if (start_k >= range_k) {
				start_k = 0;
				if (++j == range_j.value) {
					j = 0;
					i += 1;
				}
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ij_kl = fxdiv_divide_size_t(linear_index, tile_range_kl);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_kl.quotient, range_j);
			const struct fxdiv_result_size_t tile_index_k_l = fxdiv_divide_size_t(tile_index_ij_kl.remainder, tile_range_l);
			const size_t start_k = tile_index_k_l.quotient * tile_k;
			const size_t start_l = tile_index_k_l.remainder * tile_l;
			task(argument, index_i_j.quotient, index_i_j.remainder, start_k, start_l, min(range_k - start_k, tile_k), min(range_l - start_l, tile_l));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_4d_tile_2d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_4d_tile_2d_with_id_t task = (pthreadpool_task_4d_tile_2d_with_id_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const uint32_t default_uarch_index = threadpool->params.parallelize_4d_tile_2d_with_uarch.default_uarch_index;
	uint32_t uarch_index = default_uarch_index;
	#if PTHREADPOOL_USE_CPUINFO
		uarch_index = cpuinfo_get_current_uarch_index_with_default(default_uarch_index);
		if (uarch_index > threadpool->params.parallelize_4d_tile_2d_with_uarch.max_uarch_index) {
			uarch_index = default_uarch_index;
		}
	#endif

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_kl = threadpool->params.parallelize_4d_tile_2d_with_uarch.tile_range_kl;
	const struct fxdiv_result_size_t tile_index_ij_kl = fxdiv_divide_size_t(range_start, tile_range_kl);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_4d_tile_2d_with_uarch.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_kl.quotient, range_j);
	const struct fxdiv_divisor_size_t tile_range_l = threadpool->params.parallelize_4d_tile_2d_with_uarch.tile_range_l;
	const struct fxdiv_result_size_t tile_index_k_l = fxdiv_divide_size_t(tile_index_ij_kl.remainder, tile_range_l);
	const size_t tile_k = threadpool->params.parallelize_4d_tile_2d_with_uarch.tile_k;
	const size_t tile_l = threadpool->params.parallelize_4d_tile_2d_with_uarch.tile_l;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t start_k = tile_index_k_l.quotient * tile_k;
	size_t start_l = tile_index_k_l.remainder * tile_l;

	const size_t range_l = threadpool->params.parallelize_4d_tile_2d_with_uarch.range_l;
	const size_t range_k = threadpool->params.parallelize_4d_tile_2d_with_uarch.range_k;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, uarch_index, i, j, start_k, start_l, min(range_k - start_k, tile_k), min(range_l - start_l, tile_l));
		start_l += tile_l;
		if (start_l >= range_l) {
			start_l = 0;
			start_k += tile_k;
			if (start_k >= range_k) {
				start_k = 0;
				if (++j == range_j.value) {
					j = 0;
					i += 1;
				}
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ij_kl = fxdiv_divide_size_t(linear_index, tile_range_kl);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(tile_index_ij_kl.quotient, range_j);
			const struct fxdiv_result_size_t tile_index_k_l = fxdiv_divide_size_t(tile_index_ij_kl.remainder, tile_range_l);
			const size_t start_k = tile_index_k_l.quotient * tile_k;
			const size_t start_l = tile_index_k_l.remainder * tile_l;
			task(argument, uarch_index, index_i_j.quotient, index_i_j.remainder, start_k, start_l, min(range_k - start_k, tile_k), min(range_l - start_l, tile_l));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_5d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_5d_t task = (pthreadpool_task_5d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t range_lm = threadpool->params.parallelize_5d.range_lm;
	const struct fxdiv_result_size_t index_ijk_lm = fxdiv_divide_size_t(range_start, range_lm);
	const struct fxdiv_divisor_size_t range_k = threadpool->params.parallelize_5d.range_k;
	const struct fxdiv_result_size_t index_ij_k = fxdiv_divide_size_t(index_ijk_lm.quotient, range_k);
	const struct fxdiv_divisor_size_t range_m = threadpool->params.parallelize_5d.range_m;
	const struct fxdiv_result_size_t index_l_m = fxdiv_divide_size_t(index_ijk_lm.remainder, range_m);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_5d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_k.quotient, range_j);
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t k = index_ij_k.remainder;
	size_t l = index_l_m.quotient;
	size_t m = index_l_m.remainder;

	const size_t range_l = threadpool->params.parallelize_5d.range_l;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, k, l, m);
		if (++m == range_m.value) {
			m = 0;
			if (++l == range_l) {
				l = 0;
				if (++k == range_k.value) {
					k = 0;
					if (++j == range_j.value) {
						j = 0;
						i += 1;
					}
				}
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t index_ijk_lm = fxdiv_divide_size_t(linear_index, range_lm);
			const struct fxdiv_result_size_t index_ij_k = fxdiv_divide_size_t(index_ijk_lm.quotient, range_k);
			const struct fxdiv_result_size_t index_l_m = fxdiv_divide_size_t(index_ijk_lm.remainder, range_m);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_k.quotient, range_j);
			task(argument, index_i_j.quotient, index_i_j.remainder, index_ij_k.remainder, index_l_m.quotient, index_l_m.remainder);
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_5d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_5d_tile_1d_t task = (pthreadpool_task_5d_tile_1d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_m = threadpool->params.parallelize_5d_tile_1d.tile_range_m;
	const struct fxdiv_result_size_t tile_index_ijkl_m = fxdiv_divide_size_t(range_start, tile_range_m);
	const struct fxdiv_divisor_size_t range_kl = threadpool->params.parallelize_5d_tile_1d.range_kl;
	const struct fxdiv_result_size_t index_ij_kl = fxdiv_divide_size_t(tile_index_ijkl_m.quotient, range_kl);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_5d_tile_1d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_kl.quotient, range_j);
	const struct fxdiv_divisor_size_t range_l = threadpool->params.parallelize_5d_tile_1d.range_l;
	const struct fxdiv_result_size_t index_k_l = fxdiv_divide_size_t(index_ij_kl.remainder, range_l);
	const size_t tile_m = threadpool->params.parallelize_5d_tile_1d.tile_m;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t k = index_k_l.quotient;
	size_t l = index_k_l.remainder;
	size_t start_m = tile_index_ijkl_m.remainder * tile_m;

	const size_t range_m = threadpool->params.parallelize_5d_tile_1d.range_m;
	const size_t range_k = threadpool->params.parallelize_5d_tile_1d.range_k;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, k, l, start_m, min(range_m - start_m, tile_m));
		start_m += tile_m;
		if (start_m >= range_m) {
			start_m = 0;
			if (++l == range_l.value) {
				l = 0;
				if (++k == range_k) {
					k = 0;
					if (++j == range_j.value) {
						j = 0;
						i += 1;
					}
				}
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ijkl_m = fxdiv_divide_size_t(linear_index, tile_range_m);
			const struct fxdiv_result_size_t index_ij_kl = fxdiv_divide_size_t(tile_index_ijkl_m.quotient, range_kl);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_kl.quotient, range_j);
			const struct fxdiv_result_size_t index_k_l = fxdiv_divide_size_t(index_ij_kl.remainder, range_l);
			size_t start_m = tile_index_ijkl_m.remainder * tile_m;
			task(argument, index_i_j.quotient, index_i_j.remainder, index_k_l.quotient, index_k_l.remainder, start_m,
				min(range_m - start_m, tile_m));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_5d_tile_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_5d_tile_2d_t task = (pthreadpool_task_5d_tile_2d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_lm = threadpool->params.parallelize_5d_tile_2d.tile_range_lm;
	const struct fxdiv_result_size_t tile_index_ijk_lm = fxdiv_divide_size_t(range_start, tile_range_lm);
	const struct fxdiv_divisor_size_t range_k = threadpool->params.parallelize_5d_tile_2d.range_k;
	const struct fxdiv_result_size_t index_ij_k = fxdiv_divide_size_t(tile_index_ijk_lm.quotient, range_k);
	const struct fxdiv_divisor_size_t tile_range_m = threadpool->params.parallelize_5d_tile_2d.tile_range_m;
	const struct fxdiv_result_size_t tile_index_l_m = fxdiv_divide_size_t(tile_index_ijk_lm.remainder, tile_range_m);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_5d_tile_2d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_k.quotient, range_j);
	const size_t tile_l = threadpool->params.parallelize_5d_tile_2d.tile_l;
	const size_t tile_m = threadpool->params.parallelize_5d_tile_2d.tile_m;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t k = index_ij_k.remainder;
	size_t start_l = tile_index_l_m.quotient * tile_l;
	size_t start_m = tile_index_l_m.remainder * tile_m;

	const size_t range_m = threadpool->params.parallelize_5d_tile_2d.range_m;
	const size_t range_l = threadpool->params.parallelize_5d_tile_2d.range_l;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, k, start_l, start_m, min(range_l - start_l, tile_l), min(range_m - start_m, tile_m));
		start_m += tile_m;
		if (start_m >= range_m) {
			start_m = 0;
			start_l += tile_l;
			if (start_l >= range_l) {
				start_l = 0;
				if (++k == range_k.value) {
					k = 0;
					if (++j == range_j.value) {
						j = 0;
						i += 1;
					}
				}
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ijk_lm = fxdiv_divide_size_t(linear_index, tile_range_lm);
			const struct fxdiv_result_size_t index_ij_k = fxdiv_divide_size_t(tile_index_ijk_lm.quotient, range_k);
			const struct fxdiv_result_size_t tile_index_l_m = fxdiv_divide_size_t(tile_index_ijk_lm.remainder, tile_range_m);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_k.quotient, range_j);
			const size_t start_l = tile_index_l_m.quotient * tile_l;
			const size_t start_m = tile_index_l_m.remainder * tile_m;
			task(argument, index_i_j.quotient, index_i_j.remainder, index_ij_k.remainder,
				start_l, start_m, min(range_l - start_l, tile_l), min(range_m - start_m, tile_m));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_6d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_6d_t task = (pthreadpool_task_6d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t range_lmn = threadpool->params.parallelize_6d.range_lmn;
	const struct fxdiv_result_size_t index_ijk_lmn = fxdiv_divide_size_t(range_start, range_lmn);
	const struct fxdiv_divisor_size_t range_k = threadpool->params.parallelize_6d.range_k;
	const struct fxdiv_result_size_t index_ij_k = fxdiv_divide_size_t(index_ijk_lmn.quotient, range_k);
	const struct fxdiv_divisor_size_t range_n = threadpool->params.parallelize_6d.range_n;
	const struct fxdiv_result_size_t index_lm_n = fxdiv_divide_size_t(index_ijk_lmn.remainder, range_n);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_6d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_k.quotient, range_j);
	const struct fxdiv_divisor_size_t range_m = threadpool->params.parallelize_6d.range_m;
	const struct fxdiv_result_size_t index_l_m = fxdiv_divide_size_t(index_lm_n.quotient, range_m);
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t k = index_ij_k.remainder;
	size_t l = index_l_m.quotient;
	size_t m = index_l_m.remainder;
	size_t n = index_lm_n.remainder;

	const size_t range_l = threadpool->params.parallelize_6d.range_l;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, k, l, m, n);
		if (++n == range_n.value) {
			n = 0;
			if (++m == range_m.value) {
				m = 0;
				if (++l == range_l) {
					l = 0;
					if (++k == range_k.value) {
						k = 0;
						if (++j == range_j.value) {
							j = 0;
							i += 1;
						}
					}
				}
			}
		}
	}


	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t index_ijk_lmn = fxdiv_divide_size_t(linear_index, range_lmn);
			const struct fxdiv_result_size_t index_ij_k = fxdiv_divide_size_t(index_ijk_lmn.quotient, range_k);
			const struct fxdiv_result_size_t index_lm_n = fxdiv_divide_size_t(index_ijk_lmn.remainder, range_n);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_k.quotient, range_j);
			const struct fxdiv_result_size_t index_l_m = fxdiv_divide_size_t(index_lm_n.quotient, range_m);
			task(argument, index_i_j.quotient, index_i_j.remainder, index_ij_k.remainder, index_l_m.quotient, index_l_m.remainder, index_lm_n.remainder);
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_6d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_6d_tile_1d_t task = (pthreadpool_task_6d_tile_1d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_lmn = threadpool->params.parallelize_6d_tile_1d.tile_range_lmn;
	const struct fxdiv_result_size_t tile_index_ijk_lmn = fxdiv_divide_size_t(range_start, tile_range_lmn);
	const struct fxdiv_divisor_size_t range_k = threadpool->params.parallelize_6d_tile_1d.range_k;
	const struct fxdiv_result_size_t index_ij_k = fxdiv_divide_size_t(tile_index_ijk_lmn.quotient, range_k);
	const struct fxdiv_divisor_size_t tile_range_n = threadpool->params.parallelize_6d_tile_1d.tile_range_n;
	const struct fxdiv_result_size_t tile_index_lm_n = fxdiv_divide_size_t(tile_index_ijk_lmn.remainder, tile_range_n);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_6d_tile_1d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_k.quotient, range_j);
	const struct fxdiv_divisor_size_t range_m = threadpool->params.parallelize_6d_tile_1d.range_m;
	const struct fxdiv_result_size_t index_l_m = fxdiv_divide_size_t(tile_index_lm_n.quotient, range_m);
	const size_t tile_n = threadpool->params.parallelize_6d_tile_1d.tile_n;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t k = index_ij_k.remainder;
	size_t l = index_l_m.quotient;
	size_t m = index_l_m.remainder;
	size_t start_n = tile_index_lm_n.remainder * tile_n;

	const size_t range_n = threadpool->params.parallelize_6d_tile_1d.range_n;
	const size_t range_l = threadpool->params.parallelize_6d_tile_1d.range_l;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, k, l, m, start_n, min(range_n - start_n, tile_n));
		start_n += tile_n;
		if (start_n >= range_n) {
			start_n = 0;
			if (++m == range_m.value) {
				m = 0;
				if (++l == range_l) {
					l = 0;
					if (++k == range_k.value) {
						k = 0;
						if (++j == range_j.value) {
							j = 0;
							i += 1;
						}
					}
				}
			}
		}
	}


	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ijk_lmn = fxdiv_divide_size_t(linear_index, tile_range_lmn);
			const struct fxdiv_result_size_t index_ij_k = fxdiv_divide_size_t(tile_index_ijk_lmn.quotient, range_k);
			const struct fxdiv_result_size_t tile_index_lm_n = fxdiv_divide_size_t(tile_index_ijk_lmn.remainder, tile_range_n);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_k.quotient, range_j);
			const struct fxdiv_result_size_t index_l_m = fxdiv_divide_size_t(tile_index_lm_n.quotient, range_m);
			const size_t start_n = tile_index_lm_n.remainder * tile_n;
			task(argument, index_i_j.quotient, index_i_j.remainder, index_ij_k.remainder, index_l_m.quotient, index_l_m.remainder,
				start_n, min(range_n - start_n, tile_n));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_6d_tile_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread)
{
	assert(threadpool != NULL);
	assert(thread != NULL);

	const pthreadpool_task_6d_tile_2d_t task = (pthreadpool_task_6d_tile_2d_t) pthreadpool_load_relaxed_void_p(&threadpool->task);
	void *const argument = pthreadpool_load_relaxed_void_p(&threadpool->argument);

	const size_t threads_count = threadpool->threads_count.value;
	const size_t range_threshold = -threads_count;

	/* Process thread's own range of items */
	const size_t range_start = pthreadpool_load_relaxed_size_t(&thread->range_start);
	const struct fxdiv_divisor_size_t tile_range_mn = threadpool->params.parallelize_6d_tile_2d.tile_range_mn;
	const struct fxdiv_result_size_t tile_index_ijkl_mn = fxdiv_divide_size_t(range_start, tile_range_mn);
	const struct fxdiv_divisor_size_t range_kl = threadpool->params.parallelize_6d_tile_2d.range_kl;
	const struct fxdiv_result_size_t index_ij_kl = fxdiv_divide_size_t(tile_index_ijkl_mn.quotient, range_kl);
	const struct fxdiv_divisor_size_t tile_range_n = threadpool->params.parallelize_6d_tile_2d.tile_range_n;
	const struct fxdiv_result_size_t tile_index_m_n = fxdiv_divide_size_t(tile_index_ijkl_mn.remainder, tile_range_n);
	const struct fxdiv_divisor_size_t range_j = threadpool->params.parallelize_6d_tile_2d.range_j;
	const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_kl.quotient, range_j);
	const struct fxdiv_divisor_size_t range_l = threadpool->params.parallelize_6d_tile_2d.range_l;
	const struct fxdiv_result_size_t index_k_l = fxdiv_divide_size_t(index_ij_kl.remainder, range_l);
	const size_t tile_m = threadpool->params.parallelize_6d_tile_2d.tile_m;
	const size_t tile_n = threadpool->params.parallelize_6d_tile_2d.tile_n;
	size_t i = index_i_j.quotient;
	size_t j = index_i_j.remainder;
	size_t k = index_k_l.quotient;
	size_t l = index_k_l.remainder;
	size_t start_m = tile_index_m_n.quotient * tile_m;
	size_t start_n = tile_index_m_n.remainder * tile_n;

	const size_t range_n = threadpool->params.parallelize_6d_tile_2d.range_n;
	const size_t range_m = threadpool->params.parallelize_6d_tile_2d.range_m;
	const size_t range_k = threadpool->params.parallelize_6d_tile_2d.range_k;
	while (pthreadpool_decrement_fetch_relaxed_size_t(&thread->range_length) < range_threshold) {
		task(argument, i, j, k, l, start_m, start_n, min(range_m - start_m, tile_m), min(range_n - start_n, tile_n));
		start_n += tile_n;
		if (start_n >= range_n) {
			start_n = 0;
			start_m += tile_m;
			if (start_m >= range_m) {
				start_m = 0;
				if (++l == range_l.value) {
					l = 0;
					if (++k == range_k) {
						k = 0;
						if (++j == range_j.value) {
							j = 0;
							i += 1;
						}
					}
				}
			}
		}
	}

	/* There still may be other threads with work */
	const size_t thread_number = thread->thread_number;
	for (size_t tid = modulo_decrement(thread_number, threads_count);
		tid != thread_number;
		tid = modulo_decrement(tid, threads_count))
	{
		struct thread_info* other_thread = &threadpool->threads[tid];
		while (pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_length) < range_threshold) {
			const size_t linear_index = pthreadpool_decrement_fetch_relaxed_size_t(&other_thread->range_end);
			const struct fxdiv_result_size_t tile_index_ijkl_mn = fxdiv_divide_size_t(linear_index, tile_range_mn);
			const struct fxdiv_result_size_t index_ij_kl = fxdiv_divide_size_t(tile_index_ijkl_mn.quotient, range_kl);
			const struct fxdiv_result_size_t tile_index_m_n = fxdiv_divide_size_t(tile_index_ijkl_mn.remainder, tile_range_n);
			const struct fxdiv_result_size_t index_i_j = fxdiv_divide_size_t(index_ij_kl.quotient, range_j);
			const struct fxdiv_result_size_t index_k_l = fxdiv_divide_size_t(index_ij_kl.remainder, range_l);
			const size_t start_m = tile_index_m_n.quotient * tile_m;
			const size_t start_n = tile_index_m_n.remainder * tile_n;
			task(argument, index_i_j.quotient, index_i_j.remainder, index_k_l.quotient, index_k_l.remainder,
				start_m, start_n, min(range_m - start_m, tile_m), min(range_n - start_n, tile_n));
		}
	}

	/* Make changes by this thread visible to other threads */
	pthreadpool_fence_release();
}
