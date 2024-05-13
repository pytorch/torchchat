/* Standard C headers */
#include <stddef.h>

/* Dependencies */
#include <fxdiv.h>

/* Public library header */
#include <pthreadpool.h>

/* Internal library headers */
#include "threadpool-utils.h"


void pthreadpool_compute_1d(
	pthreadpool_t threadpool,
	pthreadpool_function_1d_t function,
	void* argument,
	size_t range)
{
	pthreadpool_parallelize_1d(threadpool,
		(pthreadpool_task_1d_t) function, argument,
		range, 0 /* flags */);
}

void pthreadpool_compute_1d_tiled(
	pthreadpool_t threadpool,
	pthreadpool_function_1d_tiled_t function,
	void* argument,
	size_t range,
	size_t tile)
{
	pthreadpool_parallelize_1d_tile_1d(threadpool,
		(pthreadpool_task_1d_tile_1d_t) function, argument,
		range, tile, 0 /* flags */);
}

void pthreadpool_compute_2d(
	pthreadpool_t threadpool,
	pthreadpool_function_2d_t function,
	void* argument,
	size_t range_i,
	size_t range_j)
{
	pthreadpool_parallelize_2d(threadpool,
		(pthreadpool_task_2d_t) function, argument,
		range_i, range_j, 0 /* flags */);
}

void pthreadpool_compute_2d_tiled(
	pthreadpool_t threadpool,
	pthreadpool_function_2d_tiled_t function,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t tile_i,
	size_t tile_j)
{
	pthreadpool_parallelize_2d_tile_2d(threadpool,
		(pthreadpool_task_2d_tile_2d_t) function, argument,
		range_i, range_j, tile_i, tile_j, 0 /* flags */);
}

struct compute_3d_tiled_context {
	pthreadpool_function_3d_tiled_t function;
	void* argument;
	struct fxdiv_divisor_size_t tile_range_j;
	struct fxdiv_divisor_size_t tile_range_k;
	size_t range_i;
	size_t range_j;
	size_t range_k;
	size_t tile_i;
	size_t tile_j;
	size_t tile_k;
};

static void compute_3d_tiled(const struct compute_3d_tiled_context* context, size_t linear_index) {
	const struct fxdiv_divisor_size_t tile_range_k = context->tile_range_k;
	const struct fxdiv_result_size_t tile_index_ij_k = fxdiv_divide_size_t(linear_index, tile_range_k);
	const struct fxdiv_divisor_size_t tile_range_j = context->tile_range_j;
	const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(tile_index_ij_k.quotient, tile_range_j);
	const size_t max_tile_i = context->tile_i;
	const size_t max_tile_j = context->tile_j;
	const size_t max_tile_k = context->tile_k;
	const size_t index_i = tile_index_i_j.quotient * max_tile_i;
	const size_t index_j = tile_index_i_j.remainder * max_tile_j;
	const size_t index_k = tile_index_ij_k.remainder * max_tile_k;
	const size_t tile_i = min(max_tile_i, context->range_i - index_i);
	const size_t tile_j = min(max_tile_j, context->range_j - index_j);
	const size_t tile_k = min(max_tile_k, context->range_k - index_k);
	context->function(context->argument, index_i, index_j, index_k, tile_i, tile_j, tile_k);
}

void pthreadpool_compute_3d_tiled(
	pthreadpool_t threadpool,
	pthreadpool_function_3d_tiled_t function,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t tile_i,
	size_t tile_j,
	size_t tile_k)
{
	if (pthreadpool_get_threads_count(threadpool) <= 1) {
		/* No thread pool used: execute function sequentially on the calling thread */
		for (size_t i = 0; i < range_i; i += tile_i) {
			for (size_t j = 0; j < range_j; j += tile_j) {
				for (size_t k = 0; k < range_k; k += tile_k) {
					function(argument, i, j, k, min(range_i - i, tile_i), min(range_j - j, tile_j), min(range_k - k, tile_k));
				}
			}
		}
	} else {
		/* Execute in parallel on the thread pool using linearized index */
		const size_t tile_range_i = divide_round_up(range_i, tile_i);
		const size_t tile_range_j = divide_round_up(range_j, tile_j);
		const size_t tile_range_k = divide_round_up(range_k, tile_k);
		struct compute_3d_tiled_context context = {
			.function = function,
			.argument = argument,
			.tile_range_j = fxdiv_init_size_t(tile_range_j),
			.tile_range_k = fxdiv_init_size_t(tile_range_k),
			.range_i = range_i,
			.range_j = range_j,
			.range_k = range_k,
			.tile_i = tile_i,
			.tile_j = tile_j,
			.tile_k = tile_k
		};
		pthreadpool_parallelize_1d(threadpool,
			(pthreadpool_task_1d_t) compute_3d_tiled, &context,
			tile_range_i * tile_range_j * tile_range_k,
			0 /* flags */);
	}
}

struct compute_4d_tiled_context {
	pthreadpool_function_4d_tiled_t function;
	void* argument;
	struct fxdiv_divisor_size_t tile_range_kl;
	struct fxdiv_divisor_size_t tile_range_j;
	struct fxdiv_divisor_size_t tile_range_l;
	size_t range_i;
	size_t range_j;
	size_t range_k;
	size_t range_l;
	size_t tile_i;
	size_t tile_j;
	size_t tile_k;
	size_t tile_l;
};

static void compute_4d_tiled(const struct compute_4d_tiled_context* context, size_t linear_index) {
	const struct fxdiv_divisor_size_t tile_range_kl = context->tile_range_kl;
	const struct fxdiv_result_size_t tile_index_ij_kl = fxdiv_divide_size_t(linear_index, tile_range_kl);
	const struct fxdiv_divisor_size_t tile_range_j = context->tile_range_j;
	const struct fxdiv_result_size_t tile_index_i_j = fxdiv_divide_size_t(tile_index_ij_kl.quotient, tile_range_j);
	const struct fxdiv_divisor_size_t tile_range_l = context->tile_range_l;
	const struct fxdiv_result_size_t tile_index_k_l = fxdiv_divide_size_t(tile_index_ij_kl.remainder, tile_range_l);
	const size_t max_tile_i = context->tile_i;
	const size_t max_tile_j = context->tile_j;
	const size_t max_tile_k = context->tile_k;
	const size_t max_tile_l = context->tile_l;
	const size_t index_i = tile_index_i_j.quotient * max_tile_i;
	const size_t index_j = tile_index_i_j.remainder * max_tile_j;
	const size_t index_k = tile_index_k_l.quotient * max_tile_k;
	const size_t index_l = tile_index_k_l.remainder * max_tile_l;
	const size_t tile_i = min(max_tile_i, context->range_i - index_i);
	const size_t tile_j = min(max_tile_j, context->range_j - index_j);
	const size_t tile_k = min(max_tile_k, context->range_k - index_k);
	const size_t tile_l = min(max_tile_l, context->range_l - index_l);
	context->function(context->argument, index_i, index_j, index_k, index_l, tile_i, tile_j, tile_k, tile_l);
}

void pthreadpool_compute_4d_tiled(
	pthreadpool_t threadpool,
	pthreadpool_function_4d_tiled_t function,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	size_t tile_i,
	size_t tile_j,
	size_t tile_k,
	size_t tile_l)
{
	if (pthreadpool_get_threads_count(threadpool) <= 1) {
		/* No thread pool used: execute function sequentially on the calling thread */
		for (size_t i = 0; i < range_i; i += tile_i) {
			for (size_t j = 0; j < range_j; j += tile_j) {
				for (size_t k = 0; k < range_k; k += tile_k) {
					for (size_t l = 0; l < range_l; l += tile_l) {
						function(argument, i, j, k, l,
							min(range_i - i, tile_i), min(range_j - j, tile_j), min(range_k - k, tile_k), min(range_l - l, tile_l));
					}
				}
			}
		}
	} else {
		/* Execute in parallel on the thread pool using linearized index */
		const size_t tile_range_i = divide_round_up(range_i, tile_i);
		const size_t tile_range_j = divide_round_up(range_j, tile_j);
		const size_t tile_range_k = divide_round_up(range_k, tile_k);
		const size_t tile_range_l = divide_round_up(range_l, tile_l);
		struct compute_4d_tiled_context context = {
			.function = function,
			.argument = argument,
			.tile_range_kl = fxdiv_init_size_t(tile_range_k * tile_range_l),
			.tile_range_j = fxdiv_init_size_t(tile_range_j),
			.tile_range_l = fxdiv_init_size_t(tile_range_l),
			.range_i = range_i,
			.range_j = range_j,
			.range_k = range_k,
			.range_l = range_l,
			.tile_i = tile_i,
			.tile_j = tile_j,
			.tile_k = tile_k,
			.tile_l = tile_l
		};
		pthreadpool_parallelize_1d(threadpool,
			(pthreadpool_task_1d_t) compute_4d_tiled, &context,
			tile_range_i * tile_range_j * tile_range_k * tile_range_l,
			0 /* flags */);
	}
}
