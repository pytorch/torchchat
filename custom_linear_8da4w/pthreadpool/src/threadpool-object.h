#pragma once

/* Standard C headers */
#include <stddef.h>
#include <stdint.h>

/* Internal headers */
#include "threadpool-common.h"
#include "threadpool-atomics.h"

/* POSIX headers */
#if PTHREADPOOL_USE_CONDVAR || PTHREADPOOL_USE_FUTEX
#include <pthread.h>
#endif

/* Mach headers */
#if PTHREADPOOL_USE_GCD
#include <dispatch/dispatch.h>
#endif

/* Windows headers */
#if PTHREADPOOL_USE_EVENT
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

/* Dependencies */
#include <fxdiv.h>

/* Library header */
#include <pthreadpool.h>


#define THREADPOOL_COMMAND_MASK UINT32_C(0x7FFFFFFF)

enum threadpool_command {
	threadpool_command_init,
	threadpool_command_parallelize,
	threadpool_command_shutdown,
};

struct PTHREADPOOL_CACHELINE_ALIGNED thread_info {
	/**
	 * Index of the first element in the work range.
	 * Before processing a new element the owning worker thread increments this value.
	 */
	pthreadpool_atomic_size_t range_start;
	/**
	 * Index of the element after the last element of the work range.
	 * Before processing a new element the stealing worker thread decrements this value.
	 */
	pthreadpool_atomic_size_t range_end;
	/**
	 * The number of elements in the work range.
	 * Due to race conditions range_length <= range_end - range_start.
	 * The owning worker thread must decrement this value before incrementing @a range_start.
	 * The stealing worker thread must decrement this value before decrementing @a range_end.
	 */
	pthreadpool_atomic_size_t range_length;
	/**
	 * Thread number in the 0..threads_count-1 range.
	 */
	size_t thread_number;
	/**
	 * Thread pool which owns the thread.
	 */
	struct pthreadpool* threadpool;
#if PTHREADPOOL_USE_CONDVAR || PTHREADPOOL_USE_FUTEX
	/**
	 * The pthread object corresponding to the thread.
	 */
	pthread_t thread_object;
#endif
#if PTHREADPOOL_USE_EVENT
	/**
	 * The Windows thread handle corresponding to the thread.
	 */
	HANDLE thread_handle;
#endif
};

PTHREADPOOL_STATIC_ASSERT(sizeof(struct thread_info) % PTHREADPOOL_CACHELINE_SIZE == 0,
	"thread_info structure must occupy an integer number of cache lines (64 bytes)");

struct pthreadpool_1d_with_uarch_params {
	/**
	 * Copy of the default_uarch_index argument passed to the pthreadpool_parallelize_1d_with_uarch function.
	 */
	uint32_t default_uarch_index;
	/**
	 * Copy of the max_uarch_index argument passed to the pthreadpool_parallelize_1d_with_uarch function.
	 */
	uint32_t max_uarch_index;
};

struct pthreadpool_1d_tile_1d_params {
	/**
	 * Copy of the range argument passed to the pthreadpool_parallelize_1d_tile_1d function.
	 */
	size_t range;
	/**
	 * Copy of the tile argument passed to the pthreadpool_parallelize_1d_tile_1d function.
	 */
	size_t tile;
};

struct pthreadpool_2d_params {
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_2d function.
	 */
	struct fxdiv_divisor_size_t range_j;
};

struct pthreadpool_2d_tile_1d_params {
	/**
	 * Copy of the range_j argument passed to the pthreadpool_parallelize_2d_tile_1d function.
	 */
	size_t range_j;
	/**
	 * Copy of the tile_j argument passed to the pthreadpool_parallelize_2d_tile_1d function.
	 */
	size_t tile_j;
	/**
	 * FXdiv divisor for the divide_round_up(range_j, tile_j) value.
	 */
	struct fxdiv_divisor_size_t tile_range_j;
};

struct pthreadpool_2d_tile_1d_with_uarch_params {
	/**
	 * Copy of the default_uarch_index argument passed to the pthreadpool_parallelize_2d_tile_1d_with_uarch function.
	 */
	uint32_t default_uarch_index;
	/**
	 * Copy of the max_uarch_index argument passed to the pthreadpool_parallelize_2d_tile_1d_with_uarch function.
	 */
	uint32_t max_uarch_index;
	/**
	 * Copy of the range_j argument passed to the pthreadpool_parallelize_2d_tile_1d function.
	 */
	size_t range_j;
	/**
	 * Copy of the tile_j argument passed to the pthreadpool_parallelize_2d_tile_1d function.
	 */
	size_t tile_j;
	/**
	 * FXdiv divisor for the divide_round_up(range_j, tile_j) value.
	 */
	struct fxdiv_divisor_size_t tile_range_j;
};

struct pthreadpool_2d_tile_2d_params {
	/**
	 * Copy of the range_i argument passed to the pthreadpool_parallelize_2d_tile_2d function.
	 */
	size_t range_i;
	/**
	 * Copy of the tile_i argument passed to the pthreadpool_parallelize_2d_tile_2d function.
	 */
	size_t tile_i;
	/**
	 * Copy of the range_j argument passed to the pthreadpool_parallelize_2d_tile_2d function.
	 */
	size_t range_j;
	/**
	 * Copy of the tile_j argument passed to the pthreadpool_parallelize_2d_tile_2d function.
	 */
	size_t tile_j;
	/**
	 * FXdiv divisor for the divide_round_up(range_j, tile_j) value.
	 */
	struct fxdiv_divisor_size_t tile_range_j;
};

struct pthreadpool_2d_tile_2d_with_uarch_params {
	/**
	 * Copy of the default_uarch_index argument passed to the pthreadpool_parallelize_2d_tile_2d_with_uarch function.
	 */
	uint32_t default_uarch_index;
	/**
	 * Copy of the max_uarch_index argument passed to the pthreadpool_parallelize_2d_tile_2d_with_uarch function.
	 */
	uint32_t max_uarch_index;
	/**
	 * Copy of the range_i argument passed to the pthreadpool_parallelize_2d_tile_2d_with_uarch function.
	 */
	size_t range_i;
	/**
	 * Copy of the tile_i argument passed to the pthreadpool_parallelize_2d_tile_2d_with_uarch function.
	 */
	size_t tile_i;
	/**
	 * Copy of the range_j argument passed to the pthreadpool_parallelize_2d_tile_2d_with_uarch function.
	 */
	size_t range_j;
	/**
	 * Copy of the tile_j argument passed to the pthreadpool_parallelize_2d_tile_2d_with_uarch function.
	 */
	size_t tile_j;
	/**
	 * FXdiv divisor for the divide_round_up(range_j, tile_j) value.
	 */
	struct fxdiv_divisor_size_t tile_range_j;
};

struct pthreadpool_3d_params {
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_3d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the range_k argument passed to the pthreadpool_parallelize_3d function.
	 */
	struct fxdiv_divisor_size_t range_k;
};

struct pthreadpool_3d_tile_1d_params {
	/**
	 * Copy of the range_k argument passed to the pthreadpool_parallelize_3d_tile_1d function.
	 */
	size_t range_k;
	/**
	 * Copy of the tile_k argument passed to the pthreadpool_parallelize_3d_tile_1d function.
	 */
	size_t tile_k;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_3d_tile_1d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the divide_round_up(range_k, tile_k) value.
	 */
	struct fxdiv_divisor_size_t tile_range_k;
};

struct pthreadpool_3d_tile_1d_with_uarch_params {
	/**
	 * Copy of the default_uarch_index argument passed to the pthreadpool_parallelize_3d_tile_1d_with_uarch function.
	 */
	uint32_t default_uarch_index;
	/**
	 * Copy of the max_uarch_index argument passed to the pthreadpool_parallelize_3d_tile_1d_with_uarch function.
	 */
	uint32_t max_uarch_index;
	/**
	 * Copy of the range_k argument passed to the pthreadpool_parallelize_3d_tile_1d_with_uarch function.
	 */
	size_t range_k;
	/**
	 * Copy of the tile_k argument passed to the pthreadpool_parallelize_3d_tile_1d_with_uarch function.
	 */
	size_t tile_k;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_3d_tile_1d_with_uarch function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the divide_round_up(range_k, tile_k) value.
	 */
	struct fxdiv_divisor_size_t tile_range_k;
};

struct pthreadpool_3d_tile_2d_params {
	/**
	 * Copy of the range_j argument passed to the pthreadpool_parallelize_3d_tile_2d function.
	 */
	size_t range_j;
	/**
	 * Copy of the tile_j argument passed to the pthreadpool_parallelize_3d_tile_2d function.
	 */
	size_t tile_j;
	/**
	 * Copy of the range_k argument passed to the pthreadpool_parallelize_3d_tile_2d function.
	 */
	size_t range_k;
	/**
	 * Copy of the tile_k argument passed to the pthreadpool_parallelize_3d_tile_2d function.
	 */
	size_t tile_k;
	/**
	 * FXdiv divisor for the divide_round_up(range_j, tile_j) value.
	 */
	struct fxdiv_divisor_size_t tile_range_j;
	/**
	 * FXdiv divisor for the divide_round_up(range_k, tile_k) value.
	 */
	struct fxdiv_divisor_size_t tile_range_k;
};

struct pthreadpool_3d_tile_2d_with_uarch_params {
	/**
	 * Copy of the default_uarch_index argument passed to the pthreadpool_parallelize_3d_tile_2d_with_uarch function.
	 */
	uint32_t default_uarch_index;
	/**
	 * Copy of the max_uarch_index argument passed to the pthreadpool_parallelize_3d_tile_2d_with_uarch function.
	 */
	uint32_t max_uarch_index;
	/**
	 * Copy of the range_j argument passed to the pthreadpool_parallelize_3d_tile_2d_with_uarch function.
	 */
	size_t range_j;
	/**
	 * Copy of the tile_j argument passed to the pthreadpool_parallelize_3d_tile_2d_with_uarch function.
	 */
	size_t tile_j;
	/**
	 * Copy of the range_k argument passed to the pthreadpool_parallelize_3d_tile_2d_with_uarch function.
	 */
	size_t range_k;
	/**
	 * Copy of the tile_k argument passed to the pthreadpool_parallelize_3d_tile_2d_with_uarch function.
	 */
	size_t tile_k;
	/**
	 * FXdiv divisor for the divide_round_up(range_j, tile_j) value.
	 */
	struct fxdiv_divisor_size_t tile_range_j;
	/**
	 * FXdiv divisor for the divide_round_up(range_k, tile_k) value.
	 */
	struct fxdiv_divisor_size_t tile_range_k;
};

struct pthreadpool_4d_params {
	/**
	 * Copy of the range_k argument passed to the pthreadpool_parallelize_4d function.
	 */
	size_t range_k;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_4d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the range_k * range_l value.
	 */
	struct fxdiv_divisor_size_t range_kl;
	/**
	 * FXdiv divisor for the range_l argument passed to the pthreadpool_parallelize_4d function.
	 */
	struct fxdiv_divisor_size_t range_l;
};

struct pthreadpool_4d_tile_1d_params {
	/**
	 * Copy of the range_k argument passed to the pthreadpool_parallelize_4d_tile_1d function.
	 */
	size_t range_k;
	/**
	 * Copy of the range_l argument passed to the pthreadpool_parallelize_4d_tile_1d function.
	 */
	size_t range_l;
	/**
	 * Copy of the tile_l argument passed to the pthreadpool_parallelize_4d_tile_1d function.
	 */
	size_t tile_l;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_4d_tile_1d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the range_k * divide_round_up(range_l, tile_l) value.
	 */
	struct fxdiv_divisor_size_t tile_range_kl;
	/**
	 * FXdiv divisor for the divide_round_up(range_l, tile_l) value.
	 */
	struct fxdiv_divisor_size_t tile_range_l;
};

struct pthreadpool_4d_tile_2d_params {
	/**
	 * Copy of the range_k argument passed to the pthreadpool_parallelize_4d_tile_2d function.
	 */
	size_t range_k;
	/**
	 * Copy of the tile_k argument passed to the pthreadpool_parallelize_4d_tile_2d function.
	 */
	size_t tile_k;
	/**
	 * Copy of the range_l argument passed to the pthreadpool_parallelize_4d_tile_2d function.
	 */
	size_t range_l;
	/**
	 * Copy of the tile_l argument passed to the pthreadpool_parallelize_4d_tile_2d function.
	 */
	size_t tile_l;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_4d_tile_2d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the divide_round_up(range_k, tile_k) * divide_round_up(range_l, tile_l) value.
	 */
	struct fxdiv_divisor_size_t tile_range_kl;
	/**
	 * FXdiv divisor for the divide_round_up(range_l, tile_l) value.
	 */
	struct fxdiv_divisor_size_t tile_range_l;
};

struct pthreadpool_4d_tile_2d_with_uarch_params {
	/**
	 * Copy of the default_uarch_index argument passed to the pthreadpool_parallelize_4d_tile_2d_with_uarch function.
	 */
	uint32_t default_uarch_index;
	/**
	 * Copy of the max_uarch_index argument passed to the pthreadpool_parallelize_4d_tile_2d_with_uarch function.
	 */
	uint32_t max_uarch_index;
	/**
	 * Copy of the range_k argument passed to the pthreadpool_parallelize_4d_tile_2d_with_uarch function.
	 */
	size_t range_k;
	/**
	 * Copy of the tile_k argument passed to the pthreadpool_parallelize_4d_tile_2d_with_uarch function.
	 */
	size_t tile_k;
	/**
	 * Copy of the range_l argument passed to the pthreadpool_parallelize_4d_tile_2d_with_uarch function.
	 */
	size_t range_l;
	/**
	 * Copy of the tile_l argument passed to the pthreadpool_parallelize_4d_tile_2d_with_uarch function.
	 */
	size_t tile_l;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_4d_tile_2d_with_uarch function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the divide_round_up(range_k, tile_k) * divide_round_up(range_l, tile_l) value.
	 */
	struct fxdiv_divisor_size_t tile_range_kl;
	/**
	 * FXdiv divisor for the divide_round_up(range_l, tile_l) value.
	 */
	struct fxdiv_divisor_size_t tile_range_l;
};

struct pthreadpool_5d_params {
	/**
	 * Copy of the range_l argument passed to the pthreadpool_parallelize_5d function.
	 */
	size_t range_l;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_5d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the range_k argument passed to the pthreadpool_parallelize_5d function.
	 */
	struct fxdiv_divisor_size_t range_k;
	/**
	 * FXdiv divisor for the range_l * range_m value.
	 */
	struct fxdiv_divisor_size_t range_lm;
	/**
	 * FXdiv divisor for the range_m argument passed to the pthreadpool_parallelize_5d function.
	 */
	struct fxdiv_divisor_size_t range_m;
};

struct pthreadpool_5d_tile_1d_params {
	/**
	 * Copy of the range_k argument passed to the pthreadpool_parallelize_5d_tile_1d function.
	 */
	size_t range_k;
	/**
	 * Copy of the range_m argument passed to the pthreadpool_parallelize_5d_tile_1d function.
	 */
	size_t range_m;
	/**
	 * Copy of the tile_m argument passed to the pthreadpool_parallelize_5d_tile_1d function.
	 */
	size_t tile_m;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_5d_tile_1d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the range_k * range_l value.
	 */
	struct fxdiv_divisor_size_t range_kl;
	/**
	 * FXdiv divisor for the range_l argument passed to the pthreadpool_parallelize_5d_tile_1d function.
	 */
	struct fxdiv_divisor_size_t range_l;
	/**
	 * FXdiv divisor for the divide_round_up(range_m, tile_m) value.
	 */
	struct fxdiv_divisor_size_t tile_range_m;
};

struct pthreadpool_5d_tile_2d_params {
	/**
	 * Copy of the range_l argument passed to the pthreadpool_parallelize_5d_tile_2d function.
	 */
	size_t range_l;
	/**
	 * Copy of the tile_l argument passed to the pthreadpool_parallelize_5d_tile_2d function.
	 */
	size_t tile_l;
	/**
	 * Copy of the range_m argument passed to the pthreadpool_parallelize_5d_tile_2d function.
	 */
	size_t range_m;
	/**
	 * Copy of the tile_m argument passed to the pthreadpool_parallelize_5d_tile_2d function.
	 */
	size_t tile_m;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_5d_tile_2d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the range_k argument passed to the pthreadpool_parallelize_5d_tile_2d function.
	 */
	struct fxdiv_divisor_size_t range_k;
	/**
	 * FXdiv divisor for the divide_round_up(range_l, tile_l) * divide_round_up(range_m, tile_m) value.
	 */
	struct fxdiv_divisor_size_t tile_range_lm;
	/**
	 * FXdiv divisor for the divide_round_up(range_m, tile_m) value.
	 */
	struct fxdiv_divisor_size_t tile_range_m;
};

struct pthreadpool_6d_params {
	/**
	 * Copy of the range_l argument passed to the pthreadpool_parallelize_6d function.
	 */
	size_t range_l;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_6d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the range_k argument passed to the pthreadpool_parallelize_6d function.
	 */
	struct fxdiv_divisor_size_t range_k;
	/**
	 * FXdiv divisor for the range_l * range_m * range_n value.
	 */
	struct fxdiv_divisor_size_t range_lmn;
	/**
	 * FXdiv divisor for the range_m argument passed to the pthreadpool_parallelize_6d function.
	 */
	struct fxdiv_divisor_size_t range_m;
	/**
	 * FXdiv divisor for the range_n argument passed to the pthreadpool_parallelize_6d function.
	 */
	struct fxdiv_divisor_size_t range_n;
};

struct pthreadpool_6d_tile_1d_params {
	/**
	 * Copy of the range_l argument passed to the pthreadpool_parallelize_6d_tile_1d function.
	 */
	size_t range_l;
	/**
	 * Copy of the range_n argument passed to the pthreadpool_parallelize_6d_tile_1d function.
	 */
	size_t range_n;
	/**
	 * Copy of the tile_n argument passed to the pthreadpool_parallelize_6d_tile_1d function.
	 */
	size_t tile_n;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_6d_tile_1d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the range_k argument passed to the pthreadpool_parallelize_6d_tile_1d function.
	 */
	struct fxdiv_divisor_size_t range_k;
	/**
	 * FXdiv divisor for the range_l * range_m * divide_round_up(range_n, tile_n) value.
	 */
	struct fxdiv_divisor_size_t tile_range_lmn;
	/**
	 * FXdiv divisor for the range_m argument passed to the pthreadpool_parallelize_6d_tile_1d function.
	 */
	struct fxdiv_divisor_size_t range_m;
	/**
	 * FXdiv divisor for the divide_round_up(range_n, tile_n) value.
	 */
	struct fxdiv_divisor_size_t tile_range_n;
};

struct pthreadpool_6d_tile_2d_params {
	/**
	 * Copy of the range_k argument passed to the pthreadpool_parallelize_6d_tile_2d function.
	 */
	size_t range_k;
	/**
	 * Copy of the range_m argument passed to the pthreadpool_parallelize_6d_tile_2d function.
	 */
	size_t range_m;
	/**
	 * Copy of the tile_m argument passed to the pthreadpool_parallelize_6d_tile_2d function.
	 */
	size_t tile_m;
	/**
	 * Copy of the range_n argument passed to the pthreadpool_parallelize_6d_tile_2d function.
	 */
	size_t range_n;
	/**
	 * Copy of the tile_n argument passed to the pthreadpool_parallelize_6d_tile_2d function.
	 */
	size_t tile_n;
	/**
	 * FXdiv divisor for the range_j argument passed to the pthreadpool_parallelize_6d_tile_2d function.
	 */
	struct fxdiv_divisor_size_t range_j;
	/**
	 * FXdiv divisor for the range_k * range_l value.
	 */
	struct fxdiv_divisor_size_t range_kl;
	/**
	 * FXdiv divisor for the range_l argument passed to the pthreadpool_parallelize_6d_tile_2d function.
	 */
	struct fxdiv_divisor_size_t range_l;
	/**
	 * FXdiv divisor for the divide_round_up(range_m, tile_m) * divide_round_up(range_n, tile_n) value.
	 */
	struct fxdiv_divisor_size_t tile_range_mn;
	/**
	 * FXdiv divisor for the divide_round_up(range_n, tile_n) value.
	 */
	struct fxdiv_divisor_size_t tile_range_n;
};

struct PTHREADPOOL_CACHELINE_ALIGNED pthreadpool {
#if !PTHREADPOOL_USE_GCD
	/**
	 * The number of threads that are processing an operation.
	 */
	pthreadpool_atomic_size_t active_threads;
#endif
#if PTHREADPOOL_USE_FUTEX
	/**
	 * Indicates if there are active threads.
	 * Only two values are possible:
	 * - has_active_threads == 0 if active_threads == 0
	 * - has_active_threads == 1 if active_threads != 0
	 */
	pthreadpool_atomic_uint32_t has_active_threads;
#endif
#if !PTHREADPOOL_USE_GCD
	/**
	 * The last command submitted to the thread pool.
	 */
	pthreadpool_atomic_uint32_t command;
#endif
	/**
	 * The entry point function to call for each thread in the thread pool for parallelization tasks.
	 */
	pthreadpool_atomic_void_p thread_function;
	/**
	 * The function to call for each item.
	 */
	pthreadpool_atomic_void_p task;
	/**
	 * The first argument to the item processing function.
	 */
	pthreadpool_atomic_void_p argument;
	/**
	 * Additional parallelization parameters.
	 * These parameters are specific for each thread_function.
	 */
	union {
		struct pthreadpool_1d_with_uarch_params parallelize_1d_with_uarch;
		struct pthreadpool_1d_tile_1d_params parallelize_1d_tile_1d;
		struct pthreadpool_2d_params parallelize_2d;
		struct pthreadpool_2d_tile_1d_params parallelize_2d_tile_1d;
		struct pthreadpool_2d_tile_1d_with_uarch_params parallelize_2d_tile_1d_with_uarch;
		struct pthreadpool_2d_tile_2d_params parallelize_2d_tile_2d;
		struct pthreadpool_2d_tile_2d_with_uarch_params parallelize_2d_tile_2d_with_uarch;
		struct pthreadpool_3d_params parallelize_3d;
		struct pthreadpool_3d_tile_1d_params parallelize_3d_tile_1d;
		struct pthreadpool_3d_tile_1d_with_uarch_params parallelize_3d_tile_1d_with_uarch;
		struct pthreadpool_3d_tile_2d_params parallelize_3d_tile_2d;
		struct pthreadpool_3d_tile_2d_with_uarch_params parallelize_3d_tile_2d_with_uarch;
		struct pthreadpool_4d_params parallelize_4d;
		struct pthreadpool_4d_tile_1d_params parallelize_4d_tile_1d;
		struct pthreadpool_4d_tile_2d_params parallelize_4d_tile_2d;
		struct pthreadpool_4d_tile_2d_with_uarch_params parallelize_4d_tile_2d_with_uarch;
		struct pthreadpool_5d_params parallelize_5d;
		struct pthreadpool_5d_tile_1d_params parallelize_5d_tile_1d;
		struct pthreadpool_5d_tile_2d_params parallelize_5d_tile_2d;
		struct pthreadpool_6d_params parallelize_6d;
		struct pthreadpool_6d_tile_1d_params parallelize_6d_tile_1d;
		struct pthreadpool_6d_tile_2d_params parallelize_6d_tile_2d;
	} params;
	/**
	 * Copy of the flags passed to a parallelization function.
	 */
	pthreadpool_atomic_uint32_t flags;
#if PTHREADPOOL_USE_CONDVAR || PTHREADPOOL_USE_FUTEX
	/**
	 * Serializes concurrent calls to @a pthreadpool_parallelize_* from different threads.
	 */
	pthread_mutex_t execution_mutex;
#endif
#if PTHREADPOOL_USE_GCD
	/**
	 * Serializes concurrent calls to @a pthreadpool_parallelize_* from different threads.
	 */
	dispatch_semaphore_t execution_semaphore;
#endif
#if PTHREADPOOL_USE_EVENT
	/**
	 * Serializes concurrent calls to @a pthreadpool_parallelize_* from different threads.
	 */
	HANDLE execution_mutex;
#endif
#if PTHREADPOOL_USE_CONDVAR
	/**
	 * Guards access to the @a active_threads variable.
	 */
	pthread_mutex_t completion_mutex;
	/**
	 * Condition variable to wait until all threads complete an operation (until @a active_threads is zero).
	 */
	pthread_cond_t completion_condvar;
	/**
	 * Guards access to the @a command variable.
	 */
	pthread_mutex_t command_mutex;
	/**
	 * Condition variable to wait for change of the @a command variable.
	 */
	pthread_cond_t command_condvar;
#endif
#if PTHREADPOOL_USE_EVENT
	/**
	 * Events to wait on until all threads complete an operation (until @a active_threads is zero).
	 * To avoid race conditions due to spin-lock synchronization, we use two events and switch event in use after every
	 * submitted command according to the high bit of the command word.
	 */
	HANDLE completion_event[2];
	/**
	 * Events to wait on for change of the @a command variable.
	 * To avoid race conditions due to spin-lock synchronization, we use two events and switch event in use after every
	 * submitted command according to the high bit of the command word.
	 */
	HANDLE command_event[2];
#endif
	/**
	 * FXdiv divisor for the number of threads in the thread pool.
	 * This struct never change after pthreadpool_create.
	 */
	struct fxdiv_divisor_size_t threads_count;
	/**
	 * Thread information structures that immediately follow this structure.
	 */
	struct thread_info threads[];
};

PTHREADPOOL_STATIC_ASSERT(sizeof(struct pthreadpool) % PTHREADPOOL_CACHELINE_SIZE == 0,
	"pthreadpool structure must occupy an integer number of cache lines (64 bytes)");

PTHREADPOOL_INTERNAL struct pthreadpool* pthreadpool_allocate(
	size_t threads_count);

PTHREADPOOL_INTERNAL void pthreadpool_deallocate(
	struct pthreadpool* threadpool);

typedef void (*thread_function_t)(struct pthreadpool* threadpool, struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_parallelize(
	struct pthreadpool* threadpool,
	thread_function_t thread_function,
	const void* params,
	size_t params_size,
	void* task,
	void* context,
	size_t linear_range,
	uint32_t flags);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_1d_with_thread_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_1d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_1d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_with_thread_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_tile_1d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_tile_1d_with_uarch_with_thread_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_tile_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_2d_tile_2d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_1d_with_thread_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_1d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_1d_with_uarch_with_thread_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_3d_tile_2d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_4d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_4d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_4d_tile_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_4d_tile_2d_with_uarch_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_5d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_5d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_5d_tile_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_6d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_6d_tile_1d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);

PTHREADPOOL_INTERNAL void pthreadpool_thread_parallelize_6d_tile_2d_fastpath(
	struct pthreadpool* threadpool,
	struct thread_info* thread);
