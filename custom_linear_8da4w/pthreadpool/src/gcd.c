/* Standard C headers */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Configuration header */
#include "threadpool-common.h"

/* Mach headers */
#include <dispatch/dispatch.h>
#include <sys/types.h>
#include <sys/sysctl.h>

/* Public library header */
#include <pthreadpool.h>

/* Internal library headers */
#include "threadpool-atomics.h"
#include "threadpool-object.h"
#include "threadpool-utils.h"

static void thread_main(void* arg, size_t thread_index) {
	struct pthreadpool* threadpool = (struct pthreadpool*) arg;
	struct thread_info* thread = &threadpool->threads[thread_index];

	const uint32_t flags = pthreadpool_load_relaxed_uint32_t(&threadpool->flags);
	const thread_function_t thread_function =
		(thread_function_t) pthreadpool_load_relaxed_void_p(&threadpool->thread_function);

	struct fpu_state saved_fpu_state = { 0 };
	if (flags & PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
		saved_fpu_state = get_fpu_state();
		disable_fpu_denormals();
	}

	thread_function(threadpool, thread);

	if (flags & PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
		set_fpu_state(saved_fpu_state);
	}
}

struct pthreadpool* pthreadpool_create(size_t threads_count) {
	if (threads_count == 0) {
		int threads = 1;
		size_t sizeof_threads = sizeof(threads);
		if (sysctlbyname("hw.logicalcpu_max", &threads, &sizeof_threads, NULL, 0) != 0) {
			return NULL;
		}

		if (threads <= 0) {
			return NULL;
		}

		threads_count = (size_t) threads;
	}

	struct pthreadpool* threadpool = pthreadpool_allocate(threads_count);
	if (threadpool == NULL) {
		return NULL;
	}
	threadpool->threads_count = fxdiv_init_size_t(threads_count);
	for (size_t tid = 0; tid < threads_count; tid++) {
		threadpool->threads[tid].thread_number = tid;
	}

	/* Thread pool with a single thread computes everything on the caller thread. */
	if (threads_count > 1) {
		threadpool->execution_semaphore = dispatch_semaphore_create(1);
	}
	return threadpool;
}

PTHREADPOOL_INTERNAL void pthreadpool_parallelize(
	struct pthreadpool* threadpool,
	thread_function_t thread_function,
	const void* params,
	size_t params_size,
	void* task,
	void* context,
	size_t linear_range,
	uint32_t flags)
{
	assert(threadpool != NULL);
	assert(thread_function != NULL);
	assert(task != NULL);
	assert(linear_range > 1);

	/* Protect the global threadpool structures */
	dispatch_semaphore_wait(threadpool->execution_semaphore, DISPATCH_TIME_FOREVER);

	/* Setup global arguments */
	pthreadpool_store_relaxed_void_p(&threadpool->thread_function, (void*) thread_function);
	pthreadpool_store_relaxed_void_p(&threadpool->task, task);
	pthreadpool_store_relaxed_void_p(&threadpool->argument, context);
	pthreadpool_store_relaxed_uint32_t(&threadpool->flags, flags);

	/* Locking of completion_mutex not needed: readers are sleeping on command_condvar */
	const struct fxdiv_divisor_size_t threads_count = threadpool->threads_count;

	if (params_size != 0) {
		memcpy(&threadpool->params, params, params_size);
	}

	/* Spread the work between threads */
	const struct fxdiv_result_size_t range_params = fxdiv_divide_size_t(linear_range, threads_count);
	size_t range_start = 0;
	for (size_t tid = 0; tid < threads_count.value; tid++) {
		struct thread_info* thread = &threadpool->threads[tid];
		const size_t range_length = range_params.quotient + (size_t) (tid < range_params.remainder);
		const size_t range_end = range_start + range_length;
		pthreadpool_store_relaxed_size_t(&thread->range_start, range_start);
		pthreadpool_store_relaxed_size_t(&thread->range_end, range_end);
		pthreadpool_store_relaxed_size_t(&thread->range_length, range_length);

		/* The next subrange starts where the previous ended */
		range_start = range_end;
	}

	dispatch_apply_f(threads_count.value, DISPATCH_APPLY_AUTO, threadpool, thread_main);

	/* Unprotect the global threadpool structures */
	dispatch_semaphore_signal(threadpool->execution_semaphore);
}

void pthreadpool_destroy(struct pthreadpool* threadpool) {
	if (threadpool != NULL) {
		if (threadpool->execution_semaphore != NULL) {
			/* Release resources */
			dispatch_release(threadpool->execution_semaphore);
		}
		pthreadpool_deallocate(threadpool);
	}
}
