/* Standard C headers */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Configuration header */
#include "threadpool-common.h"

/* Windows headers */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

/* Public library header */
#include <pthreadpool.h>

/* Internal library headers */
#include "threadpool-atomics.h"
#include "threadpool-object.h"
#include "threadpool-utils.h"


static void checkin_worker_thread(struct pthreadpool* threadpool, uint32_t event_index) {
	if (pthreadpool_decrement_fetch_acquire_release_size_t(&threadpool->active_threads) == 0) {
		SetEvent(threadpool->completion_event[event_index]);
	}
}

static void wait_worker_threads(struct pthreadpool* threadpool, uint32_t event_index) {
	/* Initial check */
	size_t active_threads = pthreadpool_load_acquire_size_t(&threadpool->active_threads);
	if (active_threads == 0) {
		return;
	}

	/* Spin-wait */
	for (uint32_t i = PTHREADPOOL_SPIN_WAIT_ITERATIONS; i != 0; i--) {
		pthreadpool_yield();

		active_threads = pthreadpool_load_acquire_size_t(&threadpool->active_threads);
		if (active_threads == 0) {
			return;
		}
	}

	/* Fall-back to event wait */
	const DWORD wait_status = WaitForSingleObject(threadpool->completion_event[event_index], INFINITE);
	assert(wait_status == WAIT_OBJECT_0);
	assert(pthreadpool_load_relaxed_size_t(&threadpool->active_threads) == 0);
}

static uint32_t wait_for_new_command(
	struct pthreadpool* threadpool,
	uint32_t last_command,
	uint32_t last_flags)
{
	uint32_t command = pthreadpool_load_acquire_uint32_t(&threadpool->command);
	if (command != last_command) {
		return command;
	}

	if ((last_flags & PTHREADPOOL_FLAG_YIELD_WORKERS) == 0) {
		/* Spin-wait loop */
		for (uint32_t i = PTHREADPOOL_SPIN_WAIT_ITERATIONS; i != 0; i--) {
			pthreadpool_yield();

			command = pthreadpool_load_acquire_uint32_t(&threadpool->command);
			if (command != last_command) {
				return command;
			}
		}
	}

	/* Spin-wait disabled or timed out, fall back to event wait */
	const uint32_t event_index = (last_command >> 31);
	const DWORD wait_status = WaitForSingleObject(threadpool->command_event[event_index], INFINITE);
	assert(wait_status == WAIT_OBJECT_0);

	command = pthreadpool_load_relaxed_uint32_t(&threadpool->command);
	assert(command != last_command);
	return command;
}

static DWORD WINAPI thread_main(LPVOID arg) {
	struct thread_info* thread = (struct thread_info*) arg;
	struct pthreadpool* threadpool = thread->threadpool;
	uint32_t last_command = threadpool_command_init;
	struct fpu_state saved_fpu_state = { 0 };
	uint32_t flags = 0;

	/* Check in */
	checkin_worker_thread(threadpool, 0);

	/* Monitor new commands and act accordingly */
	for (;;) {
		uint32_t command = wait_for_new_command(threadpool, last_command, flags);
		pthreadpool_fence_acquire();

		flags = pthreadpool_load_relaxed_uint32_t(&threadpool->flags);

		/* Process command */
		switch (command & THREADPOOL_COMMAND_MASK) {
			case threadpool_command_parallelize:
			{
				const thread_function_t thread_function =
					(thread_function_t) pthreadpool_load_relaxed_void_p(&threadpool->thread_function);
				if (flags & PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
					saved_fpu_state = get_fpu_state();
					disable_fpu_denormals();
				}

				thread_function(threadpool, thread);
				if (flags & PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
					set_fpu_state(saved_fpu_state);
				}
				break;
			}
			case threadpool_command_shutdown:
				/* Exit immediately: the master thread is waiting on pthread_join */
				return 0;
			case threadpool_command_init:
				/* To inhibit compiler warning */
				break;
		}
		/* Notify the master thread that we finished processing */
		const uint32_t event_index = command >> 31;
		checkin_worker_thread(threadpool, event_index);
		/* Update last command */
		last_command = command;
	};
	return 0;
}

struct pthreadpool* pthreadpool_create(size_t threads_count) {
	if (threads_count == 0) {
		SYSTEM_INFO system_info;
		ZeroMemory(&system_info, sizeof(system_info));
		GetSystemInfo(&system_info);
		threads_count = (size_t) system_info.dwNumberOfProcessors;
	}

	struct pthreadpool* threadpool = pthreadpool_allocate(threads_count);
	if (threadpool == NULL) {
		return NULL;
	}
	threadpool->threads_count = fxdiv_init_size_t(threads_count);
	for (size_t tid = 0; tid < threads_count; tid++) {
		threadpool->threads[tid].thread_number = tid;
		threadpool->threads[tid].threadpool = threadpool;
	}

	/* Thread pool with a single thread computes everything on the caller thread. */
	if (threads_count > 1) {
		threadpool->execution_mutex = CreateMutexW(
			NULL /* mutex attributes */,
			FALSE /* initially owned */,
			NULL /* name */);
		for (size_t i = 0; i < 2; i++) {
			threadpool->completion_event[i] = CreateEventW(
				NULL /* event attributes */,
				TRUE /* manual-reset event: yes */,
				FALSE /* initial state: nonsignaled */,
				NULL /* name */);
			threadpool->command_event[i] = CreateEventW(
				NULL /* event attributes */,
				TRUE /* manual-reset event: yes */,
				FALSE /* initial state: nonsignaled */,
				NULL /* name */);
		}

		pthreadpool_store_relaxed_size_t(&threadpool->active_threads, threads_count - 1 /* caller thread */);

		/* Caller thread serves as worker #0. Thus, we create system threads starting with worker #1. */
		for (size_t tid = 1; tid < threads_count; tid++) {
			threadpool->threads[tid].thread_handle = CreateThread(
				NULL /* thread attributes */,
				0 /* stack size: default */,
				&thread_main,
				&threadpool->threads[tid],
				0 /* creation flags */,
				NULL /* thread id */);
		}

		/* Wait until all threads initialize */
		wait_worker_threads(threadpool, 0);
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
	const DWORD wait_status = WaitForSingleObject(threadpool->execution_mutex, INFINITE);
	assert(wait_status == WAIT_OBJECT_0);

	/* Setup global arguments */
	pthreadpool_store_relaxed_void_p(&threadpool->thread_function, (void*) thread_function);
	pthreadpool_store_relaxed_void_p(&threadpool->task, task);
	pthreadpool_store_relaxed_void_p(&threadpool->argument, context);
	pthreadpool_store_relaxed_uint32_t(&threadpool->flags, flags);

	const struct fxdiv_divisor_size_t threads_count = threadpool->threads_count;
	pthreadpool_store_relaxed_size_t(&threadpool->active_threads, threads_count.value - 1 /* caller thread */);

	if (params_size != 0) {
		CopyMemory(&threadpool->params, params, params_size);
		pthreadpool_fence_release();
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

	/*
	 * Update the threadpool command.
	 * Imporantly, do it after initializing command parameters (range, task, argument, flags)
	 * ~(threadpool->command | THREADPOOL_COMMAND_MASK) flips the bits not in command mask
	 * to ensure the unmasked command is different then the last command, because worker threads
	 * monitor for change in the unmasked command.
	 */
	const uint32_t old_command = pthreadpool_load_relaxed_uint32_t(&threadpool->command);
	const uint32_t new_command = ~(old_command | THREADPOOL_COMMAND_MASK) | threadpool_command_parallelize;

	/*
	 * Reset the command event for the next command.
	 * It is important to reset the event before writing out the new command, because as soon as the worker threads
	 * observe the new command, they may process it and switch to waiting on the next command event.
	 *
	 * Note: the event is different from the command event signalled in this update.
	 */
	const uint32_t event_index = (old_command >> 31);
	BOOL reset_event_status = ResetEvent(threadpool->command_event[event_index ^ 1]);
	assert(reset_event_status != FALSE);

	/*
	 * Store the command with release semantics to guarantee that if a worker thread observes
	 * the new command value, it also observes the updated command parameters.
	 *
	 * Note: release semantics is necessary, because the workers might be waiting in a spin-loop
	 * rather than on the event object.
	 */
	pthreadpool_store_release_uint32_t(&threadpool->command, new_command);

	/*
	 * Signal the event to wake up the threads.
	 * Event in use must be switched after every submitted command to avoid race conditions.
	 * Choose the event based on the high bit of the command, which is flipped on every update.
	 */
	const BOOL set_event_status = SetEvent(threadpool->command_event[event_index]);
	assert(set_event_status != FALSE);

	/* Save and modify FPU denormals control, if needed */
	struct fpu_state saved_fpu_state = { 0 };
	if (flags & PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
		saved_fpu_state = get_fpu_state();
		disable_fpu_denormals();
	}

	/* Do computations as worker #0 */
	thread_function(threadpool, &threadpool->threads[0]);

	/* Restore FPU denormals control, if needed */
	if (flags & PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
		set_fpu_state(saved_fpu_state);
	}

	/*
	 * Wait until the threads finish computation
	 * Use the complementary event because it corresponds to the new command.
	 */
	wait_worker_threads(threadpool, event_index ^ 1);

	/*
	 * Reset the completion event for the next command.
	 * Note: the event is different from the one used for waiting in this update.
	 */
	reset_event_status = ResetEvent(threadpool->completion_event[event_index]);
	assert(reset_event_status != FALSE);

	/* Make changes by other threads visible to this thread */
	pthreadpool_fence_acquire();

	/* Unprotect the global threadpool structures */
	const BOOL release_mutex_status = ReleaseMutex(threadpool->execution_mutex);
	assert(release_mutex_status != FALSE);
}

void pthreadpool_destroy(struct pthreadpool* threadpool) {
	if (threadpool != NULL) {
		const size_t threads_count = threadpool->threads_count.value;
		if (threads_count > 1) {
			pthreadpool_store_relaxed_size_t(&threadpool->active_threads, threads_count - 1 /* caller thread */);

			/*
			 * Store the command with release semantics to guarantee that if a worker thread observes
			 * the new command value, it also observes the updated active_threads values.
			 */
			const uint32_t old_command = pthreadpool_load_relaxed_uint32_t(&threadpool->command);
			pthreadpool_store_release_uint32_t(&threadpool->command, threadpool_command_shutdown);

			/*
			 * Signal the event to wake up the threads.
			 * Event in use must be switched after every submitted command to avoid race conditions.
			 * Choose the event based on the high bit of the command, which is flipped on every update.
			 */
			const uint32_t event_index = (old_command >> 31);
			const BOOL set_event_status = SetEvent(threadpool->command_event[event_index]);
			assert(set_event_status != FALSE);

			/* Wait until all threads return */
			for (size_t tid = 1; tid < threads_count; tid++) {
				const HANDLE thread_handle = threadpool->threads[tid].thread_handle;
				if (thread_handle != NULL) {
					const DWORD wait_status = WaitForSingleObject(thread_handle, INFINITE);
					assert(wait_status == WAIT_OBJECT_0);

					const BOOL close_status = CloseHandle(thread_handle);
					assert(close_status != FALSE);
				}
			}

			/* Release resources */
			if (threadpool->execution_mutex != NULL) {
				const BOOL close_status = CloseHandle(threadpool->execution_mutex);
				assert(close_status != FALSE);
			}
			for (size_t i = 0; i < 2; i++) {
				if (threadpool->command_event[i] != NULL) {
					const BOOL close_status = CloseHandle(threadpool->command_event[i]);
					assert(close_status != FALSE);
				}
				if (threadpool->completion_event[i] != NULL) {
					const BOOL close_status = CloseHandle(threadpool->completion_event[i]);
					assert(close_status != FALSE);
				}
			}
		}
		pthreadpool_deallocate(threadpool);
	}
}
