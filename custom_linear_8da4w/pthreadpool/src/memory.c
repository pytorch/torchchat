/* Standard C headers */
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* POSIX headers */
#ifdef __ANDROID__
	#include <malloc.h>
#endif

/* Windows headers */
#ifdef _WIN32
	#include <malloc.h>
#endif

/* Internal library headers */
#include "threadpool-common.h"
#include "threadpool-object.h"


PTHREADPOOL_INTERNAL struct pthreadpool* pthreadpool_allocate(
	size_t threads_count)
{
	assert(threads_count >= 1);

	const size_t threadpool_size = sizeof(struct pthreadpool) + threads_count * sizeof(struct thread_info);
	struct pthreadpool* threadpool = NULL;
	#if defined(__ANDROID__)
		/*
		 * Android didn't get posix_memalign until API level 17 (Android 4.2).
		 * Use (otherwise obsolete) memalign function on Android platform.
		 */
		threadpool = memalign(PTHREADPOOL_CACHELINE_SIZE, threadpool_size);
		if (threadpool == NULL) {
			return NULL;
		}
	#elif defined(_WIN32)
		threadpool = _aligned_malloc(threadpool_size, PTHREADPOOL_CACHELINE_SIZE);
		if (threadpool == NULL) {
			return NULL;
		}
	#else
		if (posix_memalign((void**) &threadpool, PTHREADPOOL_CACHELINE_SIZE, threadpool_size) != 0) {
			return NULL;
		}
	#endif
	memset(threadpool, 0, threadpool_size);
	return threadpool;
}


PTHREADPOOL_INTERNAL void pthreadpool_deallocate(
	struct pthreadpool* threadpool)
{
	assert(threadpool != NULL);

	const size_t threadpool_size = sizeof(struct pthreadpool) + threadpool->threads_count.value * sizeof(struct thread_info);
	memset(threadpool, 0, threadpool_size);

	#ifdef _WIN32
		_aligned_free(threadpool);
	#else
		free(threadpool);
	#endif
}
