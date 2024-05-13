#pragma once

#ifndef PTHREADPOOL_USE_CPUINFO
	#define PTHREADPOOL_USE_CPUINFO 0
#endif

#ifndef PTHREADPOOL_USE_FUTEX
	#if defined(__linux__)
		#define PTHREADPOOL_USE_FUTEX 1
	#elif defined(__EMSCRIPTEN__)
		#define PTHREADPOOL_USE_FUTEX 1
	#else
		#define PTHREADPOOL_USE_FUTEX 0
	#endif
#endif

#ifndef PTHREADPOOL_USE_GCD
	#if defined(__APPLE__)
		#define PTHREADPOOL_USE_GCD 1
	#else
		#define PTHREADPOOL_USE_GCD 0
	#endif
#endif

#ifndef PTHREADPOOL_USE_EVENT
	#if defined(_WIN32) || defined(__CYGWIN__)
		#define PTHREADPOOL_USE_EVENT 1
	#else
		#define PTHREADPOOL_USE_EVENT 0
	#endif
#endif

#ifndef PTHREADPOOL_USE_CONDVAR
	#if PTHREADPOOL_USE_GCD || PTHREADPOOL_USE_FUTEX || PTHREADPOOL_USE_EVENT
		#define PTHREADPOOL_USE_CONDVAR 0
	#else
		#define PTHREADPOOL_USE_CONDVAR 1
	#endif
#endif


/* Number of iterations in spin-wait loop before going into futex/condvar wait */
#define PTHREADPOOL_SPIN_WAIT_ITERATIONS 1000000

#define PTHREADPOOL_CACHELINE_SIZE 64
#if defined(__GNUC__)
	#define PTHREADPOOL_CACHELINE_ALIGNED __attribute__((__aligned__(PTHREADPOOL_CACHELINE_SIZE)))
#elif defined(_MSC_VER)
	#define PTHREADPOOL_CACHELINE_ALIGNED __declspec(align(PTHREADPOOL_CACHELINE_SIZE))
#else
	#error "Platform-specific implementation of PTHREADPOOL_CACHELINE_ALIGNED required"
#endif

#if defined(__clang__)
	#if __has_extension(c_static_assert) || __has_feature(c_static_assert)
		#define PTHREADPOOL_STATIC_ASSERT(predicate, message) _Static_assert((predicate), message)
	#else
		#define PTHREADPOOL_STATIC_ASSERT(predicate, message)
	#endif
#elif defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4) && (__GNUC_MINOR__ >= 6))
	/* Static assert is supported by gcc >= 4.6 */
	#define PTHREADPOOL_STATIC_ASSERT(predicate, message) _Static_assert((predicate), message)
#else
	#define PTHREADPOOL_STATIC_ASSERT(predicate, message)
#endif

#ifndef PTHREADPOOL_INTERNAL
	#if defined(__ELF__)
		#define PTHREADPOOL_INTERNAL __attribute__((__visibility__("internal")))
	#elif defined(__MACH__)
		#define PTHREADPOOL_INTERNAL __attribute__((__visibility__("hidden")))
	#else
		#define PTHREADPOOL_INTERNAL
	#endif
#endif
