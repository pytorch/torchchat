#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* SSE-specific headers */
#if defined(__i386__) || defined(__i686__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64) && !defined(_M_ARM64EC)
	#include <xmmintrin.h>
#endif

/* ARM-specific headers */
#if defined(__ARM_ACLE)
	#include <arm_acle.h>
#endif

/* MSVC-specific headers */
#ifdef _MSC_VER
	#include <intrin.h>
#endif


#if defined(__wasm__) && defined(__clang__)
	/*
	 * Clang for WebAssembly target lacks stdatomic.h header,
	 * even though it supports the necessary low-level intrinsics.
	 * Thus, we implement pthreadpool atomic functions on top of
	 * low-level Clang-specific interfaces for this target.
	 */

	typedef _Atomic(uint32_t) pthreadpool_atomic_uint32_t;
	typedef _Atomic(size_t)   pthreadpool_atomic_size_t;
	typedef _Atomic(void*)    pthreadpool_atomic_void_p;

	static inline uint32_t pthreadpool_load_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return __c11_atomic_load(address, __ATOMIC_RELAXED);
	}

	static inline size_t pthreadpool_load_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return __c11_atomic_load(address, __ATOMIC_RELAXED);
	}

	static inline void* pthreadpool_load_relaxed_void_p(
		pthreadpool_atomic_void_p* address)
	{
		return __c11_atomic_load(address, __ATOMIC_RELAXED);
	}

	static inline uint32_t pthreadpool_load_acquire_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return __c11_atomic_load(address, __ATOMIC_ACQUIRE);
	}

	static inline size_t pthreadpool_load_acquire_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return __c11_atomic_load(address, __ATOMIC_ACQUIRE);
	}

	static inline void pthreadpool_store_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		__c11_atomic_store(address, value, __ATOMIC_RELAXED);
	}

	static inline void pthreadpool_store_relaxed_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		__c11_atomic_store(address, value, __ATOMIC_RELAXED);
	}

	static inline void pthreadpool_store_relaxed_void_p(
		pthreadpool_atomic_void_p* address,
		void* value)
	{
		__c11_atomic_store(address, value, __ATOMIC_RELAXED);
	}

	static inline void pthreadpool_store_release_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		__c11_atomic_store(address, value, __ATOMIC_RELEASE);
	}

	static inline void pthreadpool_store_release_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		__c11_atomic_store(address, value, __ATOMIC_RELEASE);
	}

	static inline size_t pthreadpool_decrement_fetch_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return __c11_atomic_fetch_sub(address, 1, __ATOMIC_RELAXED) - 1;
	}

	static inline size_t pthreadpool_decrement_fetch_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return __c11_atomic_fetch_sub(address, 1, __ATOMIC_RELEASE) - 1;
	}

	static inline size_t pthreadpool_decrement_fetch_acquire_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return __c11_atomic_fetch_sub(address, 1, __ATOMIC_ACQ_REL) - 1;
	}

	static inline bool pthreadpool_try_decrement_relaxed_size_t(
		pthreadpool_atomic_size_t* value)
	{
		size_t actual_value = __c11_atomic_load(value, __ATOMIC_RELAXED);
		while (actual_value != 0) {
			if (__c11_atomic_compare_exchange_weak(
				value, &actual_value, actual_value - 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
			{
				return true;
			}
		}
		return false;
	}

	static inline void pthreadpool_fence_acquire() {
		__c11_atomic_thread_fence(__ATOMIC_ACQUIRE);
	}

	static inline void pthreadpool_fence_release() {
		__c11_atomic_thread_fence(__ATOMIC_RELEASE);
	}
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) && !defined(__STDC_NO_ATOMICS__)
	#include <stdatomic.h>

	typedef _Atomic(uint32_t) pthreadpool_atomic_uint32_t;
	typedef _Atomic(size_t)   pthreadpool_atomic_size_t;
	typedef _Atomic(void*)    pthreadpool_atomic_void_p;

	static inline uint32_t pthreadpool_load_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return atomic_load_explicit(address, memory_order_relaxed);
	}

	static inline size_t pthreadpool_load_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return atomic_load_explicit(address, memory_order_relaxed);
	}

	static inline void* pthreadpool_load_relaxed_void_p(
		pthreadpool_atomic_void_p* address)
	{
		return atomic_load_explicit(address, memory_order_relaxed);
	}

	static inline uint32_t pthreadpool_load_acquire_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return atomic_load_explicit(address, memory_order_acquire);
	}

	static inline size_t pthreadpool_load_acquire_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return atomic_load_explicit(address, memory_order_acquire);
	}

	static inline void pthreadpool_store_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		atomic_store_explicit(address, value, memory_order_relaxed);
	}

	static inline void pthreadpool_store_relaxed_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		atomic_store_explicit(address, value, memory_order_relaxed);
	}

	static inline void pthreadpool_store_relaxed_void_p(
		pthreadpool_atomic_void_p* address,
		void* value)
	{
		atomic_store_explicit(address, value, memory_order_relaxed);
	}

	static inline void pthreadpool_store_release_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		atomic_store_explicit(address, value, memory_order_release);
	}

	static inline void pthreadpool_store_release_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		atomic_store_explicit(address, value, memory_order_release);
	}

	static inline size_t pthreadpool_decrement_fetch_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return atomic_fetch_sub_explicit(address, 1, memory_order_relaxed) - 1;
	}

	static inline size_t pthreadpool_decrement_fetch_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return atomic_fetch_sub_explicit(address, 1, memory_order_release) - 1;
	}

	static inline size_t pthreadpool_decrement_fetch_acquire_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return atomic_fetch_sub_explicit(address, 1, memory_order_acq_rel) - 1;
	}

	static inline bool pthreadpool_try_decrement_relaxed_size_t(
		pthreadpool_atomic_size_t* value)
	{
		#if defined(__clang__) && (defined(__arm__) || defined(__aarch64__))
			size_t actual_value;
			do {
				actual_value = __builtin_arm_ldrex((const volatile size_t*) value);
				if (actual_value == 0) {
					__builtin_arm_clrex();
					return false;
				}
			} while (__builtin_arm_strex(actual_value - 1, (volatile size_t*) value) != 0);
			return true;
		#else
			size_t actual_value = pthreadpool_load_relaxed_size_t(value);
			while (actual_value != 0) {
				if (atomic_compare_exchange_weak_explicit(
					value, &actual_value, actual_value - 1, memory_order_relaxed, memory_order_relaxed))
				{
					return true;
				}
			}
			return false;
		#endif
	}

	static inline void pthreadpool_fence_acquire() {
		atomic_thread_fence(memory_order_acquire);
	}

	static inline void pthreadpool_fence_release() {
		atomic_thread_fence(memory_order_release);
	}
#elif defined(__GNUC__)
	typedef uint32_t volatile pthreadpool_atomic_uint32_t;
	typedef size_t volatile   pthreadpool_atomic_size_t;
	typedef void* volatile    pthreadpool_atomic_void_p;

	static inline uint32_t pthreadpool_load_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return *address;
	}

	static inline size_t pthreadpool_load_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return *address;
	}

	static inline void* pthreadpool_load_relaxed_void_p(
		pthreadpool_atomic_void_p* address)
	{
		return *address;
	}

	static inline uint32_t pthreadpool_load_acquire_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return *address;
	}

	static inline size_t pthreadpool_load_acquire_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return *address;
	}

	static inline void pthreadpool_store_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		*address = value;
	}

	static inline void pthreadpool_store_relaxed_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		*address = value;
	}

	static inline void pthreadpool_store_relaxed_void_p(
		pthreadpool_atomic_void_p* address,
		void* value)
	{
		*address = value;
	}

	static inline void pthreadpool_store_release_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		*address = value;
	}

	static inline void pthreadpool_store_release_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		*address = value;
	}

	static inline size_t pthreadpool_decrement_fetch_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return __sync_sub_and_fetch(address, 1);
	}

	static inline size_t pthreadpool_decrement_fetch_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return __sync_sub_and_fetch(address, 1);
	}

	static inline size_t pthreadpool_decrement_fetch_acquire_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return __sync_sub_and_fetch(address, 1);
	}

	static inline bool pthreadpool_try_decrement_relaxed_size_t(
		pthreadpool_atomic_size_t* value)
	{
		size_t actual_value = *value;
		while (actual_value != 0) {
			const size_t new_value = actual_value - 1;
			const size_t expected_value = actual_value;
			actual_value = __sync_val_compare_and_swap(value, expected_value, new_value);
			if (actual_value == expected_value) {
				return true;
			}
		}
		return false;
	}

	static inline void pthreadpool_fence_acquire() {
		__sync_synchronize();
	}

	static inline void pthreadpool_fence_release() {
		__sync_synchronize();
	}
#elif defined(_MSC_VER) && defined(_M_ARM)
	typedef volatile uint32_t pthreadpool_atomic_uint32_t;
	typedef volatile size_t   pthreadpool_atomic_size_t;
	typedef void *volatile    pthreadpool_atomic_void_p;

	static inline uint32_t pthreadpool_load_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return (uint32_t) __iso_volatile_load32((const volatile __int32*) address);
	}

	static inline size_t pthreadpool_load_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) __iso_volatile_load32((const volatile __int32*) address);
	}

	static inline void* pthreadpool_load_relaxed_void_p(
		pthreadpool_atomic_void_p* address)
	{
		return (void*) __iso_volatile_load32((const volatile __int32*) address);
	}

	static inline uint32_t pthreadpool_load_acquire_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		const uint32_t value = (uint32_t) __iso_volatile_load32((const volatile __int32*) address);
		__dmb(_ARM_BARRIER_ISH);
		_ReadBarrier();
		return value;
	}

	static inline size_t pthreadpool_load_acquire_size_t(
		pthreadpool_atomic_size_t* address)
	{
		const size_t value = (size_t) __iso_volatile_load32((const volatile __int32*) address);
		__dmb(_ARM_BARRIER_ISH);
		_ReadBarrier();
		return value;
	}

	static inline void pthreadpool_store_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		__iso_volatile_store32((volatile __int32*) address, (__int32) value);
	}

	static inline void pthreadpool_store_relaxed_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		__iso_volatile_store32((volatile __int32*) address, (__int32) value);
	}

	static inline void pthreadpool_store_relaxed_void_p(
		pthreadpool_atomic_void_p* address,
		void* value)
	{
		__iso_volatile_store32((volatile __int32*) address, (__int32) value);
	}

	static inline void pthreadpool_store_release_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		_WriteBarrier();
		__dmb(_ARM_BARRIER_ISH);
		__iso_volatile_store32((volatile __int32*) address, (__int32) value);
	}

	static inline void pthreadpool_store_release_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		_WriteBarrier();
		__dmb(_ARM_BARRIER_ISH);
		__iso_volatile_store32((volatile __int32*) address, (__int32) value);
	}

	static inline size_t pthreadpool_decrement_fetch_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement_nf((volatile long*) address);
	}

	static inline size_t pthreadpool_decrement_fetch_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement_rel((volatile long*) address);
	}

	static inline size_t pthreadpool_decrement_fetch_acquire_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement((volatile long*) address);
	}

	static inline bool pthreadpool_try_decrement_relaxed_size_t(
		pthreadpool_atomic_size_t* value)
	{
		size_t actual_value = (size_t) __iso_volatile_load32((const volatile __int32*) value);
		while (actual_value != 0) {
			const size_t new_value = actual_value - 1;
			const size_t expected_value = actual_value;
			actual_value = _InterlockedCompareExchange_nf(
				(volatile long*) value, (long) new_value, (long) expected_value);
			if (actual_value == expected_value) {
				return true;
			}
		}
		return false;
	}

	static inline void pthreadpool_fence_acquire() {
		__dmb(_ARM_BARRIER_ISH);
		_ReadBarrier();
	}

	static inline void pthreadpool_fence_release() {
		_WriteBarrier();
		__dmb(_ARM_BARRIER_ISH);
	}
#elif defined(_MSC_VER) && defined(_M_ARM64)
	typedef volatile uint32_t pthreadpool_atomic_uint32_t;
	typedef volatile size_t   pthreadpool_atomic_size_t;
	typedef void *volatile    pthreadpool_atomic_void_p;

	static inline uint32_t pthreadpool_load_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return (uint32_t) __iso_volatile_load32((const volatile __int32*) address);
	}

	static inline size_t pthreadpool_load_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) __iso_volatile_load64((const volatile __int64*) address);
	}

	static inline void* pthreadpool_load_relaxed_void_p(
		pthreadpool_atomic_void_p* address)
	{
		return (void*) __iso_volatile_load64((const volatile __int64*) address);
	}

	static inline uint32_t pthreadpool_load_acquire_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return (uint32_t) __ldar32((volatile unsigned __int32*) address);
	}

	static inline size_t pthreadpool_load_acquire_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) __ldar64((volatile unsigned __int64*) address);
	}

	static inline void pthreadpool_store_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		__iso_volatile_store32((volatile __int32*) address, (__int32) value);
	}

	static inline void pthreadpool_store_relaxed_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		__iso_volatile_store64((volatile __int64*) address, (__int64) value);
	}

	static inline void pthreadpool_store_relaxed_void_p(
		pthreadpool_atomic_void_p* address,
		void* value)
	{
		__iso_volatile_store64((volatile __int64*) address, (__int64) value);
	}

	static inline void pthreadpool_store_release_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		_WriteBarrier();
		__stlr32((unsigned __int32 volatile*) address, (unsigned __int32) value);
	}

	static inline void pthreadpool_store_release_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		_WriteBarrier();
		__stlr64((unsigned __int64 volatile*) address, (unsigned __int64) value);
	}

	static inline size_t pthreadpool_decrement_fetch_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement64_nf((volatile __int64*) address);
	}

	static inline size_t pthreadpool_decrement_fetch_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement64_rel((volatile __int64*) address);
	}

	static inline size_t pthreadpool_decrement_fetch_acquire_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement64((volatile __int64*) address);
	}

	static inline bool pthreadpool_try_decrement_relaxed_size_t(
		pthreadpool_atomic_size_t* value)
	{
		size_t actual_value = (size_t) __iso_volatile_load64((const volatile __int64*) value);
		while (actual_value != 0) {
			const size_t new_value = actual_value - 1;
			const size_t expected_value = actual_value;
			actual_value = _InterlockedCompareExchange64_nf(
				(volatile __int64*) value, (__int64) new_value, (__int64) expected_value);
			if (actual_value == expected_value) {
				return true;
			}
		}
		return false;
	}

	static inline void pthreadpool_fence_acquire() {
		__dmb(_ARM64_BARRIER_ISHLD);
		_ReadBarrier();
	}

	static inline void pthreadpool_fence_release() {
		_WriteBarrier();
		__dmb(_ARM64_BARRIER_ISH);
	}
#elif defined(_MSC_VER) && defined(_M_IX86)
	typedef volatile uint32_t pthreadpool_atomic_uint32_t;
	typedef volatile size_t   pthreadpool_atomic_size_t;
	typedef void *volatile    pthreadpool_atomic_void_p;

	static inline uint32_t pthreadpool_load_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return *address;
	}

	static inline size_t pthreadpool_load_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return *address;
	}

	static inline void* pthreadpool_load_relaxed_void_p(
		pthreadpool_atomic_void_p* address)
	{
		return *address;
	}

	static inline uint32_t pthreadpool_load_acquire_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		/* x86 loads always have acquire semantics; use only a compiler barrier */
		const uint32_t value = *address;
		_ReadBarrier();
		return value;
	}

	static inline size_t pthreadpool_load_acquire_size_t(
		pthreadpool_atomic_size_t* address)
	{
		/* x86 loads always have acquire semantics; use only a compiler barrier */
		const size_t value = *address;
		_ReadBarrier();
		return value;
	}

	static inline void pthreadpool_store_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		*address = value;
	}

	static inline void pthreadpool_store_relaxed_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		*address = value;
	}

	static inline void pthreadpool_store_relaxed_void_p(
		pthreadpool_atomic_void_p* address,
		void* value)
	{
		*address = value;
	}

	static inline void pthreadpool_store_release_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		/* x86 stores always have release semantics; use only a compiler barrier */
		_WriteBarrier();
		*address = value;
	}

	static inline void pthreadpool_store_release_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		/* x86 stores always have release semantics; use only a compiler barrier */
		_WriteBarrier();
		*address = value;
	}

	static inline size_t pthreadpool_decrement_fetch_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement((volatile long*) address);
	}

	static inline size_t pthreadpool_decrement_fetch_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement((volatile long*) address);
	}

	static inline size_t pthreadpool_decrement_fetch_acquire_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement((volatile long*) address);
	}

	static inline bool pthreadpool_try_decrement_relaxed_size_t(
		pthreadpool_atomic_size_t* value)
	{
		size_t actual_value = *value;
		while (actual_value != 0) {
			const size_t new_value = actual_value - 1;
			const size_t expected_value = actual_value;
			actual_value = _InterlockedCompareExchange(
				(volatile long*) value, (long) new_value, (long) expected_value);
			if (actual_value == expected_value) {
				return true;
			}
		}
		return false;
	}

	static inline void pthreadpool_fence_acquire() {
		_mm_lfence();
	}

	static inline void pthreadpool_fence_release() {
		_mm_sfence();
	}
#elif defined(_MSC_VER) && defined(_M_X64)
	typedef volatile uint32_t pthreadpool_atomic_uint32_t;
	typedef volatile size_t   pthreadpool_atomic_size_t;
	typedef void *volatile    pthreadpool_atomic_void_p;

	static inline uint32_t pthreadpool_load_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		return *address;
	}

	static inline size_t pthreadpool_load_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return *address;
	}

	static inline void* pthreadpool_load_relaxed_void_p(
		pthreadpool_atomic_void_p* address)
	{
		return *address;
	}

	static inline uint32_t pthreadpool_load_acquire_uint32_t(
		pthreadpool_atomic_uint32_t* address)
	{
		/* x86-64 loads always have acquire semantics; use only a compiler barrier */
		const uint32_t value = *address;
		_ReadBarrier();
		return value;
	}

	static inline size_t pthreadpool_load_acquire_size_t(
		pthreadpool_atomic_size_t* address)
	{
		/* x86-64 loads always have acquire semantics; use only a compiler barrier */
		const size_t value = *address;
		_ReadBarrier();
		return value;
	}

	static inline void pthreadpool_store_relaxed_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		*address = value;
	}

	static inline void pthreadpool_store_relaxed_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		*address = value;
	}

	static inline void pthreadpool_store_relaxed_void_p(
		pthreadpool_atomic_void_p* address,
		void* value)
	{
		*address = value;
	}

	static inline void pthreadpool_store_release_uint32_t(
		pthreadpool_atomic_uint32_t* address,
		uint32_t value)
	{
		/* x86-64 stores always have release semantics; use only a compiler barrier */
		_WriteBarrier();
		*address = value;
	}

	static inline void pthreadpool_store_release_size_t(
		pthreadpool_atomic_size_t* address,
		size_t value)
	{
		/* x86-64 stores always have release semantics; use only a compiler barrier */
		_WriteBarrier();
		*address = value;
	}

	static inline size_t pthreadpool_decrement_fetch_relaxed_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement64((volatile __int64*) address);
	}

	static inline size_t pthreadpool_decrement_fetch_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement64((volatile __int64*) address);
	}

	static inline size_t pthreadpool_decrement_fetch_acquire_release_size_t(
		pthreadpool_atomic_size_t* address)
	{
		return (size_t) _InterlockedDecrement64((volatile __int64*) address);
	}

	static inline bool pthreadpool_try_decrement_relaxed_size_t(
		pthreadpool_atomic_size_t* value)
	{
		size_t actual_value = *value;
		while (actual_value != 0) {
			const size_t new_value = actual_value - 1;
			const size_t expected_value = actual_value;
			actual_value = _InterlockedCompareExchange64(
				(volatile __int64*) value, (__int64) new_value, (__int64) expected_value);
			if (actual_value == expected_value) {
				return true;
			}
		}
		return false;
	}

	static inline void pthreadpool_fence_acquire() {
		_mm_lfence();
		_ReadBarrier();
	}

	static inline void pthreadpool_fence_release() {
		_WriteBarrier();
		_mm_sfence();
	}
#else
	#error "Platform-specific implementation of threadpool-atomics.h required"
#endif

#if defined(__ARM_ACLE) || defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64) || defined(_M_ARM64EC))
	static inline void pthreadpool_yield() {
		__yield();
	}
#elif defined(__GNUC__) && (defined(__ARM_ARCH) && (__ARM_ARCH >= 7) || (defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6KZ__)) && !defined(__thumb__))
	static inline void pthreadpool_yield() {
		__asm__ __volatile__("yield");
	}
#elif defined(__i386__) || defined(__i686__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
	static inline void pthreadpool_yield() {
		_mm_pause();
	}
#else
	static inline void pthreadpool_yield() {
		pthreadpool_fence_acquire();
	}
#endif
