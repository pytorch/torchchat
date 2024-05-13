#include <gtest/gtest.h>

#include <pthreadpool.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>


typedef std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_pthreadpool_t;


const size_t kParallelize1DRange = 1223;
const size_t kParallelize1DTile1DRange = 1303;
const size_t kParallelize1DTile1DTile = 11;
const size_t kParallelize2DRangeI = 41;
const size_t kParallelize2DRangeJ = 43;
const size_t kParallelize2DTile1DRangeI = 43;
const size_t kParallelize2DTile1DRangeJ = 53;
const size_t kParallelize2DTile1DTileJ = 5;
const size_t kParallelize2DTile2DRangeI = 53;
const size_t kParallelize2DTile2DRangeJ = 59;
const size_t kParallelize2DTile2DTileI = 5;
const size_t kParallelize2DTile2DTileJ = 7;
const size_t kParallelize3DRangeI = 13;
const size_t kParallelize3DRangeJ = 17;
const size_t kParallelize3DRangeK = 19;
const size_t kParallelize3DTile1DRangeI = 17;
const size_t kParallelize3DTile1DRangeJ = 19;
const size_t kParallelize3DTile1DRangeK = 23;
const size_t kParallelize3DTile1DTileK = 5;
const size_t kParallelize3DTile2DRangeI = 19;
const size_t kParallelize3DTile2DRangeJ = 23;
const size_t kParallelize3DTile2DRangeK = 29;
const size_t kParallelize3DTile2DTileJ = 2;
const size_t kParallelize3DTile2DTileK = 3;
const size_t kParallelize4DRangeI = 11;
const size_t kParallelize4DRangeJ = 13;
const size_t kParallelize4DRangeK = 17;
const size_t kParallelize4DRangeL = 19;
const size_t kParallelize4DTile1DRangeI = 13;
const size_t kParallelize4DTile1DRangeJ = 17;
const size_t kParallelize4DTile1DRangeK = 19;
const size_t kParallelize4DTile1DRangeL = 23;
const size_t kParallelize4DTile1DTileL = 5;
const size_t kParallelize4DTile2DRangeI = 17;
const size_t kParallelize4DTile2DRangeJ = 19;
const size_t kParallelize4DTile2DRangeK = 23;
const size_t kParallelize4DTile2DRangeL = 29;
const size_t kParallelize4DTile2DTileK = 2;
const size_t kParallelize4DTile2DTileL = 3;
const size_t kParallelize5DRangeI = 7;
const size_t kParallelize5DRangeJ = 11;
const size_t kParallelize5DRangeK = 13;
const size_t kParallelize5DRangeL = 17;
const size_t kParallelize5DRangeM = 19;
const size_t kParallelize5DTile1DRangeI = 11;
const size_t kParallelize5DTile1DRangeJ = 13;
const size_t kParallelize5DTile1DRangeK = 17;
const size_t kParallelize5DTile1DRangeL = 19;
const size_t kParallelize5DTile1DRangeM = 23;
const size_t kParallelize5DTile1DTileM = 5;
const size_t kParallelize5DTile2DRangeI = 13;
const size_t kParallelize5DTile2DRangeJ = 17;
const size_t kParallelize5DTile2DRangeK = 19;
const size_t kParallelize5DTile2DRangeL = 23;
const size_t kParallelize5DTile2DRangeM = 29;
const size_t kParallelize5DTile2DTileL = 3;
const size_t kParallelize5DTile2DTileM = 2;
const size_t kParallelize6DRangeI = 3;
const size_t kParallelize6DRangeJ = 5;
const size_t kParallelize6DRangeK = 7;
const size_t kParallelize6DRangeL = 11;
const size_t kParallelize6DRangeM = 13;
const size_t kParallelize6DRangeN = 17;
const size_t kParallelize6DTile1DRangeI = 5;
const size_t kParallelize6DTile1DRangeJ = 7;
const size_t kParallelize6DTile1DRangeK = 11;
const size_t kParallelize6DTile1DRangeL = 13;
const size_t kParallelize6DTile1DRangeM = 17;
const size_t kParallelize6DTile1DRangeN = 19;
const size_t kParallelize6DTile1DTileN = 5;
const size_t kParallelize6DTile2DRangeI = 7;
const size_t kParallelize6DTile2DRangeJ = 11;
const size_t kParallelize6DTile2DRangeK = 13;
const size_t kParallelize6DTile2DRangeL = 17;
const size_t kParallelize6DTile2DRangeM = 19;
const size_t kParallelize6DTile2DRangeN = 23;
const size_t kParallelize6DTile2DTileM = 3;
const size_t kParallelize6DTile2DTileN = 2;

const size_t kIncrementIterations = 101;
const size_t kIncrementIterations5D = 7;
const size_t kIncrementIterations6D = 3;

const uint32_t kMaxUArchIndex = 0;
const uint32_t kDefaultUArchIndex = 42;


TEST(CreateAndDestroy, NullThreadPool) {
	pthreadpool* threadpool = nullptr;
	pthreadpool_destroy(threadpool);
}

TEST(CreateAndDestroy, SingleThreadPool) {
	pthreadpool* threadpool = pthreadpool_create(1);
	ASSERT_TRUE(threadpool);
	pthreadpool_destroy(threadpool);
}

TEST(CreateAndDestroy, MultiThreadPool) {
	pthreadpool* threadpool = pthreadpool_create(0);
	ASSERT_TRUE(threadpool);
	pthreadpool_destroy(threadpool);
}

static void ComputeNothing1D(void*, size_t) {
}

TEST(Parallelize1D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d(threadpool.get(),
		ComputeNothing1D,
		nullptr,
		kParallelize1DRange,
		0 /* flags */);
}

TEST(Parallelize1D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d(
		threadpool.get(),
		ComputeNothing1D,
		nullptr,
		kParallelize1DRange,
		0 /* flags */);
}

static void CheckBounds1D(void*, size_t i) {
	EXPECT_LT(i, kParallelize1DRange);
}

TEST(Parallelize1D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d(
		threadpool.get(),
		CheckBounds1D,
		nullptr,
		kParallelize1DRange,
		0 /* flags */);
}

TEST(Parallelize1D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d(
		threadpool.get(),
		CheckBounds1D,
		nullptr,
		kParallelize1DRange,
		0 /* flags */);
}

static void SetTrue1D(std::atomic_bool* processed_indicators, size_t i) {
	processed_indicators[i].store(true, std::memory_order_relaxed);
}

TEST(Parallelize1D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_t>(SetTrue1D),
		static_cast<void*>(indicators.data()),
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_TRUE(indicators[i].load(std::memory_order_relaxed))
			<< "Element " << i << " not processed";
	}
}

TEST(Parallelize1D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_t>(SetTrue1D),
		static_cast<void*>(indicators.data()),
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_TRUE(indicators[i].load(std::memory_order_relaxed))
			<< "Element " << i << " not processed";
	}
}

static void Increment1D(std::atomic_int* processed_counters, size_t i) {
	processed_counters[i].fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize1D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_t>(Increment1D),
		static_cast<void*>(counters.data()),
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), 1)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times (expected: 1)";
	}
}

TEST(Parallelize1D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_t>(Increment1D),
		static_cast<void*>(counters.data()),
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), 1)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times (expected: 1)";
	}
}

TEST(Parallelize1D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_1d_t>(Increment1D),
			static_cast<void*>(counters.data()),
			kParallelize1DRange,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), kIncrementIterations)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times "
			<< "(expected: " << kIncrementIterations << ")";
	}
}

TEST(Parallelize1D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_1d_t>(Increment1D),
			static_cast<void*>(counters.data()),
			kParallelize1DRange,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), kIncrementIterations)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times "
			<< "(expected: " << kIncrementIterations << ")";
	}
}

static void IncrementSame1D(std::atomic_int* num_processed_items, size_t i) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize1D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_t>(IncrementSame1D),
		static_cast<void*>(&num_processed_items),
		kParallelize1DRange,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize1DRange);
}

static void WorkImbalance1D(std::atomic_int* num_processed_items, size_t i) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
	if (i == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize1DRange) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize1D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_t>(WorkImbalance1D),
		static_cast<void*>(&num_processed_items),
		kParallelize1DRange,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize1DRange);
}

static void ComputeNothing1DWithThread(void*, size_t, size_t) {
}

TEST(Parallelize1DWithThread, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_with_thread(threadpool.get(),
		ComputeNothing1DWithThread,
		nullptr,
		kParallelize1DRange,
		0 /* flags */);
}

TEST(Parallelize1DWithThread, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_thread(
		threadpool.get(),
		ComputeNothing1DWithThread,
		nullptr,
		kParallelize1DRange,
		0 /* flags */);
}

static void CheckBounds1DWithThread(void*, size_t, size_t i) {
	EXPECT_LT(i, kParallelize1DRange);
}

TEST(Parallelize1DWithThread, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_with_thread(
		threadpool.get(),
		CheckBounds1DWithThread,
		nullptr,
		kParallelize1DRange,
		0 /* flags */);
}

TEST(Parallelize1DWithThread, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_thread(
		threadpool.get(),
		CheckBounds1DWithThread,
		nullptr,
		kParallelize1DRange,
		0 /* flags */);
}

static void SetTrue1DWithThread(std::atomic_bool* processed_indicators, size_t, size_t i) {
	processed_indicators[i].store(true, std::memory_order_relaxed);
}

TEST(Parallelize1DWithThread, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_thread_t>(SetTrue1DWithThread),
		static_cast<void*>(indicators.data()),
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_TRUE(indicators[i].load(std::memory_order_relaxed))
			<< "Element " << i << " not processed";
	}
}

TEST(Parallelize1DWithThread, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_thread_t>(SetTrue1DWithThread),
		static_cast<void*>(indicators.data()),
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_TRUE(indicators[i].load(std::memory_order_relaxed))
			<< "Element " << i << " not processed";
	}
}

static void Increment1DWithThread(std::atomic_int* processed_counters, size_t, size_t i) {
	processed_counters[i].fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize1DWithThread, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_thread_t>(Increment1DWithThread),
		static_cast<void*>(counters.data()),
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), 1)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times (expected: 1)";
	}
}

TEST(Parallelize1DWithThread, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_thread_t>(Increment1DWithThread),
		static_cast<void*>(counters.data()),
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), 1)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times (expected: 1)";
	}
}

TEST(Parallelize1DWithThread, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_1d_with_thread(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_1d_with_thread_t>(Increment1DWithThread),
			static_cast<void*>(counters.data()),
			kParallelize1DRange,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), kIncrementIterations)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times "
			<< "(expected: " << kIncrementIterations << ")";
	}
}

TEST(Parallelize1DWithThread, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_1d_with_thread(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_1d_with_thread_t>(Increment1DWithThread),
			static_cast<void*>(counters.data()),
			kParallelize1DRange,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), kIncrementIterations)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times "
			<< "(expected: " << kIncrementIterations << ")";
	}
}

static void IncrementSame1DWithThread(std::atomic_int* num_processed_items, size_t, size_t i) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize1DWithThread, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_thread_t>(IncrementSame1DWithThread),
		static_cast<void*>(&num_processed_items),
		kParallelize1DRange,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize1DRange);
}

static void WorkImbalance1DWithThread(std::atomic_int* num_processed_items, size_t, size_t i) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
	if (i == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize1DRange) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize1DWithThread, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_thread_t>(WorkImbalance1DWithThread),
		static_cast<void*>(&num_processed_items),
		kParallelize1DRange,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize1DRange);
}

static void CheckThreadIndexValid1DWithThread(const size_t* num_threads, size_t thread_index, size_t) {
	EXPECT_LE(thread_index, *num_threads);
}

TEST(Parallelize1DWithThread, MultiThreadPoolThreadIndexValid) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	size_t num_threads = pthreadpool_get_threads_count(threadpool.get());
	if (num_threads <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_thread_t>(CheckThreadIndexValid1DWithThread),
		static_cast<void*>(&num_threads),
		kParallelize1DRange,
		0 /* flags */);
}

static void ComputeNothing1DWithUArch(void*, uint32_t, size_t) {
}

TEST(Parallelize1DWithUArch, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_with_uarch(threadpool.get(),
		ComputeNothing1DWithUArch,
		nullptr,
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);
}

TEST(Parallelize1DWithUArch, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_uarch(
		threadpool.get(),
		ComputeNothing1DWithUArch,
		nullptr,
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);
}

static void CheckUArch1DWithUArch(void*, uint32_t uarch_index, size_t) {
	if (uarch_index != kDefaultUArchIndex) {
		EXPECT_LE(uarch_index, kMaxUArchIndex);
	}
}

TEST(Parallelize1DWithUArch, SingleThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_with_uarch(threadpool.get(),
		CheckUArch1DWithUArch,
		nullptr,
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);
}

TEST(Parallelize1DWithUArch, MultiThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_uarch(
		threadpool.get(),
		CheckUArch1DWithUArch,
		nullptr,
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);
}

static void CheckBounds1DWithUArch(void*, uint32_t, size_t i) {
	EXPECT_LT(i, kParallelize1DRange);
}

TEST(Parallelize1DWithUArch, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_with_uarch(
		threadpool.get(),
		CheckBounds1DWithUArch,
		nullptr,
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);
}

TEST(Parallelize1DWithUArch, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_uarch(
		threadpool.get(),
		CheckBounds1DWithUArch,
		nullptr,
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);
}

static void SetTrue1DWithUArch(std::atomic_bool* processed_indicators, uint32_t, size_t i) {
	processed_indicators[i].store(true, std::memory_order_relaxed);
}

TEST(Parallelize1DWithUArch, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_id_t>(SetTrue1DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_TRUE(indicators[i].load(std::memory_order_relaxed))
			<< "Element " << i << " not processed";
	}
}

TEST(Parallelize1DWithUArch, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_id_t>(SetTrue1DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_TRUE(indicators[i].load(std::memory_order_relaxed))
			<< "Element " << i << " not processed";
	}
}

static void Increment1DWithUArch(std::atomic_int* processed_counters, uint32_t, size_t i) {
	processed_counters[i].fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize1DWithUArch, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_id_t>(Increment1DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), 1)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times (expected: 1)";
	}
}

TEST(Parallelize1DWithUArch, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_id_t>(Increment1DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), 1)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times (expected: 1)";
	}
}

TEST(Parallelize1DWithUArch, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_1d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_1d_with_id_t>(Increment1DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex,
			kMaxUArchIndex,
			kParallelize1DRange,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), kIncrementIterations)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times "
			<< "(expected: " << kIncrementIterations << ")";
	}
}

TEST(Parallelize1DWithUArch, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_1d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_1d_with_id_t>(Increment1DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex,
			kMaxUArchIndex,
			kParallelize1DRange,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), kIncrementIterations)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times "
			<< "(expected: " << kIncrementIterations << ")";
	}
}

static void IncrementSame1DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t i) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize1DWithUArch, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_id_t>(IncrementSame1DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize1DRange);
}

static void WorkImbalance1DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t i) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
	if (i == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize1DRange) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize1DWithUArch, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_with_id_t>(WorkImbalance1DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex,
		kMaxUArchIndex,
		kParallelize1DRange,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize1DRange);
}

static void ComputeNothing1DTile1D(void*, size_t, size_t) {
}

TEST(Parallelize1DTile1D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_tile_1d(threadpool.get(),
		ComputeNothing1DTile1D,
		nullptr,
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);
}

TEST(Parallelize1DTile1D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		ComputeNothing1DTile1D,
		nullptr,
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);
}

static void CheckBounds1DTile1D(void*, size_t start_i, size_t tile_i) {
	EXPECT_LT(start_i, kParallelize1DTile1DRange);
	EXPECT_LE(start_i + tile_i, kParallelize1DTile1DRange);
}

TEST(Parallelize1DTile1D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		CheckBounds1DTile1D,
		nullptr,
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);
}

TEST(Parallelize1DTile1D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		CheckBounds1DTile1D,
		nullptr,
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);
}

static void CheckTiling1DTile1D(void*, size_t start_i, size_t tile_i) {
	EXPECT_GT(tile_i, 0);
	EXPECT_LE(tile_i, kParallelize1DTile1DTile);
	EXPECT_EQ(start_i % kParallelize1DTile1DTile, 0);
	EXPECT_EQ(tile_i, std::min<size_t>(kParallelize1DTile1DTile, kParallelize1DTile1DRange - start_i));
}

TEST(Parallelize1DTile1D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		CheckTiling1DTile1D,
		nullptr,
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);
}

TEST(Parallelize1DTile1D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		CheckTiling1DTile1D,
		nullptr,
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);
}

static void SetTrue1DTile1D(std::atomic_bool* processed_indicators, size_t start_i, size_t tile_i) {
	for (size_t i = start_i; i < start_i + tile_i; i++) {
		processed_indicators[i].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize1DTile1D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize1DTile1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(SetTrue1DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DTile1DRange; i++) {
		EXPECT_TRUE(indicators[i].load(std::memory_order_relaxed))
			<< "Element " << i << " not processed";
	}
}

TEST(Parallelize1DTile1D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize1DTile1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(SetTrue1DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DTile1DRange; i++) {
		EXPECT_TRUE(indicators[i].load(std::memory_order_relaxed))
			<< "Element " << i << " not processed";
	}
}

static void Increment1DTile1D(std::atomic_int* processed_counters, size_t start_i, size_t tile_i) {
	for (size_t i = start_i; i < start_i + tile_i; i++) {
		processed_counters[i].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize1DTile1D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize1DTile1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(Increment1DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DTile1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), 1)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times (expected: 1)";
	}
}

TEST(Parallelize1DTile1D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize1DTile1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(Increment1DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize1DTile1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), 1)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times (expected: 1)";
	}
}

TEST(Parallelize1DTile1D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize1DTile1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_1d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(Increment1DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize1DTile1DRange, kParallelize1DTile1DTile,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize1DTile1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), kIncrementIterations)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times "
			<< "(expected: " << kIncrementIterations << ")";
	}
}

TEST(Parallelize1DTile1D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize1DTile1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_1d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(Increment1DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize1DTile1DRange, kParallelize1DTile1DTile,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize1DTile1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), kIncrementIterations)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times "
			<< "(expected: " << kIncrementIterations << ")";
	}
}

static void IncrementSame1DTile1D(std::atomic_int* num_processed_items, size_t start_i, size_t tile_i) {
	for (size_t i = start_i; i < start_i + tile_i; i++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize1DTile1D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(IncrementSame1DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize1DTile1DRange);
}

static void WorkImbalance1DTile1D(std::atomic_int* num_processed_items, size_t start_i, size_t tile_i) {
	num_processed_items->fetch_add(tile_i, std::memory_order_relaxed);
	if (start_i == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize1DTile1DRange) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize1DTile1D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(WorkImbalance1DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize1DTile1DRange, kParallelize1DTile1DTile,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize1DTile1DRange);
}

static void ComputeNothing2D(void*, size_t, size_t) {
}

TEST(Parallelize2D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d(threadpool.get(),
		ComputeNothing2D,
		nullptr,
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
}

TEST(Parallelize2D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d(
		threadpool.get(),
		ComputeNothing2D,
		nullptr,
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
}

static void CheckBounds2D(void*, size_t i, size_t j) {
	EXPECT_LT(i, kParallelize2DRangeI);
	EXPECT_LT(j, kParallelize2DRangeJ);
}

TEST(Parallelize2D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d(
		threadpool.get(),
		CheckBounds2D,
		nullptr,
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
}

TEST(Parallelize2D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d(
		threadpool.get(),
		CheckBounds2D,
		nullptr,
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
}

static void SetTrue2D(std::atomic_bool* processed_indicators, size_t i, size_t j) {
	const size_t linear_idx = i * kParallelize2DRangeJ + j;
	processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
}

TEST(Parallelize2D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_t>(SetTrue2D),
		static_cast<void*>(indicators.data()),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

TEST(Parallelize2D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_t>(SetTrue2D),
		static_cast<void*>(indicators.data()),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

static void Increment2D(std::atomic_int* processed_counters, size_t i, size_t j) {
	const size_t linear_idx = i * kParallelize2DRangeJ + j;
	processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize2D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_t>(Increment2D),
		static_cast<void*>(counters.data()),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_t>(Increment2D),
		static_cast<void*>(counters.data()),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_t>(Increment2D),
			static_cast<void*>(counters.data()),
			kParallelize2DRangeI, kParallelize2DRangeJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

TEST(Parallelize2D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_t>(Increment2D),
			static_cast<void*>(counters.data()),
			kParallelize2DRangeI, kParallelize2DRangeJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

static void IncrementSame2D(std::atomic_int* num_processed_items, size_t i, size_t j) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize2D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_t>(IncrementSame2D),
		static_cast<void*>(&num_processed_items),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DRangeI * kParallelize2DRangeJ);
}

static void WorkImbalance2D(std::atomic_int* num_processed_items, size_t i, size_t j) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
	if (i == 0 && j == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize2DRangeI * kParallelize2DRangeJ) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize2D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_t>(WorkImbalance2D),
		static_cast<void*>(&num_processed_items),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DRangeI * kParallelize2DRangeJ);
}

static void ComputeNothing2DWithThread(void*, size_t, size_t, size_t) {
}

TEST(Parallelize2DWithThread, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_with_thread(threadpool.get(),
		ComputeNothing2DWithThread,
		nullptr,
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
}

TEST(Parallelize2DWithThread, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_with_thread(
		threadpool.get(),
		ComputeNothing2DWithThread,
		nullptr,
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
}

static void CheckBounds2DWithThread(void*, size_t, size_t i, size_t j) {
	EXPECT_LT(i, kParallelize2DRangeI);
	EXPECT_LT(j, kParallelize2DRangeJ);
}

TEST(Parallelize2DWithThread, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_with_thread(
		threadpool.get(),
		CheckBounds2DWithThread,
		nullptr,
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
}

TEST(Parallelize2DWithThread, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_with_thread(
		threadpool.get(),
		CheckBounds2DWithThread,
		nullptr,
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
}

static void SetTrue2DWithThread(std::atomic_bool* processed_indicators, size_t, size_t i, size_t j) {
	const size_t linear_idx = i * kParallelize2DRangeJ + j;
	processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
}

TEST(Parallelize2DWithThread, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_with_thread_t>(SetTrue2DWithThread),
		static_cast<void*>(indicators.data()),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

TEST(Parallelize2DWithThread, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_with_thread_t>(SetTrue2DWithThread),
		static_cast<void*>(indicators.data()),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

static void Increment2DWithThread(std::atomic_int* processed_counters, size_t, size_t i, size_t j) {
	const size_t linear_idx = i * kParallelize2DRangeJ + j;
	processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize2DWithThread, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_with_thread_t>(Increment2DWithThread),
		static_cast<void*>(counters.data()),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DWithThread, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_with_thread_t>(Increment2DWithThread),
		static_cast<void*>(counters.data()),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DWithThread, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_with_thread(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_with_thread_t>(Increment2DWithThread),
			static_cast<void*>(counters.data()),
			kParallelize2DRangeI, kParallelize2DRangeJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

TEST(Parallelize2DWithThread, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_with_thread(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_with_thread_t>(Increment2DWithThread),
			static_cast<void*>(counters.data()),
			kParallelize2DRangeI, kParallelize2DRangeJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

static void IncrementSame2DWithThread(std::atomic_int* num_processed_items, size_t, size_t i, size_t j) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize2DWithThread, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_with_thread_t>(IncrementSame2DWithThread),
		static_cast<void*>(&num_processed_items),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DRangeI * kParallelize2DRangeJ);
}

static void WorkImbalance2DWithThread(std::atomic_int* num_processed_items, size_t, size_t i, size_t j) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
	if (i == 0 && j == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize2DRangeI * kParallelize2DRangeJ) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize2DWithThread, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_with_thread_t>(WorkImbalance2DWithThread),
		static_cast<void*>(&num_processed_items),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DRangeI * kParallelize2DRangeJ);
}

static void CheckThreadIndexValid2DWithThread(const size_t* num_threads, size_t thread_index, size_t, size_t) {
	EXPECT_LE(thread_index, *num_threads);
}

TEST(Parallelize2DWithThread, MultiThreadPoolThreadIndexValid) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	size_t num_threads = pthreadpool_get_threads_count(threadpool.get());
	if (num_threads <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_with_thread_t>(CheckThreadIndexValid2DWithThread),
		static_cast<void*>(&num_threads),
		kParallelize2DRangeI, kParallelize2DRangeJ,
		0 /* flags */);
}

static void ComputeNothing2DTile1D(void*, size_t, size_t, size_t) {
}

TEST(Parallelize2DTile1D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d(threadpool.get(),
		ComputeNothing2DTile1D,
		nullptr,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		ComputeNothing2DTile1D,
		nullptr,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void CheckBounds2DTile1D(void*, size_t i, size_t start_j, size_t tile_j) {
	EXPECT_LT(i, kParallelize2DTile1DRangeI);
	EXPECT_LT(start_j, kParallelize2DTile1DRangeJ);
	EXPECT_LE(start_j + tile_j, kParallelize2DTile1DRangeJ);
}

TEST(Parallelize2DTile1D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		CheckBounds2DTile1D,
		nullptr,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		CheckBounds2DTile1D,
		nullptr,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void CheckTiling2DTile1D(void*, size_t i, size_t start_j, size_t tile_j) {
	EXPECT_GT(tile_j, 0);
	EXPECT_LE(tile_j, kParallelize2DTile1DTileJ);
	EXPECT_EQ(start_j % kParallelize2DTile1DTileJ, 0);
	EXPECT_EQ(tile_j, std::min<size_t>(kParallelize2DTile1DTileJ, kParallelize2DTile1DRangeJ - start_j));
}

TEST(Parallelize2DTile1D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		CheckTiling2DTile1D,
		nullptr,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		CheckTiling2DTile1D,
		nullptr,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void SetTrue2DTile1D(std::atomic_bool* processed_indicators, size_t i, size_t start_j, size_t tile_j) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
		processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize2DTile1D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_t>(SetTrue2DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

TEST(Parallelize2DTile1D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_t>(SetTrue2DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

static void Increment2DTile1D(std::atomic_int* processed_counters, size_t i, size_t start_j, size_t tile_j) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
		processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize2DTile1D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_t>(Increment2DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile1D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_t>(Increment2DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile1D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_tile_1d_t>(Increment2DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

TEST(Parallelize2DTile1D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_tile_1d_t>(Increment2DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

static void IncrementSame2DTile1D(std::atomic_int* num_processed_items, size_t i, size_t start_j, size_t tile_j) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize2DTile1D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_t>(IncrementSame2DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);
}

static void WorkImbalance2DTile1D(std::atomic_int* num_processed_items, size_t i, size_t start_j, size_t tile_j) {
	num_processed_items->fetch_add(tile_j, std::memory_order_relaxed);
	if (i == 0 && start_j == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize2DTile1D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_t>(WorkImbalance2DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);
}

static void ComputeNothing2DTile1DWithUArch(void*, uint32_t, size_t, size_t, size_t) {
}

TEST(Parallelize2DTile1DWithUArch, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch(threadpool.get(),
		ComputeNothing2DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1DWithUArch, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		ComputeNothing2DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void CheckUArch2DTile1DWithUArch(void*, uint32_t uarch_index, size_t, size_t, size_t) {
	if (uarch_index != kDefaultUArchIndex) {
		EXPECT_LE(uarch_index, kMaxUArchIndex);
	}
}

TEST(Parallelize2DTile1DWithUArch, SingleThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		CheckUArch2DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1DWithUArch, MultiThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		CheckUArch2DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void CheckBounds2DTile1DWithUArch(void*, uint32_t, size_t i, size_t start_j, size_t tile_j) {
	EXPECT_LT(i, kParallelize2DTile1DRangeI);
	EXPECT_LT(start_j, kParallelize2DTile1DRangeJ);
	EXPECT_LE(start_j + tile_j, kParallelize2DTile1DRangeJ);
}

TEST(Parallelize2DTile1DWithUArch, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		CheckBounds2DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1DWithUArch, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		CheckBounds2DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void CheckTiling2DTile1DWithUArch(void*, uint32_t, size_t i, size_t start_j, size_t tile_j) {
	EXPECT_GT(tile_j, 0);
	EXPECT_LE(tile_j, kParallelize2DTile1DTileJ);
	EXPECT_EQ(start_j % kParallelize2DTile1DTileJ, 0);
	EXPECT_EQ(tile_j, std::min<size_t>(kParallelize2DTile1DTileJ, kParallelize2DTile1DRangeJ - start_j));
}

TEST(Parallelize2DTile1DWithUArch, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		CheckTiling2DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1DWithUArch, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		CheckTiling2DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void SetTrue2DTile1DWithUArch(std::atomic_bool* processed_indicators, uint32_t, size_t i, size_t start_j, size_t tile_j) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
		processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize2DTile1DWithUArch, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_t>(SetTrue2DTile1DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

TEST(Parallelize2DTile1DWithUArch, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_t>(SetTrue2DTile1DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

static void Increment2DTile1DWithUArch(std::atomic_int* processed_counters, uint32_t, size_t i, size_t start_j, size_t tile_j) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
		processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize2DTile1DWithUArch, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_t>(Increment2DTile1DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile1DWithUArch, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_t>(Increment2DTile1DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile1DWithUArch, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_tile_1d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_t>(Increment2DTile1DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

TEST(Parallelize2DTile1DWithUArch, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_tile_1d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_t>(Increment2DTile1DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

static void IncrementSame2DTile1DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t i, size_t start_j, size_t tile_j) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize2DTile1DWithUArch, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_t>(IncrementSame2DTile1DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);
}

static void WorkImbalance2DTile1DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t i, size_t start_j, size_t tile_j) {
	num_processed_items->fetch_add(tile_j, std::memory_order_relaxed);
	if (i == 0 && start_j == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize2DTile1DWithUArch, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_t>(WorkImbalance2DTile1DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);
}

static void ComputeNothing2DTile1DWithUArchWithThread(void*, uint32_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize2DTile1DWithUArchWithThread, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(threadpool.get(),
		ComputeNothing2DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1DWithUArchWithThread, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		ComputeNothing2DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void CheckUArch2DTile1DWithUArchWithThread(void*, uint32_t uarch_index, size_t, size_t, size_t, size_t) {
	if (uarch_index != kDefaultUArchIndex) {
		EXPECT_LE(uarch_index, kMaxUArchIndex);
	}
}

TEST(Parallelize2DTile1DWithUArchWithThread, SingleThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckUArch2DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1DWithUArchWithThread, MultiThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckUArch2DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void CheckBounds2DTile1DWithUArchWithThread(void*, uint32_t, size_t, size_t i, size_t start_j, size_t tile_j) {
	EXPECT_LT(i, kParallelize2DTile1DRangeI);
	EXPECT_LT(start_j, kParallelize2DTile1DRangeJ);
	EXPECT_LE(start_j + tile_j, kParallelize2DTile1DRangeJ);
}

TEST(Parallelize2DTile1DWithUArchWithThread, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckBounds2DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1DWithUArchWithThread, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckBounds2DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void CheckTiling2DTile1DWithUArchWithThread(void*, uint32_t, size_t, size_t i, size_t start_j, size_t tile_j) {
	EXPECT_GT(tile_j, 0);
	EXPECT_LE(tile_j, kParallelize2DTile1DTileJ);
	EXPECT_EQ(start_j % kParallelize2DTile1DTileJ, 0);
	EXPECT_EQ(tile_j, std::min<size_t>(kParallelize2DTile1DTileJ, kParallelize2DTile1DRangeJ - start_j));
}

TEST(Parallelize2DTile1DWithUArchWithThread, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckTiling2DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile1DWithUArchWithThread, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckTiling2DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void SetTrue2DTile1DWithUArchWithThread(std::atomic_bool* processed_indicators, uint32_t, size_t, size_t i, size_t start_j, size_t tile_j) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
		processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize2DTile1DWithUArchWithThread, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_with_thread_t>(SetTrue2DTile1DWithUArchWithThread),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

TEST(Parallelize2DTile1DWithUArchWithThread, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_with_thread_t>(SetTrue2DTile1DWithUArchWithThread),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

static void Increment2DTile1DWithUArchWithThread(std::atomic_int* processed_counters, uint32_t, size_t, size_t i, size_t start_j, size_t tile_j) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
		processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize2DTile1DWithUArchWithThread, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_with_thread_t>(Increment2DTile1DWithUArchWithThread),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile1DWithUArchWithThread, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_with_thread_t>(Increment2DTile1DWithUArchWithThread),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile1DWithUArchWithThread, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_with_thread_t>(Increment2DTile1DWithUArchWithThread),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

TEST(Parallelize2DTile1DWithUArchWithThread, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_with_thread_t>(Increment2DTile1DWithUArchWithThread),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

static void IncrementSame2DTile1DWithUArchWithThread(std::atomic_int* num_processed_items, uint32_t, size_t, size_t i, size_t start_j, size_t tile_j) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize2DTile1DWithUArchWithThread, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_with_thread_t>(IncrementSame2DTile1DWithUArchWithThread),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);
}

static void WorkImbalance2DTile1DWithUArchWithThread(std::atomic_int* num_processed_items, uint32_t, size_t, size_t i, size_t start_j, size_t tile_j) {
	num_processed_items->fetch_add(tile_j, std::memory_order_relaxed);
	if (i == 0 && start_j == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize2DTile1DWithUArchWithThread, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_with_thread_t>(WorkImbalance2DTile1DWithUArchWithThread),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);
}

static void SetThreadTrue2DTile1DWithUArchWithThread(const size_t* num_threads, uint32_t, size_t thread_index, size_t i, size_t start_j, size_t tile_j) {
	EXPECT_LE(thread_index, *num_threads);
}

TEST(Parallelize2DTile1DWithUArchWithThread, MultiThreadPoolThreadIndexValid) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	size_t num_threads = pthreadpool_get_threads_count(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_1d_with_id_with_thread_t>(SetThreadTrue2DTile1DWithUArchWithThread),
		static_cast<void*>(&num_threads),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ,
		0 /* flags */);
}

static void ComputeNothing2DTile2D(void*, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize2DTile2D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d(threadpool.get(),
		ComputeNothing2DTile2D,
		nullptr,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile2D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		ComputeNothing2DTile2D,
		nullptr,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

static void CheckBounds2DTile2D(void*, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	EXPECT_LT(start_i, kParallelize2DTile2DRangeI);
	EXPECT_LT(start_j, kParallelize2DTile2DRangeJ);
	EXPECT_LE(start_i + tile_i, kParallelize2DTile2DRangeI);
	EXPECT_LE(start_j + tile_j, kParallelize2DTile2DRangeJ);
}

TEST(Parallelize2DTile2D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		CheckBounds2DTile2D,
		nullptr,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile2D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		CheckBounds2DTile2D,
		nullptr,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

static void CheckTiling2DTile2D(void*, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	EXPECT_GT(tile_i, 0);
	EXPECT_LE(tile_i, kParallelize2DTile2DTileI);
	EXPECT_EQ(start_i % kParallelize2DTile2DTileI, 0);
	EXPECT_EQ(tile_i, std::min<size_t>(kParallelize2DTile2DTileI, kParallelize2DTile2DRangeI - start_i));

	EXPECT_GT(tile_j, 0);
	EXPECT_LE(tile_j, kParallelize2DTile2DTileJ);
	EXPECT_EQ(start_j % kParallelize2DTile2DTileJ, 0);
	EXPECT_EQ(tile_j, std::min<size_t>(kParallelize2DTile2DTileJ, kParallelize2DTile2DRangeJ - start_j));
}

TEST(Parallelize2DTile2D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		CheckTiling2DTile2D,
		nullptr,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile2D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		CheckTiling2DTile2D,
		nullptr,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

static void SetTrue2DTile2D(std::atomic_bool* processed_indicators, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	for (size_t i = start_i; i < start_i + tile_i; i++) {
		for (size_t j = start_j; j < start_j + tile_j; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize2DTile2D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_t>(SetTrue2DTile2D),
		static_cast<void*>(indicators.data()),
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

TEST(Parallelize2DTile2D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_t>(SetTrue2DTile2D),
		static_cast<void*>(indicators.data()),
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

static void Increment2DTile2D(std::atomic_int* processed_counters, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	for (size_t i = start_i; i < start_i + tile_i; i++) {
		for (size_t j = start_j; j < start_j + tile_j; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize2DTile2D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_t>(Increment2DTile2D),
		static_cast<void*>(counters.data()),
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile2D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_t>(Increment2DTile2D),
		static_cast<void*>(counters.data()),
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile2D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_tile_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_tile_2d_t>(Increment2DTile2D),
			static_cast<void*>(counters.data()),
			kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
			kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

TEST(Parallelize2DTile2D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_tile_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_tile_2d_t>(Increment2DTile2D),
			static_cast<void*>(counters.data()),
			kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
			kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

static void IncrementSame2DTile2D(std::atomic_int* num_processed_items, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	for (size_t i = start_i; i < start_i + tile_i; i++) {
		for (size_t j = start_j; j < start_j + tile_j; j++) {
			num_processed_items->fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize2DTile2D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_t>(IncrementSame2DTile2D),
		static_cast<void*>(&num_processed_items),
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);
}

static void WorkImbalance2DTile2D(std::atomic_int* num_processed_items, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	num_processed_items->fetch_add(tile_i * tile_j, std::memory_order_relaxed);
	if (start_i == 0 && start_j == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize2DTile2D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_t>(WorkImbalance2DTile2D),
		static_cast<void*>(&num_processed_items),
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);
}

static void ComputeNothing2DTile2DWithUArch(void*, uint32_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize2DTile2DWithUArch, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d_with_uarch(threadpool.get(),
		ComputeNothing2DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile2DWithUArch, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		ComputeNothing2DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

static void CheckUArch2DTile2DWithUArch(void*, uint32_t uarch_index, size_t, size_t, size_t, size_t) {
	if (uarch_index != kDefaultUArchIndex) {
		EXPECT_LE(uarch_index, kMaxUArchIndex);
	}
}

TEST(Parallelize2DTile2DWithUArch, SingleThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		CheckUArch2DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile2DWithUArch, MultiThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		CheckUArch2DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

static void CheckBounds2DTile2DWithUArch(void*, uint32_t, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	EXPECT_LT(start_i, kParallelize2DTile2DRangeI);
	EXPECT_LT(start_j, kParallelize2DTile2DRangeJ);
	EXPECT_LE(start_i + tile_i, kParallelize2DTile2DRangeI);
	EXPECT_LE(start_j + tile_j, kParallelize2DTile2DRangeJ);
}

TEST(Parallelize2DTile2DWithUArch, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		CheckBounds2DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile2DWithUArch, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		CheckBounds2DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

static void CheckTiling2DTile2DWithUArch(void*, uint32_t, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	EXPECT_GT(tile_i, 0);
	EXPECT_LE(tile_i, kParallelize2DTile2DTileI);
	EXPECT_EQ(start_i % kParallelize2DTile2DTileI, 0);
	EXPECT_EQ(tile_i, std::min<size_t>(kParallelize2DTile2DTileI, kParallelize2DTile2DRangeI - start_i));

	EXPECT_GT(tile_j, 0);
	EXPECT_LE(tile_j, kParallelize2DTile2DTileJ);
	EXPECT_EQ(start_j % kParallelize2DTile2DTileJ, 0);
	EXPECT_EQ(tile_j, std::min<size_t>(kParallelize2DTile2DTileJ, kParallelize2DTile2DRangeJ - start_j));
}

TEST(Parallelize2DTile2DWithUArch, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		CheckTiling2DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

TEST(Parallelize2DTile2DWithUArch, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		CheckTiling2DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
}

static void SetTrue2DTile2DWithUArch(std::atomic_bool* processed_indicators, uint32_t, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	for (size_t i = start_i; i < start_i + tile_i; i++) {
		for (size_t j = start_j; j < start_j + tile_j; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize2DTile2DWithUArch, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_with_id_t>(SetTrue2DTile2DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

TEST(Parallelize2DTile2DWithUArch, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_with_id_t>(SetTrue2DTile2DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

static void Increment2DTile2DWithUArch(std::atomic_int* processed_counters, uint32_t, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	for (size_t i = start_i; i < start_i + tile_i; i++) {
		for (size_t j = start_j; j < start_j + tile_j; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize2DTile2DWithUArch, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_with_id_t>(Increment2DTile2DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile2DWithUArch, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_with_id_t>(Increment2DTile2DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile2DWithUArch, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_tile_2d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_tile_2d_with_id_t>(Increment2DTile2DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
			kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

TEST(Parallelize2DTile2DWithUArch, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_2d_tile_2d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_2d_tile_2d_with_id_t>(Increment2DTile2DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
			kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
				<< "(expected: " << kIncrementIterations << ")";
		}
	}
}

static void IncrementSame2DTile2DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	for (size_t i = start_i; i < start_i + tile_i; i++) {
		for (size_t j = start_j; j < start_j + tile_j; j++) {
			num_processed_items->fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize2DTile2DWithUArch, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_with_id_t>(IncrementSame2DTile2DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);
}

static void WorkImbalance2DTile2DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
	num_processed_items->fetch_add(tile_i * tile_j, std::memory_order_relaxed);
	if (start_i == 0 && start_j == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize2DTile2DWithUArch, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_2d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_2d_tile_2d_with_id_t>(WorkImbalance2DTile2DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);
}

static void ComputeNothing3D(void*, size_t, size_t, size_t) {
}

TEST(Parallelize3D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d(threadpool.get(),
		ComputeNothing3D,
		nullptr,
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
		0 /* flags */);
}

TEST(Parallelize3D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d(
		threadpool.get(),
		ComputeNothing3D,
		nullptr,
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
		0 /* flags */);
}

static void CheckBounds3D(void*, size_t i, size_t j, size_t k) {
	EXPECT_LT(i, kParallelize3DRangeI);
	EXPECT_LT(j, kParallelize3DRangeJ);
	EXPECT_LT(k, kParallelize3DRangeK);
}

TEST(Parallelize3D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d(
		threadpool.get(),
		CheckBounds3D,
		nullptr,
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
		0 /* flags */);
}

TEST(Parallelize3D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d(
		threadpool.get(),
		CheckBounds3D,
		nullptr,
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
		0 /* flags */);
}

static void SetTrue3D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t k) {
	const size_t linear_idx = (i * kParallelize3DRangeJ + j) * kParallelize3DRangeK + k;
	processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
}

TEST(Parallelize3D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_t>(SetTrue3D),
		static_cast<void*>(indicators.data()),
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DRangeJ + j) * kParallelize3DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

TEST(Parallelize3D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_t>(SetTrue3D),
		static_cast<void*>(indicators.data()),
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DRangeJ + j) * kParallelize3DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

static void Increment3D(std::atomic_int* processed_counters, size_t i, size_t j, size_t k) {
	const size_t linear_idx = (i * kParallelize3DRangeJ + j) * kParallelize3DRangeK + k;
	processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize3D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_t>(Increment3D),
		static_cast<void*>(counters.data()),
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DRangeJ + j) * kParallelize3DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_t>(Increment3D),
		static_cast<void*>(counters.data()),
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DRangeJ + j) * kParallelize3DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_t>(Increment3D),
			static_cast<void*>(counters.data()),
			kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
				0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DRangeJ + j) * kParallelize3DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

TEST(Parallelize3D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_t>(Increment3D),
			static_cast<void*>(counters.data()),
			kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
				0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DRangeJ + j) * kParallelize3DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

static void IncrementSame3D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize3D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_t>(IncrementSame3D),
		static_cast<void*>(&num_processed_items),
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK);
}

static void WorkImbalance3D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
	if (i == 0 && j == 0 && k == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize3D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_t>(WorkImbalance3D),
		static_cast<void*>(&num_processed_items),
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK);
}

static void ComputeNothing3DTile1D(void*, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize3DTile1D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d(threadpool.get(),
		ComputeNothing3DTile1D,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		ComputeNothing3DTile1D,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void CheckBounds3DTile1D(void*, size_t i, size_t j, size_t start_k, size_t tile_k) {
	EXPECT_LT(i, kParallelize3DTile1DRangeI);
	EXPECT_LT(j, kParallelize3DTile1DRangeJ);
	EXPECT_LT(start_k, kParallelize3DTile1DRangeK);
	EXPECT_LE(start_k + tile_k, kParallelize3DTile1DRangeK);
}

TEST(Parallelize3DTile1D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		CheckBounds3DTile1D,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		CheckBounds3DTile1D,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void CheckTiling3DTile1D(void*, size_t i, size_t j, size_t start_k, size_t tile_k) {
	EXPECT_GT(tile_k, 0);
	EXPECT_LE(tile_k, kParallelize3DTile1DTileK);
	EXPECT_EQ(start_k % kParallelize3DTile1DTileK, 0);
	EXPECT_EQ(tile_k, std::min<size_t>(kParallelize3DTile1DTileK, kParallelize3DTile1DRangeK - start_k));
}

TEST(Parallelize3DTile1D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		CheckTiling3DTile1D,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		CheckTiling3DTile1D,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void SetTrue3DTile1D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
		processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_t>(SetTrue3DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

TEST(Parallelize3DTile1D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_t>(SetTrue3DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

static void Increment3DTile1D(std::atomic_int* processed_counters, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
		processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_t>(Increment3DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile1D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_t>(Increment3DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile1D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_1d_t>(Increment3DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
			kParallelize3DTile1DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

TEST(Parallelize3DTile1D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_1d_t>(Increment3DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
			kParallelize3DTile1DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

static void IncrementSame3DTile1D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_t>(IncrementSame3DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);
}

static void WorkImbalance3DTile1D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t start_k, size_t tile_k) {
	num_processed_items->fetch_add(tile_k, std::memory_order_relaxed);
	if (i == 0 && j == 0 && start_k == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize3DTile1D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_t>(WorkImbalance3DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);
}

static void ComputeNothing3DTile1DWithThread(void*, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize3DTile1DWithThread, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_thread(threadpool.get(),
		ComputeNothing3DTile1DWithThread,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithThread, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		ComputeNothing3DTile1DWithThread,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void CheckBounds3DTile1DWithThread(void*, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	EXPECT_LT(i, kParallelize3DTile1DRangeI);
	EXPECT_LT(j, kParallelize3DTile1DRangeJ);
	EXPECT_LT(start_k, kParallelize3DTile1DRangeK);
	EXPECT_LE(start_k + tile_k, kParallelize3DTile1DRangeK);
}

TEST(Parallelize3DTile1DWithThread, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		CheckBounds3DTile1DWithThread,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithThread, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		CheckBounds3DTile1DWithThread,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void CheckTiling3DTile1DWithThread(void*, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	EXPECT_GT(tile_k, 0);
	EXPECT_LE(tile_k, kParallelize3DTile1DTileK);
	EXPECT_EQ(start_k % kParallelize3DTile1DTileK, 0);
	EXPECT_EQ(tile_k, std::min<size_t>(kParallelize3DTile1DTileK, kParallelize3DTile1DRangeK - start_k));
}

TEST(Parallelize3DTile1DWithThread, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		CheckTiling3DTile1DWithThread,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithThread, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		CheckTiling3DTile1DWithThread,
		nullptr,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void SetTrue3DTile1DWithThread(std::atomic_bool* processed_indicators, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
		processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1DWithThread, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_thread_t>(SetTrue3DTile1DWithThread),
		static_cast<void*>(indicators.data()),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithThread, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_thread_t>(SetTrue3DTile1DWithThread),
		static_cast<void*>(indicators.data()),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

static void Increment3DTile1DWithThread(std::atomic_int* processed_counters, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
		processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1DWithThread, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_thread_t>(Increment3DTile1DWithThread),
		static_cast<void*>(counters.data()),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithThread, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_thread_t>(Increment3DTile1DWithThread),
		static_cast<void*>(counters.data()),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithThread, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_1d_with_thread(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_1d_with_thread_t>(Increment3DTile1DWithThread),
			static_cast<void*>(counters.data()),
			kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
			kParallelize3DTile1DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithThread, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_1d_with_thread(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_1d_with_thread_t>(Increment3DTile1DWithThread),
			static_cast<void*>(counters.data()),
			kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
			kParallelize3DTile1DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

static void IncrementSame3DTile1DWithThread(std::atomic_int* num_processed_items, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1DWithThread, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_thread_t>(IncrementSame3DTile1DWithThread),
		static_cast<void*>(&num_processed_items),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);
}

static void WorkImbalance3DTile1DWithThread(std::atomic_int* num_processed_items, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	num_processed_items->fetch_add(tile_k, std::memory_order_relaxed);
	if (i == 0 && j == 0 && start_k == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize3DTile1DWithThread, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_thread_t>(WorkImbalance3DTile1DWithThread),
		static_cast<void*>(&num_processed_items),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);
}

static void CheckThreadIndexValid3DTile1DWithThread(const size_t* num_threads, size_t thread_index, size_t i, size_t j, size_t start_k, size_t tile_k) {
	EXPECT_LE(thread_index, *num_threads);
}

TEST(Parallelize3DTile1DWithThread, MultiThreadPoolThreadIndexValid) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	size_t num_threads = pthreadpool_get_threads_count(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_thread_t>(CheckThreadIndexValid3DTile1DWithThread),
		static_cast<void*>(&num_threads),
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void ComputeNothing3DTile1DWithUArch(void*, uint32_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize3DTile1DWithUArch, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch(threadpool.get(),
		ComputeNothing3DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithUArch, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		ComputeNothing3DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void CheckUArch3DTile1DWithUArch(void*, uint32_t uarch_index, size_t, size_t, size_t, size_t) {
	if (uarch_index != kDefaultUArchIndex) {
		EXPECT_LE(uarch_index, kMaxUArchIndex);
	}
}

TEST(Parallelize3DTile1DWithUArch, SingleThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		CheckUArch3DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithUArch, MultiThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		CheckUArch3DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void CheckBounds3DTile1DWithUArch(void*, uint32_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	EXPECT_LT(i, kParallelize3DTile1DRangeI);
	EXPECT_LT(j, kParallelize3DTile1DRangeJ);
	EXPECT_LT(start_k, kParallelize3DTile1DRangeK);
	EXPECT_LE(start_k + tile_k, kParallelize3DTile1DRangeK);
}

TEST(Parallelize3DTile1DWithUArch, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		CheckBounds3DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithUArch, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		CheckBounds3DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void CheckTiling3DTile1DWithUArch(void*, uint32_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	EXPECT_GT(tile_k, 0);
	EXPECT_LE(tile_k, kParallelize3DTile1DTileK);
	EXPECT_EQ(start_k % kParallelize3DTile1DTileK, 0);
	EXPECT_EQ(tile_k, std::min<size_t>(kParallelize3DTile1DTileK, kParallelize3DTile1DRangeK - start_k));
}

TEST(Parallelize3DTile1DWithUArch, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		CheckTiling3DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithUArch, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		CheckTiling3DTile1DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void SetTrue3DTile1DWithUArch(std::atomic_bool* processed_indicators, uint32_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
		processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1DWithUArch, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_t>(SetTrue3DTile1DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithUArch, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_t>(SetTrue3DTile1DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

static void Increment3DTile1DWithUArch(std::atomic_int* processed_counters, uint32_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
		processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1DWithUArch, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_t>(Increment3DTile1DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithUArch, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_t>(Increment3DTile1DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithUArch, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_1d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_t>(Increment3DTile1DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
			kParallelize3DTile1DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithUArch, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_1d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_t>(Increment3DTile1DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
			kParallelize3DTile1DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

static void IncrementSame3DTile1DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1DWithUArch, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_t>(IncrementSame3DTile1DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);
}

static void WorkImbalance3DTile1DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	num_processed_items->fetch_add(tile_k, std::memory_order_relaxed);
	if (i == 0 && j == 0 && start_k == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize3DTile1DWithUArch, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_t>(WorkImbalance3DTile1DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);
}

static void ComputeNothing3DTile1DWithUArchWithThread(void*, uint32_t, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize3DTile1DWithUArchWithThread, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(threadpool.get(),
		ComputeNothing3DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithUArchWithThread, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		ComputeNothing3DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void CheckUArch3DTile1DWithUArchWithThread(void*, uint32_t uarch_index, size_t, size_t, size_t, size_t, size_t) {
	if (uarch_index != kDefaultUArchIndex) {
		EXPECT_LE(uarch_index, kMaxUArchIndex);
	}
}

TEST(Parallelize3DTile1DWithUArchWithThread, SingleThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckUArch3DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithUArchWithThread, MultiThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckUArch3DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void CheckBounds3DTile1DWithUArchWithThread(void*, uint32_t, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	EXPECT_LT(i, kParallelize3DTile1DRangeI);
	EXPECT_LT(j, kParallelize3DTile1DRangeJ);
	EXPECT_LT(start_k, kParallelize3DTile1DRangeK);
	EXPECT_LE(start_k + tile_k, kParallelize3DTile1DRangeK);
}

TEST(Parallelize3DTile1DWithUArchWithThread, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckBounds3DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithUArchWithThread, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckBounds3DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void CheckTiling3DTile1DWithUArchWithThread(void*, uint32_t, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	EXPECT_GT(tile_k, 0);
	EXPECT_LE(tile_k, kParallelize3DTile1DTileK);
	EXPECT_EQ(start_k % kParallelize3DTile1DTileK, 0);
	EXPECT_EQ(tile_k, std::min<size_t>(kParallelize3DTile1DTileK, kParallelize3DTile1DRangeK - start_k));
}

TEST(Parallelize3DTile1DWithUArchWithThread, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckTiling3DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile1DWithUArchWithThread, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		CheckTiling3DTile1DWithUArchWithThread,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void SetTrue3DTile1DWithUArchWithThread(std::atomic_bool* processed_indicators, uint32_t, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
		processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1DWithUArchWithThread, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_with_thread_t>(SetTrue3DTile1DWithUArchWithThread),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithUArchWithThread, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_with_thread_t>(SetTrue3DTile1DWithUArchWithThread),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

static void Increment3DTile1DWithUArchWithThread(std::atomic_int* processed_counters, uint32_t, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
		processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1DWithUArchWithThread, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_with_thread_t>(Increment3DTile1DWithUArchWithThread),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithUArchWithThread, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_with_thread_t>(Increment3DTile1DWithUArchWithThread),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithUArchWithThread, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_with_thread_t>(Increment3DTile1DWithUArchWithThread),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
			kParallelize3DTile1DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

TEST(Parallelize3DTile1DWithUArchWithThread, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_with_thread_t>(Increment3DTile1DWithUArchWithThread),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
			kParallelize3DTile1DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile1DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

static void IncrementSame3DTile1DWithUArchWithThread(std::atomic_int* num_processed_items, uint32_t, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize3DTile1DWithUArchWithThread, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_with_thread_t>(IncrementSame3DTile1DWithUArchWithThread),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);
}

static void WorkImbalance3DTile1DWithUArchWithThread(std::atomic_int* num_processed_items, uint32_t, size_t, size_t i, size_t j, size_t start_k, size_t tile_k) {
	num_processed_items->fetch_add(tile_k, std::memory_order_relaxed);
	if (i == 0 && j == 0 && start_k == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize3DTile1DWithUArchWithThread, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_with_thread_t>(WorkImbalance3DTile1DWithUArchWithThread),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);
}

static void SetThreadTrue3DTile1DWithUArchWithThread(const size_t* num_threads, uint32_t, size_t thread_index, size_t i, size_t j, size_t start_k, size_t tile_k) {
	EXPECT_LE(thread_index, *num_threads);
}

TEST(Parallelize3DTile1DWithUArchWithThread, MultiThreadPoolThreadIndexValid) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	size_t num_threads = pthreadpool_get_threads_count(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_1d_with_id_with_thread_t>(SetThreadTrue3DTile1DWithUArchWithThread),
		static_cast<void*>(&num_threads),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK,
		0 /* flags */);
}

static void ComputeNothing3DTile2D(void*, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize3DTile2D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d(threadpool.get(),
		ComputeNothing3DTile2D,
		nullptr,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile2D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		ComputeNothing3DTile2D,
		nullptr,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

static void CheckBounds3DTile2D(void*, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	EXPECT_LT(i, kParallelize3DTile2DRangeI);
	EXPECT_LT(start_j, kParallelize3DTile2DRangeJ);
	EXPECT_LT(start_k, kParallelize3DTile2DRangeK);
	EXPECT_LE(start_j + tile_j, kParallelize3DTile2DRangeJ);
	EXPECT_LE(start_k + tile_k, kParallelize3DTile2DRangeK);
}

TEST(Parallelize3DTile2D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		CheckBounds3DTile2D,
		nullptr,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile2D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		CheckBounds3DTile2D,
		nullptr,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

static void CheckTiling3DTile2D(void*, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	EXPECT_GT(tile_j, 0);
	EXPECT_LE(tile_j, kParallelize3DTile2DTileJ);
	EXPECT_EQ(start_j % kParallelize3DTile2DTileJ, 0);
	EXPECT_EQ(tile_j, std::min<size_t>(kParallelize3DTile2DTileJ, kParallelize3DTile2DRangeJ - start_j));

	EXPECT_GT(tile_k, 0);
	EXPECT_LE(tile_k, kParallelize3DTile2DTileK);
	EXPECT_EQ(start_k % kParallelize3DTile2DTileK, 0);
	EXPECT_EQ(tile_k, std::min<size_t>(kParallelize3DTile2DTileK, kParallelize3DTile2DRangeK - start_k));
}

TEST(Parallelize3DTile2D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		CheckTiling3DTile2D,
		nullptr,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile2D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		CheckTiling3DTile2D,
		nullptr,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

static void SetTrue3DTile2D(std::atomic_bool* processed_indicators, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		for (size_t k = start_k; k < start_k + tile_k; k++) {
			const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
			processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize3DTile2D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_t>(SetTrue3DTile2D),
		static_cast<void*>(indicators.data()),
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

TEST(Parallelize3DTile2D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_t>(SetTrue3DTile2D),
		static_cast<void*>(indicators.data()),
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

static void Increment3DTile2D(std::atomic_int* processed_counters, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		for (size_t k = start_k; k < start_k + tile_k; k++) {
			const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
			processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize3DTile2D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_t>(Increment3DTile2D),
		static_cast<void*>(counters.data()),
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile2D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_t>(Increment3DTile2D),
		static_cast<void*>(counters.data()),
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile2D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_2d_t>(Increment3DTile2D),
			static_cast<void*>(counters.data()),
			kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
			kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

TEST(Parallelize3DTile2D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_2d_t>(Increment3DTile2D),
			static_cast<void*>(counters.data()),
			kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
			kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

static void IncrementSame3DTile2D(std::atomic_int* num_processed_items, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		for (size_t k = start_k; k < start_k + tile_k; k++) {
			num_processed_items->fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize3DTile2D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_t>(IncrementSame3DTile2D),
		static_cast<void*>(&num_processed_items),
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);
}

static void WorkImbalance3DTile2D(std::atomic_int* num_processed_items, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	num_processed_items->fetch_add(tile_j * tile_k, std::memory_order_relaxed);
	if (i == 0 && start_j == 0 && start_k == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize3DTile2D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_t>(WorkImbalance3DTile2D),
		static_cast<void*>(&num_processed_items),
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);
}

static void ComputeNothing3DTile2DWithUArch(void*, uint32_t, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize3DTile2DWithUArch, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d_with_uarch(threadpool.get(),
		ComputeNothing3DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile2DWithUArch, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		ComputeNothing3DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

static void CheckUArch3DTile2DWithUArch(void*, uint32_t uarch_index, size_t, size_t, size_t, size_t, size_t) {
	if (uarch_index != kDefaultUArchIndex) {
		EXPECT_LE(uarch_index, kMaxUArchIndex);
	}
}

TEST(Parallelize3DTile2DWithUArch, SingleThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		CheckUArch3DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile2DWithUArch, MultiThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		CheckUArch3DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

static void CheckBounds3DTile2DWithUArch(void*, uint32_t, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	EXPECT_LT(i, kParallelize3DTile2DRangeI);
	EXPECT_LT(start_j, kParallelize3DTile2DRangeJ);
	EXPECT_LT(start_k, kParallelize3DTile2DRangeK);
	EXPECT_LE(start_j + tile_j, kParallelize3DTile2DRangeJ);
	EXPECT_LE(start_k + tile_k, kParallelize3DTile2DRangeK);
}

TEST(Parallelize3DTile2DWithUArch, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		CheckBounds3DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile2DWithUArch, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		CheckBounds3DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

static void CheckTiling3DTile2DWithUArch(void*, uint32_t, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	EXPECT_GT(tile_j, 0);
	EXPECT_LE(tile_j, kParallelize3DTile2DTileJ);
	EXPECT_EQ(start_j % kParallelize3DTile2DTileJ, 0);
	EXPECT_EQ(tile_j, std::min<size_t>(kParallelize3DTile2DTileJ, kParallelize3DTile2DRangeJ - start_j));

	EXPECT_GT(tile_k, 0);
	EXPECT_LE(tile_k, kParallelize3DTile2DTileK);
	EXPECT_EQ(start_k % kParallelize3DTile2DTileK, 0);
	EXPECT_EQ(tile_k, std::min<size_t>(kParallelize3DTile2DTileK, kParallelize3DTile2DRangeK - start_k));
}

TEST(Parallelize3DTile2DWithUArch, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		CheckTiling3DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

TEST(Parallelize3DTile2DWithUArch, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		CheckTiling3DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
}

static void SetTrue3DTile2DWithUArch(std::atomic_bool* processed_indicators, uint32_t, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		for (size_t k = start_k; k < start_k + tile_k; k++) {
			const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
			processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize3DTile2DWithUArch, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_with_id_t>(SetTrue3DTile2DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

TEST(Parallelize3DTile2DWithUArch, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_with_id_t>(SetTrue3DTile2DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
					<< "Element (" << i << ", " << j << ", " << k << ") not processed";
			}
		}
	}
}

static void Increment3DTile2DWithUArch(std::atomic_int* processed_counters, uint32_t, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		for (size_t k = start_k; k < start_k + tile_k; k++) {
			const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
			processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize3DTile2DWithUArch, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_with_id_t>(Increment3DTile2DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile2DWithUArch, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_with_id_t>(Increment3DTile2DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
			}
		}
	}
}

TEST(Parallelize3DTile2DWithUArch, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_2d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_2d_with_id_t>(Increment3DTile2DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
			kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

TEST(Parallelize3DTile2DWithUArch, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_3d_tile_2d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_3d_tile_2d_with_id_t>(Increment3DTile2DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
			kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize3DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize3DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize3DTile2DRangeK; k++) {
				const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
				EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
					<< "Element (" << i << ", " << j << ", " << k << ") was processed "
					<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
					<< "(expected: " << kIncrementIterations << ")";
			}
		}
	}
}

static void IncrementSame3DTile2DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	for (size_t j = start_j; j < start_j + tile_j; j++) {
		for (size_t k = start_k; k < start_k + tile_k; k++) {
			num_processed_items->fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize3DTile2DWithUArch, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_with_id_t>(IncrementSame3DTile2DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);
}

static void WorkImbalance3DTile2DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
	num_processed_items->fetch_add(tile_j * tile_k, std::memory_order_relaxed);
	if (i == 0 && start_j == 0 && start_k == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize3DTile2DWithUArch, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_3d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_3d_tile_2d_with_id_t>(WorkImbalance3DTile2DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);
}

static void ComputeNothing4D(void*, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize4D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d(threadpool.get(),
		ComputeNothing4D,
		nullptr,
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
		0 /* flags */);
}

TEST(Parallelize4D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d(
		threadpool.get(),
		ComputeNothing4D,
		nullptr,
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
		0 /* flags */);
}

static void CheckBounds4D(void*, size_t i, size_t j, size_t k, size_t l) {
	EXPECT_LT(i, kParallelize4DRangeI);
	EXPECT_LT(j, kParallelize4DRangeJ);
	EXPECT_LT(k, kParallelize4DRangeK);
	EXPECT_LT(l, kParallelize4DRangeL);
}

TEST(Parallelize4D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d(
		threadpool.get(),
		CheckBounds4D,
		nullptr,
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
		0 /* flags */);
}

TEST(Parallelize4D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d(
		threadpool.get(),
		CheckBounds4D,
		nullptr,
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
		0 /* flags */);
}

static void SetTrue4D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t k, size_t l) {
	const size_t linear_idx = ((i * kParallelize4DRangeJ + j) * kParallelize4DRangeK + k) * kParallelize4DRangeL + l;
	processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
}

TEST(Parallelize4D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_t>(SetTrue4D),
		static_cast<void*>(indicators.data()),
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DRangeJ + j) * kParallelize4DRangeK + k) * kParallelize4DRangeL + l;
					EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") not processed";
				}
			}
		}
	}
}

TEST(Parallelize4D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_t>(SetTrue4D),
		static_cast<void*>(indicators.data()),
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DRangeJ + j) * kParallelize4DRangeK + k) * kParallelize4DRangeL + l;
					EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") not processed";
				}
			}
		}
	}
}

static void Increment4D(std::atomic_int* processed_counters, size_t i, size_t j, size_t k, size_t l) {
	const size_t linear_idx = ((i * kParallelize4DRangeJ + j) * kParallelize4DRangeK + k) * kParallelize4DRangeL + l;
	processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize4D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_t>(Increment4D),
		static_cast<void*>(counters.data()),
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DRangeJ + j) * kParallelize4DRangeK + k) * kParallelize4DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
				}
			}
		}
	}
}

TEST(Parallelize4D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_t>(Increment4D),
		static_cast<void*>(counters.data()),
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DRangeJ + j) * kParallelize4DRangeK + k) * kParallelize4DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
				}
			}
		}
	}
}

TEST(Parallelize4D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_4d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_4d_t>(Increment4D),
			static_cast<void*>(counters.data()),
			kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
				0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize4DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DRangeJ + j) * kParallelize4DRangeK + k) * kParallelize4DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
						<< "(expected: " << kIncrementIterations << ")";
				}
			}
		}
	}
}

TEST(Parallelize4D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_4d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_4d_t>(Increment4D),
			static_cast<void*>(counters.data()),
			kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
				0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize4DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DRangeJ + j) * kParallelize4DRangeK + k) * kParallelize4DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
						<< "(expected: " << kIncrementIterations << ")";
				}
			}
		}
	}
}

static void IncrementSame4D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize4D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_t>(IncrementSame4D),
		static_cast<void*>(&num_processed_items),
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL);
}

static void WorkImbalance4D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
	if (i == 0 && j == 0 && k == 0 && l == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize4D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_t>(WorkImbalance4D),
		static_cast<void*>(&num_processed_items),
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL);
}

static void ComputeNothing4DTile1D(void*, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize4DTile1D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_1d(threadpool.get(),
		ComputeNothing4DTile1D,
		nullptr,
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);
}

TEST(Parallelize4DTile1D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		ComputeNothing4DTile1D,
		nullptr,
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);
}

static void CheckBounds4DTile1D(void*, size_t i, size_t j, size_t k, size_t start_l, size_t tile_l) {
	EXPECT_LT(i, kParallelize4DTile1DRangeI);
	EXPECT_LT(j, kParallelize4DTile1DRangeJ);
	EXPECT_LT(k, kParallelize4DTile1DRangeK);
	EXPECT_LT(start_l, kParallelize4DTile1DRangeL);
	EXPECT_LE(start_l + tile_l, kParallelize4DTile1DRangeL);
}

TEST(Parallelize4DTile1D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		CheckBounds4DTile1D,
		nullptr,
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);
}

TEST(Parallelize4DTile1D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		CheckBounds4DTile1D,
		nullptr,
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);
}

static void CheckTiling4DTile1D(void*, size_t i, size_t j, size_t k, size_t start_l, size_t tile_l) {
	EXPECT_GT(tile_l, 0);
	EXPECT_LE(tile_l, kParallelize4DTile1DTileL);
	EXPECT_EQ(start_l % kParallelize4DTile1DTileL, 0);
	EXPECT_EQ(tile_l, std::min<size_t>(kParallelize4DTile1DTileL, kParallelize4DTile1DRangeL - start_l));
}

TEST(Parallelize4DTile1D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		CheckTiling4DTile1D,
		nullptr,
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);
}

TEST(Parallelize4DTile1D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		CheckTiling4DTile1D,
		nullptr,
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);
}

static void SetTrue4DTile1D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t k, size_t start_l, size_t tile_l) {
	for (size_t l = start_l; l < start_l + tile_l; l++) {
		const size_t linear_idx = ((i * kParallelize4DTile1DRangeJ + j) * kParallelize4DTile1DRangeK + k) * kParallelize4DTile1DRangeL + l;
		processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize4DTile1D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_1d_t>(SetTrue4DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile1DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile1DRangeJ + j) * kParallelize4DTile1DRangeK + k) * kParallelize4DTile1DRangeL + l;
					EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") not processed";
				}
			}
		}
	}
}

TEST(Parallelize4DTile1D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_1d_t>(SetTrue4DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile1DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile1DRangeJ + j) * kParallelize4DTile1DRangeK + k) * kParallelize4DTile1DRangeL + l;
					EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") not processed";
				}
			}
		}
	}
}

static void Increment4DTile1D(std::atomic_int* processed_counters, size_t i, size_t j, size_t k, size_t start_l, size_t tile_l) {
	for (size_t l = start_l; l < start_l + tile_l; l++) {
		const size_t linear_idx = ((i * kParallelize4DTile1DRangeJ + j) * kParallelize4DTile1DRangeK + k) * kParallelize4DTile1DRangeL + l;
		processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize4DTile1D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_1d_t>(Increment4DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile1DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile1DRangeJ + j) * kParallelize4DTile1DRangeK + k) * kParallelize4DTile1DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
				}
			}
		}
	}
}

TEST(Parallelize4DTile1D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_1d_t>(Increment4DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile1DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile1DRangeJ + j) * kParallelize4DTile1DRangeK + k) * kParallelize4DTile1DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
				}
			}
		}
	}
}

TEST(Parallelize4DTile1D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_4d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_4d_tile_1d_t>(Increment4DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
			kParallelize4DTile1DTileL,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize4DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile1DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile1DRangeJ + j) * kParallelize4DTile1DRangeK + k) * kParallelize4DTile1DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
						<< "(expected: " << kIncrementIterations << ")";
				}
			}
		}
	}
}

TEST(Parallelize4DTile1D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_4d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_4d_tile_1d_t>(Increment4DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
			kParallelize4DTile1DTileL,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize4DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile1DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile1DRangeJ + j) * kParallelize4DTile1DRangeK + k) * kParallelize4DTile1DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
						<< "(expected: " << kIncrementIterations << ")";
				}
			}
		}
	}
}

static void IncrementSame4DTile1D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t start_l, size_t tile_l) {
	for (size_t l = start_l; l < start_l + tile_l; l++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize4DTile1D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_1d_t>(IncrementSame4DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL);
}

static void WorkImbalance4DTile1D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t start_l, size_t tile_l) {
	num_processed_items->fetch_add(tile_l, std::memory_order_relaxed);
	if (i == 0 && j == 0 && k == 0 && start_l == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize4DTile1D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_1d_t>(WorkImbalance4DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL);
}

static void ComputeNothing4DTile2D(void*, size_t, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize4DTile2D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d(threadpool.get(),
		ComputeNothing4DTile2D,
		nullptr,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

TEST(Parallelize4DTile2D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		ComputeNothing4DTile2D,
		nullptr,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

static void CheckBounds4DTile2D(void*, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	EXPECT_LT(i, kParallelize4DTile2DRangeI);
	EXPECT_LT(j, kParallelize4DTile2DRangeJ);
	EXPECT_LT(start_k, kParallelize4DTile2DRangeK);
	EXPECT_LT(start_l, kParallelize4DTile2DRangeL);
	EXPECT_LE(start_k + tile_k, kParallelize4DTile2DRangeK);
	EXPECT_LE(start_l + tile_l, kParallelize4DTile2DRangeL);
}

TEST(Parallelize4DTile2D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		CheckBounds4DTile2D,
		nullptr,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

TEST(Parallelize4DTile2D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		CheckBounds4DTile2D,
		nullptr,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

static void CheckTiling4DTile2D(void*, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	EXPECT_GT(tile_k, 0);
	EXPECT_LE(tile_k, kParallelize4DTile2DTileK);
	EXPECT_EQ(start_k % kParallelize4DTile2DTileK, 0);
	EXPECT_EQ(tile_k, std::min<size_t>(kParallelize4DTile2DTileK, kParallelize4DTile2DRangeK - start_k));

	EXPECT_GT(tile_l, 0);
	EXPECT_LE(tile_l, kParallelize4DTile2DTileL);
	EXPECT_EQ(start_l % kParallelize4DTile2DTileL, 0);
	EXPECT_EQ(tile_l, std::min<size_t>(kParallelize4DTile2DTileL, kParallelize4DTile2DRangeL - start_l));
}

TEST(Parallelize4DTile2D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		CheckTiling4DTile2D,
		nullptr,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

TEST(Parallelize4DTile2D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		CheckTiling4DTile2D,
		nullptr,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

static void SetTrue4DTile2D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		for (size_t l = start_l; l < start_l + tile_l; l++) {
			const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
			processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize4DTile2D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_t>(SetTrue4DTile2D),
		static_cast<void*>(indicators.data()),
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") not processed";
				}
			}
		}
	}
}

TEST(Parallelize4DTile2D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_t>(SetTrue4DTile2D),
		static_cast<void*>(indicators.data()),
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") not processed";
				}
			}
		}
	}
}

static void Increment4DTile2D(std::atomic_int* processed_counters, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		for (size_t l = start_l; l < start_l + tile_l; l++) {
			const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
			processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize4DTile2D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_t>(Increment4DTile2D),
		static_cast<void*>(counters.data()),
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
				}
			}
		}
	}
}

TEST(Parallelize4DTile2D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_t>(Increment4DTile2D),
		static_cast<void*>(counters.data()),
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
				}
			}
		}
	}
}

TEST(Parallelize4DTile2D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_4d_tile_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_4d_tile_2d_t>(Increment4DTile2D),
			static_cast<void*>(counters.data()),
			kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
			kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
						<< "(expected: " << kIncrementIterations << ")";
				}
			}
		}
	}
}

TEST(Parallelize4DTile2D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_4d_tile_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_4d_tile_2d_t>(Increment4DTile2D),
			static_cast<void*>(counters.data()),
			kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
			kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
						<< "(expected: " << kIncrementIterations << ")";
				}
			}
		}
	}
}

static void IncrementSame4DTile2D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		for (size_t l = start_l; l < start_l + tile_l; l++) {
			num_processed_items->fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize4DTile2D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_t>(IncrementSame4DTile2D),
		static_cast<void*>(&num_processed_items),
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);
}

static void WorkImbalance4DTile2D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	num_processed_items->fetch_add(tile_k * tile_l, std::memory_order_relaxed);
	if (i == 0 && j == 0 && start_k == 0 && start_l == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize4DTile2D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_t>(WorkImbalance4DTile2D),
		static_cast<void*>(&num_processed_items),
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);
}

static void ComputeNothing4DTile2DWithUArch(void*, uint32_t, size_t, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize4DTile2DWithUArch, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d_with_uarch(threadpool.get(),
		ComputeNothing4DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

TEST(Parallelize4DTile2DWithUArch, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		ComputeNothing4DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

static void CheckUArch4DTile2DWithUArch(void*, uint32_t uarch_index, size_t, size_t, size_t, size_t, size_t, size_t) {
	if (uarch_index != kDefaultUArchIndex) {
		EXPECT_LE(uarch_index, kMaxUArchIndex);
	}
}

TEST(Parallelize4DTile2DWithUArch, SingleThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		CheckUArch4DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

TEST(Parallelize4DTile2DWithUArch, MultiThreadPoolUArchInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		CheckUArch4DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

static void CheckBounds4DTile2DWithUArch(void*, uint32_t, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	EXPECT_LT(i, kParallelize4DTile2DRangeI);
	EXPECT_LT(j, kParallelize4DTile2DRangeJ);
	EXPECT_LT(start_k, kParallelize4DTile2DRangeK);
	EXPECT_LT(start_l, kParallelize4DTile2DRangeL);
	EXPECT_LE(start_k + tile_k, kParallelize4DTile2DRangeK);
	EXPECT_LE(start_l + tile_l, kParallelize4DTile2DRangeL);
}

TEST(Parallelize4DTile2DWithUArch, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		CheckBounds4DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

TEST(Parallelize4DTile2DWithUArch, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		CheckBounds4DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

static void CheckTiling4DTile2DWithUArch(void*, uint32_t, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	EXPECT_GT(tile_k, 0);
	EXPECT_LE(tile_k, kParallelize4DTile2DTileK);
	EXPECT_EQ(start_k % kParallelize4DTile2DTileK, 0);
	EXPECT_EQ(tile_k, std::min<size_t>(kParallelize4DTile2DTileK, kParallelize4DTile2DRangeK - start_k));

	EXPECT_GT(tile_l, 0);
	EXPECT_LE(tile_l, kParallelize4DTile2DTileL);
	EXPECT_EQ(start_l % kParallelize4DTile2DTileL, 0);
	EXPECT_EQ(tile_l, std::min<size_t>(kParallelize4DTile2DTileL, kParallelize4DTile2DRangeL - start_l));
}

TEST(Parallelize4DTile2DWithUArch, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		CheckTiling4DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

TEST(Parallelize4DTile2DWithUArch, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		CheckTiling4DTile2DWithUArch,
		nullptr,
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
}

static void SetTrue4DTile2DWithUArch(std::atomic_bool* processed_indicators, uint32_t, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		for (size_t l = start_l; l < start_l + tile_l; l++) {
			const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
			processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize4DTile2DWithUArch, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_with_id_t>(SetTrue4DTile2DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") not processed";
				}
			}
		}
	}
}

TEST(Parallelize4DTile2DWithUArch, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_with_id_t>(SetTrue4DTile2DWithUArch),
		static_cast<void*>(indicators.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") not processed";
				}
			}
		}
	}
}

static void Increment4DTile2DWithUArch(std::atomic_int* processed_counters, uint32_t, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		for (size_t l = start_l; l < start_l + tile_l; l++) {
			const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
			processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize4DTile2DWithUArch, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_with_id_t>(Increment4DTile2DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
				}
			}
		}
	}
}

TEST(Parallelize4DTile2DWithUArch, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_with_id_t>(Increment4DTile2DWithUArch),
		static_cast<void*>(counters.data()),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
				}
			}
		}
	}
}

TEST(Parallelize4DTile2DWithUArch, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_4d_tile_2d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_4d_tile_2d_with_id_t>(Increment4DTile2DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
			kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
						<< "(expected: " << kIncrementIterations << ")";
				}
			}
		}
	}
}

TEST(Parallelize4DTile2DWithUArch, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations; iteration++) {
		pthreadpool_parallelize_4d_tile_2d_with_uarch(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_4d_tile_2d_with_id_t>(Increment4DTile2DWithUArch),
			static_cast<void*>(counters.data()),
			kDefaultUArchIndex, kMaxUArchIndex,
			kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
			kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize4DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize4DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize4DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize4DTile2DRangeL; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations)
						<< "Element (" << i << ", " << j << ", " << k << ", " << l << ") was processed "
						<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
						<< "(expected: " << kIncrementIterations << ")";
				}
			}
		}
	}
}

static void IncrementSame4DTile2DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	for (size_t k = start_k; k < start_k + tile_k; k++) {
		for (size_t l = start_l; l < start_l + tile_l; l++) {
			num_processed_items->fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize4DTile2DWithUArch, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_with_id_t>(IncrementSame4DTile2DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);
}

static void WorkImbalance4DTile2DWithUArch(std::atomic_int* num_processed_items, uint32_t, size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
	num_processed_items->fetch_add(tile_k * tile_l, std::memory_order_relaxed);
	if (i == 0 && j == 0 && start_k == 0 && start_l == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize4DTile2DWithUArch, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_4d_tile_2d_with_uarch(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_4d_tile_2d_with_id_t>(WorkImbalance4DTile2DWithUArch),
		static_cast<void*>(&num_processed_items),
		kDefaultUArchIndex, kMaxUArchIndex,
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);
}

static void ComputeNothing5D(void*, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize5D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d(threadpool.get(),
		ComputeNothing5D,
		nullptr,
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
		0 /* flags */);
}

TEST(Parallelize5D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d(
		threadpool.get(),
		ComputeNothing5D,
		nullptr,
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
		0 /* flags */);
}

static void CheckBounds5D(void*, size_t i, size_t j, size_t k, size_t l, size_t m) {
	EXPECT_LT(i, kParallelize5DRangeI);
	EXPECT_LT(j, kParallelize5DRangeJ);
	EXPECT_LT(k, kParallelize5DRangeK);
	EXPECT_LT(l, kParallelize5DRangeL);
	EXPECT_LT(m, kParallelize5DRangeM);
}

TEST(Parallelize5D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d(
		threadpool.get(),
		CheckBounds5D,
		nullptr,
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
		0 /* flags */);
}

TEST(Parallelize5D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d(
		threadpool.get(),
		CheckBounds5D,
		nullptr,
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
		0 /* flags */);
}

static void SetTrue5D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t k, size_t l, size_t m) {
	const size_t linear_idx = (((i * kParallelize5DRangeJ + j) * kParallelize5DRangeK + k) * kParallelize5DRangeL + l) * kParallelize5DRangeM + m;
	processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
}

TEST(Parallelize5D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_t>(SetTrue5D),
		static_cast<void*>(indicators.data()),
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DRangeJ + j) * kParallelize5DRangeK + k) * kParallelize5DRangeL + l) * kParallelize5DRangeM + m;
						EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") not processed";
					}
				}
			}
		}
	}
}

TEST(Parallelize5D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_t>(SetTrue5D),
		static_cast<void*>(indicators.data()),
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DRangeJ + j) * kParallelize5DRangeK + k) * kParallelize5DRangeL + l) * kParallelize5DRangeM + m;
						EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") not processed";
					}
				}
			}
		}
	}
}

static void Increment5D(std::atomic_int* processed_counters, size_t i, size_t j, size_t k, size_t l, size_t m) {
	const size_t linear_idx = (((i * kParallelize5DRangeJ + j) * kParallelize5DRangeK + k) * kParallelize5DRangeL + l) * kParallelize5DRangeM + m;
	processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize5D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_t>(Increment5D),
		static_cast<void*>(counters.data()),
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DRangeJ + j) * kParallelize5DRangeK + k) * kParallelize5DRangeL + l) * kParallelize5DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
					}
				}
			}
		}
	}
}

TEST(Parallelize5D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_t>(Increment5D),
		static_cast<void*>(counters.data()),
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DRangeJ + j) * kParallelize5DRangeK + k) * kParallelize5DRangeL + l) * kParallelize5DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
					}
				}
			}
		}
	}
}

TEST(Parallelize5D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations5D; iteration++) {
		pthreadpool_parallelize_5d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_5d_t>(Increment5D),
			static_cast<void*>(counters.data()),
			kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
				0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize5DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DRangeJ + j) * kParallelize5DRangeK + k) * kParallelize5DRangeL + l) * kParallelize5DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations5D)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
							<< "(expected: " << kIncrementIterations5D << ")";
					}
				}
			}
		}
	}
}

TEST(Parallelize5D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations5D; iteration++) {
		pthreadpool_parallelize_5d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_5d_t>(Increment5D),
			static_cast<void*>(counters.data()),
			kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
				0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize5DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DRangeJ + j) * kParallelize5DRangeK + k) * kParallelize5DRangeL + l) * kParallelize5DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations5D)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
							<< "(expected: " << kIncrementIterations5D << ")";
					}
				}
			}
		}
	}
}

static void IncrementSame5D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l, size_t m) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize5D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_t>(IncrementSame5D),
		static_cast<void*>(&num_processed_items),
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM);
}

static void WorkImbalance5D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l, size_t m) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
	if (i == 0 && j == 0 && k == 0 && l == 0 && m == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize5D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_t>(WorkImbalance5D),
		static_cast<void*>(&num_processed_items),
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM);
}

static void ComputeNothing5DTile1D(void*, size_t, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize5DTile1D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_1d(threadpool.get(),
		ComputeNothing5DTile1D,
		nullptr,
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);
}

TEST(Parallelize5DTile1D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		ComputeNothing5DTile1D,
		nullptr,
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);
}

static void CheckBounds5DTile1D(void*, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t tile_m) {
	EXPECT_LT(i, kParallelize5DTile1DRangeI);
	EXPECT_LT(j, kParallelize5DTile1DRangeJ);
	EXPECT_LT(k, kParallelize5DTile1DRangeK);
	EXPECT_LT(l, kParallelize5DTile1DRangeL);
	EXPECT_LT(start_m, kParallelize5DTile1DRangeM);
	EXPECT_LE(start_m + tile_m, kParallelize5DTile1DRangeM);
}

TEST(Parallelize5DTile1D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		CheckBounds5DTile1D,
		nullptr,
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);
}

TEST(Parallelize5DTile1D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		CheckBounds5DTile1D,
		nullptr,
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);
}

static void CheckTiling5DTile1D(void*, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t tile_m) {
	EXPECT_GT(tile_m, 0);
	EXPECT_LE(tile_m, kParallelize5DTile1DTileM);
	EXPECT_EQ(start_m % kParallelize5DTile1DTileM, 0);
	EXPECT_EQ(tile_m, std::min<size_t>(kParallelize5DTile1DTileM, kParallelize5DTile1DRangeM - start_m));
}

TEST(Parallelize5DTile1D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		CheckTiling5DTile1D,
		nullptr,
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);
}

TEST(Parallelize5DTile1D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		CheckTiling5DTile1D,
		nullptr,
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);
}

static void SetTrue5DTile1D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t tile_m) {
	for (size_t m = start_m; m < start_m + tile_m; m++) {
		const size_t linear_idx = (((i * kParallelize5DTile1DRangeJ + j) * kParallelize5DTile1DRangeK + k) * kParallelize5DTile1DRangeL + l) * kParallelize5DTile1DRangeM + m;
		processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize5DTile1D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_1d_t>(SetTrue5DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile1DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile1DRangeJ + j) * kParallelize5DTile1DRangeK + k) * kParallelize5DTile1DRangeL + l) * kParallelize5DTile1DRangeM + m;
						EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") not processed";
					}
				}
			}
		}
	}
}

TEST(Parallelize5DTile1D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_1d_t>(SetTrue5DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile1DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile1DRangeJ + j) * kParallelize5DTile1DRangeK + k) * kParallelize5DTile1DRangeL + l) * kParallelize5DTile1DRangeM + m;
						EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") not processed";
					}
				}
			}
		}
	}
}

static void Increment5DTile1D(std::atomic_int* processed_counters, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t tile_m) {
	for (size_t m = start_m; m < start_m + tile_m; m++) {
		const size_t linear_idx = (((i * kParallelize5DTile1DRangeJ + j) * kParallelize5DTile1DRangeK + k) * kParallelize5DTile1DRangeL + l) * kParallelize5DTile1DRangeM + m;
		processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize5DTile1D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_1d_t>(Increment5DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile1DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile1DRangeJ + j) * kParallelize5DTile1DRangeK + k) * kParallelize5DTile1DRangeL + l) * kParallelize5DTile1DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
					}
				}
			}
		}
	}
}

TEST(Parallelize5DTile1D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_1d_t>(Increment5DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile1DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile1DRangeJ + j) * kParallelize5DTile1DRangeK + k) * kParallelize5DTile1DRangeL + l) * kParallelize5DTile1DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
					}
				}
			}
		}
	}
}

TEST(Parallelize5DTile1D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations5D; iteration++) {
		pthreadpool_parallelize_5d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_5d_tile_1d_t>(Increment5DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
			kParallelize5DTile1DTileM,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize5DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile1DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile1DRangeJ + j) * kParallelize5DTile1DRangeK + k) * kParallelize5DTile1DRangeL + l) * kParallelize5DTile1DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations5D)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
							<< "(expected: " << kIncrementIterations5D << ")";
					}
				}
			}
		}
	}
}

TEST(Parallelize5DTile1D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations5D; iteration++) {
		pthreadpool_parallelize_5d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_5d_tile_1d_t>(Increment5DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
			kParallelize5DTile1DTileM,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize5DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile1DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile1DRangeJ + j) * kParallelize5DTile1DRangeK + k) * kParallelize5DTile1DRangeL + l) * kParallelize5DTile1DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations5D)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
							<< "(expected: " << kIncrementIterations5D << ")";
					}
				}
			}
		}
	}
}

static void IncrementSame5DTile1D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t tile_m) {
	for (size_t m = start_m; m < start_m + tile_m; m++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize5DTile1D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_1d_t>(IncrementSame5DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM);
}

static void WorkImbalance5DTile1D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t tile_m) {
	num_processed_items->fetch_add(tile_m, std::memory_order_relaxed);
	if (i == 0 && j == 0 && k == 0 && l == 0 && start_m == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize5DTile1D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_1d_t>(WorkImbalance5DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM);
}

static void ComputeNothing5DTile2D(void*, size_t, size_t, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize5DTile2D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_2d(threadpool.get(),
		ComputeNothing5DTile2D,
		nullptr,
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);
}

TEST(Parallelize5DTile2D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		ComputeNothing5DTile2D,
		nullptr,
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);
}

static void CheckBounds5DTile2D(void*, size_t i, size_t j, size_t k, size_t start_l, size_t start_m, size_t tile_l, size_t tile_m) {
	EXPECT_LT(i, kParallelize5DTile2DRangeI);
	EXPECT_LT(j, kParallelize5DTile2DRangeJ);
	EXPECT_LT(k, kParallelize5DTile2DRangeK);
	EXPECT_LT(start_l, kParallelize5DTile2DRangeL);
	EXPECT_LT(start_m, kParallelize5DTile2DRangeM);
	EXPECT_LE(start_l + tile_l, kParallelize5DTile2DRangeL);
	EXPECT_LE(start_m + tile_m, kParallelize5DTile2DRangeM);
}

TEST(Parallelize5DTile2D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		CheckBounds5DTile2D,
		nullptr,
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);
}

TEST(Parallelize5DTile2D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		CheckBounds5DTile2D,
		nullptr,
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);
}

static void CheckTiling5DTile2D(void*, size_t i, size_t j, size_t k, size_t start_l, size_t start_m, size_t tile_l, size_t tile_m) {
	EXPECT_GT(tile_l, 0);
	EXPECT_LE(tile_l, kParallelize5DTile2DTileL);
	EXPECT_EQ(start_l % kParallelize5DTile2DTileL, 0);
	EXPECT_EQ(tile_l, std::min<size_t>(kParallelize5DTile2DTileL, kParallelize5DTile2DRangeL - start_l));

	EXPECT_GT(tile_m, 0);
	EXPECT_LE(tile_m, kParallelize5DTile2DTileM);
	EXPECT_EQ(start_m % kParallelize5DTile2DTileM, 0);
	EXPECT_EQ(tile_m, std::min<size_t>(kParallelize5DTile2DTileM, kParallelize5DTile2DRangeM - start_m));
}

TEST(Parallelize5DTile2D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		CheckTiling5DTile2D,
		nullptr,
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);
}

TEST(Parallelize5DTile2D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		CheckTiling5DTile2D,
		nullptr,
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);
}

static void SetTrue5DTile2D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t k, size_t start_l, size_t start_m, size_t tile_l, size_t tile_m) {
	for (size_t l = start_l; l < start_l + tile_l; l++) {
		for (size_t m = start_m; m < start_m + tile_m; m++) {
			const size_t linear_idx = (((i * kParallelize5DTile2DRangeJ + j) * kParallelize5DTile2DRangeK + k) * kParallelize5DTile2DRangeL + l) * kParallelize5DTile2DRangeM + m;
			processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize5DTile2D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_2d_t>(SetTrue5DTile2D),
		static_cast<void*>(indicators.data()),
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile2DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile2DRangeJ + j) * kParallelize5DTile2DRangeK + k) * kParallelize5DTile2DRangeL + l) * kParallelize5DTile2DRangeM + m;
						EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") not processed";
					}
				}
			}
		}
	}
}

TEST(Parallelize5DTile2D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_2d_t>(SetTrue5DTile2D),
		static_cast<void*>(indicators.data()),
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile2DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile2DRangeJ + j) * kParallelize5DTile2DRangeK + k) * kParallelize5DTile2DRangeL + l) * kParallelize5DTile2DRangeM + m;
						EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") not processed";
					}
				}
			}
		}
	}
}

static void Increment5DTile2D(std::atomic_int* processed_counters, size_t i, size_t j, size_t k, size_t start_l, size_t start_m, size_t tile_l, size_t tile_m) {
	for (size_t l = start_l; l < start_l + tile_l; l++) {
		for (size_t m = start_m; m < start_m + tile_m; m++) {
			const size_t linear_idx = (((i * kParallelize5DTile2DRangeJ + j) * kParallelize5DTile2DRangeK + k) * kParallelize5DTile2DRangeL + l) * kParallelize5DTile2DRangeM + m;
			processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize5DTile2D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_2d_t>(Increment5DTile2D),
		static_cast<void*>(counters.data()),
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile2DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile2DRangeJ + j) * kParallelize5DTile2DRangeK + k) * kParallelize5DTile2DRangeL + l) * kParallelize5DTile2DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
					}
				}
			}
		}
	}
}

TEST(Parallelize5DTile2D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_2d_t>(Increment5DTile2D),
		static_cast<void*>(counters.data()),
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize5DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile2DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile2DRangeJ + j) * kParallelize5DTile2DRangeK + k) * kParallelize5DTile2DRangeL + l) * kParallelize5DTile2DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
					}
				}
			}
		}
	}
}

TEST(Parallelize5DTile2D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations5D; iteration++) {
		pthreadpool_parallelize_5d_tile_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_5d_tile_2d_t>(Increment5DTile2D),
			static_cast<void*>(counters.data()),
			kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
			kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize5DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile2DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile2DRangeJ + j) * kParallelize5DTile2DRangeK + k) * kParallelize5DTile2DRangeL + l) * kParallelize5DTile2DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations5D)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
							<< "(expected: " << kIncrementIterations5D << ")";
					}
				}
			}
		}
	}
}

TEST(Parallelize5DTile2D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations5D; iteration++) {
		pthreadpool_parallelize_5d_tile_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_5d_tile_2d_t>(Increment5DTile2D),
			static_cast<void*>(counters.data()),
			kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
			kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize5DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize5DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize5DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize5DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize5DTile2DRangeM; m++) {
						const size_t linear_idx = (((i * kParallelize5DTile2DRangeJ + j) * kParallelize5DTile2DRangeK + k) * kParallelize5DTile2DRangeL + l) * kParallelize5DTile2DRangeM + m;
						EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations5D)
							<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ") was processed "
							<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
							<< "(expected: " << kIncrementIterations5D << ")";
					}
				}
			}
		}
	}
}

static void IncrementSame5DTile2D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t start_l, size_t start_m, size_t tile_l, size_t tile_m) {
	for (size_t l = start_l; l < start_l + tile_l; l++) {
		for (size_t m = start_m; m < start_m + tile_m; m++) {
			num_processed_items->fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize5DTile2D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_2d_t>(IncrementSame5DTile2D),
		static_cast<void*>(&num_processed_items),
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM);
}

static void WorkImbalance5DTile2D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t start_l, size_t start_m, size_t tile_l, size_t tile_m) {
	num_processed_items->fetch_add(tile_l * tile_m, std::memory_order_relaxed);
	if (i == 0 && j == 0 && k == 0 && start_l == 0 && start_m == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize5DTile2D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_5d_tile_2d_t>(WorkImbalance5DTile2D),
		static_cast<void*>(&num_processed_items),
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM);
}

static void ComputeNothing6D(void*, size_t, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize6D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d(threadpool.get(),
		ComputeNothing6D,
		nullptr,
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
		0 /* flags */);
}

TEST(Parallelize6D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d(
		threadpool.get(),
		ComputeNothing6D,
		nullptr,
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
		0 /* flags */);
}

static void CheckBounds6D(void*, size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) {
	EXPECT_LT(i, kParallelize6DRangeI);
	EXPECT_LT(j, kParallelize6DRangeJ);
	EXPECT_LT(k, kParallelize6DRangeK);
	EXPECT_LT(l, kParallelize6DRangeL);
	EXPECT_LT(m, kParallelize6DRangeM);
	EXPECT_LT(n, kParallelize6DRangeN);
}

TEST(Parallelize6D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d(
		threadpool.get(),
		CheckBounds6D,
		nullptr,
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
		0 /* flags */);
}

TEST(Parallelize6D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d(
		threadpool.get(),
		CheckBounds6D,
		nullptr,
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
		0 /* flags */);
}

static void SetTrue6D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) {
	const size_t linear_idx = ((((i * kParallelize6DRangeJ + j) * kParallelize6DRangeK + k) * kParallelize6DRangeL + l) * kParallelize6DRangeM + m) * kParallelize6DRangeN + n;
	processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
}

TEST(Parallelize6D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_t>(SetTrue6D),
		static_cast<void*>(indicators.data()),
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DRangeJ + j) * kParallelize6DRangeK + k) * kParallelize6DRangeL + l) * kParallelize6DRangeM + m) * kParallelize6DRangeN + n;
							EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") not processed";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_t>(SetTrue6D),
		static_cast<void*>(indicators.data()),
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DRangeJ + j) * kParallelize6DRangeK + k) * kParallelize6DRangeL + l) * kParallelize6DRangeM + m) * kParallelize6DRangeN + n;
							EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") not processed";
						}
					}
				}
			}
		}
	}
}

static void Increment6D(std::atomic_int* processed_counters, size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) {
	const size_t linear_idx = ((((i * kParallelize6DRangeJ + j) * kParallelize6DRangeK + k) * kParallelize6DRangeL + l) * kParallelize6DRangeM + m) * kParallelize6DRangeN + n;
	processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize6D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_t>(Increment6D),
		static_cast<void*>(counters.data()),
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DRangeJ + j) * kParallelize6DRangeK + k) * kParallelize6DRangeL + l) * kParallelize6DRangeM + m) * kParallelize6DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_t>(Increment6D),
		static_cast<void*>(counters.data()),
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DRangeJ + j) * kParallelize6DRangeK + k) * kParallelize6DRangeL + l) * kParallelize6DRangeM + m) * kParallelize6DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations6D; iteration++) {
		pthreadpool_parallelize_6d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_6d_t>(Increment6D),
			static_cast<void*>(counters.data()),
			kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
				0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize6DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DRangeJ + j) * kParallelize6DRangeK + k) * kParallelize6DRangeL + l) * kParallelize6DRangeM + m) * kParallelize6DRangeN;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations6D)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
								<< "(expected: " << kIncrementIterations6D << ")";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations6D; iteration++) {
		pthreadpool_parallelize_6d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_6d_t>(Increment6D),
			static_cast<void*>(counters.data()),
			kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
				0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize6DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DRangeJ + j) * kParallelize6DRangeK + k) * kParallelize6DRangeL + l) * kParallelize6DRangeM + m) * kParallelize6DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations6D)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
								<< "(expected: " << kIncrementIterations6D << ")";
						}
					}
				}
			}
		}
	}
}

static void IncrementSame6D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
}

TEST(Parallelize6D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_t>(IncrementSame6D),
		static_cast<void*>(&num_processed_items),
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN);
}

static void WorkImbalance6D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) {
	num_processed_items->fetch_add(1, std::memory_order_relaxed);
	if (i == 0 && j == 0 && k == 0 && l == 0 && m == 0 && n == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize6D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_t>(WorkImbalance6D),
		static_cast<void*>(&num_processed_items),
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN);
}

static void ComputeNothing6DTile1D(void*, size_t, size_t, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize6DTile1D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_1d(threadpool.get(),
		ComputeNothing6DTile1D,
		nullptr,
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);
}

TEST(Parallelize6DTile1D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		ComputeNothing6DTile1D,
		nullptr,
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);
}

static void CheckBounds6DTile1D(void*, size_t i, size_t j, size_t k, size_t l, size_t m, size_t start_n, size_t tile_n) {
	EXPECT_LT(i, kParallelize6DTile1DRangeI);
	EXPECT_LT(j, kParallelize6DTile1DRangeJ);
	EXPECT_LT(k, kParallelize6DTile1DRangeK);
	EXPECT_LT(l, kParallelize6DTile1DRangeL);
	EXPECT_LT(m, kParallelize6DTile1DRangeM);
	EXPECT_LT(start_n, kParallelize6DTile1DRangeN);
	EXPECT_LE(start_n + tile_n, kParallelize6DTile1DRangeN);
}

TEST(Parallelize6DTile1D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		CheckBounds6DTile1D,
		nullptr,
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);
}

TEST(Parallelize6DTile1D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		CheckBounds6DTile1D,
		nullptr,
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);
}

static void CheckTiling6DTile1D(void*, size_t i, size_t j, size_t k, size_t l, size_t m, size_t start_n, size_t tile_n) {
	EXPECT_GT(tile_n, 0);
	EXPECT_LE(tile_n, kParallelize6DTile1DTileN);
	EXPECT_EQ(start_n % kParallelize6DTile1DTileN, 0);
	EXPECT_EQ(tile_n, std::min<size_t>(kParallelize6DTile1DTileN, kParallelize6DTile1DRangeN - start_n));
}

TEST(Parallelize6DTile1D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		CheckTiling6DTile1D,
		nullptr,
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);
}

TEST(Parallelize6DTile1D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		CheckTiling6DTile1D,
		nullptr,
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);
}

static void SetTrue6DTile1D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t k, size_t l, size_t m, size_t start_n, size_t tile_n) {
	for (size_t n = start_n; n < start_n + tile_n; n++) {
		const size_t linear_idx = ((((i * kParallelize6DTile1DRangeJ + j) * kParallelize6DTile1DRangeK + k) * kParallelize6DTile1DRangeL + l) * kParallelize6DTile1DRangeM + m) * kParallelize6DTile1DRangeN + n;
		processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
	}
}

TEST(Parallelize6DTile1D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_1d_t>(SetTrue6DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile1DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile1DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile1DRangeJ + j) * kParallelize6DTile1DRangeK + k) * kParallelize6DTile1DRangeL + l) * kParallelize6DTile1DRangeM + m) * kParallelize6DTile1DRangeN + n;
							EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") not processed";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6DTile1D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_1d_t>(SetTrue6DTile1D),
		static_cast<void*>(indicators.data()),
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile1DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile1DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile1DRangeJ + j) * kParallelize6DTile1DRangeK + k) * kParallelize6DTile1DRangeL + l) * kParallelize6DTile1DRangeM + m) * kParallelize6DTile1DRangeN + n;
							EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") not processed";
						}
					}
				}
			}
		}
	}
}

static void Increment6DTile1D(std::atomic_int* processed_counters, size_t i, size_t j, size_t k, size_t l, size_t m, size_t start_n, size_t tile_n) {
	for (size_t n = start_n; n < start_n + tile_n; n++) {
		const size_t linear_idx = ((((i * kParallelize6DTile1DRangeJ + j) * kParallelize6DTile1DRangeK + k) * kParallelize6DTile1DRangeL + l) * kParallelize6DTile1DRangeM + m) * kParallelize6DTile1DRangeN + n;
		processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize6DTile1D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_1d_t>(Increment6DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile1DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile1DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile1DRangeJ + j) * kParallelize6DTile1DRangeK + k) * kParallelize6DTile1DRangeL + l) * kParallelize6DTile1DRangeM + m) * kParallelize6DTile1DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6DTile1D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_1d_t>(Increment6DTile1D),
		static_cast<void*>(counters.data()),
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile1DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile1DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile1DRangeJ + j) * kParallelize6DTile1DRangeK + k) * kParallelize6DTile1DRangeL + l) * kParallelize6DTile1DRangeM + m) * kParallelize6DTile1DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6DTile1D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations6D; iteration++) {
		pthreadpool_parallelize_6d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_6d_tile_1d_t>(Increment6DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
			kParallelize6DTile1DTileN,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize6DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile1DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile1DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile1DRangeJ + j) * kParallelize6DTile1DRangeK + k) * kParallelize6DTile1DRangeL + l) * kParallelize6DTile1DRangeM + m) * kParallelize6DTile1DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations6D)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
								<< "(expected: " << kIncrementIterations6D << ")";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6DTile1D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations6D; iteration++) {
		pthreadpool_parallelize_6d_tile_1d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_6d_tile_1d_t>(Increment6DTile1D),
			static_cast<void*>(counters.data()),
			kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
			kParallelize6DTile1DTileN,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize6DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile1DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile1DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile1DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile1DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile1DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile1DRangeJ + j) * kParallelize6DTile1DRangeK + k) * kParallelize6DTile1DRangeL + l) * kParallelize6DTile1DRangeM + m) * kParallelize6DTile1DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations6D)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
								<< "(expected: " << kIncrementIterations6D << ")";
						}
					}
				}
			}
		}
	}
}

static void IncrementSame6DTile1D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l, size_t m, size_t start_n, size_t tile_n) {
	for (size_t n = start_n; n < start_n + tile_n; n++) {
		num_processed_items->fetch_add(1, std::memory_order_relaxed);
	}
}

TEST(Parallelize6DTile1D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_1d_t>(IncrementSame6DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN);
}

static void WorkImbalance6DTile1D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l, size_t m, size_t start_n, size_t tile_n) {
	num_processed_items->fetch_add(tile_n, std::memory_order_relaxed);
	if (i == 0 && j == 0 && k == 0 && l == 0 && m == 0 && start_n == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize6DTile1D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_1d_t>(WorkImbalance6DTile1D),
		static_cast<void*>(&num_processed_items),
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN);
}

static void ComputeNothing6DTile2D(void*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t) {
}

TEST(Parallelize6DTile2D, SingleThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_2d(threadpool.get(),
		ComputeNothing6DTile2D,
		nullptr,
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);
}

TEST(Parallelize6DTile2D, MultiThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		ComputeNothing6DTile2D,
		nullptr,
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);
}

static void CheckBounds6DTile2D(void*, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t start_n, size_t tile_m, size_t tile_n) {
	EXPECT_LT(i, kParallelize6DTile2DRangeI);
	EXPECT_LT(j, kParallelize6DTile2DRangeJ);
	EXPECT_LT(k, kParallelize6DTile2DRangeK);
	EXPECT_LT(l, kParallelize6DTile2DRangeL);
	EXPECT_LT(start_m, kParallelize6DTile2DRangeM);
	EXPECT_LT(start_n, kParallelize6DTile2DRangeN);
	EXPECT_LE(start_m + tile_m, kParallelize6DTile2DRangeM);
	EXPECT_LE(start_n + tile_n, kParallelize6DTile2DRangeN);
}

TEST(Parallelize6DTile2D, SingleThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		CheckBounds6DTile2D,
		nullptr,
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);
}

TEST(Parallelize6DTile2D, MultiThreadPoolAllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		CheckBounds6DTile2D,
		nullptr,
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);
}

static void CheckTiling6DTile2D(void*, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t start_n, size_t tile_m, size_t tile_n) {
	EXPECT_GT(tile_m, 0);
	EXPECT_LE(tile_m, kParallelize6DTile2DTileM);
	EXPECT_EQ(start_m % kParallelize6DTile2DTileM, 0);
	EXPECT_EQ(tile_m, std::min<size_t>(kParallelize6DTile2DTileM, kParallelize6DTile2DRangeM - start_m));

	EXPECT_GT(tile_n, 0);
	EXPECT_LE(tile_n, kParallelize6DTile2DTileN);
	EXPECT_EQ(start_n % kParallelize6DTile2DTileN, 0);
	EXPECT_EQ(tile_n, std::min<size_t>(kParallelize6DTile2DTileN, kParallelize6DTile2DRangeN - start_n));
}

TEST(Parallelize6DTile2D, SingleThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		CheckTiling6DTile2D,
		nullptr,
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);
}

TEST(Parallelize6DTile2D, MultiThreadPoolUniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		CheckTiling6DTile2D,
		nullptr,
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);
}

static void SetTrue6DTile2D(std::atomic_bool* processed_indicators, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t start_n, size_t tile_m, size_t tile_n) {
	for (size_t m = start_m; m < start_m + tile_m; m++) {
		for (size_t n = start_n; n < start_n + tile_n; n++) {
			const size_t linear_idx = ((((i * kParallelize6DTile2DRangeJ + j) * kParallelize6DTile2DRangeK + k) * kParallelize6DTile2DRangeL + l) * kParallelize6DTile2DRangeM + m) * kParallelize6DTile2DRangeN + n;
			processed_indicators[linear_idx].store(true, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize6DTile2D, SingleThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_2d_t>(SetTrue6DTile2D),
		static_cast<void*>(indicators.data()),
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile2DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile2DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile2DRangeJ + j) * kParallelize6DTile2DRangeK + k) * kParallelize6DTile2DRangeL + l) * kParallelize6DTile2DRangeM + m) * kParallelize6DTile2DRangeN + n;
							EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") not processed";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6DTile2D, MultiThreadPoolAllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_2d_t>(SetTrue6DTile2D),
		static_cast<void*>(indicators.data()),
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile2DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile2DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile2DRangeJ + j) * kParallelize6DTile2DRangeK + k) * kParallelize6DTile2DRangeL + l) * kParallelize6DTile2DRangeM + m) * kParallelize6DTile2DRangeN + n;
							EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") not processed";
						}
					}
				}
			}
		}
	}
}

static void Increment6DTile2D(std::atomic_int* processed_counters, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t start_n, size_t tile_m, size_t tile_n) {
	for (size_t m = start_m; m < start_m + tile_m; m++) {
		for (size_t n = start_n; n < start_n + tile_n; n++) {
			const size_t linear_idx = ((((i * kParallelize6DTile2DRangeJ + j) * kParallelize6DTile2DRangeK + k) * kParallelize6DTile2DRangeL + l) * kParallelize6DTile2DRangeM + m) * kParallelize6DTile2DRangeN + n;
			processed_counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize6DTile2D, SingleThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_2d_t>(Increment6DTile2D),
		static_cast<void*>(counters.data()),
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile2DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile2DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile2DRangeJ + j) * kParallelize6DTile2DRangeK + k) * kParallelize6DTile2DRangeL + l) * kParallelize6DTile2DRangeM + m) * kParallelize6DTile2DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6DTile2D, MultiThreadPoolEachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_2d_t>(Increment6DTile2D),
		static_cast<void*>(counters.data()),
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);

	for (size_t i = 0; i < kParallelize6DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile2DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile2DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile2DRangeJ + j) * kParallelize6DTile2DRangeK + k) * kParallelize6DTile2DRangeL + l) * kParallelize6DTile2DRangeM + m) * kParallelize6DTile2DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6DTile2D, SingleThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	for (size_t iteration = 0; iteration < kIncrementIterations6D; iteration++) {
		pthreadpool_parallelize_6d_tile_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_6d_tile_2d_t>(Increment6DTile2D),
			static_cast<void*>(counters.data()),
			kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
			kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize6DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile2DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile2DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile2DRangeJ + j) * kParallelize6DTile2DRangeK + k) * kParallelize6DTile2DRangeL + l) * kParallelize6DTile2DRangeM + m) * kParallelize6DTile2DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations6D)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
								<< "(expected: " << kIncrementIterations6D << ")";
						}
					}
				}
			}
		}
	}
}

TEST(Parallelize6DTile2D, MultiThreadPoolEachItemProcessedMultipleTimes) {
	std::vector<std::atomic_int> counters(kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	for (size_t iteration = 0; iteration < kIncrementIterations6D; iteration++) {
		pthreadpool_parallelize_6d_tile_2d(
			threadpool.get(),
			reinterpret_cast<pthreadpool_task_6d_tile_2d_t>(Increment6DTile2D),
			static_cast<void*>(counters.data()),
			kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
			kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
			0 /* flags */);
	}

	for (size_t i = 0; i < kParallelize6DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize6DTile2DRangeJ; j++) {
			for (size_t k = 0; k < kParallelize6DTile2DRangeK; k++) {
				for (size_t l = 0; l < kParallelize6DTile2DRangeL; l++) {
					for (size_t m = 0; m < kParallelize6DTile2DRangeM; m++) {
						for (size_t n = 0; n < kParallelize6DTile2DRangeN; n++) {
							const size_t linear_idx = ((((i * kParallelize6DTile2DRangeJ + j) * kParallelize6DTile2DRangeK + k) * kParallelize6DTile2DRangeL + l) * kParallelize6DTile2DRangeM + m) * kParallelize6DTile2DRangeN + n;
							EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), kIncrementIterations6D)
								<< "Element (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ") was processed "
								<< counters[linear_idx].load(std::memory_order_relaxed) << " times "
								<< "(expected: " << kIncrementIterations6D << ")";
						}
					}
				}
			}
		}
	}
}

static void IncrementSame6DTile2D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t start_n, size_t tile_m, size_t tile_n) {
	for (size_t m = start_m; m < start_m + tile_m; m++) {
		for (size_t n = start_n; n < start_n + tile_n; n++) {
			num_processed_items->fetch_add(1, std::memory_order_relaxed);
		}
	}
}

TEST(Parallelize6DTile2D, MultiThreadPoolHighContention) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_2d_t>(IncrementSame6DTile2D),
		static_cast<void*>(&num_processed_items),
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN);
}

static void WorkImbalance6DTile2D(std::atomic_int* num_processed_items, size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t start_n, size_t tile_m, size_t tile_n) {
	num_processed_items->fetch_add(tile_m * tile_n, std::memory_order_relaxed);
	if (i == 0 && j == 0 && k == 0 && l == 0 && start_m == 0 && start_n == 0) {
		/* Spin-wait until all items are computed */
		while (num_processed_items->load(std::memory_order_relaxed) != kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN) {
			std::atomic_thread_fence(std::memory_order_acquire);
		}
	}
}

TEST(Parallelize6DTile2D, MultiThreadPoolWorkStealing) {
	std::atomic_int num_processed_items = ATOMIC_VAR_INIT(0);

	auto_pthreadpool_t threadpool(pthreadpool_create(0), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	if (pthreadpool_get_threads_count(threadpool.get()) <= 1) {
		GTEST_SKIP();
	}

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		reinterpret_cast<pthreadpool_task_6d_tile_2d_t>(WorkImbalance6DTile2D),
		static_cast<void*>(&num_processed_items),
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN,
		0 /* flags */);
	EXPECT_EQ(num_processed_items.load(std::memory_order_relaxed), kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN);
}
