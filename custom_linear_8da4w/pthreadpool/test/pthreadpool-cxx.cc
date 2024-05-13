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


TEST(Parallelize1D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d(
		threadpool.get(),
		[](size_t) { },
		kParallelize1DRange);
}

TEST(Parallelize1D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d(
		threadpool.get(),
		[](size_t i) {
			EXPECT_LT(i, kParallelize1DRange);
		},
		kParallelize1DRange);
}

TEST(Parallelize1D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d(
		threadpool.get(),
		[&indicators](size_t i) {
			indicators[i].store(true, std::memory_order_relaxed);
		},
		kParallelize1DRange);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_TRUE(indicators[i].load(std::memory_order_relaxed))
			<< "Element " << i << " not processed";
	}
}


TEST(Parallelize1D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d(
		threadpool.get(),
		[&counters](size_t i) {
			counters[i].fetch_add(1, std::memory_order_relaxed);
		},
		kParallelize1DRange);

	for (size_t i = 0; i < kParallelize1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), 1)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times (expected: 1)";
	}
}

TEST(Parallelize1DTile1D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		[](size_t, size_t) { },
		kParallelize1DTile1DRange, kParallelize1DTile1DTile);
}

TEST(Parallelize1DTile1D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		[](size_t start_i, size_t tile_i) {
			EXPECT_LT(start_i, kParallelize1DTile1DRange);
			EXPECT_LE(start_i + tile_i, kParallelize1DTile1DRange);
		},
		kParallelize1DTile1DRange, kParallelize1DTile1DTile);
}

TEST(Parallelize1DTile1D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		[](size_t start_i, size_t tile_i) {
			EXPECT_GT(tile_i, 0);
			EXPECT_LE(tile_i, kParallelize1DTile1DTile);
			EXPECT_EQ(start_i % kParallelize1DTile1DTile, 0);
			EXPECT_EQ(tile_i, std::min<size_t>(kParallelize1DTile1DTile, kParallelize1DTile1DRange - start_i));
		},
		kParallelize1DTile1DRange, kParallelize1DTile1DTile);
}

TEST(Parallelize1DTile1D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize1DTile1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		[&indicators](size_t start_i, size_t tile_i) {
			for (size_t i = start_i; i < start_i + tile_i; i++) {
				indicators[i].store(true, std::memory_order_relaxed);
			}
		},
		kParallelize1DTile1DRange, kParallelize1DTile1DTile);

	for (size_t i = 0; i < kParallelize1DTile1DRange; i++) {
		EXPECT_TRUE(indicators[i].load(std::memory_order_relaxed))
			<< "Element " << i << " not processed";
	}
}

TEST(Parallelize1DTile1D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize1DTile1DRange);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_1d_tile_1d(
		threadpool.get(),
		[&counters](size_t start_i, size_t tile_i) {
			for (size_t i = start_i; i < start_i + tile_i; i++) {
				counters[i].fetch_add(1, std::memory_order_relaxed);
			}
		},
		kParallelize1DTile1DRange, kParallelize1DTile1DTile);

	for (size_t i = 0; i < kParallelize1DTile1DRange; i++) {
		EXPECT_EQ(counters[i].load(std::memory_order_relaxed), 1)
			<< "Element " << i << " was processed " << counters[i].load(std::memory_order_relaxed) << " times (expected: 1)";
	}
}

TEST(Parallelize2D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d(
		threadpool.get(),
		[](size_t, size_t) { },
		kParallelize2DRangeI, kParallelize2DRangeJ);
}

TEST(Parallelize2D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d(
		threadpool.get(),
		[](size_t i, size_t j) {
			EXPECT_LT(i, kParallelize2DRangeI);
			EXPECT_LT(j, kParallelize2DRangeJ);
		},
		kParallelize2DRangeI, kParallelize2DRangeJ);
}

TEST(Parallelize2D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d(
		threadpool.get(),
		[&indicators](size_t i, size_t j) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			indicators[linear_idx].store(true, std::memory_order_relaxed);
		},
		kParallelize2DRangeI, kParallelize2DRangeJ);

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

TEST(Parallelize2D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DRangeI * kParallelize2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d(
		threadpool.get(),
		[&counters](size_t i, size_t j) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		},
		kParallelize2DRangeI, kParallelize2DRangeJ);

	for (size_t i = 0; i < kParallelize2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile1D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		[](size_t, size_t, size_t) { },
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ);
}

TEST(Parallelize2DTile1D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		[](size_t i, size_t start_j, size_t tile_j) {
			EXPECT_LT(i, kParallelize2DTile1DRangeI);
			EXPECT_LT(start_j, kParallelize2DTile1DRangeJ);
			EXPECT_LE(start_j + tile_j, kParallelize2DTile1DRangeJ);
		},
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ);
}

TEST(Parallelize2DTile1D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		[](size_t i, size_t start_j, size_t tile_j) {
			EXPECT_GT(tile_j, 0);
			EXPECT_LE(tile_j, kParallelize2DTile1DTileJ);
			EXPECT_EQ(start_j % kParallelize2DTile1DTileJ, 0);
			EXPECT_EQ(tile_j, std::min<size_t>(kParallelize2DTile1DTileJ, kParallelize2DTile1DRangeJ - start_j));
		},
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ);
}

TEST(Parallelize2DTile1D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		[&indicators](size_t i, size_t start_j, size_t tile_j) {
			for (size_t j = start_j; j < start_j + tile_j; j++) {
				const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
				indicators[linear_idx].store(true, std::memory_order_relaxed);
			}
		},
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

TEST(Parallelize2DTile1D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile1DRangeI * kParallelize2DTile1DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_1d(
		threadpool.get(),
		[&counters](size_t i, size_t start_j, size_t tile_j) {
			for (size_t j = start_j; j < start_j + tile_j; j++) {
				const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
				counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
			}
		},
		kParallelize2DTile1DRangeI, kParallelize2DTile1DRangeJ, kParallelize2DTile1DTileJ);

	for (size_t i = 0; i < kParallelize2DTile1DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile1DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile1DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize2DTile2D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		[](size_t, size_t, size_t, size_t) { },
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ);
}

TEST(Parallelize2DTile2D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		[](size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
			EXPECT_LT(start_i, kParallelize2DTile2DRangeI);
			EXPECT_LT(start_j, kParallelize2DTile2DRangeJ);
			EXPECT_LE(start_i + tile_i, kParallelize2DTile2DRangeI);
			EXPECT_LE(start_j + tile_j, kParallelize2DTile2DRangeJ);
		},
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ);
}

TEST(Parallelize2DTile2D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		[](size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
			EXPECT_GT(tile_i, 0);
			EXPECT_LE(tile_i, kParallelize2DTile2DTileI);
			EXPECT_EQ(start_i % kParallelize2DTile2DTileI, 0);
			EXPECT_EQ(tile_i, std::min<size_t>(kParallelize2DTile2DTileI, kParallelize2DTile2DRangeI - start_i));

			EXPECT_GT(tile_j, 0);
			EXPECT_LE(tile_j, kParallelize2DTile2DTileJ);
			EXPECT_EQ(start_j % kParallelize2DTile2DTileJ, 0);
			EXPECT_EQ(tile_j, std::min<size_t>(kParallelize2DTile2DTileJ, kParallelize2DTile2DRangeJ - start_j));
		},
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ);
}

TEST(Parallelize2DTile2D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		[&indicators](size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
			for (size_t i = start_i; i < start_i + tile_i; i++) {
				for (size_t j = start_j; j < start_j + tile_j; j++) {
					const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
					indicators[linear_idx].store(true, std::memory_order_relaxed);
				}
			}
		},
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ);

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_TRUE(indicators[linear_idx].load(std::memory_order_relaxed))
				<< "Element (" << i << ", " << j << ") not processed";
		}
	}
}

TEST(Parallelize2DTile2D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize2DTile2DRangeI * kParallelize2DTile2DRangeJ);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_2d_tile_2d(
		threadpool.get(),
		[&counters](size_t start_i, size_t start_j, size_t tile_i, size_t tile_j) {
			for (size_t i = start_i; i < start_i + tile_i; i++) {
				for (size_t j = start_j; j < start_j + tile_j; j++) {
					const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
					counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
				}
			}
		},
		kParallelize2DTile2DRangeI, kParallelize2DTile2DRangeJ,
		kParallelize2DTile2DTileI, kParallelize2DTile2DTileJ);

	for (size_t i = 0; i < kParallelize2DTile2DRangeI; i++) {
		for (size_t j = 0; j < kParallelize2DTile2DRangeJ; j++) {
			const size_t linear_idx = i * kParallelize2DTile2DRangeJ + j;
			EXPECT_EQ(counters[linear_idx].load(std::memory_order_relaxed), 1)
				<< "Element (" << i << ", " << j << ") was processed "
				<< counters[linear_idx].load(std::memory_order_relaxed) << " times (expected: 1)";
		}
	}
}

TEST(Parallelize3D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d(
		threadpool.get(),
		[](size_t, size_t, size_t) { },
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK);
}

TEST(Parallelize3D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k) {
			EXPECT_LT(i, kParallelize3DRangeI);
			EXPECT_LT(j, kParallelize3DRangeJ);
			EXPECT_LT(k, kParallelize3DRangeK);
		},
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK);
}

TEST(Parallelize3D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t k) {
			const size_t linear_idx = (i * kParallelize3DRangeJ + j) * kParallelize3DRangeK + k;
			indicators[linear_idx].store(true, std::memory_order_relaxed);
		},
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK);

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

TEST(Parallelize3D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DRangeI * kParallelize3DRangeJ * kParallelize3DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t k) {
			const size_t linear_idx = (i * kParallelize3DRangeJ + j) * kParallelize3DRangeK + k;
			counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		},
		kParallelize3DRangeI, kParallelize3DRangeJ, kParallelize3DRangeK);

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

TEST(Parallelize3DTile1D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		[](size_t, size_t, size_t, size_t) { },
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK);
}

TEST(Parallelize3DTile1D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		[](size_t i, size_t j, size_t start_k, size_t tile_k) {
			EXPECT_LT(i, kParallelize3DTile1DRangeI);
			EXPECT_LT(j, kParallelize3DTile1DRangeJ);
			EXPECT_LT(start_k, kParallelize3DTile1DRangeK);
			EXPECT_LE(start_k + tile_k, kParallelize3DTile1DRangeK);
		},
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK);
}

TEST(Parallelize3DTile1D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		[](size_t i, size_t j, size_t start_k, size_t tile_k) {
			EXPECT_GT(tile_k, 0);
			EXPECT_LE(tile_k, kParallelize3DTile1DTileK);
			EXPECT_EQ(start_k % kParallelize3DTile1DTileK, 0);
			EXPECT_EQ(tile_k, std::min<size_t>(kParallelize3DTile1DTileK, kParallelize3DTile1DRangeK - start_k));
		},
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK);
}

TEST(Parallelize3DTile1D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t start_k, size_t tile_k) {
			for (size_t k = start_k; k < start_k + tile_k; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				indicators[linear_idx].store(true, std::memory_order_relaxed);
			}
		},
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK);

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

TEST(Parallelize3DTile1D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile1DRangeI * kParallelize3DTile1DRangeJ * kParallelize3DTile1DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_1d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t start_k, size_t tile_k) {
			for (size_t k = start_k; k < start_k + tile_k; k++) {
				const size_t linear_idx = (i * kParallelize3DTile1DRangeJ + j) * kParallelize3DTile1DRangeK + k;
				counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
			}
		},
		kParallelize3DTile1DRangeI, kParallelize3DTile1DRangeJ, kParallelize3DTile1DRangeK,
		kParallelize3DTile1DTileK);

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

TEST(Parallelize3DTile2D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		[](size_t, size_t, size_t, size_t, size_t) { },
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK);
}

TEST(Parallelize3DTile2D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		[](size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
			EXPECT_LT(i, kParallelize3DTile2DRangeI);
			EXPECT_LT(start_j, kParallelize3DTile2DRangeJ);
			EXPECT_LT(start_k, kParallelize3DTile2DRangeK);
			EXPECT_LE(start_j + tile_j, kParallelize3DTile2DRangeJ);
			EXPECT_LE(start_k + tile_k, kParallelize3DTile2DRangeK);
		},
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK);
}

TEST(Parallelize3DTile2D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		[](size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
			EXPECT_GT(tile_j, 0);
			EXPECT_LE(tile_j, kParallelize3DTile2DTileJ);
			EXPECT_EQ(start_j % kParallelize3DTile2DTileJ, 0);
			EXPECT_EQ(tile_j, std::min<size_t>(kParallelize3DTile2DTileJ, kParallelize3DTile2DRangeJ - start_j));

			EXPECT_GT(tile_k, 0);
			EXPECT_LE(tile_k, kParallelize3DTile2DTileK);
			EXPECT_EQ(start_k % kParallelize3DTile2DTileK, 0);
			EXPECT_EQ(tile_k, std::min<size_t>(kParallelize3DTile2DTileK, kParallelize3DTile2DRangeK - start_k));
		},
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK);
}

TEST(Parallelize3DTile2D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		[&indicators](size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
			for (size_t j = start_j; j < start_j + tile_j; j++) {
				for (size_t k = start_k; k < start_k + tile_k; k++) {
					const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
					indicators[linear_idx].store(true, std::memory_order_relaxed);
				}
			}
		},
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK);

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

TEST(Parallelize3DTile2D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize3DTile2DRangeI * kParallelize3DTile2DRangeJ * kParallelize3DTile2DRangeK);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_3d_tile_2d(
		threadpool.get(),
		[&counters](size_t i, size_t start_j, size_t start_k, size_t tile_j, size_t tile_k) {
			for (size_t j = start_j; j < start_j + tile_j; j++) {
				for (size_t k = start_k; k < start_k + tile_k; k++) {
					const size_t linear_idx = (i * kParallelize3DTile2DRangeJ + j) * kParallelize3DTile2DRangeK + k;
					counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
				}
			}
		},
		kParallelize3DTile2DRangeI, kParallelize3DTile2DRangeJ, kParallelize3DTile2DRangeK,
		kParallelize3DTile2DTileJ, kParallelize3DTile2DTileK);

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

TEST(Parallelize4D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d(
		threadpool.get(),
		[](size_t, size_t, size_t, size_t) { },
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL);
}

TEST(Parallelize4D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t l) {
			EXPECT_LT(i, kParallelize4DRangeI);
			EXPECT_LT(j, kParallelize4DRangeJ);
			EXPECT_LT(k, kParallelize4DRangeK);
			EXPECT_LT(l, kParallelize4DRangeL);
		},
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL);
}

TEST(Parallelize4D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t k, size_t l) {
			const size_t linear_idx = ((i * kParallelize4DRangeJ + j) * kParallelize4DRangeK + k) * kParallelize4DRangeL + l;
			indicators[linear_idx].store(true, std::memory_order_relaxed);
		},
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL);

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

TEST(Parallelize4D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DRangeI * kParallelize4DRangeJ * kParallelize4DRangeK * kParallelize4DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t k, size_t l) {
			const size_t linear_idx = ((i * kParallelize4DRangeJ + j) * kParallelize4DRangeK + k) * kParallelize4DRangeL + l;
			counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		},
		kParallelize4DRangeI, kParallelize4DRangeJ, kParallelize4DRangeK, kParallelize4DRangeL);

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

TEST(Parallelize4DTile1D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		[](size_t, size_t, size_t, size_t, size_t) { },
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL);
}

TEST(Parallelize4DTile1D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t start_l, size_t tile_l) {
			EXPECT_LT(i, kParallelize4DTile1DRangeI);
			EXPECT_LT(j, kParallelize4DTile1DRangeJ);
			EXPECT_LT(k, kParallelize4DTile1DRangeK);
			EXPECT_LT(start_l, kParallelize4DTile1DRangeL);
			EXPECT_LE(start_l + tile_l, kParallelize4DTile1DRangeL);
		},
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL);
}

TEST(Parallelize4DTile1D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t start_l, size_t tile_l) {
			EXPECT_GT(tile_l, 0);
			EXPECT_LE(tile_l, kParallelize4DTile1DTileL);
			EXPECT_EQ(start_l % kParallelize4DTile1DTileL, 0);
			EXPECT_EQ(tile_l, std::min<size_t>(kParallelize4DTile1DTileL, kParallelize4DTile1DRangeL - start_l));
		},
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL);
}

TEST(Parallelize4DTile1D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t k, size_t start_l, size_t tile_l) {
			for (size_t l = start_l; l < start_l + tile_l; l++) {
				const size_t linear_idx = ((i * kParallelize4DTile1DRangeJ + j) * kParallelize4DTile1DRangeK + k) * kParallelize4DTile1DRangeL + l;
				indicators[linear_idx].store(true, std::memory_order_relaxed);
			}
		},
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL);

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

TEST(Parallelize4DTile1D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DTile1DRangeI * kParallelize4DTile1DRangeJ * kParallelize4DTile1DRangeK * kParallelize4DTile1DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_1d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t k, size_t start_l, size_t tile_l) {
			for (size_t l = start_l; l < start_l + tile_l; l++) {
				const size_t linear_idx = ((i * kParallelize4DTile1DRangeJ + j) * kParallelize4DTile1DRangeK + k) * kParallelize4DTile1DRangeL + l;
				counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
			}
		},
		kParallelize4DTile1DRangeI, kParallelize4DTile1DRangeJ, kParallelize4DTile1DRangeK, kParallelize4DTile1DRangeL,
		kParallelize4DTile1DTileL);

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

TEST(Parallelize4DTile2D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		[](size_t, size_t, size_t, size_t, size_t, size_t) { },
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL);
}

TEST(Parallelize4DTile2D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		[](size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
			EXPECT_LT(i, kParallelize4DTile2DRangeI);
			EXPECT_LT(j, kParallelize4DTile2DRangeJ);
			EXPECT_LT(start_k, kParallelize4DTile2DRangeK);
			EXPECT_LT(start_l, kParallelize4DTile2DRangeL);
			EXPECT_LE(start_k + tile_k, kParallelize4DTile2DRangeK);
			EXPECT_LE(start_l + tile_l, kParallelize4DTile2DRangeL);
		},
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL);
}

TEST(Parallelize4DTile2D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		[](size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
			EXPECT_GT(tile_k, 0);
			EXPECT_LE(tile_k, kParallelize4DTile2DTileK);
			EXPECT_EQ(start_k % kParallelize4DTile2DTileK, 0);
			EXPECT_EQ(tile_k, std::min<size_t>(kParallelize4DTile2DTileK, kParallelize4DTile2DRangeK - start_k));

			EXPECT_GT(tile_l, 0);
			EXPECT_LE(tile_l, kParallelize4DTile2DTileL);
			EXPECT_EQ(start_l % kParallelize4DTile2DTileL, 0);
			EXPECT_EQ(tile_l, std::min<size_t>(kParallelize4DTile2DTileL, kParallelize4DTile2DRangeL - start_l));
		},
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL);
}

TEST(Parallelize4DTile2D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
			for (size_t k = start_k; k < start_k + tile_k; k++) {
				for (size_t l = start_l; l < start_l + tile_l; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					indicators[linear_idx].store(true, std::memory_order_relaxed);
				}
			}
		},
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL);

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

TEST(Parallelize4DTile2D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize4DTile2DRangeI * kParallelize4DTile2DRangeJ * kParallelize4DTile2DRangeK * kParallelize4DTile2DRangeL);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_4d_tile_2d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t start_k, size_t start_l, size_t tile_k, size_t tile_l) {
			for (size_t k = start_k; k < start_k + tile_k; k++) {
				for (size_t l = start_l; l < start_l + tile_l; l++) {
					const size_t linear_idx = ((i * kParallelize4DTile2DRangeJ + j) * kParallelize4DTile2DRangeK + k) * kParallelize4DTile2DRangeL + l;
					counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
				}
			}
		},
		kParallelize4DTile2DRangeI, kParallelize4DTile2DRangeJ, kParallelize4DTile2DRangeK, kParallelize4DTile2DRangeL,
		kParallelize4DTile2DTileK, kParallelize4DTile2DTileL);

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

TEST(Parallelize5D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d(
		threadpool.get(),
		[](size_t, size_t, size_t, size_t, size_t) { },
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM);
}

TEST(Parallelize5D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t l, size_t m) {
			EXPECT_LT(i, kParallelize5DRangeI);
			EXPECT_LT(j, kParallelize5DRangeJ);
			EXPECT_LT(k, kParallelize5DRangeK);
			EXPECT_LT(l, kParallelize5DRangeL);
			EXPECT_LT(m, kParallelize5DRangeM);
		},
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM);
}

TEST(Parallelize5D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t k, size_t l, size_t m) {
			const size_t linear_idx = (((i * kParallelize5DRangeJ + j) * kParallelize5DRangeK + k) * kParallelize5DRangeL + l) * kParallelize5DRangeM + m;
			indicators[linear_idx].store(true, std::memory_order_relaxed);
		},
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM);

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

TEST(Parallelize5D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize5DRangeI * kParallelize5DRangeJ * kParallelize5DRangeK * kParallelize5DRangeL * kParallelize5DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t k, size_t l, size_t m) {
			const size_t linear_idx = (((i * kParallelize5DRangeJ + j) * kParallelize5DRangeK + k) * kParallelize5DRangeL + l) * kParallelize5DRangeM + m;
			counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		},
		kParallelize5DRangeI, kParallelize5DRangeJ, kParallelize5DRangeK, kParallelize5DRangeL, kParallelize5DRangeM);

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

TEST(Parallelize5DTile1D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		[](size_t, size_t, size_t, size_t, size_t, size_t) { },
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM);
}

TEST(Parallelize5DTile1D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t tile_m) {
			EXPECT_LT(i, kParallelize5DTile1DRangeI);
			EXPECT_LT(j, kParallelize5DTile1DRangeJ);
			EXPECT_LT(k, kParallelize5DTile1DRangeK);
			EXPECT_LT(l, kParallelize5DTile1DRangeL);
			EXPECT_LT(start_m, kParallelize5DTile1DRangeM);
			EXPECT_LE(start_m + tile_m, kParallelize5DTile1DRangeM);
		},
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM);
}

TEST(Parallelize5DTile1D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t tile_m) {
			EXPECT_GT(tile_m, 0);
			EXPECT_LE(tile_m, kParallelize5DTile1DTileM);
			EXPECT_EQ(start_m % kParallelize5DTile1DTileM, 0);
			EXPECT_EQ(tile_m, std::min<size_t>(kParallelize5DTile1DTileM, kParallelize5DTile1DRangeM - start_m));
		},
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM);
}

TEST(Parallelize5DTile1D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t tile_m) {
			for (size_t m = start_m; m < start_m + tile_m; m++) {
				const size_t linear_idx = (((i * kParallelize5DTile1DRangeJ + j) * kParallelize5DTile1DRangeK + k) * kParallelize5DTile1DRangeL + l) * kParallelize5DTile1DRangeM + m;
				indicators[linear_idx].store(true, std::memory_order_relaxed);
			}
		},
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM);

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

TEST(Parallelize5DTile1D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize5DTile1DRangeI * kParallelize5DTile1DRangeJ * kParallelize5DTile1DRangeK * kParallelize5DTile1DRangeL * kParallelize5DTile1DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_1d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t tile_m) {
			for (size_t m = start_m; m < start_m + tile_m; m++) {
				const size_t linear_idx = (((i * kParallelize5DTile1DRangeJ + j) * kParallelize5DTile1DRangeK + k) * kParallelize5DTile1DRangeL + l) * kParallelize5DTile1DRangeM + m;
				counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
			}
		},
		kParallelize5DTile1DRangeI, kParallelize5DTile1DRangeJ, kParallelize5DTile1DRangeK, kParallelize5DTile1DRangeL, kParallelize5DTile1DRangeM,
		kParallelize5DTile1DTileM);

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

TEST(Parallelize5DTile2D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		[](size_t, size_t, size_t, size_t, size_t, size_t, size_t) { },
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM);
}

TEST(Parallelize5DTile2D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t start_l, size_t start_m, size_t tile_l, size_t tile_m) {
			EXPECT_LT(i, kParallelize5DTile2DRangeI);
			EXPECT_LT(j, kParallelize5DTile2DRangeJ);
			EXPECT_LT(k, kParallelize5DTile2DRangeK);
			EXPECT_LT(start_l, kParallelize5DTile2DRangeL);
			EXPECT_LT(start_m, kParallelize5DTile2DRangeM);
			EXPECT_LE(start_l + tile_l, kParallelize5DTile2DRangeL);
			EXPECT_LE(start_m + tile_m, kParallelize5DTile2DRangeM);
		},
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM);
}

TEST(Parallelize5DTile2D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t start_l, size_t start_m, size_t tile_l, size_t tile_m) {
			EXPECT_GT(tile_l, 0);
			EXPECT_LE(tile_l, kParallelize5DTile2DTileL);
			EXPECT_EQ(start_l % kParallelize5DTile2DTileL, 0);
			EXPECT_EQ(tile_l, std::min<size_t>(kParallelize5DTile2DTileL, kParallelize5DTile2DRangeL - start_l));

			EXPECT_GT(tile_m, 0);
			EXPECT_LE(tile_m, kParallelize5DTile2DTileM);
			EXPECT_EQ(start_m % kParallelize5DTile2DTileM, 0);
			EXPECT_EQ(tile_m, std::min<size_t>(kParallelize5DTile2DTileM, kParallelize5DTile2DRangeM - start_m));
		},
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM);
}

TEST(Parallelize5DTile2D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t k, size_t start_l, size_t start_m, size_t tile_l, size_t tile_m) {
			for (size_t l = start_l; l < start_l + tile_l; l++) {
				for (size_t m = start_m; m < start_m + tile_m; m++) {
					const size_t linear_idx = (((i * kParallelize5DTile2DRangeJ + j) * kParallelize5DTile2DRangeK + k) * kParallelize5DTile2DRangeL + l) * kParallelize5DTile2DRangeM + m;
					indicators[linear_idx].store(true, std::memory_order_relaxed);
				}
			}
		},
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM);

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

TEST(Parallelize5DTile2D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize5DTile2DRangeI * kParallelize5DTile2DRangeJ * kParallelize5DTile2DRangeK * kParallelize5DTile2DRangeL * kParallelize5DTile2DRangeM);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_5d_tile_2d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t k, size_t start_l, size_t start_m, size_t tile_l, size_t tile_m) {
			for (size_t l = start_l; l < start_l + tile_l; l++) {
				for (size_t m = start_m; m < start_m + tile_m; m++) {
					const size_t linear_idx = (((i * kParallelize5DTile2DRangeJ + j) * kParallelize5DTile2DRangeK + k) * kParallelize5DTile2DRangeL + l) * kParallelize5DTile2DRangeM + m;
					counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
				}
			}
		},
		kParallelize5DTile2DRangeI, kParallelize5DTile2DRangeJ, kParallelize5DTile2DRangeK, kParallelize5DTile2DRangeL, kParallelize5DTile2DRangeM,
		kParallelize5DTile2DTileL, kParallelize5DTile2DTileM);

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

TEST(Parallelize6D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d(threadpool.get(),
		[](size_t, size_t, size_t, size_t, size_t, size_t) { },
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN);
}

TEST(Parallelize6D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) {
			EXPECT_LT(i, kParallelize6DRangeI);
			EXPECT_LT(j, kParallelize6DRangeJ);
			EXPECT_LT(k, kParallelize6DRangeK);
			EXPECT_LT(l, kParallelize6DRangeL);
			EXPECT_LT(m, kParallelize6DRangeM);
			EXPECT_LT(n, kParallelize6DRangeN);
		},
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN);
}

TEST(Parallelize6D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) {
			const size_t linear_idx = ((((i * kParallelize6DRangeJ + j) * kParallelize6DRangeK + k) * kParallelize6DRangeL + l) * kParallelize6DRangeM + m) * kParallelize6DRangeN + n;
			indicators[linear_idx].store(true, std::memory_order_relaxed);
		},
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN);

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

TEST(Parallelize6D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize6DRangeI * kParallelize6DRangeJ * kParallelize6DRangeK * kParallelize6DRangeL * kParallelize6DRangeM * kParallelize6DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) {
			const size_t linear_idx = ((((i * kParallelize6DRangeJ + j) * kParallelize6DRangeK + k) * kParallelize6DRangeL + l) * kParallelize6DRangeM + m) * kParallelize6DRangeN + n;
			counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
		},
		kParallelize6DRangeI, kParallelize6DRangeJ, kParallelize6DRangeK, kParallelize6DRangeL, kParallelize6DRangeM, kParallelize6DRangeN);

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

TEST(Parallelize6DTile1D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_1d(threadpool.get(),
		[](size_t, size_t, size_t, size_t, size_t, size_t, size_t) { },
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN);
}

TEST(Parallelize6DTile1D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t l, size_t m, size_t start_n, size_t tile_n) {
			EXPECT_LT(i, kParallelize6DTile1DRangeI);
			EXPECT_LT(j, kParallelize6DTile1DRangeJ);
			EXPECT_LT(k, kParallelize6DTile1DRangeK);
			EXPECT_LT(l, kParallelize6DTile1DRangeL);
			EXPECT_LT(m, kParallelize6DTile1DRangeM);
			EXPECT_LT(start_n, kParallelize6DTile1DRangeN);
			EXPECT_LE(start_n + tile_n, kParallelize6DTile1DRangeN);
		},
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN);
}

TEST(Parallelize6DTile1D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t l, size_t m, size_t start_n, size_t tile_n) {
			EXPECT_GT(tile_n, 0);
			EXPECT_LE(tile_n, kParallelize6DTile1DTileN);
			EXPECT_EQ(start_n % kParallelize6DTile1DTileN, 0);
			EXPECT_EQ(tile_n, std::min<size_t>(kParallelize6DTile1DTileN, kParallelize6DTile1DRangeN - start_n));
		},
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN);
}

TEST(Parallelize6DTile1D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t k, size_t l, size_t m, size_t start_n, size_t tile_n) {
			for (size_t n = start_n; n < start_n + tile_n; n++) {
				const size_t linear_idx = ((((i * kParallelize6DTile1DRangeJ + j) * kParallelize6DTile1DRangeK + k) * kParallelize6DTile1DRangeL + l) * kParallelize6DTile1DRangeM + m) * kParallelize6DTile1DRangeN + n;
				indicators[linear_idx].store(true, std::memory_order_relaxed);
			}
		},
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN);

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

TEST(Parallelize6DTile1D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize6DTile1DRangeI * kParallelize6DTile1DRangeJ * kParallelize6DTile1DRangeK * kParallelize6DTile1DRangeL * kParallelize6DTile1DRangeM * kParallelize6DTile1DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_1d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t k, size_t l, size_t m, size_t start_n, size_t tile_n) {
			for (size_t n = start_n; n < start_n + tile_n; n++) {
				const size_t linear_idx = ((((i * kParallelize6DTile1DRangeJ + j) * kParallelize6DTile1DRangeK + k) * kParallelize6DTile1DRangeL + l) * kParallelize6DTile1DRangeM + m) * kParallelize6DTile1DRangeN + n;
				counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
			}
		},
		kParallelize6DTile1DRangeI, kParallelize6DTile1DRangeJ, kParallelize6DTile1DRangeK, kParallelize6DTile1DRangeL, kParallelize6DTile1DRangeM, kParallelize6DTile1DRangeN,
		kParallelize6DTile1DTileN);

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

TEST(Parallelize6DTile2D, ThreadPoolCompletes) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_2d(threadpool.get(),
		[](size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t) { },
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN);
}

TEST(Parallelize6DTile2D, AllItemsInBounds) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t start_n, size_t tile_m, size_t tile_n) {
			EXPECT_LT(i, kParallelize6DTile2DRangeI);
			EXPECT_LT(j, kParallelize6DTile2DRangeJ);
			EXPECT_LT(k, kParallelize6DTile2DRangeK);
			EXPECT_LT(l, kParallelize6DTile2DRangeL);
			EXPECT_LT(start_m, kParallelize6DTile2DRangeM);
			EXPECT_LT(start_n, kParallelize6DTile2DRangeN);
			EXPECT_LE(start_m + tile_m, kParallelize6DTile2DRangeM);
			EXPECT_LE(start_n + tile_n, kParallelize6DTile2DRangeN);
		},
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN);
}

TEST(Parallelize6DTile2D, UniformTiling) {
	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		[](size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t start_n, size_t tile_m, size_t tile_n) {
			EXPECT_GT(tile_m, 0);
			EXPECT_LE(tile_m, kParallelize6DTile2DTileM);
			EXPECT_EQ(start_m % kParallelize6DTile2DTileM, 0);
			EXPECT_EQ(tile_m, std::min<size_t>(kParallelize6DTile2DTileM, kParallelize6DTile2DRangeM - start_m));

			EXPECT_GT(tile_n, 0);
			EXPECT_LE(tile_n, kParallelize6DTile2DTileN);
			EXPECT_EQ(start_n % kParallelize6DTile2DTileN, 0);
			EXPECT_EQ(tile_n, std::min<size_t>(kParallelize6DTile2DTileN, kParallelize6DTile2DRangeN - start_n));
		},
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN);
}

TEST(Parallelize6DTile2D, AllItemsProcessed) {
	std::vector<std::atomic_bool> indicators(kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		[&indicators](size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t start_n, size_t tile_m, size_t tile_n) {
			for (size_t m = start_m; m < start_m + tile_m; m++) {
				for (size_t n = start_n; n < start_n + tile_n; n++) {
					const size_t linear_idx = ((((i * kParallelize6DTile2DRangeJ + j) * kParallelize6DTile2DRangeK + k) * kParallelize6DTile2DRangeL + l) * kParallelize6DTile2DRangeM + m) * kParallelize6DTile2DRangeN + n;
					indicators[linear_idx].store(true, std::memory_order_relaxed);
				}
			}
		},
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN);

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

TEST(Parallelize6DTile2D, EachItemProcessedOnce) {
	std::vector<std::atomic_int> counters(kParallelize6DTile2DRangeI * kParallelize6DTile2DRangeJ * kParallelize6DTile2DRangeK * kParallelize6DTile2DRangeL * kParallelize6DTile2DRangeM * kParallelize6DTile2DRangeN);

	auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
	ASSERT_TRUE(threadpool.get());

	pthreadpool_parallelize_6d_tile_2d(
		threadpool.get(),
		[&counters](size_t i, size_t j, size_t k, size_t l, size_t start_m, size_t start_n, size_t tile_m, size_t tile_n) {
			for (size_t m = start_m; m < start_m + tile_m; m++) {
				for (size_t n = start_n; n < start_n + tile_n; n++) {
					const size_t linear_idx = ((((i * kParallelize6DTile2DRangeJ + j) * kParallelize6DTile2DRangeK + k) * kParallelize6DTile2DRangeL + l) * kParallelize6DTile2DRangeM + m) * kParallelize6DTile2DRangeN + n;
					counters[linear_idx].fetch_add(1, std::memory_order_relaxed);
				}
			}
		},
		kParallelize6DTile2DRangeI, kParallelize6DTile2DRangeJ, kParallelize6DTile2DRangeK, kParallelize6DTile2DRangeL, kParallelize6DTile2DRangeM, kParallelize6DTile2DRangeN,
		kParallelize6DTile2DTileM, kParallelize6DTile2DTileN);

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
