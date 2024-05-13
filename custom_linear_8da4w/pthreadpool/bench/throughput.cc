#include <benchmark/benchmark.h>

#include <pthreadpool.h>


static void compute_1d(void*, size_t) {
}

static void pthreadpool_parallelize_1d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_1d(
			threadpool,
			compute_1d,
			nullptr /* context */,
			items * threads,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_1d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_1d_tile_1d(void*, size_t, size_t) {
}

static void pthreadpool_parallelize_1d_tile_1d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_1d_tile_1d(
			threadpool,
			compute_1d_tile_1d,
			nullptr /* context */,
			items * threads, 1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_1d_tile_1d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_2d(void*, size_t, size_t) {
}

static void pthreadpool_parallelize_2d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_2d(
			threadpool,
			compute_2d,
			nullptr /* context */,
			threads, items,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_2d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_2d_tile_1d(void*, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_2d_tile_1d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_2d_tile_1d(
			threadpool,
			compute_2d_tile_1d,
			nullptr /* context */,
			threads, items,
			1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_2d_tile_1d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_2d_tile_2d(void*, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_2d_tile_2d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_2d_tile_2d(
			threadpool,
			compute_2d_tile_2d,
			nullptr /* context */,
			threads, items,
			1, 1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_2d_tile_2d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_3d(void*, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_3d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_3d(
			threadpool,
			compute_3d,
			nullptr /* context */,
			1, threads, items,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_3d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_3d_tile_1d(void*, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_3d_tile_1d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_3d_tile_1d(
			threadpool,
			compute_3d_tile_1d,
			nullptr /* context */,
			1, threads, items,
			1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_3d_tile_1d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_3d_tile_2d(void*, size_t, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_3d_tile_2d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_3d_tile_2d(
			threadpool,
			compute_3d_tile_2d,
			nullptr /* context */,
			1, threads, items,
			1, 1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_3d_tile_2d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_4d(void*, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_4d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_4d(
			threadpool,
			compute_4d,
			nullptr /* context */,
			1, 1, threads, items,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_4d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_4d_tile_1d(void*, size_t, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_4d_tile_1d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_4d_tile_1d(
			threadpool,
			compute_4d_tile_1d,
			nullptr /* context */,
			1, 1, threads, items,
			1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_4d_tile_1d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_4d_tile_2d(void*, size_t, size_t, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_4d_tile_2d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_4d_tile_2d(
			threadpool,
			compute_4d_tile_2d,
			nullptr /* context */,
			1, 1, threads, items,
			1, 1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_4d_tile_2d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_5d(void*, size_t, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_5d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_5d(
			threadpool,
			compute_5d,
			nullptr /* context */,
			1, 1, 1, threads, items,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_5d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_5d_tile_1d(void*, size_t, size_t, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_5d_tile_1d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_5d_tile_1d(
			threadpool,
			compute_5d_tile_1d,
			nullptr /* context */,
			1, 1, 1, threads, items,
			1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_5d_tile_1d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_5d_tile_2d(void*, size_t, size_t, size_t, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_5d_tile_2d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_5d_tile_2d(
			threadpool,
			compute_5d_tile_2d,
			nullptr /* context */,
			1, 1, 1, threads, items,
			1, 1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_5d_tile_2d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_6d(void*, size_t, size_t, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_6d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_6d(
			threadpool,
			compute_6d,
			nullptr /* context */,
			1, 1, 1, 1, threads, items,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_6d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_6d_tile_1d(void*, size_t, size_t, size_t, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_6d_tile_1d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_6d_tile_1d(
			threadpool,
			compute_6d_tile_1d,
			nullptr /* context */,
			1, 1, 1, 1, threads, items,
			1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_6d_tile_1d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


static void compute_6d_tile_2d(void*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t) {
}

static void pthreadpool_parallelize_6d_tile_2d(benchmark::State& state) {
	pthreadpool_t threadpool = pthreadpool_create(2);
	const size_t threads = pthreadpool_get_threads_count(threadpool);
	const size_t items = static_cast<size_t>(state.range(0));
	while (state.KeepRunning()) {
		pthreadpool_parallelize_6d_tile_2d(
			threadpool,
			compute_6d_tile_2d,
			nullptr /* context */,
			1, 1, 1, 1, threads, items,
			1, 1,
			0 /* flags */);
	}
	pthreadpool_destroy(threadpool);

	/* Do not normalize by thread */
	state.SetItemsProcessed(int64_t(state.iterations()) * items);
}
BENCHMARK(pthreadpool_parallelize_6d_tile_2d)->UseRealTime()->RangeMultiplier(10)->Range(10, 1000000);


BENCHMARK_MAIN();
