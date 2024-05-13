#include <benchmark/benchmark.h>

#include <cpuinfo.h>


static void cpuinfo_get_current_processor(benchmark::State& state) {
	cpuinfo_initialize();
	while (state.KeepRunning()) {
		const cpuinfo_processor* current_processor = cpuinfo_get_current_processor();
		benchmark::DoNotOptimize(current_processor);
	}
}
BENCHMARK(cpuinfo_get_current_processor)->Unit(benchmark::kNanosecond);

static void cpuinfo_get_current_core(benchmark::State& state) {
	cpuinfo_initialize();
	while (state.KeepRunning()) {
		const cpuinfo_core* current_core = cpuinfo_get_current_core();
		benchmark::DoNotOptimize(current_core);
	}
}
BENCHMARK(cpuinfo_get_current_core)->Unit(benchmark::kNanosecond);

static void cpuinfo_get_current_uarch_index(benchmark::State& state) {
	cpuinfo_initialize();
	while (state.KeepRunning()) {
		const uint32_t uarch_index = cpuinfo_get_current_uarch_index();
		benchmark::DoNotOptimize(uarch_index);
	}
}
BENCHMARK(cpuinfo_get_current_uarch_index)->Unit(benchmark::kNanosecond);

static void cpuinfo_get_current_uarch_index_with_default(benchmark::State& state) {
	cpuinfo_initialize();
	while (state.KeepRunning()) {
		const uint32_t uarch_index = cpuinfo_get_current_uarch_index_with_default(0);
		benchmark::DoNotOptimize(uarch_index);
	}
}
BENCHMARK(cpuinfo_get_current_uarch_index_with_default)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
