#include <benchmark/benchmark.h>

#include <cpuinfo.h>


static void cpuinfo_initialize(benchmark::State& state) {
	while (state.KeepRunning()) {
		cpuinfo_initialize();
	}
}
BENCHMARK(cpuinfo_initialize)->Iterations(1)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
