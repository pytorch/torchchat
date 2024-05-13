#include <gtest/gtest.h>

#include <cpuinfo.h>


TEST(CURRENT_PROCESSOR, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());

	const struct cpuinfo_processor* current_processor = cpuinfo_get_current_processor();
	if (current_processor == nullptr) {
		GTEST_SKIP();
	}

	const struct cpuinfo_processor* processors_begin = cpuinfo_get_processors();
	const struct cpuinfo_processor* processors_end = processors_begin + cpuinfo_get_processors_count();
	ASSERT_GE(current_processor, processors_begin);
	ASSERT_LT(current_processor, processors_end);
}

TEST(CURRENT_CORE, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());

	const struct cpuinfo_core* current_core = cpuinfo_get_current_core();
	if (current_core == nullptr) {
		GTEST_SKIP();
	}

	const struct cpuinfo_core* cores_begin = cpuinfo_get_cores();
	const struct cpuinfo_core* cores_end = cores_begin + cpuinfo_get_cores_count();
	ASSERT_GE(current_core, cores_begin);
	ASSERT_LT(current_core, cores_end);
}

TEST(CURRENT_UARCH_INDEX, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());

	ASSERT_LT(cpuinfo_get_current_uarch_index(), cpuinfo_get_uarchs_count());
}

TEST(CURRENT_UARCH_INDEX_WITH_DEFAULT, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());

	ASSERT_LE(cpuinfo_get_current_uarch_index_with_default(cpuinfo_get_uarchs_count()), cpuinfo_get_uarchs_count());
}
