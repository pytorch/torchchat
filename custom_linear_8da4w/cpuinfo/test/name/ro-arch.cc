#include <gtest/gtest.h>

#include <stdint.h>
#include <string.h>

#include <string>

#define CPUINFO_BUILD_PROP_VALUE_MAX 92
#define CPUINFO_ARM_CHIPSET_NAME_MAX 48

extern "C" void cpuinfo_arm_android_parse_ro_arch(
	const char arch[CPUINFO_BUILD_PROP_VALUE_MAX],
	uint32_t cores,
	uint32_t max_cpu_freq_max,
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX]);

inline std::string parse_ro_arch(
	std::string arch, uint32_t cores=1, uint32_t max_cpu_freq_max=0)
{
	char arch_buffer[CPUINFO_BUILD_PROP_VALUE_MAX];
	strncpy(arch_buffer, arch.c_str(), CPUINFO_BUILD_PROP_VALUE_MAX);
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX];
	cpuinfo_arm_android_parse_ro_arch(
		arch_buffer, cores, max_cpu_freq_max, chipset_name);
	return std::string(chipset_name, strnlen(chipset_name, CPUINFO_ARM_CHIPSET_NAME_MAX));
}

TEST(RO_ARCH, samsung_exynos) {
	EXPECT_EQ("Samsung Exynos 3470",
		parse_ro_arch("exynos3470"));
	EXPECT_EQ("Samsung Exynos 3475",
		parse_ro_arch("exynos3475"));
	EXPECT_EQ("Samsung Exynos 4415",
		parse_ro_arch("exynos4415"));
	EXPECT_EQ("Samsung Exynos 5260",
		parse_ro_arch("exynos5260"));
	EXPECT_EQ("Samsung Exynos 5410",
		parse_ro_arch("exynos5410"));
	EXPECT_EQ("Samsung Exynos 5420",
		parse_ro_arch("exynos5420", 4));
	EXPECT_EQ("Samsung Exynos 5422",
		parse_ro_arch("exynos5422"));
	EXPECT_EQ("Samsung Exynos 5430",
		parse_ro_arch("exynos5430"));
	EXPECT_EQ("Samsung Exynos 5433",
		parse_ro_arch("exynos5433"));
	EXPECT_EQ("Samsung Exynos 7420",
		parse_ro_arch("exynos7420"));
	EXPECT_EQ("Samsung Exynos 7570",
		parse_ro_arch("exynos7570"));
	EXPECT_EQ("Samsung Exynos 7580",
		parse_ro_arch("exynos7580", 8));
	EXPECT_EQ("Samsung Exynos 7870",
		parse_ro_arch("exynos7870"));
	EXPECT_EQ("Samsung Exynos 7872",
		parse_ro_arch("exynos7872"));
	EXPECT_EQ("Samsung Exynos 7880",
		parse_ro_arch("exynos7880"));
	EXPECT_EQ("Samsung Exynos 7885",
		parse_ro_arch("exynos7885"));
	EXPECT_EQ("Samsung Exynos 8890",
		parse_ro_arch("exynos8890"));
	EXPECT_EQ("Samsung Exynos 8895",
		parse_ro_arch("exynos8895"));
	EXPECT_EQ("Samsung Exynos 9810",
		parse_ro_arch("exynos9810"));
}
