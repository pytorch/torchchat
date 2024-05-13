#include <gtest/gtest.h>

#include <stdint.h>
#include <string.h>

#include <string>

#define CPUINFO_BUILD_PROP_VALUE_MAX 92
#define CPUINFO_ARM_CHIPSET_NAME_MAX 48

extern "C" void cpuinfo_arm_android_parse_ro_chipname(
	const char chipname[CPUINFO_BUILD_PROP_VALUE_MAX],
	uint32_t cores,
	uint32_t max_cpu_freq_max,
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX]);

inline std::string parse_ro_chipname(
	std::string chipname, uint32_t cores=1, uint32_t max_cpu_freq_max=0)
{
	char chipname_buffer[CPUINFO_BUILD_PROP_VALUE_MAX];
	strncpy(chipname_buffer, chipname.c_str(), CPUINFO_BUILD_PROP_VALUE_MAX);
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX];
	cpuinfo_arm_android_parse_ro_chipname(
		chipname_buffer, cores, max_cpu_freq_max, chipset_name);
	return std::string(chipset_name, strnlen(chipset_name, CPUINFO_ARM_CHIPSET_NAME_MAX));
}

TEST(RO_CHIPNAME, qualcomm_msm) {
	EXPECT_EQ("Qualcomm MSM7630",
		parse_ro_chipname("MSM7630_SURF"));
	EXPECT_EQ("Qualcomm MSM8210",
		parse_ro_chipname("MSM8210"));
	EXPECT_EQ("Qualcomm MSM8226",
		parse_ro_chipname("MSM8226"));
	EXPECT_EQ("Qualcomm MSM8228",
		parse_ro_chipname("MSM8228"));
	EXPECT_EQ("Qualcomm MSM8230AB",
		parse_ro_chipname("MSM8230AB"));
	EXPECT_EQ("Qualcomm MSM8230VV",
		parse_ro_chipname("MSM8230VV"));
	EXPECT_EQ("Qualcomm MSM8239",
		parse_ro_chipname("MSM8239"));
	EXPECT_EQ("Qualcomm MSM8260A",
		parse_ro_chipname("MSM8260A"));
	EXPECT_EQ("Qualcomm MSM8274",
		parse_ro_chipname("MSM8274"));
	EXPECT_EQ("Qualcomm MSM8610",
		parse_ro_chipname("MSM8610", 2));
	EXPECT_EQ("Qualcomm MSM8626",
		parse_ro_chipname("MSM8626"));
	EXPECT_EQ("Qualcomm MSM8660",
		parse_ro_chipname("MSM8660_SURF"));
	EXPECT_EQ("Qualcomm MSM8674",
		parse_ro_chipname("MSM8674"));
	EXPECT_EQ("Qualcomm MSM8674PRO",
		parse_ro_chipname("MSM8674PRO"));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_ro_chipname("MSM8216"));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_ro_chipname("MSM8916", 4));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_ro_chipname("msm8916", 4));
	EXPECT_EQ("Qualcomm MSM8917",
		parse_ro_chipname("MSM8937", 4));
	EXPECT_EQ("Qualcomm MSM8926",
		parse_ro_chipname("MSM8926"));
	EXPECT_EQ("Qualcomm MSM8928",
		parse_ro_chipname("MSM8928"));
	EXPECT_EQ("Qualcomm MSM8929",
		parse_ro_chipname("MSM8929"));
	EXPECT_EQ("Qualcomm MSM8930",
		parse_ro_chipname("MSM8930"));
	EXPECT_EQ("Qualcomm MSM8930AB",
		parse_ro_chipname("MSM8930AB"));
	EXPECT_EQ("Qualcomm MSM8930VV",
		parse_ro_chipname("MSM8930VV"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_ro_chipname("MSM8939"));
	EXPECT_EQ("Qualcomm MSM8952",
		parse_ro_chipname("MSM8952"));
	EXPECT_EQ("Qualcomm MSM8953",
		parse_ro_chipname("MSM8953"));
	EXPECT_EQ("Qualcomm MSM8960",
		parse_ro_chipname("MSM8960", 2));
	EXPECT_EQ("Qualcomm MSM8974",
		parse_ro_chipname("MSM8974"));
	EXPECT_EQ("Qualcomm MSM8974PRO",
		parse_ro_chipname("MSM8974PRO"));
	EXPECT_EQ("Qualcomm MSM8976",
		parse_ro_chipname("MSM8976"));
	EXPECT_EQ("Qualcomm MSM8996",
		parse_ro_chipname("MSM8996", 4));
	EXPECT_EQ("Qualcomm MSM8998",
		parse_ro_chipname("MSM8998"));
}

TEST(RO_CHIPNAME, qualcomm_apq) {
	EXPECT_EQ("Qualcomm APQ8016",
		parse_ro_chipname("APQ8016"));
	EXPECT_EQ("Qualcomm APQ8026",
		parse_ro_chipname("APQ8026"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_ro_chipname("apq8064"));
	EXPECT_EQ("Qualcomm APQ8074",
		parse_ro_chipname("APQ8074"));
	EXPECT_EQ("Qualcomm APQ8076",
		parse_ro_chipname("APQ8076"));
	EXPECT_EQ("Qualcomm APQ8084",
		parse_ro_chipname("APQ8084"));
}

TEST(RO_CHIPNAME, mediatek_mt) {
	EXPECT_EQ("MediaTek MT6737T",
		parse_ro_chipname("MT6737T"));
	EXPECT_EQ("MediaTek MT6757",
		parse_ro_chipname("MT6757"));
}

TEST(RO_CHIPNAME, samsung_exynos) {
	EXPECT_EQ("Samsung Exynos 3470",
		parse_ro_chipname("exynos3470"));
	EXPECT_EQ("Samsung Exynos 3475",
		parse_ro_chipname("exynos3475"));
	EXPECT_EQ("Samsung Exynos 4415",
		parse_ro_chipname("exynos4415"));
	EXPECT_EQ("Samsung Exynos 5260",
		parse_ro_chipname("exynos5260"));
	EXPECT_EQ("Samsung Exynos 5410",
		parse_ro_chipname("exynos5410"));
	EXPECT_EQ("Samsung Exynos 5420",
		parse_ro_chipname("exynos5420", 4));
	EXPECT_EQ("Samsung Exynos 5422",
		parse_ro_chipname("exynos5422"));
	EXPECT_EQ("Samsung Exynos 5430",
		parse_ro_chipname("exynos5430"));
	EXPECT_EQ("Samsung Exynos 5433",
		parse_ro_chipname("exynos5433"));
	EXPECT_EQ("Samsung Exynos 7420",
		parse_ro_chipname("exynos7420"));
	EXPECT_EQ("Samsung Exynos 7570",
		parse_ro_chipname("exynos7570"));
	EXPECT_EQ("Samsung Exynos 7578",
		parse_ro_chipname("exynos7580", 4));
	EXPECT_EQ("Samsung Exynos 7580",
		parse_ro_chipname("exynos7580", 8));
	EXPECT_EQ("Samsung Exynos 7870",
		parse_ro_chipname("exynos7870"));
	EXPECT_EQ("Samsung Exynos 7880",
		parse_ro_chipname("exynos7880"));
	EXPECT_EQ("Samsung Exynos 8890",
		parse_ro_chipname("exynos8890"));
	EXPECT_EQ("Samsung Exynos 8895",
		parse_ro_chipname("exynos8895"));
}

#if CPUINFO_ARCH_ARM
	TEST(RO_CHIPNAME, marvell_pxa) {
		EXPECT_EQ("Marvell PXA1088",
			parse_ro_chipname("PXA1088"));
		EXPECT_EQ("Marvell PXA986",
			parse_ro_chipname("PXA986"));
	}

	TEST(RO_CHIPNAME, renesas) {
		EXPECT_EQ("Renesas MP5232",
			parse_ro_chipname("mp523x"));
	}
#endif /* CPUINFO_ARCH_ARM */

TEST(RO_CHIPNAME, spreadtrum) {
	EXPECT_EQ("Spreadtrum SC6815AS",
		parse_ro_chipname("SC6815AS"));
	EXPECT_EQ("Spreadtrum SC7715A",
		parse_ro_chipname("SC7715A"));
	EXPECT_EQ("Spreadtrum SC7715T",
		parse_ro_chipname("SC7715T"));
	EXPECT_EQ("Spreadtrum SC7727S",
		parse_ro_chipname("SC7727S"));
	EXPECT_EQ("Spreadtrum SC7727S",
		parse_ro_chipname("sc7727s"));
	EXPECT_EQ("Spreadtrum SC7727SE",
		parse_ro_chipname("SC7727SE"));
	EXPECT_EQ("Spreadtrum SC7730S",
		parse_ro_chipname("sc7730s"));
	EXPECT_EQ("Spreadtrum SC7730SE",
		parse_ro_chipname("SC7730SE"));
	EXPECT_EQ("Spreadtrum SC7730SW",
		parse_ro_chipname("SC7730SW"));
	EXPECT_EQ("Spreadtrum SC7735S",
		parse_ro_chipname("sc7735s"));
	EXPECT_EQ("Spreadtrum SC9830I",
		parse_ro_chipname("SC9830I"));
}
