#include <gtest/gtest.h>

#include <stdint.h>
#include <string.h>

#include <string>

#define CPUINFO_BUILD_PROP_VALUE_MAX 92
#define CPUINFO_ARM_CHIPSET_NAME_MAX 48

extern "C" void cpuinfo_arm_android_parse_ro_board_platform(
	const char platform[CPUINFO_BUILD_PROP_VALUE_MAX],
	uint32_t cores,
	uint32_t max_cpu_freq_max,
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX]);

inline std::string parse_ro_board_platform(
	std::string platform, uint32_t cores=1, uint32_t max_cpu_freq_max=0)
{
	char platform_buffer[CPUINFO_BUILD_PROP_VALUE_MAX];
	strncpy(platform_buffer, platform.c_str(), CPUINFO_BUILD_PROP_VALUE_MAX);
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX];
	cpuinfo_arm_android_parse_ro_board_platform(
		platform_buffer, cores, max_cpu_freq_max, chipset_name);
	return std::string(chipset_name, strnlen(chipset_name, CPUINFO_ARM_CHIPSET_NAME_MAX));
}

TEST(RO_BOARD_PLATFORM, qualcomm_msm) {
	EXPECT_EQ("Qualcomm APQ8064",
		parse_ro_board_platform("msm8960", 4));
	EXPECT_EQ("Qualcomm MSM7627A",
		parse_ro_board_platform("msm7627a"));
	EXPECT_EQ("Qualcomm MSM8084",
		parse_ro_board_platform("msm8084"));
	EXPECT_EQ("Qualcomm MSM8226",
		parse_ro_board_platform("msm8226"));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Qualcomm MSM8610",
		parse_ro_board_platform("msm8610", 2));
	EXPECT_EQ("Qualcomm MSM8612",
		parse_ro_board_platform("msm8610", 4));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("Qualcomm MSM8612",
		parse_ro_board_platform("MSM8612"));
	EXPECT_EQ("Qualcomm MSM8660",
		parse_ro_board_platform("msm8660"));
	EXPECT_EQ("Qualcomm MSM8909",
		parse_ro_board_platform("msm8909"));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_ro_board_platform("msm8916", 4));
	EXPECT_EQ("Qualcomm MSM8917",
		parse_ro_board_platform("msm8937", 4));
	EXPECT_EQ("Qualcomm MSM8937",
		parse_ro_board_platform("msm8937", 8));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_ro_board_platform("msm8916", 8));
	EXPECT_EQ("Qualcomm MSM8952",
		parse_ro_board_platform("msm8952"));
	EXPECT_EQ("Qualcomm MSM8953",
		parse_ro_board_platform("msm8953"));
	EXPECT_EQ("Qualcomm MSM8960",
		parse_ro_board_platform("msm8960", 2));
	EXPECT_EQ("Qualcomm MSM8974",
		parse_ro_board_platform("msm8974"));
	EXPECT_EQ("Qualcomm MSM8992",
		parse_ro_board_platform("msm8992"));
	EXPECT_EQ("Qualcomm MSM8994",
		parse_ro_board_platform("msm8994"));
	EXPECT_EQ("Qualcomm MSM8996",
		parse_ro_board_platform("msm8996", 4));
	EXPECT_EQ("Qualcomm MSM8998",
		parse_ro_board_platform("msm8998"));
}

TEST(RO_BOARD_PLATFORM, qualcomm_apq) {
	EXPECT_EQ("Qualcomm APQ8084",
		parse_ro_board_platform("apq8084"));
}

TEST(RO_BOARD_PLATFORM, mediatek_mt) {
	EXPECT_EQ("MediaTek MT5861",
		parse_ro_board_platform("mt5861"));
	EXPECT_EQ("MediaTek MT5882",
		parse_ro_board_platform("mt5882"));
	EXPECT_EQ("MediaTek MT6570",
		parse_ro_board_platform("mt6570"));
	EXPECT_EQ("MediaTek MT6572",
		parse_ro_board_platform("mt6572"));
	EXPECT_EQ("MediaTek MT6572A",
		parse_ro_board_platform("MT6572A"));
	EXPECT_EQ("MediaTek MT6575",
		parse_ro_board_platform("mt6575"));
	EXPECT_EQ("MediaTek MT6577",
		parse_ro_board_platform("MT6577"));
	EXPECT_EQ("MediaTek MT6577",
		parse_ro_board_platform("mt6577"));
	EXPECT_EQ("MediaTek MT6580",
		parse_ro_board_platform("mt6580"));
	EXPECT_EQ("MediaTek MT6582",
		parse_ro_board_platform("MTK6582"));
	EXPECT_EQ("MediaTek MT6582",
		parse_ro_board_platform("mt6582"));
	EXPECT_EQ("MediaTek MT6582M",
		parse_ro_board_platform("MTK6582M"));
	EXPECT_EQ("MediaTek MT6589",
		parse_ro_board_platform("MT6589"));
	EXPECT_EQ("MediaTek MT6589",
		parse_ro_board_platform("MTK6589"));
	EXPECT_EQ("MediaTek MT6592",
		parse_ro_board_platform("mt6592"));
	EXPECT_EQ("MediaTek MT6592T",
		parse_ro_board_platform("MTK6592T"));
	EXPECT_EQ("MediaTek MT6595",
		parse_ro_board_platform("mt6595"));
	EXPECT_EQ("MediaTek MT6732",
		parse_ro_board_platform("mt6752", 4));
	EXPECT_EQ("MediaTek MT6735",
		parse_ro_board_platform("mt6735"));
	EXPECT_EQ("MediaTek MT6735M",
		parse_ro_board_platform("mt6735m"));
	EXPECT_EQ("MediaTek MT6737",
		parse_ro_board_platform("mt6737"));
	EXPECT_EQ("MediaTek MT6737M",
		parse_ro_board_platform("mt6737m"));
	EXPECT_EQ("MediaTek MT6737T",
		parse_ro_board_platform("mt6737t"));
	EXPECT_EQ("MediaTek MT6750",
		parse_ro_board_platform("mt6750"));
	EXPECT_EQ("MediaTek MT6752",
		parse_ro_board_platform("mt6752", 8));
	EXPECT_EQ("MediaTek MT6753",
		parse_ro_board_platform("mt6753"));
	EXPECT_EQ("MediaTek MT6755",
		parse_ro_board_platform("mt6755"));
	EXPECT_EQ("MediaTek MT6757",
		parse_ro_board_platform("mt6757"));
	EXPECT_EQ("MediaTek MT6795",
		parse_ro_board_platform("mt6795"));
	EXPECT_EQ("MediaTek MT6797",
		parse_ro_board_platform("mt6797"));
	EXPECT_EQ("MediaTek MT8111",
		parse_ro_board_platform("MT8111"));
	EXPECT_EQ("MediaTek MT8127",
		parse_ro_board_platform("MT8127"));
	EXPECT_EQ("MediaTek MT8127",
		parse_ro_board_platform("mt8127"));
	EXPECT_EQ("MediaTek MT8135",
		parse_ro_board_platform("mt8135"));
	EXPECT_EQ("MediaTek MT8151",
		parse_ro_board_platform("mt8151"));
	EXPECT_EQ("MediaTek MT8163",
		parse_ro_board_platform("mt8163"));
	EXPECT_EQ("MediaTek MT8167",
		parse_ro_board_platform("mt8167"));
	EXPECT_EQ("MediaTek MT8173",
		parse_ro_board_platform("mt8173"));
	EXPECT_EQ("MediaTek MT8312",
		parse_ro_board_platform("MT8312"));
	EXPECT_EQ("MediaTek MT8382",
		parse_ro_board_platform("MT8382"));
	EXPECT_EQ("MediaTek MT8382V",
		parse_ro_board_platform("MT8382V"));
	EXPECT_EQ("MediaTek MT8392",
		parse_ro_board_platform("MT8392"));
}

TEST(RO_BOARD_PLATFORM, samsung) {
	EXPECT_EQ("Samsung Exynos 4412",
		parse_ro_board_platform("exynos4412"));
}

TEST(RO_BOARD_PLATFORM, hisilicon) {
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("HiSilicon K3V2",
		parse_ro_board_platform("k3v200"));
	EXPECT_EQ("HiSilicon K3V2",
		parse_ro_board_platform("k3v2oem1"));
#endif
	EXPECT_EQ("HiSilicon Kirin 620",
		parse_ro_board_platform("hi6210sft"));
	EXPECT_EQ("HiSilicon Kirin 650",
		parse_ro_board_platform("hi6250"));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("HiSilicon Kirin 910T",
		parse_ro_board_platform("hi6620oem"));
	EXPECT_EQ("HiSilicon Kirin 920",
		parse_ro_board_platform("hi3630"));
#endif
	EXPECT_EQ("HiSilicon Kirin 930",
		parse_ro_board_platform("hi3635"));
	EXPECT_EQ("HiSilicon Kirin 950",
		parse_ro_board_platform("hi3650"));
	EXPECT_EQ("HiSilicon Kirin 960",
		parse_ro_board_platform("hi3660"));
	EXPECT_EQ("HiSilicon Kirin 970",
		parse_ro_board_platform("kirin970"));
}

TEST(RO_BOARD_PLATFORM, amlogic) {
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Amlogic AML8726-M",
		parse_ro_board_platform("meson3"));
	EXPECT_EQ("Amlogic AML8726-MX",
		parse_ro_board_platform("meson6"));
	EXPECT_EQ("Amlogic S805",
		parse_ro_board_platform("meson8"));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("Amlogic S905",
		parse_ro_board_platform("gxbaby"));
	EXPECT_EQ("Amlogic S905X",
		parse_ro_board_platform("gxl"));
	EXPECT_EQ("Amlogic S912",
		parse_ro_board_platform("gxm"));
}

#if CPUINFO_ARCH_ARM
	TEST(RO_BOARD_PLATFORM, broadcom) {
		EXPECT_EQ("Broadcom BCM21654",
			parse_ro_board_platform("rhea", 1, 849999));
		EXPECT_EQ("Broadcom BCM21654G",
			parse_ro_board_platform("rhea", 1, 999999));
		EXPECT_EQ("Broadcom BCM21663",
			parse_ro_board_platform("hawaii", 1, 999999));
		EXPECT_EQ("Broadcom BCM21664",
			parse_ro_board_platform("hawaii", 2, 999999));
		EXPECT_EQ("Broadcom BCM21664T",
			parse_ro_board_platform("hawaii", 2, 1200000));
		EXPECT_EQ("Broadcom BCM23550",
			parse_ro_board_platform("java", 4, 1200000));
		EXPECT_EQ("Broadcom BCM28155",
			parse_ro_board_platform("capri", 2, 1200000));
		EXPECT_EQ("Broadcom BCM28155",
			parse_ro_board_platform("capri", 2, 1399999));
	}

	TEST(RO_BOARD_PLATFORM, leadcore) {
		EXPECT_EQ("Leadcore LC1860",
			parse_ro_board_platform("lc1860"));
	}

	TEST(RO_BOARD_PLATFORM, novathor) {
		EXPECT_EQ("NovaThor U8500",
			parse_ro_board_platform("montblanc"));
	}
#endif /* CPUINFO_ARCH_ARM */

TEST(RO_BOARD_PLATFORM, nvidia) {
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Nvidia Tegra T114",
		parse_ro_board_platform("tegra4"));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("Nvidia Tegra T132",
		parse_ro_board_platform("tegra132"));
	EXPECT_EQ("Nvidia Tegra T210",
		parse_ro_board_platform("tegra210_dragon"));
}

TEST(RO_BOARD_PLATFORM, pinecone) {
	EXPECT_EQ("Pinecone Surge S1",
		parse_ro_board_platform("song"));
}

TEST(RO_BOARD_PLATFORM, rockchip_rk) {
	EXPECT_EQ("Rockchip RK2928",
		parse_ro_board_platform("rk2928"));
	EXPECT_EQ("Rockchip RK3026",
		parse_ro_board_platform("rk3026"));
	EXPECT_EQ("Rockchip RK3066",
		parse_ro_board_platform("rk3066"));
	EXPECT_EQ("Rockchip RK3188",
		parse_ro_board_platform("rk3188"));
	EXPECT_EQ("Rockchip RK3228",
		parse_ro_board_platform("rk3228"));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Rockchip RK3229",
		parse_ro_board_platform("rk322x"));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("Rockchip RK3288",
		parse_ro_board_platform("rk3288", 4));
	EXPECT_EQ("Rockchip RK3399",
		parse_ro_board_platform("rk3288", 6));
	EXPECT_EQ("Rockchip RK3328",
		parse_ro_board_platform("rk3328"));
	EXPECT_EQ("Rockchip RK3368",
		parse_ro_board_platform("rk3368"));
	EXPECT_EQ("Rockchip RK3399",
		parse_ro_board_platform("rk3399"));
}

TEST(RO_BOARD_PLATFORM, spreadtrum_sc) {
	EXPECT_EQ("Spreadtrum SC6820I",
		parse_ro_board_platform("sc6820i"));
	EXPECT_EQ("Spreadtrum SC7731",
		parse_ro_board_platform("SC7731"));
	EXPECT_EQ("Spreadtrum SC7731",
		parse_ro_board_platform("sc7731"));
	EXPECT_EQ("Spreadtrum SC7731G",
		parse_ro_board_platform("sc7731g"));
	EXPECT_EQ("Spreadtrum SC8810",
		parse_ro_board_platform("sc8810"));
	EXPECT_EQ("Spreadtrum SC8825",
		parse_ro_board_platform("sc8825"));
	EXPECT_EQ("Spreadtrum SC8830",
		parse_ro_board_platform("sc8830"));
}

#if CPUINFO_ARCH_ARM
	TEST(RO_BOARD_PLATFORM, texas_instruments_omap) {
		EXPECT_EQ("Texas Instruments OMAP4430",
			parse_ro_board_platform("omap4", 2, 1008000));
	}
#endif /* CPUINFO_ARCH_ARM */
