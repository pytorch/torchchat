#include <gtest/gtest.h>

#include <stdint.h>
#include <string.h>

#include <string>

#define CPUINFO_HARDWARE_VALUE_MAX 64
#define CPUINFO_BUILD_PROP_VALUE_MAX 92
#define CPUINFO_ARM_CHIPSET_NAME_MAX 48

extern "C" void cpuinfo_arm_android_parse_chipset_properties(
	const char proc_cpuinfo_hardware[CPUINFO_HARDWARE_VALUE_MAX],
	const char ro_product_board[CPUINFO_BUILD_PROP_VALUE_MAX],
	const char ro_board_platform[CPUINFO_BUILD_PROP_VALUE_MAX],
	const char ro_mediatek_platform[CPUINFO_BUILD_PROP_VALUE_MAX],
	const char ro_arch[CPUINFO_HARDWARE_VALUE_MAX],
	const char ro_chipname[CPUINFO_BUILD_PROP_VALUE_MAX],
	uint32_t cores,
	uint32_t max_cpu_freq_max,
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX]);

inline std::string parse_chipset(
	std::string hardware,
	std::string product_board,
	std::string board_platform,
	std::string mediatek_platform,
	std::string arch,
	std::string chipname,
	uint32_t cores=1,
	uint32_t max_cpu_freq_max=0)
{
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX];
	cpuinfo_arm_android_parse_chipset_properties(
		hardware.c_str(), product_board.c_str(), board_platform.c_str(), mediatek_platform.c_str(), arch.c_str(), chipname.c_str(),
		cores, max_cpu_freq_max, chipset_name);
	return std::string(chipset_name, strnlen(chipset_name, CPUINFO_ARM_CHIPSET_NAME_MAX));
}

TEST(ANDROID_PROPERTIES, disambiguate_chipset) {
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Amlogic S812",
		parse_chipset("Amlogic Meson8", "n200C", "meson8", "", "", ""));
	EXPECT_EQ("HiSilicon Kirin 925",
		parse_chipset("Kirin925", "MT7-L09", "hi3630", "", "", ""));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("HiSilicon Kirin 955",
		parse_chipset("Hisilicon Kirin 955", "EVA-L09", "hi3650", "", "", ""));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Marvell PXA986",
		parse_chipset("PXA988", "PXA986", "mrvl", "", "", ""));
	EXPECT_EQ("Marvell PXA986",
		parse_chipset("PXA988", "PXA986", "mrvl", "", "", "PXA986"));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("MediaTek MT6735P",
		parse_chipset("MT6735P", "MT6735P", "mt6735m", "MT6735", "", ""));
	EXPECT_EQ("MediaTek MT8382",
		parse_chipset("MT8382", "MT8382", "", "MT6582", "", ""));
	EXPECT_EQ("MediaTek MT6735P",
		parse_chipset("MT6735P", "unknown", "mt6735m", "MT6735", "", ""));
	EXPECT_EQ("MediaTek MT8382",
		parse_chipset("MT8382", "LenovoTAB2A7-30HC", "", "MT6582", "", ""));
	EXPECT_EQ("Qualcomm MSM8926",
		parse_chipset("Qualcomm MSM 8226 (Flattened Device Tree)", "MSM8226", "msm8226", "", "", "MSM8926"));
	EXPECT_EQ("Qualcomm MSM8926",
		parse_chipset("Qualcomm MSM8926", "draconis", "msm8226", "", "", ""));
	EXPECT_EQ("Qualcomm MSM8930AB",
		parse_chipset("SAMSUNG SERRANO", "MSM8960", "msm8960", "", "", "MSM8930AB", 2));
	EXPECT_EQ("Qualcomm MSM8940",
		parse_chipset("Qualcomm Technologies, Inc MSM8940", "msm8937_32", "msm8937", "", "", "", 8));
	EXPECT_EQ("Spreadtrum SC6815AS",
		parse_chipset("scx15", "SC6815AS", "scx15", "", "", "SC6815AS"));
	EXPECT_EQ("Spreadtrum SC7727S",
		parse_chipset("sc8830", "SC7727S", "sc8830", "", "", "SC7727S"));
	EXPECT_EQ("Spreadtrum SC7731",
		parse_chipset("sc7731", "SC7731", "sc8830", "", "", ""));
	EXPECT_EQ("Spreadtrum SC7731C",
		parse_chipset("sc7731c", "sp7731cea", "sc8830", "", "", ""));
}

TEST(ANDROID_PROPERTIES, ambiguous_vendors) {
	EXPECT_EQ("",
		parse_chipset("MTK6580", "sp7731ceb", "sc8830", "", "", ""));
	EXPECT_EQ("",
		parse_chipset("", "universal5410", "msm8974", "", "", ""));
	EXPECT_EQ("",
		parse_chipset("MT6580", "universal8895", "mt6580", "MT6580", "", ""));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("",
		parse_chipset("", "smdk4x12", "msm8974", "", "", "", 2));
#endif /* CPUINFO_ARCH_ARM */
}

TEST(ANDROID_PROPERTIES, unambiguous_chipset) {
	EXPECT_EQ("Samsung Exynos 3470",
		parse_chipset("UNIVERSAL_GARDA", "universal_garda", "exynos3", "", "exynos3470", "exynos3470"));
	EXPECT_EQ("MediaTek MT6582",
		parse_chipset("APPLE A8", "APPLE A8", "", "MT6582", "", ""));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("NovaThor U8500",
		parse_chipset("SAMSUNG GOLDEN", "DB8520H", "montblanc", "", "", ""));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("MediaTek MT6580",
		parse_chipset("Qualcomm", "unknown", "mt6580", "MT6580", "", ""));
	EXPECT_EQ("HiSilicon Kirin 650",
		parse_chipset("", "hi6250", "", "", "", ""));
	EXPECT_EQ("Samsung Exynos 8890",
		parse_chipset("", "universal8890", "exynos5", "", "exynos8890", "exynos8890"));
	EXPECT_EQ("MediaTek MT6582",
		parse_chipset("", "MT6582", "", "MT6582", "", ""));
	EXPECT_EQ("Qualcomm MSM8994",
		parse_chipset("", "msm8994", "msm8994", "", "", ""));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_chipset("SAMSUNG JF", "MSM8960", "msm8960", "", "", "apq8064", 4));
	EXPECT_EQ("MediaTek MT6795",
		parse_chipset("", "mt6795", "mt6795", "MT6795", "", ""));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Marvell PXA1908",
		parse_chipset("PXA1908", "PXA19xx", "mrvl", "", "", "PXA19xx"));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("Spreadtrum SC7715A",
		parse_chipset("scx15", "SM-G928G", "scx15", "", "", "SC7715A"));
	EXPECT_EQ("MediaTek MT6592",
		parse_chipset("MT6592", "lcsh92_cwet_htc_kk", "", "MT6592", "", ""));
	EXPECT_EQ("HiSilicon Kirin 620",
		parse_chipset("hi6210sft", "BalongV8R1SFT", "hi6210sft", "", "", ""));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_chipset("PANTECH APQ8064 EF52L", "VEGA", "msm8960", "", "", "apq8064", 4));
	EXPECT_EQ("MediaTek MT6580M",
		parse_chipset("MT6580M", "unknown", "mt6580", "MT6580", "", ""));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Samsung Exynos 4412",
		parse_chipset("SMDK4x12", "smdk4x12", "exynos4", "", "", "smdk4x12", 4));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("Samsung Exynos 7420",
		parse_chipset("SAMSUNG Exynos7420", "universal7420", "exynos5", "", "exynos7420", "exynos7420"));
	EXPECT_EQ("MediaTek MT6582",
		parse_chipset("MT6582", "MT6582", "", "MT6582", "", ""));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_chipset("Qualcomm Technologies, Inc MSM8916", "msm8916", "msm8916", "", "", "", 4));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_chipset("Qualcomm Technologies, Inc MSM8916", "MSM8916", "msm8916", "", "", "MSM8916", 4));
	EXPECT_EQ("MediaTek MT6735",
		parse_chipset("MT6735", "mt6735", "mt6735", "MT6735", "", ""));
	EXPECT_EQ("MediaTek MT6737T",
		parse_chipset("Samsung GrandPrimePlus LTE CIS rev04 board based on MT6737T", "MT6737T", "mt6737t", "MT6737T", "", "MT6737T"));
}
