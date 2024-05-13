#include <gtest/gtest.h>

#include <stdint.h>
#include <string.h>

#include <string>

#define CPUINFO_BUILD_PROP_VALUE_MAX 92
#define CPUINFO_ARM_CHIPSET_NAME_MAX 48

extern "C" void cpuinfo_arm_android_parse_ro_product_board(
	const char board[CPUINFO_BUILD_PROP_VALUE_MAX],
	uint32_t cores,
	uint32_t max_cpu_freq_max,
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX]);

inline std::string parse_ro_product_board(
	std::string board, uint32_t cores=1, uint32_t max_cpu_freq_max=0)
{
	char board_buffer[CPUINFO_BUILD_PROP_VALUE_MAX];
	strncpy(board_buffer, board.c_str(), CPUINFO_BUILD_PROP_VALUE_MAX);
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX];
	cpuinfo_arm_android_parse_ro_product_board(
		board_buffer, cores, max_cpu_freq_max, chipset_name);
	return std::string(chipset_name, strnlen(chipset_name, CPUINFO_ARM_CHIPSET_NAME_MAX));
}

TEST(RO_PRODUCT_BOARD, qualcomm_msm) {
	EXPECT_EQ("Qualcomm APQ8064",
		parse_ro_product_board("MSM8960", 4));
	EXPECT_EQ("Qualcomm MSM7630",
		parse_ro_product_board("MSM7630_SURF"));
	EXPECT_EQ("Qualcomm MSM8209",
		parse_ro_product_board("MSM8209"));
	EXPECT_EQ("Qualcomm MSM8210",
		parse_ro_product_board("MSM8210"));
	EXPECT_EQ("Qualcomm MSM8212",
		parse_ro_product_board("MSM8212"));
	EXPECT_EQ("Qualcomm MSM8225",
		parse_ro_product_board("MSM8225"));
	EXPECT_EQ("Qualcomm MSM8226",
		parse_ro_product_board("MSM8226"));
	EXPECT_EQ("Qualcomm MSM8227",
		parse_ro_product_board("MSM8227"));
	EXPECT_EQ("Qualcomm MSM8228",
		parse_ro_product_board("MSM8228"));
	EXPECT_EQ("Qualcomm MSM8230",
		parse_ro_product_board("MSM8230"));
	EXPECT_EQ("Qualcomm MSM8260A",
		parse_ro_product_board("MSM8260A"));
	EXPECT_EQ("Qualcomm MSM8274",
		parse_ro_product_board("MSM8274"));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Qualcomm MSM8610",
		parse_ro_product_board("MSM8610", 2));
	EXPECT_EQ("Qualcomm MSM8612",
		parse_ro_product_board("MSM8610", 4));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("Qualcomm MSM8612",
		parse_ro_product_board("MSM8612"));
	EXPECT_EQ("Qualcomm MSM8625",
		parse_ro_product_board("MSM8625"));
	EXPECT_EQ("Qualcomm MSM8626",
		parse_ro_product_board("MSM8626"));
	EXPECT_EQ("Qualcomm MSM8660",
		parse_ro_product_board("MSM8660_SURF"));
	EXPECT_EQ("Qualcomm MSM8674",
		parse_ro_product_board("MSM8674"));
	EXPECT_EQ("Qualcomm MSM8909",
		parse_ro_product_board("MSM8909"));
	EXPECT_EQ("Qualcomm MSM8909",
		parse_ro_product_board("msm8909"));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_ro_product_board("MSM8216"));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_ro_product_board("MSM8916", 4));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_ro_product_board("msm8916", 4));
	EXPECT_EQ("Qualcomm MSM8917",
		parse_ro_product_board("MSM8917"));
	EXPECT_EQ("Qualcomm MSM8917",
		parse_ro_product_board("msm8937", 4));
	EXPECT_EQ("Qualcomm MSM8917",
		parse_ro_product_board("msm8937_32", 4));
	EXPECT_EQ("Qualcomm MSM8926",
		parse_ro_product_board("MSM8926"));
	EXPECT_EQ("Qualcomm MSM8928",
		parse_ro_product_board("MSM8928"));
	EXPECT_EQ("Qualcomm MSM8929",
		parse_ro_product_board("MSM8929"));
	EXPECT_EQ("Qualcomm MSM8929",
		parse_ro_product_board("msm8929"));
	EXPECT_EQ("Qualcomm MSM8937",
		parse_ro_product_board("MSM8937", 8));
	EXPECT_EQ("Qualcomm MSM8937",
		parse_ro_product_board("msm8937", 8));
	EXPECT_EQ("Qualcomm MSM8937",
		parse_ro_product_board("msm8937_32", 8));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_ro_product_board("MSM8916", 8));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_ro_product_board("MSM8939"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_ro_product_board("msm8916", 8));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_ro_product_board("msm8939"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_ro_product_board("msm8939_64"));
	EXPECT_EQ("Qualcomm MSM8940",
		parse_ro_product_board("MSM8940"));
	EXPECT_EQ("Qualcomm MSM8952",
		parse_ro_product_board("MSM8952"));
	EXPECT_EQ("Qualcomm MSM8952",
		parse_ro_product_board("msm8952"));
	EXPECT_EQ("Qualcomm MSM8953",
		parse_ro_product_board("msm8953"));
	EXPECT_EQ("Qualcomm MSM8960",
		parse_ro_product_board("MSM8960", 2));
	EXPECT_EQ("Qualcomm MSM8974",
		parse_ro_product_board("MSM8974"));
	EXPECT_EQ("Qualcomm MSM8974",
		parse_ro_product_board("msm8974"));
	EXPECT_EQ("Qualcomm MSM8976",
		parse_ro_product_board("MSM8976"));
	EXPECT_EQ("Qualcomm MSM8992",
		parse_ro_product_board("MSM8992"));
	EXPECT_EQ("Qualcomm MSM8992",
		parse_ro_product_board("msm8992"));
	EXPECT_EQ("Qualcomm MSM8994",
		parse_ro_product_board("MSM8994"));
	EXPECT_EQ("Qualcomm MSM8994",
		parse_ro_product_board("msm8994"));
	EXPECT_EQ("Qualcomm MSM8996",
		parse_ro_product_board("msm8996", 4));
	EXPECT_EQ("Qualcomm MSM8998",
		parse_ro_product_board("msm8998"));
}

TEST(RO_PRODUCT_BOARD, qualcomm_apq) {
	EXPECT_EQ("Qualcomm APQ8064",
		parse_ro_product_board("APQ8064"));
	EXPECT_EQ("Qualcomm APQ8064A",
		parse_ro_product_board("APQ8064A"));
	EXPECT_EQ("Qualcomm APQ8064PRO",
		parse_ro_product_board("APQ8064Pro"));
	EXPECT_EQ("Qualcomm APQ8084",
		parse_ro_product_board("APQ8084"));
}

TEST(RO_PRODUCT_BOARD, qualcomm_special) {
	EXPECT_EQ("Qualcomm MSM8996PRO-AB",
		parse_ro_product_board("marlin"));
	EXPECT_EQ("Qualcomm MSM8996PRO-AB",
		parse_ro_product_board("sailfish"));
}

TEST(RO_PRODUCT_BOARD, mediatek_mt) {
	EXPECT_EQ("MediaTek MT5861",
		parse_ro_product_board("mt5861"));
	EXPECT_EQ("MediaTek MT5882",
		parse_ro_product_board("mt5882"));
	EXPECT_EQ("MediaTek MT6572",
		parse_ro_product_board("mt6572"));
	EXPECT_EQ("MediaTek MT6572M",
		parse_ro_product_board("MT6572M"));
	EXPECT_EQ("MediaTek MT6575",
		parse_ro_product_board("MTK6575"));
	EXPECT_EQ("MediaTek MT6575",
		parse_ro_product_board("mt6575"));
	EXPECT_EQ("MediaTek MT6577",
		parse_ro_product_board("MTK6577"));
	EXPECT_EQ("MediaTek MT6577",
		parse_ro_product_board("mt6577"));
	EXPECT_EQ("MediaTek MT6580",
		parse_ro_product_board("MT6580"));
	EXPECT_EQ("MediaTek MT6580",
		parse_ro_product_board("mt6580"));
	EXPECT_EQ("MediaTek MT6580A",
		parse_ro_product_board("MT6580A"));
	EXPECT_EQ("MediaTek MT6580M",
		parse_ro_product_board("MT6580M"));
	EXPECT_EQ("MediaTek MT6582",
		parse_ro_product_board("MT6582"));
	EXPECT_EQ("MediaTek MT6582",
		parse_ro_product_board("mt6582"));
	EXPECT_EQ("MediaTek MT6582M",
		parse_ro_product_board("MTK6582M"));
	EXPECT_EQ("MediaTek MT6582V",
		parse_ro_product_board("MT6582V"));
	EXPECT_EQ("MediaTek MT6582W",
		parse_ro_product_board("MT6582W"));
	EXPECT_EQ("MediaTek MT6589",
		parse_ro_product_board("MT6589"));
	EXPECT_EQ("MediaTek MT6589T",
		parse_ro_product_board("MT6589T"));
	EXPECT_EQ("MediaTek MT6592",
		parse_ro_product_board("MT6592"));
	EXPECT_EQ("MediaTek MT6592",
		parse_ro_product_board("mt6592"));
	EXPECT_EQ("MediaTek MT6592M",
		parse_ro_product_board("MT6592M"));
	EXPECT_EQ("MediaTek MT6595",
		parse_ro_product_board("MT6595"));
	EXPECT_EQ("MediaTek MT6732",
		parse_ro_product_board("MT6732"));
	EXPECT_EQ("MediaTek MT6735",
		parse_ro_product_board("MT6735"));
	EXPECT_EQ("MediaTek MT6735",
		parse_ro_product_board("mt6735"));
	EXPECT_EQ("MediaTek MT6735M",
		parse_ro_product_board("MT6735M"));
	EXPECT_EQ("MediaTek MT6735M",
		parse_ro_product_board("mt6735m"));
	EXPECT_EQ("MediaTek MT6735P",
		parse_ro_product_board("MT6735P"));
	EXPECT_EQ("MediaTek MT6735V",
		parse_ro_product_board("MT6735V"));
	EXPECT_EQ("MediaTek MT6737",
		parse_ro_product_board("MT6737"));
	EXPECT_EQ("MediaTek MT6737M",
		parse_ro_product_board("mt6737m"));
	EXPECT_EQ("MediaTek MT6737T",
		parse_ro_product_board("MT6737T"));
	EXPECT_EQ("MediaTek MT6750",
		parse_ro_product_board("MT6750"));
	EXPECT_EQ("MediaTek MT6750",
		parse_ro_product_board("mt6750"));
	EXPECT_EQ("MediaTek MT6752",
		parse_ro_product_board("MT6752", 8));
	EXPECT_EQ("MediaTek MT6753",
		parse_ro_product_board("MT6753"));
	EXPECT_EQ("MediaTek MT6753",
		parse_ro_product_board("mt6753"));
	EXPECT_EQ("MediaTek MT6755",
		parse_ro_product_board("mt6755"));
	EXPECT_EQ("MediaTek MT6755M",
		parse_ro_product_board("MT6755M"));
	EXPECT_EQ("MediaTek MT6757",
		parse_ro_product_board("MT6757"));
	EXPECT_EQ("MediaTek MT6795",
		parse_ro_product_board("mt6795"));
	EXPECT_EQ("MediaTek MT6797",
		parse_ro_product_board("MT6797"));
	EXPECT_EQ("MediaTek MT8127",
		parse_ro_product_board("mt8127"));
	EXPECT_EQ("MediaTek MT8151",
		parse_ro_product_board("mt8151"));
	EXPECT_EQ("MediaTek MT8163",
		parse_ro_product_board("mt8163"));
	EXPECT_EQ("MediaTek MT8312",
		parse_ro_product_board("MT8312"));
	EXPECT_EQ("MediaTek MT8321",
		parse_ro_product_board("MT8321"));
	EXPECT_EQ("MediaTek MT8382",
		parse_ro_product_board("MT8382"));
	EXPECT_EQ("MediaTek MT8382V",
		parse_ro_product_board("MT8382V"));
	EXPECT_EQ("MediaTek MT8389",
		parse_ro_product_board("MT8389"));
	EXPECT_EQ("MediaTek MT8735M",
		parse_ro_product_board("MT8735m"));
	EXPECT_EQ("MediaTek MT8735P",
		parse_ro_product_board("MT8735P"));
	EXPECT_EQ("MediaTek MT8783",
		parse_ro_product_board("MT8783"));
}

TEST(RO_PRODUCT_BOARD, samsung_universal) {
	EXPECT_EQ("Samsung Exynos 3470",
		parse_ro_product_board("universal3470"));
	EXPECT_EQ("Samsung Exynos 3475",
		parse_ro_product_board("universal3475"));
	EXPECT_EQ("Samsung Exynos 4415",
		parse_ro_product_board("universal4415"));
	EXPECT_EQ("Samsung Exynos 5260",
		parse_ro_product_board("universal5260"));
	EXPECT_EQ("Samsung Exynos 5410",
		parse_ro_product_board("universal5410"));
	EXPECT_EQ("Samsung Exynos 5420",
		parse_ro_product_board("universal5420", 4));
	EXPECT_EQ("Samsung Exynos 5422",
		parse_ro_product_board("universal5422"));
	EXPECT_EQ("Samsung Exynos 5430",
		parse_ro_product_board("universal5430"));
	EXPECT_EQ("Samsung Exynos 5433",
		parse_ro_product_board("universal5433"));
	EXPECT_EQ("Samsung Exynos 7420",
		parse_ro_product_board("universal7420"));
	EXPECT_EQ("Samsung Exynos 7570",
		parse_ro_product_board("universal7570"));
	EXPECT_EQ("Samsung Exynos 7578",
		parse_ro_product_board("universal7580", 4));
	EXPECT_EQ("Samsung Exynos 7580",
		parse_ro_product_board("universal7580", 8));
	EXPECT_EQ("Samsung Exynos 7870",
		parse_ro_product_board("universal7870"));
	EXPECT_EQ("Samsung Exynos 7880",
		parse_ro_product_board("universal7880"));
	EXPECT_EQ("Samsung Exynos 8890",
		parse_ro_product_board("universal8890"));
	EXPECT_EQ("Samsung Exynos 8895",
		parse_ro_product_board("universal8895"));
}

#if CPUINFO_ARCH_ARM
	TEST(RO_PRODUCT_BOARD, samsung_smdk) {
		EXPECT_EQ("Samsung Exynos 4212",
			parse_ro_product_board("smdk4x12", 2));
		EXPECT_EQ("Samsung Exynos 4412",
			parse_ro_product_board("smdk4x12", 4));
	}
#endif

TEST(RO_PRODUCT_BOARD, hisilicon_huawei) {
	EXPECT_EQ("HiSilicon Kirin 659",
		parse_ro_product_board("BAC"));
	EXPECT_EQ("HiSilicon Kirin 950",
		parse_ro_product_board("FRD"));
	EXPECT_EQ("HiSilicon Kirin 950",
		parse_ro_product_board("FRD-L09"));
	EXPECT_EQ("HiSilicon Kirin 950",
		parse_ro_product_board("NXT"));
	EXPECT_EQ("HiSilicon Kirin 950",
		parse_ro_product_board("NXT-AL10"));
	EXPECT_EQ("HiSilicon Kirin 950",
		parse_ro_product_board("NXT-L09"));
	EXPECT_EQ("HiSilicon Kirin 950",
		parse_ro_product_board("NXT-L29"));
	EXPECT_EQ("HiSilicon Kirin 955",
		parse_ro_product_board("EVA"));
	EXPECT_EQ("HiSilicon Kirin 955",
		parse_ro_product_board("EVA-AL10"));
	EXPECT_EQ("HiSilicon Kirin 955",
		parse_ro_product_board("EVA-L09"));
	EXPECT_EQ("HiSilicon Kirin 955",
		parse_ro_product_board("EVA-L19"));
	EXPECT_EQ("HiSilicon Kirin 955",
		parse_ro_product_board("VIE-L09"));
	EXPECT_EQ("HiSilicon Kirin 955",
		parse_ro_product_board("VIE-L29"));
	EXPECT_EQ("HiSilicon Kirin 960",
		parse_ro_product_board("DUK"));
	EXPECT_EQ("HiSilicon Kirin 960",
		parse_ro_product_board("LON"));
	EXPECT_EQ("HiSilicon Kirin 960",
		parse_ro_product_board("MHA"));
	EXPECT_EQ("HiSilicon Kirin 960",
		parse_ro_product_board("STF"));
	EXPECT_EQ("HiSilicon Kirin 960",
		parse_ro_product_board("VKY"));
	EXPECT_EQ("HiSilicon Kirin 960",
		parse_ro_product_board("VTR"));
}

TEST(RO_PRODUCT_BOARD, hisilicon_special) {
	EXPECT_EQ("HiSilicon Kirin 620",
		parse_ro_product_board("hi6210sft"));
	EXPECT_EQ("HiSilicon Kirin 650",
		parse_ro_product_board("hi6250"));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("HiSilicon Kirin 920",
		parse_ro_product_board("hi3630"));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("HiSilicon Kirin 930",
		parse_ro_product_board("hi3635"));
	EXPECT_EQ("HiSilicon Kirin 950",
		parse_ro_product_board("hi3650"));
	EXPECT_EQ("HiSilicon Kirin 960",
		parse_ro_product_board("hi3660"));
	EXPECT_EQ("HiSilicon Kirin 950",
		parse_ro_product_board("BEETHOVEN"));
}

#if CPUINFO_ARCH_ARM
	TEST(RO_PRODUCT_BOARD, broadcom) {
		EXPECT_EQ("Broadcom BCM28155",
			parse_ro_product_board("capri", 2, 1200000));
		EXPECT_EQ("Broadcom BCM28155",
			parse_ro_product_board("capri", 2, 1300000));
		EXPECT_EQ("Broadcom BCM28155",
			parse_ro_product_board("capri", 2, 1399999));
		EXPECT_EQ("Broadcom BCM28155",
			parse_ro_product_board("capri", 2, 1399999));
		EXPECT_EQ("Broadcom BCM23550",
			parse_ro_product_board("java", 4, 1200000));
		EXPECT_EQ("Broadcom BCM23550",
			parse_ro_product_board("java", 4, 1300000));
		EXPECT_EQ("Broadcom BCM21654",
			parse_ro_product_board("rhea", 1, 849999));
		EXPECT_EQ("Broadcom BCM21654G",
			parse_ro_product_board("rhea", 1, 999999));
		EXPECT_EQ("Broadcom BCM21663",
			parse_ro_product_board("hawaii", 1, 999999));
		EXPECT_EQ("Broadcom BCM21664",
			parse_ro_product_board("hawaii", 2, 999999));
		EXPECT_EQ("Broadcom BCM21664T",
			parse_ro_product_board("hawaii", 2, 1200000));
	}

	TEST(RO_PRODUCT_BOARD, leadcore_lc) {
		EXPECT_EQ("Leadcore LC1810",
			parse_ro_product_board("lc1810"));
	}

	TEST(RO_PRODUCT_BOARD, marvell_pxa) {
		EXPECT_EQ("Marvell PXA1088",
			parse_ro_product_board("PXA1088"));
		EXPECT_EQ("Marvell PXA986",
			parse_ro_product_board("PXA986"));
		EXPECT_EQ("Marvell PXA988",
			parse_ro_product_board("PXA988"));
	}

	TEST(RO_PRODUCT_BOARD, nvidia) {
		EXPECT_EQ("Nvidia Tegra SL460N",
			parse_ro_product_board("g2mv"));
		EXPECT_EQ("Nvidia Tegra T132",
			parse_ro_product_board("flounder"));
		EXPECT_EQ("Nvidia Tegra T210",
			parse_ro_product_board("dragon"));
		EXPECT_EQ("Nvidia Tegra T30L",
			parse_ro_product_board("grouper"));
	}

	TEST(RO_PRODUCT_BOARD, renesas) {
		EXPECT_EQ("Renesas MP5232",
			parse_ro_product_board("mp523x"));
	}

	TEST(RO_PRODUCT_BOARD, rockchip) {
		EXPECT_EQ("Rockchip RK3066",
			parse_ro_product_board("T7H"));
		EXPECT_EQ("Rockchip RK3168",
			parse_ro_product_board("hws7701u"));
		EXPECT_EQ("Rockchip RK3188",
			parse_ro_product_board("K00F"));
	}
#endif

TEST(RO_PRODUCT_BOARD, spreadtrum) {
	EXPECT_EQ("Spreadtrum SC6815AS",
		parse_ro_product_board("SC6815AS"));
	EXPECT_EQ("Spreadtrum SC7715",
		parse_ro_product_board("SC7715"));
	EXPECT_EQ("Spreadtrum SC7715A",
		parse_ro_product_board("SC7715A"));
	EXPECT_EQ("Spreadtrum SC7715T",
		parse_ro_product_board("SC7715T"));
	EXPECT_EQ("Spreadtrum SC7727S",
		parse_ro_product_board("SC7727S"));
	EXPECT_EQ("Spreadtrum SC7727S",
		parse_ro_product_board("sc7727s"));
	EXPECT_EQ("Spreadtrum SC7727SE",
		parse_ro_product_board("SC7727SE"));
	EXPECT_EQ("Spreadtrum SC7730S",
		parse_ro_product_board("sc7730s"));
	EXPECT_EQ("Spreadtrum SC7730SE",
		parse_ro_product_board("SC7730SE"));
	EXPECT_EQ("Spreadtrum SC7730SW",
		parse_ro_product_board("SC7730SW"));
	EXPECT_EQ("Spreadtrum SC7731",
		parse_ro_product_board("SC7731"));
	EXPECT_EQ("Spreadtrum SC7731C",
		parse_ro_product_board("SC7731C"));
	EXPECT_EQ("Spreadtrum SC7731G",
		parse_ro_product_board("SC7731G"));
	EXPECT_EQ("Spreadtrum SC7735S",
		parse_ro_product_board("sc7735s"));
	EXPECT_EQ("Spreadtrum SC9830A",
		parse_ro_product_board("SC9830A"));
	EXPECT_EQ("Spreadtrum SC9830I",
		parse_ro_product_board("SC9830I"));
}

#if CPUINFO_ARCH_ARM
	TEST(RO_PRODUCT_BOARD, texas_instruments) {
		EXPECT_EQ("Texas Instruments OMAP4460",
			parse_ro_product_board("tuna"));
	}
#endif /* CPUINFO_ARCH_ARM */
