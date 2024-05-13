#include <gtest/gtest.h>

#include <stdint.h>
#include <string.h>

#include <string>

#define CPUINFO_BUILD_PROP_VALUE_MAX 92
#define CPUINFO_ARM_CHIPSET_NAME_MAX 48

extern "C" void cpuinfo_arm_android_parse_ro_mediatek_platform(
	const char platform[CPUINFO_BUILD_PROP_VALUE_MAX],
	uint32_t cores,
	uint32_t max_cpu_freq_max,
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX]);

inline std::string parse_ro_mediatek_platform(
	std::string platform, uint32_t cores=1, uint32_t max_cpu_freq_max=0)
{
	char platform_buffer[CPUINFO_BUILD_PROP_VALUE_MAX];
	strncpy(platform_buffer, platform.c_str(), CPUINFO_BUILD_PROP_VALUE_MAX);
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX];
	cpuinfo_arm_android_parse_ro_mediatek_platform(
		platform_buffer, cores, max_cpu_freq_max, chipset_name);
	return std::string(chipset_name, strnlen(chipset_name, CPUINFO_ARM_CHIPSET_NAME_MAX));
}

TEST(RO_MEDIATEK_PLATFORM, qualcomm) {
	EXPECT_EQ("", parse_ro_mediatek_platform("MSM8225Q"));
	EXPECT_EQ("", parse_ro_mediatek_platform("Qualcomm Snapdragon 805"));
	EXPECT_EQ("", parse_ro_mediatek_platform("Qualcomm Snapdragon 820"));
}

TEST(RO_MEDIATEK_PLATFORM, samsung) {
	EXPECT_EQ("", parse_ro_mediatek_platform("EXYNOS5420"));
	EXPECT_EQ("", parse_ro_mediatek_platform("Samsung  Exynos 5420"));
}

TEST(RO_MEDIATEK_PLATFORM, apple) {
	EXPECT_EQ("", parse_ro_mediatek_platform("Apple A9"));
	EXPECT_EQ("", parse_ro_mediatek_platform("Apple A10"));
}

TEST(RO_MEDIATEK_PLATFORM, mediatek_mt) {
	EXPECT_EQ("MediaTek MT5861",
		parse_ro_mediatek_platform("mt5861"));
	EXPECT_EQ("MediaTek MT5882",
		parse_ro_mediatek_platform("mt5882"));
	EXPECT_EQ("MediaTek MT6570",
		parse_ro_mediatek_platform("mt6570"));
	EXPECT_EQ("MediaTek MT6572",
		parse_ro_mediatek_platform("mt6572"));
	EXPECT_EQ("MediaTek MT6572A",
		parse_ro_mediatek_platform("MT6572A"));
	EXPECT_EQ("MediaTek MT6575",
		parse_ro_mediatek_platform("mt6575"));
	EXPECT_EQ("MediaTek MT6577",
		parse_ro_mediatek_platform("MT6577"));
	EXPECT_EQ("MediaTek MT6577",
		parse_ro_mediatek_platform("mt6577"));
	EXPECT_EQ("MediaTek MT6580",
		parse_ro_mediatek_platform("mt6580"));
	EXPECT_EQ("MediaTek MT6582",
		parse_ro_mediatek_platform("MTK6582"));
	EXPECT_EQ("MediaTek MT6582",
		parse_ro_mediatek_platform("mt6582"));
	EXPECT_EQ("MediaTek MT6582M",
		parse_ro_mediatek_platform("MTK6582M"));
	EXPECT_EQ("MediaTek MT6589",
		parse_ro_mediatek_platform("MT6589"));
	EXPECT_EQ("MediaTek MT6589",
		parse_ro_mediatek_platform("MTK6589"));
	EXPECT_EQ("MediaTek MT6592",
		parse_ro_mediatek_platform("mt6592"));
	EXPECT_EQ("MediaTek MT6592T",
		parse_ro_mediatek_platform("MTK6592T"));
	EXPECT_EQ("MediaTek MT6595",
		parse_ro_mediatek_platform("mt6595"));
	EXPECT_EQ("MediaTek MT6732",
		parse_ro_mediatek_platform("mt6752", 4));
	EXPECT_EQ("MediaTek MT6735",
		parse_ro_mediatek_platform("mt6735"));
	EXPECT_EQ("MediaTek MT6735M",
		parse_ro_mediatek_platform("mt6735m"));
	EXPECT_EQ("MediaTek MT6737",
		parse_ro_mediatek_platform("mt6737"));
	EXPECT_EQ("MediaTek MT6737M",
		parse_ro_mediatek_platform("mt6737m"));
	EXPECT_EQ("MediaTek MT6737T",
		parse_ro_mediatek_platform("mt6737t"));
	EXPECT_EQ("MediaTek MT6750",
		parse_ro_mediatek_platform("mt6750"));
	EXPECT_EQ("MediaTek MT6752",
		parse_ro_mediatek_platform("mt6752", 8));
	EXPECT_EQ("MediaTek MT6753",
		parse_ro_mediatek_platform("mt6753"));
	EXPECT_EQ("MediaTek MT6755",
		parse_ro_mediatek_platform("mt6755"));
	EXPECT_EQ("MediaTek MT6757",
		parse_ro_mediatek_platform("mt6757"));
	EXPECT_EQ("MediaTek MT6795",
		parse_ro_mediatek_platform("mt6795"));
	EXPECT_EQ("MediaTek MT6797",
		parse_ro_mediatek_platform("mt6797"));
	EXPECT_EQ("MediaTek MT8111",
		parse_ro_mediatek_platform("MT8111"));
	EXPECT_EQ("MediaTek MT8127",
		parse_ro_mediatek_platform("MT8127"));
	EXPECT_EQ("MediaTek MT8127",
		parse_ro_mediatek_platform("mt8127"));
	EXPECT_EQ("MediaTek MT8135",
		parse_ro_mediatek_platform("mt8135"));
	EXPECT_EQ("MediaTek MT8151",
		parse_ro_mediatek_platform("mt8151"));
	EXPECT_EQ("MediaTek MT8163",
		parse_ro_mediatek_platform("mt8163"));
	EXPECT_EQ("MediaTek MT8167",
		parse_ro_mediatek_platform("mt8167"));
	EXPECT_EQ("MediaTek MT8173",
		parse_ro_mediatek_platform("mt8173"));
	EXPECT_EQ("MediaTek MT8312",
		parse_ro_mediatek_platform("MT8312"));
	EXPECT_EQ("MediaTek MT8382",
		parse_ro_mediatek_platform("MT8382"));
	EXPECT_EQ("MediaTek MT8382V",
		parse_ro_mediatek_platform("MT8382V"));
	EXPECT_EQ("MediaTek MT8392",
		parse_ro_mediatek_platform("MT8392"));
}
