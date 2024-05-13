#include <gtest/gtest.h>

#include <stdint.h>
#include <string.h>

#include <string>

#define CPUINFO_HARDWARE_VALUE_MAX 64
#define CPUINFO_ARM_CHIPSET_NAME_MAX 48

extern "C" void cpuinfo_arm_android_parse_proc_cpuinfo_hardware(
	const char hardware[CPUINFO_HARDWARE_VALUE_MAX],
	uint32_t cores,
	uint32_t max_cpu_freq_max,
	bool is_tegra,
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX]);

inline std::string parse_proc_cpuinfo_hardware(
	std::string hardware,
	uint32_t cores=1,
	uint32_t max_cpu_freq_max=0)
{
	char hardware_buffer[CPUINFO_HARDWARE_VALUE_MAX];
	strncpy(hardware_buffer, hardware.c_str(), CPUINFO_HARDWARE_VALUE_MAX);
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX];
	cpuinfo_arm_android_parse_proc_cpuinfo_hardware(
		hardware_buffer, cores, max_cpu_freq_max, false, chipset_name);
	return std::string(chipset_name, strnlen(chipset_name, CPUINFO_ARM_CHIPSET_NAME_MAX));
}

inline std::string parse_proc_cpuinfo_hardware_tegra(
	std::string hardware,
	uint32_t cores=1,
	uint32_t max_cpu_freq_max=0)
{
	char hardware_buffer[CPUINFO_HARDWARE_VALUE_MAX];
	strncpy(hardware_buffer, hardware.c_str(), CPUINFO_HARDWARE_VALUE_MAX);
	char chipset_name[CPUINFO_ARM_CHIPSET_NAME_MAX];
	cpuinfo_arm_android_parse_proc_cpuinfo_hardware(
		hardware_buffer, cores, max_cpu_freq_max, true, chipset_name);
	return std::string(chipset_name, strnlen(chipset_name, CPUINFO_ARM_CHIPSET_NAME_MAX));
}

TEST(PROC_CPUINFO_HARDWARE, qualcomm_msm) {
	EXPECT_EQ("Qualcomm MSM7225AB",
		parse_proc_cpuinfo_hardware("LG MSM7225AB"));
	EXPECT_EQ("Qualcomm MSM7225AB",
		parse_proc_cpuinfo_hardware("LG MSM7225AB V1"));
	EXPECT_EQ("Qualcomm MSM7625A",
		parse_proc_cpuinfo_hardware("QCT MSM7625a FFA"));
	EXPECT_EQ("Qualcomm MSM8208",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8208"));
	EXPECT_EQ("Qualcomm MSM8209",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8209"));
	EXPECT_EQ("Qualcomm MSM8210",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8210"));
	EXPECT_EQ("Qualcomm MSM8212",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8212 (Flattened Device Tree)"));
	EXPECT_EQ("Qualcomm MSM8212",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8212"));
	EXPECT_EQ("Qualcomm MSM8225",
		parse_proc_cpuinfo_hardware("QCT MSM8225 SURF"));
	EXPECT_EQ("Qualcomm MSM8226",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8226 (Flattened Device Tree)"));
	EXPECT_EQ("Qualcomm MSM8226",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8226"));
	EXPECT_EQ("Qualcomm MSM8228",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8228"));
	EXPECT_EQ("Qualcomm MSM8230",
		parse_proc_cpuinfo_hardware("LGE MSM8230 L9II"));
	EXPECT_EQ("Qualcomm MSM8239",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8239"));
	EXPECT_EQ("Qualcomm MSM8609",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8609"));
	EXPECT_EQ("Qualcomm MSM8610",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8610 (Flattened Device Tree)", 2));
	EXPECT_EQ("Qualcomm MSM8610",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8610", 2));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Qualcomm MSM8612",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8610 (Flattened Device Tree)", 4));
	EXPECT_EQ("Qualcomm MSM8612",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8612 (Flattened Device Tree)"));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("Qualcomm MSM8625",
		parse_proc_cpuinfo_hardware("LG MSM8625 V7"));
	EXPECT_EQ("Qualcomm MSM8625",
		parse_proc_cpuinfo_hardware("QCT MSM8625 FFA"));
	EXPECT_EQ("Qualcomm MSM8625",
		parse_proc_cpuinfo_hardware("QCT MSM8625 SURF"));
	EXPECT_EQ("Qualcomm MSM8625Q",
		parse_proc_cpuinfo_hardware("QRD MSM8625Q SKUD"));
	EXPECT_EQ("Qualcomm MSM8626",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8626"));
	EXPECT_EQ("Qualcomm MSM8627",
		parse_proc_cpuinfo_hardware("QCT MSM8627 MTP"));
	EXPECT_EQ("Qualcomm MSM8628",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8628"));
	EXPECT_EQ("Qualcomm MSM8909",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8909"));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8216"));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8916", 4));
	EXPECT_EQ("Qualcomm MSM8916",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8916MSM8916", 4));
	EXPECT_EQ("Qualcomm MSM8917",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8917", 4));
	EXPECT_EQ("Qualcomm MSM8920",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8920"));
	EXPECT_EQ("Qualcomm MSM8926",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8926 (Flattened Device Tree)"));
	EXPECT_EQ("Qualcomm MSM8926",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8926"));
	EXPECT_EQ("Qualcomm MSM8928",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8928"));
	EXPECT_EQ("Qualcomm MSM8928",
		parse_proc_cpuinfo_hardware("Qualcomm msm 8928"));
	EXPECT_EQ("Qualcomm MSM8929",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8929"));
	EXPECT_EQ("Qualcomm MSM8930",
		parse_proc_cpuinfo_hardware("LGE MSM8930 FX3"));
	EXPECT_EQ("Qualcomm MSM8930",
		parse_proc_cpuinfo_hardware("QCT MSM8930 CDP"));
	EXPECT_EQ("Qualcomm MSM8930",
		parse_proc_cpuinfo_hardware("QCT MSM8930 MTP"));
	EXPECT_EQ("Qualcomm MSM8937",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8937", 8));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI ALE_L04"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI ATH-UL01"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI KII-L05"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI KIW-L21"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI KIW-L22"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI KIW-L23"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI KIW-L24"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI KIW-L33"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI RIO-L01_VB"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI RIO-L02"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI RIO-L03"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8939 HUAWEI TEXAS-A1"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8939"));
	EXPECT_EQ("Qualcomm MSM8939",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8939_BC"));
	EXPECT_EQ("Qualcomm MSM8940",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8940"));
	EXPECT_EQ("Qualcomm MSM8952",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8952"));
	EXPECT_EQ("Qualcomm MSM8952",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8952MSM8952"));
	EXPECT_EQ("Qualcomm MSM8952",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc. MSM8952 QRD SKUM"));
	EXPECT_EQ("Qualcomm MSM8953",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8953"));
	EXPECT_EQ("Qualcomm MSM8953PRO",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8953Pro"));
	EXPECT_EQ("Qualcomm MSM8956",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8956"));
	EXPECT_EQ("Qualcomm MSM8960",
		parse_proc_cpuinfo_hardware("LGE MSM8960 D1L KR", 2));
	EXPECT_EQ("Qualcomm MSM8960",
		parse_proc_cpuinfo_hardware("LGE MSM8960 FX1", 2));
	EXPECT_EQ("Qualcomm MSM8960",
		parse_proc_cpuinfo_hardware("LGE MSM8960 Lx", 2));
	EXPECT_EQ("Qualcomm MSM8960",
		parse_proc_cpuinfo_hardware("LGE MSM8960 VU2", 2));
	EXPECT_EQ("Qualcomm MSM8960",
		parse_proc_cpuinfo_hardware("QCT MSM8960 CDP", 2));
	EXPECT_EQ("Qualcomm MSM8960",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8960", 2));
	EXPECT_EQ("Qualcomm MSM8960DT",
		parse_proc_cpuinfo_hardware("msm8960dt"));
	EXPECT_EQ("Qualcomm MSM8974",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8974 (Flattened Device Tree)"));
	EXPECT_EQ("Qualcomm MSM8974",
		parse_proc_cpuinfo_hardware("Qualcomm MSM 8974 HAMMERHEAD (Flattened Device Tree)"));
	EXPECT_EQ("Qualcomm MSM8974",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8974"));
	EXPECT_EQ("Qualcomm MSM8974PRO-AA",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8974PRO-AA"));
	EXPECT_EQ("Qualcomm MSM8974PRO-AB",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8974PRO-AB"));
	EXPECT_EQ("Qualcomm MSM8974PRO-AC",
		parse_proc_cpuinfo_hardware("Qualcomm MSM8974PRO-AC"));
	EXPECT_EQ("Qualcomm MSM8976",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8976"));
	EXPECT_EQ("Qualcomm MSM8976PRO",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8976SG"));
	EXPECT_EQ("Qualcomm MSM8992",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8992"));
	EXPECT_EQ("Qualcomm MSM8994",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8994"));
	EXPECT_EQ("Qualcomm MSM8994V",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc. MSM8994v2.1 MTP"));
	EXPECT_EQ("Qualcomm MSM8996",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8996", 4));
	EXPECT_EQ("Qualcomm MSM8996PRO-AB",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8996pro", 4, 1593600 /* LITTLE core */));
	EXPECT_EQ("Qualcomm MSM8996PRO-AB",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8996pro", 4, 2150400 /* big core */));
	EXPECT_EQ("Qualcomm MSM8996PRO-AC",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8996pro", 4, 2188800 /* LITTLE core */));
	EXPECT_EQ("Qualcomm MSM8996PRO-AC",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8996pro", 4, 2342400 /* big core */));
	EXPECT_EQ("Qualcomm MSM8998",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc MSM8998"));
}

TEST(PROC_CPUINFO_HARDWARE, qualcomm_apq) {
	EXPECT_EQ("Qualcomm APQ8009",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc APQ8009"));
	EXPECT_EQ("Qualcomm APQ8016",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc APQ8016"));
	EXPECT_EQ("Qualcomm APQ8016",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc APQ8016APQ8016"));
	EXPECT_EQ("Qualcomm APQ8017",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc APQ8017"));
	EXPECT_EQ("Qualcomm APQ8026",
		parse_proc_cpuinfo_hardware("Qualcomm APQ8026"));
	EXPECT_EQ("Qualcomm APQ8028",
		parse_proc_cpuinfo_hardware("Qualcomm APQ8028"));
	EXPECT_EQ("Qualcomm APQ8039",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc APQ8039"));
	EXPECT_EQ("Qualcomm APQ8053",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc APQ8053"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("PANTECH APQ8064 EF48S"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("PANTECH APQ8064 EF49K"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("PANTECH APQ8064 EF50L"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("PANTECH APQ8064 EF51K"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("PANTECH APQ8064 EF51L"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("PANTECH APQ8064 EF51S"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("PANTECH APQ8064 EF52K"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("PANTECH APQ8064 EF52L"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("PANTECH APQ8064 EF52S"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("QCT APQ8064 AWIFI"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("QCT APQ8064 DEB"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("QCT APQ8064 DUMA"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("QCT APQ8064 FLO"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("QCT APQ8064 LEOPARDCAT"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("QCT APQ8064 MAKO"));
	EXPECT_EQ("Qualcomm APQ8064",
		parse_proc_cpuinfo_hardware("QCT APQ8064 MTP"));
	EXPECT_EQ("Qualcomm APQ8074PRO-AB",
		parse_proc_cpuinfo_hardware("Qualcomm APQ8074PRO-AB"));
	EXPECT_EQ("Qualcomm APQ8076",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc APQ8076"));
	EXPECT_EQ("Qualcomm APQ8084",
		parse_proc_cpuinfo_hardware("Qualcomm APQ 8084 (Flattened Device Tree)"));
	EXPECT_EQ("Qualcomm APQ8084",
		parse_proc_cpuinfo_hardware("Qualcomm APQ8084"));
	EXPECT_EQ("Qualcomm APQ8094",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc APQ8094"));
	EXPECT_EQ("Qualcomm APQ8096",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc APQ8096"));
}

TEST(PROC_CPUINFO_HARDWARE, qualcomm_sdm) {
	EXPECT_EQ("Qualcomm Snapdragon 630",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc SDM630"));
	EXPECT_EQ("Qualcomm Snapdragon 660",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc SDM660"));
}

TEST(PROC_CPUINFO_HARDWARE, qualcomm_sm) {
	EXPECT_EQ("Qualcomm Snapdragon 8150",
		parse_proc_cpuinfo_hardware("Qualcomm Technologies, Inc SM8150"));
}

TEST(PROC_CPUINFO_HARDWARE, mediatek_mt) {
	EXPECT_EQ("MediaTek MT5507",
		parse_proc_cpuinfo_hardware("MT5507"));
	EXPECT_EQ("MediaTek MT5508",
		parse_proc_cpuinfo_hardware("MT5508"));
	EXPECT_EQ("MediaTek MT6517",
		parse_proc_cpuinfo_hardware("MT6517"));
	EXPECT_EQ("MediaTek MT6570",
		parse_proc_cpuinfo_hardware("MT6570"));
	EXPECT_EQ("MediaTek MT6571",
		parse_proc_cpuinfo_hardware("MT6571"));
	EXPECT_EQ("MediaTek MT6572",
		parse_proc_cpuinfo_hardware("MT6572"));
	EXPECT_EQ("MediaTek MT6575",
		parse_proc_cpuinfo_hardware("MT6575"));
	EXPECT_EQ("MediaTek MT6577",
		parse_proc_cpuinfo_hardware("MT6577"));
	EXPECT_EQ("MediaTek MT6580",
		parse_proc_cpuinfo_hardware("MT6580"));
	EXPECT_EQ("MediaTek MT6580M",
		parse_proc_cpuinfo_hardware("MT6580M"));
	EXPECT_EQ("MediaTek MT6581",
		parse_proc_cpuinfo_hardware("MT6581"));
	EXPECT_EQ("MediaTek MT6582",
		parse_proc_cpuinfo_hardware("MT6582"));
	EXPECT_EQ("MediaTek MT6582",
		parse_proc_cpuinfo_hardware("Mediatek MT6582"));
	EXPECT_EQ("MediaTek MT6588",
		parse_proc_cpuinfo_hardware("MT6588"));
	EXPECT_EQ("MediaTek MT6589",
		parse_proc_cpuinfo_hardware("MT6589"));
	EXPECT_EQ("MediaTek MT6591",
		parse_proc_cpuinfo_hardware("MT6591"));
	EXPECT_EQ("MediaTek MT6592",
		parse_proc_cpuinfo_hardware("MT6592"));
	EXPECT_EQ("MediaTek MT6592T",
		parse_proc_cpuinfo_hardware("MT6592T"));
	EXPECT_EQ("MediaTek MT6592T",
		parse_proc_cpuinfo_hardware("MT6592trubo"));
	EXPECT_EQ("MediaTek MT6592T",
		parse_proc_cpuinfo_hardware("MT6592turbo"));
	EXPECT_EQ("MediaTek MT6595",
		parse_proc_cpuinfo_hardware("MT6595"));
	EXPECT_EQ("MediaTek MT6732",
		parse_proc_cpuinfo_hardware("MT6732"));
	EXPECT_EQ("MediaTek MT6732",
		parse_proc_cpuinfo_hardware("MT6752", 4));
	EXPECT_EQ("MediaTek MT6732M",
		parse_proc_cpuinfo_hardware("MT6732M"));
	EXPECT_EQ("MediaTek MT6735",
		parse_proc_cpuinfo_hardware("MT6735"));
	EXPECT_EQ("MediaTek MT6735M",
		parse_proc_cpuinfo_hardware("MT6735M"));
	EXPECT_EQ("MediaTek MT6735P",
		parse_proc_cpuinfo_hardware("MT6735P"));
	EXPECT_EQ("MediaTek MT6737",
		parse_proc_cpuinfo_hardware("MT6737"));
	EXPECT_EQ("MediaTek MT6737M",
		parse_proc_cpuinfo_hardware("MT6737M"));
	EXPECT_EQ("MediaTek MT6737T",
		parse_proc_cpuinfo_hardware("MT6737T"));
	EXPECT_EQ("MediaTek MT6737T",
		parse_proc_cpuinfo_hardware("Samsung GrandPrimePlus LTE CIS rev04 board based on MT6737T"));
	EXPECT_EQ("MediaTek MT6737T",
		parse_proc_cpuinfo_hardware("Samsung GrandPrimePlus LTE LTN DTV rev04 board based on MT6737T"));
	EXPECT_EQ("MediaTek MT6737T",
		parse_proc_cpuinfo_hardware("Samsung GrandPrimePlus LTE LTN OPEN rev04 board based on MT6737T"));
	EXPECT_EQ("MediaTek MT6738",
		parse_proc_cpuinfo_hardware("MT6738"));
	EXPECT_EQ("MediaTek MT6750",
		parse_proc_cpuinfo_hardware("MT6750"));
	EXPECT_EQ("MediaTek MT6750T",
		parse_proc_cpuinfo_hardware("MT6750T"));
	EXPECT_EQ("MediaTek MT6752",
		parse_proc_cpuinfo_hardware("MT6752", 8));
	EXPECT_EQ("MediaTek MT6752M",
		parse_proc_cpuinfo_hardware("MT6752M", 8));
	EXPECT_EQ("MediaTek MT6753",
		parse_proc_cpuinfo_hardware("MT6753"));
	EXPECT_EQ("MediaTek MT6753T",
		parse_proc_cpuinfo_hardware("MT6753T"));
	EXPECT_EQ("MediaTek MT6755",
		parse_proc_cpuinfo_hardware("MT6755"));
	EXPECT_EQ("MediaTek MT6755BM",
		parse_proc_cpuinfo_hardware("MT6755BM"));
	EXPECT_EQ("MediaTek MT6755M",
		parse_proc_cpuinfo_hardware("MT6755M"));
	EXPECT_EQ("MediaTek MT6755V/B",
		parse_proc_cpuinfo_hardware("MT6755V/B"));
	EXPECT_EQ("MediaTek MT6755V/BM",
		parse_proc_cpuinfo_hardware("MT6755V/BM"));
	EXPECT_EQ("MediaTek MT6755V/C",
		parse_proc_cpuinfo_hardware("MT6755V/C"));
	EXPECT_EQ("MediaTek MT6755V/CM",
		parse_proc_cpuinfo_hardware("MT6755V/CM"));
	EXPECT_EQ("MediaTek MT6755V/W",
		parse_proc_cpuinfo_hardware("MT6755V/W"));
	EXPECT_EQ("MediaTek MT6755V/WM",
		parse_proc_cpuinfo_hardware("MT6755V/WM"));
	EXPECT_EQ("MediaTek MT6755V/WT",
		parse_proc_cpuinfo_hardware("MT6755V/WT"));
	EXPECT_EQ("MediaTek MT6757",
		parse_proc_cpuinfo_hardware("MT6757"));
	EXPECT_EQ("MediaTek MT6757",
		parse_proc_cpuinfo_hardware("Samsung J7 Max LTE SWA rev02a board based on MT6757"));
	EXPECT_EQ("MediaTek MT6757CD",
		parse_proc_cpuinfo_hardware("MT6757CD"));
	EXPECT_EQ("MediaTek MT6757CH",
		parse_proc_cpuinfo_hardware("MT6757CH"));
	EXPECT_EQ("MediaTek MT6795",
		parse_proc_cpuinfo_hardware("MT6795"));
	EXPECT_EQ("MediaTek MT6795M",
		parse_proc_cpuinfo_hardware("MT6795M"));
	EXPECT_EQ("MediaTek MT6795MM",
		parse_proc_cpuinfo_hardware("MT6795MM"));
	EXPECT_EQ("MediaTek MT6795T",
		parse_proc_cpuinfo_hardware("MT6795T"));
	EXPECT_EQ("MediaTek MT6797",
		parse_proc_cpuinfo_hardware("MT6797"));
	EXPECT_EQ("MediaTek MT6797M",
		parse_proc_cpuinfo_hardware("MT6797M"));
	EXPECT_EQ("MediaTek MT6797T",
		parse_proc_cpuinfo_hardware("MT6797T"));
	EXPECT_EQ("MediaTek MT6797X",
		parse_proc_cpuinfo_hardware("MT6797X"));
	EXPECT_EQ("MediaTek MT8111",
		parse_proc_cpuinfo_hardware("MT8111"));
	EXPECT_EQ("MediaTek MT8121",
		parse_proc_cpuinfo_hardware("MT8121"));
	EXPECT_EQ("MediaTek MT8125",
		parse_proc_cpuinfo_hardware("MT8125"));
	EXPECT_EQ("MediaTek MT8127",
		parse_proc_cpuinfo_hardware("MT8127"));
	EXPECT_EQ("MediaTek MT8135",
		parse_proc_cpuinfo_hardware("MT8135"));
	EXPECT_EQ("MediaTek MT8151",
		parse_proc_cpuinfo_hardware("MT8151"));
	EXPECT_EQ("MediaTek MT8161",
		parse_proc_cpuinfo_hardware("MT8161"));
	EXPECT_EQ("MediaTek MT8161A",
		parse_proc_cpuinfo_hardware("MT8161A"));
	EXPECT_EQ("MediaTek MT8161P",
		parse_proc_cpuinfo_hardware("MT8161P"));
	EXPECT_EQ("MediaTek MT8163",
		parse_proc_cpuinfo_hardware("MT8163"));
	EXPECT_EQ("MediaTek MT8165",
		parse_proc_cpuinfo_hardware("MT8165"));
	EXPECT_EQ("MediaTek MT8167A",
		parse_proc_cpuinfo_hardware("MT8167A"));
	EXPECT_EQ("MediaTek MT8167B",
		parse_proc_cpuinfo_hardware("MT8167B"));
	EXPECT_EQ("MediaTek MT8173",
		parse_proc_cpuinfo_hardware("MT8173"));
	EXPECT_EQ("MediaTek MT8176",
		parse_proc_cpuinfo_hardware("MT8176"));
	EXPECT_EQ("MediaTek MT8312",
		parse_proc_cpuinfo_hardware("MT8312"));
	EXPECT_EQ("MediaTek MT8312C",
		parse_proc_cpuinfo_hardware("MT8312C"));
	EXPECT_EQ("MediaTek MT8312D",
		parse_proc_cpuinfo_hardware("MT8312D"));
	EXPECT_EQ("MediaTek MT8317",
		parse_proc_cpuinfo_hardware("MT8317"));
	EXPECT_EQ("MediaTek MT8321",
		parse_proc_cpuinfo_hardware("MT8321"));
	EXPECT_EQ("MediaTek MT8321M",
		parse_proc_cpuinfo_hardware("MT8321M"));
	EXPECT_EQ("MediaTek MT8377",
		parse_proc_cpuinfo_hardware("MT8377"));
	EXPECT_EQ("MediaTek MT8382",
		parse_proc_cpuinfo_hardware("MT8382"));
	EXPECT_EQ("MediaTek MT8389",
		parse_proc_cpuinfo_hardware("MT8389"));
	EXPECT_EQ("MediaTek MT8389Q",
		parse_proc_cpuinfo_hardware("MT8389Q"));
	EXPECT_EQ("MediaTek MT8392",
		parse_proc_cpuinfo_hardware("MT8392"));
	EXPECT_EQ("MediaTek MT8685",
		parse_proc_cpuinfo_hardware("MT8685"));
	EXPECT_EQ("MediaTek MT8732",
		parse_proc_cpuinfo_hardware("MT8732"));
	EXPECT_EQ("MediaTek MT8732T",
		parse_proc_cpuinfo_hardware("MT8732T"));
	EXPECT_EQ("MediaTek MT8735",
		parse_proc_cpuinfo_hardware("MT8735"));
	EXPECT_EQ("MediaTek MT8735A",
		parse_proc_cpuinfo_hardware("MT8735A"));
	EXPECT_EQ("MediaTek MT8735B",
		parse_proc_cpuinfo_hardware("MT8735B"));
	EXPECT_EQ("MediaTek MT8735D",
		parse_proc_cpuinfo_hardware("MT8735D"));
	EXPECT_EQ("MediaTek MT8735M",
		parse_proc_cpuinfo_hardware("MT8735M"));
	EXPECT_EQ("MediaTek MT8735P",
		parse_proc_cpuinfo_hardware("MT8735P"));
	EXPECT_EQ("MediaTek MT8735T",
		parse_proc_cpuinfo_hardware("MT8735T"));
	EXPECT_EQ("MediaTek MT8752",
		parse_proc_cpuinfo_hardware("MT8752"));
	EXPECT_EQ("MediaTek MT8783",
		parse_proc_cpuinfo_hardware("MT8783"));
	EXPECT_EQ("MediaTek MT8783T",
		parse_proc_cpuinfo_hardware("MT8783T"));
}

TEST(PROC_CPUINFO_HARDWARE, samsung_exynos) {
	EXPECT_EQ("Samsung Exynos 4415",
		parse_proc_cpuinfo_hardware("Samsung EXYNOS4415"));
	EXPECT_EQ("Samsung Exynos 5420",
		parse_proc_cpuinfo_hardware("Samsung EXYNOS5420", 4));
	EXPECT_EQ("Samsung Exynos 5430",
		parse_proc_cpuinfo_hardware("Samsung EXYNOS5430"));
	EXPECT_EQ("Samsung Exynos 5433",
		parse_proc_cpuinfo_hardware("Samsung EXYNOS5433"));
	EXPECT_EQ("Samsung Exynos 7420",
		parse_proc_cpuinfo_hardware("SAMSUNG Exynos7420"));
	EXPECT_EQ("Samsung Exynos 7578",
		parse_proc_cpuinfo_hardware("SAMSUNG Exynos7580", 4));
	EXPECT_EQ("Samsung Exynos 7580",
		parse_proc_cpuinfo_hardware("SAMSUNG Exynos7580", 8));
}

TEST(PROC_CPUINFO_HARDWARE, samsung_universal) {
	EXPECT_EQ("Samsung Exynos 3470",
		parse_proc_cpuinfo_hardware("UNIVERSAL3470"));
	EXPECT_EQ("Samsung Exynos 3475",
		parse_proc_cpuinfo_hardware("UNIVERSAL3475"));
	EXPECT_EQ("Samsung Exynos 5260",
		parse_proc_cpuinfo_hardware("UNIVERSAL5260"));
	EXPECT_EQ("Samsung Exynos 5410",
		parse_proc_cpuinfo_hardware("UNIVERSAL5410"));
	EXPECT_EQ("Samsung Exynos 5420",
		parse_proc_cpuinfo_hardware("UNIVERSAL5420", 4));
	EXPECT_EQ("Samsung Exynos 5422",
		parse_proc_cpuinfo_hardware("universal5422"));
	EXPECT_EQ("Samsung Exynos 5430",
		parse_proc_cpuinfo_hardware("UNIVERSAL5430"));
}

#if CPUINFO_ARCH_ARM
	TEST(PROC_CPUINFO_HARDWARE, samsung_smdk) {
		EXPECT_EQ("Samsung Exynos 4210",
			parse_proc_cpuinfo_hardware("SMDK4210"));
		EXPECT_EQ("Samsung Exynos 4212",
			parse_proc_cpuinfo_hardware("SMDK4x12", 2));
		EXPECT_EQ("Samsung Exynos 4412",
			parse_proc_cpuinfo_hardware("SMDK4x12", 4));
	}

	TEST(PROC_CPUINFO_HARDWARE, samsung_special) {
		EXPECT_EQ("Samsung Exynos 5250",
			parse_proc_cpuinfo_hardware("Manta"));
	}
#endif /* CPUINFO_ARCH_ARM */

TEST(PROC_CPUINFO_HARDWARE, hisilicon_kirin) {
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("HiSilicon Kirin 920",
		parse_proc_cpuinfo_hardware("Hisilicon Kirin 920"));
	EXPECT_EQ("HiSilicon Kirin 920",
		parse_proc_cpuinfo_hardware("Kirin920"));
	EXPECT_EQ("HiSilicon Kirin 925",
		parse_proc_cpuinfo_hardware("Hisilicon Kirin 925"));
	EXPECT_EQ("HiSilicon Kirin 925",
		parse_proc_cpuinfo_hardware("Kirin925"));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("HiSilicon Kirin 930",
		parse_proc_cpuinfo_hardware("Hisilicon Kirin 930"));
	EXPECT_EQ("HiSilicon Kirin 935",
		parse_proc_cpuinfo_hardware("Hisilicon Kirin 935"));
	EXPECT_EQ("HiSilicon Kirin 950",
		parse_proc_cpuinfo_hardware("Hisilicon Kirin 950"));
	EXPECT_EQ("HiSilicon Kirin 955",
		parse_proc_cpuinfo_hardware("Hisilicon Kirin 955"));
}

TEST(PROC_CPUINFO_HARDWARE, hisilicon_special) {
	EXPECT_EQ("HiSilicon Hi3751",
		parse_proc_cpuinfo_hardware("hi3751"));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("HiSilicon K3V2",
		parse_proc_cpuinfo_hardware("k3v2oem1"));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("HiSilicon Kirin 620",
		parse_proc_cpuinfo_hardware("hi6210sft"));
	EXPECT_EQ("HiSilicon Kirin 650",
		parse_proc_cpuinfo_hardware("hi6250"));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("HiSilicon Kirin 910T",
		parse_proc_cpuinfo_hardware("hi6620oem"));
#endif /* CPUINFO_ARCH_ARM */
}

#if CPUINFO_ARCH_ARM
	TEST(PROC_CPUINFO_HARDWARE, actions) {
		EXPECT_EQ("Actions ATM7029",
			parse_proc_cpuinfo_hardware("gs702a"));
		EXPECT_EQ("Actions ATM7029B",
			parse_proc_cpuinfo_hardware("gs702c"));
		EXPECT_EQ("Actions ATM7059A",
			parse_proc_cpuinfo_hardware("gs705a"));
	}
#endif /* CPUINFO_ARCH_ARM */

TEST(PROC_CPUINFO_HARDWARE, allwinner_sunxi) {
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Allwinner A10",
		parse_proc_cpuinfo_hardware("sun4i", 1));
	EXPECT_EQ("Allwinner A13",
		parse_proc_cpuinfo_hardware("sun5i", 1));
	EXPECT_EQ("Allwinner A20",
		parse_proc_cpuinfo_hardware("sun7i", 2));
	EXPECT_EQ("Allwinner A23",
		parse_proc_cpuinfo_hardware("sun8i", 2));
	EXPECT_EQ("Allwinner A31",
		parse_proc_cpuinfo_hardware("sun6i", 4));
	EXPECT_EQ("Allwinner A33",
		parse_proc_cpuinfo_hardware("sun8i", 4));
#endif /* CPUINFO_ARCH_ARM */
	EXPECT_EQ("Allwinner A64",
		parse_proc_cpuinfo_hardware("sun50iw1", 4));
	EXPECT_EQ("Allwinner A64",
		parse_proc_cpuinfo_hardware("sun50iw1p1", 4));
	EXPECT_EQ("Allwinner A64",
		parse_proc_cpuinfo_hardware("sun50iw2", 4));
#if CPUINFO_ARCH_ARM
	EXPECT_EQ("Allwinner A80",
		parse_proc_cpuinfo_hardware("sun9i", 8));
	EXPECT_EQ("Allwinner A83T",
		parse_proc_cpuinfo_hardware("sun8i", 8));
#endif /* CPUINFO_ARCH_ARM */
}

#if CPUINFO_ARCH_ARM
	TEST(PROC_CPUINFO_HARDWARE, amlogic) {
		EXPECT_EQ("Amlogic S805",
			parse_proc_cpuinfo_hardware("Amlogic Meson8B"));
		EXPECT_EQ("Amlogic S812",
			parse_proc_cpuinfo_hardware("Amlogic Meson8"));
	}

	TEST(PROC_CPUINFO_HARDWARE, lg) {
		EXPECT_EQ("LG Nuclun 7111",
			parse_proc_cpuinfo_hardware("Odin"));
	}

	TEST(PROC_CPUINFO_HARDWARE, marvell_pxa) {
		EXPECT_EQ("Marvell PXA1088",
			parse_proc_cpuinfo_hardware("PXA1088"));
		EXPECT_EQ("Marvell PXA1088",
			parse_proc_cpuinfo_hardware("PXA1L88"));
		EXPECT_EQ("Marvell PXA1908",
			parse_proc_cpuinfo_hardware("PXA1908"));
		EXPECT_EQ("Marvell PXA1928",
			parse_proc_cpuinfo_hardware("PXA1928"));
		EXPECT_EQ("Marvell PXA988",
			parse_proc_cpuinfo_hardware("PXA988"));
	}

	TEST(PROC_CPUINFO_HARDWARE, mstar) {
		EXPECT_EQ("MStar 6A338",
			parse_proc_cpuinfo_hardware("Madison"));
	}

	TEST(PROC_CPUINFO_HARDWARE, nvidia) {
		EXPECT_EQ("Nvidia Tegra AP20H",
			parse_proc_cpuinfo_hardware_tegra("picasso"));
		EXPECT_EQ("Nvidia Tegra AP20H",
			parse_proc_cpuinfo_hardware_tegra("picasso_e"));
		EXPECT_EQ("Nvidia Tegra AP20H",
			parse_proc_cpuinfo_hardware_tegra("stingray"));
		EXPECT_EQ("Nvidia Tegra AP33",
			parse_proc_cpuinfo_hardware_tegra("endeavoru"));
		EXPECT_EQ("Nvidia Tegra AP33",
			parse_proc_cpuinfo_hardware_tegra("x3"));
		EXPECT_EQ("Nvidia Tegra SL460N",
			parse_proc_cpuinfo_hardware_tegra("Ceres"));
		EXPECT_EQ("Nvidia Tegra T114",
			parse_proc_cpuinfo_hardware_tegra("macallan"));
		EXPECT_EQ("Nvidia Tegra T114",
			parse_proc_cpuinfo_hardware_tegra("mozart"));
		EXPECT_EQ("Nvidia Tegra T114",
			parse_proc_cpuinfo_hardware_tegra("tostab12BA"));
		EXPECT_EQ("Nvidia Tegra T124",
			parse_proc_cpuinfo_hardware_tegra("mocha"));
		EXPECT_EQ("Nvidia Tegra T124",
			parse_proc_cpuinfo_hardware_tegra("tn8"));
		EXPECT_EQ("Nvidia Tegra T20",
			parse_proc_cpuinfo_hardware_tegra("nbx03"));
		EXPECT_EQ("Nvidia Tegra T20",
			parse_proc_cpuinfo_hardware_tegra("p3"));
		EXPECT_EQ("Nvidia Tegra T20",
			parse_proc_cpuinfo_hardware_tegra("ventana"));
		EXPECT_EQ("Nvidia Tegra T30",
			parse_proc_cpuinfo_hardware_tegra("cardhu"));
		EXPECT_EQ("Nvidia Tegra T30",
			parse_proc_cpuinfo_hardware_tegra("chagall"));
		EXPECT_EQ("Nvidia Tegra T30",
			parse_proc_cpuinfo_hardware_tegra("picasso_m"));
		EXPECT_EQ("Nvidia Tegra T30",
			parse_proc_cpuinfo_hardware_tegra("picasso_mf"));
		EXPECT_EQ("Nvidia Tegra T30L",
			parse_proc_cpuinfo_hardware_tegra("BIRCH"));
		EXPECT_EQ("Nvidia Tegra T30L",
			parse_proc_cpuinfo_hardware_tegra("NS_14T004"));
		EXPECT_EQ("Nvidia Tegra T30L",
			parse_proc_cpuinfo_hardware_tegra("avalon"));
		EXPECT_EQ("Nvidia Tegra T30L",
			parse_proc_cpuinfo_hardware_tegra("picasso_e2"));
		EXPECT_EQ("Nvidia Tegra T30L",
			parse_proc_cpuinfo_hardware_tegra("tostab12BL"));
		EXPECT_EQ("Nvidia Tegra T30L",
			parse_proc_cpuinfo_hardware_tegra("txs03"));
		EXPECT_EQ("Nvidia Tegra T33",
			parse_proc_cpuinfo_hardware_tegra("bobsleigh"));
		EXPECT_EQ("Nvidia Tegra T33",
			parse_proc_cpuinfo_hardware_tegra("enrc2b"));
		EXPECT_EQ("Nvidia Tegra T33",
			parse_proc_cpuinfo_hardware_tegra("evitareul"));
		EXPECT_EQ("Nvidia Tegra T33",
			parse_proc_cpuinfo_hardware_tegra("tegra_fjdev103"));
	}
#endif /* CPUINFO_ARCH_ARM */

TEST(PROC_CPUINFO_HARDWARE, rockchip_rk) {
	EXPECT_EQ("Rockchip RK3126",
		parse_proc_cpuinfo_hardware("Rockchip RK3126"));
	EXPECT_EQ("Rockchip RK3128",
		parse_proc_cpuinfo_hardware("Rockchip RK3128"));
	EXPECT_EQ("Rockchip RK3188",
		parse_proc_cpuinfo_hardware("Rockchip RK3188"));
	EXPECT_EQ("Rockchip RK3228H",
		parse_proc_cpuinfo_hardware("rockchip,rk3228h"));
	EXPECT_EQ("Rockchip RK3229",
		parse_proc_cpuinfo_hardware("Rockchip RK3229"));
	EXPECT_EQ("Rockchip RK3328",
		parse_proc_cpuinfo_hardware("rockchip,rk3328"));
	EXPECT_EQ("Rockchip RK3368",
		parse_proc_cpuinfo_hardware("rockchip,rk3368"));
}

TEST(PROC_CPUINFO_HARDWARE, spreadtrum_sc) {
	EXPECT_EQ("Spreadtrum SC5735",
		parse_proc_cpuinfo_hardware("sc5735"));
	EXPECT_EQ("Spreadtrum SC6820I",
		parse_proc_cpuinfo_hardware("sc6820i"));
	EXPECT_EQ("Spreadtrum SC7715",
		parse_proc_cpuinfo_hardware("scx15"));
	EXPECT_EQ("Spreadtrum SC7730",
		parse_proc_cpuinfo_hardware("sc7730"));
	EXPECT_EQ("Spreadtrum SC7731",
		parse_proc_cpuinfo_hardware("sc7731"));
	EXPECT_EQ("Spreadtrum SC7731C",
		parse_proc_cpuinfo_hardware("sc7731c"));
	EXPECT_EQ("Spreadtrum SC7731G",
		parse_proc_cpuinfo_hardware("sc7731g"));
	EXPECT_EQ("Spreadtrum SC8825",
		parse_proc_cpuinfo_hardware("sc8825"));
	EXPECT_EQ("Spreadtrum SC8830",
		parse_proc_cpuinfo_hardware("sc8830"));
	EXPECT_EQ("Spreadtrum SC9830",
		parse_proc_cpuinfo_hardware("sc9830"));
	EXPECT_EQ("Spreadtrum SC9832",
		parse_proc_cpuinfo_hardware("sc9832"));
	EXPECT_EQ("Spreadtrum SC9832A",
		parse_proc_cpuinfo_hardware("sc9832a"));
}

TEST(PROC_CPUINFO_HARDWARE, telechips) {
	EXPECT_EQ("Telechips TCC892X",
		parse_proc_cpuinfo_hardware("tcc892x"));
	EXPECT_EQ("Telechips TCC893X",
		parse_proc_cpuinfo_hardware("tcc893x"));
}

#if CPUINFO_ARCH_ARM
	TEST(PROC_CPUINFO_HARDWARE, texas_instruments_omap) {
		EXPECT_EQ("Texas Instruments OMAP4430",
			parse_proc_cpuinfo_hardware("OMAP4430"));
		EXPECT_EQ("Texas Instruments OMAP4460",
			parse_proc_cpuinfo_hardware("OMAP4460"));
	}

	TEST(PROC_CPUINFO_HARDWARE, texas_instruments_special) {
		EXPECT_EQ("Texas Instruments OMAP4430",
			parse_proc_cpuinfo_hardware("mapphone_CDMA"));
		EXPECT_EQ("Texas Instruments OMAP4460",
			parse_proc_cpuinfo_hardware("Tuna"));
	}

	TEST(PROC_CPUINFO_HARDWARE, wondermedia) {
		EXPECT_EQ("WonderMedia WM8850",
			parse_proc_cpuinfo_hardware("WMT", 1, 1200000));
		EXPECT_EQ("WonderMedia WM8880",
			parse_proc_cpuinfo_hardware("WMT", 2, 1500000));
		EXPECT_EQ("WonderMedia WM8950",
			parse_proc_cpuinfo_hardware("WMT", 1, 1008000));
	}
#endif
