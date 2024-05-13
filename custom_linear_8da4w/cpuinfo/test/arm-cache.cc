#include <gtest/gtest.h>

#include <cstdint>

#include <cpuinfo.h>
extern "C" {
	#include <arm/api.h>
}


TEST(QUALCOMM, snapdragon_410_msm) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8916,
	};

	struct cpuinfo_cache l1i = { 0 };
	struct cpuinfo_cache l1d = { 0 };
	struct cpuinfo_cache l2 = { 0 };
	struct cpuinfo_cache l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD030),
		&chipset, 0, 8,
		&l1i, &l1d, &l2, &l3);
	EXPECT_EQ(32 * 1024, l1i.size);
	EXPECT_EQ(32 * 1024, l1d.size);
	EXPECT_EQ(512 * 1024, l2.size);
	EXPECT_EQ(0, l3.size);
}

TEST(QUALCOMM, snapdragon_410_apq) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_apq,
		.model = 8016,
	};

	struct cpuinfo_cache l1i = { 0 };
	struct cpuinfo_cache l1d = { 0 };
	struct cpuinfo_cache l2 = { 0 };
	struct cpuinfo_cache l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD030),
		&chipset, 0, 8,
		&l1i, &l1d, &l2, &l3);
	EXPECT_EQ(32 * 1024, l1i.size);
	EXPECT_EQ(32 * 1024, l1d.size);
	EXPECT_EQ(512 * 1024, l2.size);
	EXPECT_EQ(0, l3.size);
}

TEST(QUALCOMM, snapdragon_415) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8929,
	};

	for (uint32_t cluster = 0; cluster < 2; cluster++) {
		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD031),
			&chipset, cluster, 8,
			&l1i, &l1d, &l2, &l3);
		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(512 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}
}

TEST(QUALCOMM, snapdragon_425) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8917,
	};

	struct cpuinfo_cache l1i = { 0 };
	struct cpuinfo_cache l1d = { 0 };
	struct cpuinfo_cache l2 = { 0 };
	struct cpuinfo_cache l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 8,
		&l1i, &l1d, &l2, &l3);
	EXPECT_EQ(32 * 1024, l1i.size);
	EXPECT_EQ(32 * 1024, l1d.size);
	EXPECT_EQ(512 * 1024, l2.size);
	EXPECT_EQ(0, l3.size);
}

TEST(QUALCOMM, snapdragon_427) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8920,
	};

	struct cpuinfo_cache l1i = { 0 };
	struct cpuinfo_cache l1d = { 0 };
	struct cpuinfo_cache l2 = { 0 };
	struct cpuinfo_cache l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 8,
		&l1i, &l1d, &l2, &l3);
	EXPECT_EQ(32 * 1024, l1i.size);
	EXPECT_EQ(32 * 1024, l1d.size);
	EXPECT_EQ(512 * 1024, l2.size);
	EXPECT_EQ(0, l3.size);
}

TEST(QUALCOMM, snapdragon_430) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8937,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_435) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8940,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_450) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_snapdragon,
		.model = 450,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_617) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8952,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(512 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(256 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_625) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8953,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_626) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8953,
		.suffix = {
			[0] = 'P',
			[1] = 'R',
			[2] = 'O',
		},
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_630) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_snapdragon,
		.model = 630,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x51AF8014),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x51AF8014),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_636) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_snapdragon,
		.model = 636,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a73, 4, UINT32_C(0x51AF8002),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x51AF8014),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(64 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(1024 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_650) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8956,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a72, 2, UINT32_C(0x410FD080),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(48 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_652) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8976,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a72, 4, UINT32_C(0x410FD080),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(48 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_653) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8976,
		.suffix = {
			[0] = 'P',
			[1] = 'R',
			[2] = 'O',
		},
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a72, 4, UINT32_C(0x410FD080),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(48 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_660) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_snapdragon,
		.model = 660,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a73, 4, UINT32_C(0x51AF8002),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x51AF8014),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(64 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(1024 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_808) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8992,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a57, 2, UINT32_C(0x410FD033),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD033),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(48 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_810) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8994,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a57, 4, UINT32_C(0x410FD033),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD033),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(48 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_820) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8996,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_kryo, 4, UINT32_C(0x511F2052),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_kryo, 4, UINT32_C(0x511F2112),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(24 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(24 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_821) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8996,
		.suffix = {
			[0] = 'P',
			[1] = 'R',
			[2] = 'O',
			[3] = '-',
			[4] = 'A',
			[5] = 'C',
		},
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_kryo, 4, UINT32_C(0x512F2051),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_kryo, 4, UINT32_C(0x512F2011),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(24 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(24 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_835) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_msm,
		.model = 8998,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a73, 4, UINT32_C(0x51AF8001),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x51AF8014),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(64 * 1024, big_l1d.size);
	EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(1024 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(QUALCOMM, snapdragon_845) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_qualcomm,
		.series = cpuinfo_arm_chipset_series_qualcomm_snapdragon,
		.model = 845,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a75, 4, UINT32_C(0x518F802D),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a55r0, 4, UINT32_C(0x518F803C),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(64 * 1024, big_l1d.size);
	EXPECT_EQ(256 * 1024, big_l2.size);
	EXPECT_EQ(2 * 1024 * 1024, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(128 * 1024, little_l2.size);
	EXPECT_EQ(2 * 1024 * 1024, little_l3.size);
}

TEST(SAMSUNG, exynos_7885) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_samsung,
		.series = cpuinfo_arm_chipset_series_samsung_exynos,
		.model = 7885,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a73, 2, UINT32_C(0x410FD092),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 6, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(512 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(256 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(SAMSUNG, exynos_8890) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_samsung,
		.series = cpuinfo_arm_chipset_series_samsung_exynos,
		.model = 8890,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_exynos_m1, 4, UINT32_C(0x531F0011),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(256 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(SAMSUNG, exynos_8895) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_samsung,
		.series = cpuinfo_arm_chipset_series_samsung_exynos,
		.model = 8890,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_exynos_m2, 4, UINT32_C(0x534F0010),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(256 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(SAMSUNG, exynos_9810) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_samsung,
		.series = cpuinfo_arm_chipset_series_samsung_exynos,
		.model = 9810,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_exynos_m3, 4, UINT32_C(0x531F0020),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a55r0, 4, UINT32_C(0x410FD051),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(64 * 1024, big_l1d.size);
	EXPECT_EQ(512 * 1024, big_l2.size);
	EXPECT_EQ(4 * 1024 * 1024, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(MEDIATEK, mediatek_mt8173) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_mediatek,
		.series = cpuinfo_arm_chipset_series_mediatek_mt,
		.model = 8173,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a72, 2, UINT32_C(0x410FD080),
		&chipset, 0, 4,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 2, UINT32_C(0x410FD032),
		&chipset, 1, 4,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(48 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(MEDIATEK, mediatek_mt8173c) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_mediatek,
		.series = cpuinfo_arm_chipset_series_mediatek_mt,
		.model = 8173,
		.suffix = {
			[0] = 'C',
		},
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a72, 2, UINT32_C(0x410FD080),
		&chipset, 0, 4,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 2, UINT32_C(0x410FD032),
		&chipset, 1, 4,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(48 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(HISILICON, kirin_650) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_hisilicon,
		.series = cpuinfo_arm_chipset_series_hisilicon_kirin,
		.model = 650,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(512 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(HISILICON, kirin_659) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_hisilicon,
		.series = cpuinfo_arm_chipset_series_hisilicon_kirin,
		.model = 659,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(512 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

#if CPUINFO_ARCH_ARM
	TEST(HISILICON, kirin_920) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_hisilicon,
			.series = cpuinfo_arm_chipset_series_hisilicon_kirin,
			.model = 920,
		};

		struct cpuinfo_cache big_l1i = { 0 };
		struct cpuinfo_cache big_l1d = { 0 };
		struct cpuinfo_cache big_l2 = { 0 };
		struct cpuinfo_cache big_l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a15, 4, UINT32_C(0x413FC0F3),
			&chipset, 0, 8,
			&big_l1i, &big_l1d, &big_l2, &big_l3);

		struct cpuinfo_cache little_l1i = { 0 };
		struct cpuinfo_cache little_l1d = { 0 };
		struct cpuinfo_cache little_l2 = { 0 };
		struct cpuinfo_cache little_l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a7, 4, UINT32_C(0x410FC075),
			&chipset, 1, 8,
			&little_l1i, &little_l1d, &little_l2, &little_l3);

		EXPECT_EQ(32 * 1024, big_l1i.size);
		EXPECT_EQ(32 * 1024, big_l1d.size);
		EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
		EXPECT_EQ(0, big_l3.size);

		EXPECT_EQ(32 * 1024, little_l1i.size); /* TODO: verify */
		EXPECT_EQ(32 * 1024, little_l1d.size); /* TODO: verify */
		EXPECT_EQ(512 * 1024, little_l2.size);
		EXPECT_EQ(0, little_l3.size);
	}

	TEST(HISILICON, kirin_925) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_hisilicon,
			.series = cpuinfo_arm_chipset_series_hisilicon_kirin,
			.model = 925,
		};

		struct cpuinfo_cache big_l1i = { 0 };
		struct cpuinfo_cache big_l1d = { 0 };
		struct cpuinfo_cache big_l2 = { 0 };
		struct cpuinfo_cache big_l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a15, 4, UINT32_C(0x413FC0F3),
			&chipset, 0, 8,
			&big_l1i, &big_l1d, &big_l2, &big_l3);

		struct cpuinfo_cache little_l1i = { 0 };
		struct cpuinfo_cache little_l1d = { 0 };
		struct cpuinfo_cache little_l2 = { 0 };
		struct cpuinfo_cache little_l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a7, 4, UINT32_C(0x410FC075),
			&chipset, 1, 8,
			&little_l1i, &little_l1d, &little_l2, &little_l3);

		EXPECT_EQ(32 * 1024, big_l1i.size);
		EXPECT_EQ(32 * 1024, big_l1d.size);
		EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
		EXPECT_EQ(0, big_l3.size);

		EXPECT_EQ(32 * 1024, little_l1i.size); /* TODO: verify */
		EXPECT_EQ(32 * 1024, little_l1d.size); /* TODO: verify */
		EXPECT_EQ(512 * 1024, little_l2.size);
		EXPECT_EQ(0, little_l3.size);
	}

	TEST(HISILICON, kirin_928) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_hisilicon,
			.series = cpuinfo_arm_chipset_series_hisilicon_kirin,
			.model = 928,
		};

		struct cpuinfo_cache big_l1i = { 0 };
		struct cpuinfo_cache big_l1d = { 0 };
		struct cpuinfo_cache big_l2 = { 0 };
		struct cpuinfo_cache big_l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a15, 4, UINT32_C(0x413FC0F3),
			&chipset, 0, 8,
			&big_l1i, &big_l1d, &big_l2, &big_l3);

		struct cpuinfo_cache little_l1i = { 0 };
		struct cpuinfo_cache little_l1d = { 0 };
		struct cpuinfo_cache little_l2 = { 0 };
		struct cpuinfo_cache little_l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a7, 4, UINT32_C(0x410FC075),
			&chipset, 1, 8,
			&little_l1i, &little_l1d, &little_l2, &little_l3);

		EXPECT_EQ(32 * 1024, big_l1i.size);
		EXPECT_EQ(32 * 1024, big_l1d.size);
		EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
		EXPECT_EQ(0, big_l3.size);

		EXPECT_EQ(32 * 1024, little_l1i.size); /* TODO: verify */
		EXPECT_EQ(32 * 1024, little_l1d.size); /* TODO: verify */
		EXPECT_EQ(512 * 1024, little_l2.size);
		EXPECT_EQ(0, little_l3.size);
	}
#endif /* CPUINFO_ARCH_ARM */

TEST(HISILICON, kirin_950) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_hisilicon,
		.series = cpuinfo_arm_chipset_series_hisilicon_kirin,
		.model = 950,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a72, 4, UINT32_C(0x410FD080),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(48 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(HISILICON, kirin_955) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_hisilicon,
		.series = cpuinfo_arm_chipset_series_hisilicon_kirin,
		.model = 955,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a72, 4, UINT32_C(0x410FD080),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(48 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(HISILICON, kirin_960) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_hisilicon,
		.series = cpuinfo_arm_chipset_series_hisilicon_kirin,
		.model = 960,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a73, 4, UINT32_C(0x410FD091),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(64 * 1024, big_l1d.size);
	EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(512 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(HISILICON, kirin_970) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_hisilicon,
		.series = cpuinfo_arm_chipset_series_hisilicon_kirin,
		.model = 970,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a73, 4, UINT32_C(0x410FD092),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(64 * 1024, big_l1d.size);
	EXPECT_EQ(2 * 1024 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(1024 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

TEST(HISILICON, kirin_980) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_hisilicon,
		.series = cpuinfo_arm_chipset_series_hisilicon_kirin,
		.model = 980,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a76, 2, UINT32_C(0x481FD400),
		&chipset, 0, 2,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache middle_l1i = { 0 };
	struct cpuinfo_cache middle_l1d = { 0 };
	struct cpuinfo_cache middle_l2 = { 0 };
	struct cpuinfo_cache middle_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a76, 2, UINT32_C(0x481FD400),
		&chipset, 1, 2,
		&middle_l1i, &middle_l1d, &middle_l2, &middle_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a55, 4, UINT32_C(0x411FD050),
		&chipset, 2, 4,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(64 * 1024, big_l1i.size);
	EXPECT_EQ(64 * 1024, big_l1d.size);
	EXPECT_EQ(512 * 1024, big_l2.size);
	EXPECT_EQ(4 * 1024 * 1024, big_l3.size);

	EXPECT_EQ(64 * 1024, middle_l1i.size);
	EXPECT_EQ(64 * 1024, middle_l1d.size);
	EXPECT_EQ(512 * 1024, middle_l2.size);
	EXPECT_EQ(4 * 1024 * 1024, middle_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(128 * 1024, little_l2.size);
	EXPECT_EQ(4 * 1024 * 1024, little_l3.size);
}

#if CPUINFO_ARCH_ARM
	TEST(NVIDIA, tegra_ap20h) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_nvidia,
			.series = cpuinfo_arm_chipset_series_nvidia_tegra_ap,
			.model = 20,
			.suffix = {
				[0] = 'H',
			},
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a9, 2, UINT32_C(0x411FC090),
			&chipset, 0, 7,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(1024 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}

	TEST(NVIDIA, tegra_t20) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_nvidia,
			.series = cpuinfo_arm_chipset_series_nvidia_tegra_t,
			.model = 20,
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a9, 2, UINT32_C(0x411FC090),
			&chipset, 0, 7,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(1024 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}

	TEST(NVIDIA, tegra_t30l) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_nvidia,
			.series = cpuinfo_arm_chipset_series_nvidia_tegra_t,
			.model = 30,
			.suffix = {
				[0] = 'L',
			},
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a9, 4, UINT32_C(0x412FC099),
			&chipset, 0, 7,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(1024 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}

	TEST(NVIDIA, tegra_t30) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_nvidia,
			.series = cpuinfo_arm_chipset_series_nvidia_tegra_t,
			.model = 30,
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a9, 4, UINT32_C(0x412FC099),
			&chipset, 0, 7,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(1024 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}

	TEST(NVIDIA, tegra_t33) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_nvidia,
			.series = cpuinfo_arm_chipset_series_nvidia_tegra_t,
			.model = 33,
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a9, 4, UINT32_C(0x412FC099),
			&chipset, 0, 7,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(1024 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}

	TEST(NVIDIA, tegra_ap33) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_nvidia,
			.series = cpuinfo_arm_chipset_series_nvidia_tegra_ap,
			.model = 33,
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a9, 4, UINT32_C(0x412FC099),
			&chipset, 0, 7,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(1024 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}

	TEST(NVIDIA, tegra_t114) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_nvidia,
			.series = cpuinfo_arm_chipset_series_nvidia_tegra_t,
			.model = 114,
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a15, 4, UINT32_C(0x412FC0F2),
			&chipset, 0, 7,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(2 * 1024 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}

	TEST(NVIDIA, tegra_sl460n) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_nvidia,
			.series = cpuinfo_arm_chipset_series_nvidia_tegra_sl,
			.model = 460,
			.suffix = {
				[0] = 'N',
			},
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a9, 4, UINT32_C(0x414FC091),
			&chipset, 0, 7,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(1 * 1024 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}

	TEST(NVIDIA, tegra_t124) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_nvidia,
			.series = cpuinfo_arm_chipset_series_nvidia_tegra_t,
			.model = 124,
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a15, 4, UINT32_C(0x413FC0F3),
			&chipset, 0, 7,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(2 * 1024 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}
#endif /* CPUINFO_ARCH_ARM */

TEST(NVIDIA, tegra_t132) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_nvidia,
		.series = cpuinfo_arm_chipset_series_nvidia_tegra_t,
		.model = 132,
	};

	struct cpuinfo_cache l1i = { 0 };
	struct cpuinfo_cache l1d = { 0 };
	struct cpuinfo_cache l2 = { 0 };
	struct cpuinfo_cache l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_denver, 2, UINT32_C(0x4E0F0000),
		&chipset, 0, 8,
		&l1i, &l1d, &l2, &l3);

	EXPECT_EQ(128 * 1024, l1i.size);
	EXPECT_EQ(64 * 1024, l1d.size);
	EXPECT_EQ(2 * 1024 * 1024, l2.size);
	EXPECT_EQ(0, l3.size);
}

TEST(NVIDIA, tegra_t210) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_nvidia,
		.series = cpuinfo_arm_chipset_series_nvidia_tegra_t,
		.model = 210,
	};

	struct cpuinfo_cache l1i = { 0 };
	struct cpuinfo_cache l1d = { 0 };
	struct cpuinfo_cache l2 = { 0 };
	struct cpuinfo_cache l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a57, 4, UINT32_C(0x411FD071),
		&chipset, 0, 8,
		&l1i, &l1d, &l2, &l3);

	EXPECT_EQ(48 * 1024, l1i.size);
	EXPECT_EQ(32 * 1024, l1d.size);
	EXPECT_EQ(2 * 1024 * 1024, l2.size);
	EXPECT_EQ(0, l3.size);
}

TEST(ROCKCHIP, rk3368) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_rockchip,
		.series = cpuinfo_arm_chipset_series_rockchip_rk,
		.model = 3368,
	};

	struct cpuinfo_cache big_l1i = { 0 };
	struct cpuinfo_cache big_l1d = { 0 };
	struct cpuinfo_cache big_l2 = { 0 };
	struct cpuinfo_cache big_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD033),
		&chipset, 0, 8,
		&big_l1i, &big_l1d, &big_l2, &big_l3);

	struct cpuinfo_cache little_l1i = { 0 };
	struct cpuinfo_cache little_l1d = { 0 };
	struct cpuinfo_cache little_l2 = { 0 };
	struct cpuinfo_cache little_l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD033),
		&chipset, 1, 8,
		&little_l1i, &little_l1d, &little_l2, &little_l3);

	EXPECT_EQ(32 * 1024, big_l1i.size);
	EXPECT_EQ(32 * 1024, big_l1d.size);
	EXPECT_EQ(512 * 1024, big_l2.size);
	EXPECT_EQ(0, big_l3.size);

	EXPECT_EQ(32 * 1024, little_l1i.size);
	EXPECT_EQ(32 * 1024, little_l1d.size);
	EXPECT_EQ(256 * 1024, little_l2.size);
	EXPECT_EQ(0, little_l3.size);
}

#if CPUINFO_ARCH_ARM
	TEST(BROADCOM, bcm2835) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_broadcom,
			.series = cpuinfo_arm_chipset_series_broadcom_bcm,
			.model = 2835,
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_arm11, 4, UINT32_C(0x410FB767),
			&chipset, 0, 4,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(16 * 1024, l1i.size);
		EXPECT_EQ(16 * 1024, l1d.size);
		EXPECT_EQ(0, l2.size);
		EXPECT_EQ(0, l3.size);
	}

	TEST(BROADCOM, bcm2836) {
		const struct cpuinfo_arm_chipset chipset = {
			.vendor = cpuinfo_arm_chipset_vendor_broadcom,
			.series = cpuinfo_arm_chipset_series_broadcom_bcm,
			.model = 2836,
		};

		struct cpuinfo_cache l1i = { 0 };
		struct cpuinfo_cache l1d = { 0 };
		struct cpuinfo_cache l2 = { 0 };
		struct cpuinfo_cache l3 = { 0 };
		cpuinfo_arm_decode_cache(
			cpuinfo_uarch_cortex_a7, 4, UINT32_C(0x410FC075),
			&chipset, 0, 4,
			&l1i, &l1d, &l2, &l3);

		EXPECT_EQ(32 * 1024, l1i.size);
		EXPECT_EQ(32 * 1024, l1d.size);
		EXPECT_EQ(512 * 1024, l2.size);
		EXPECT_EQ(0, l3.size);
	}
#endif /* CPUINFO_ARCH_ARM */

TEST(BROADCOM, bcm2837) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_broadcom,
		.series = cpuinfo_arm_chipset_series_broadcom_bcm,
		.model = 2837,
	};

	struct cpuinfo_cache l1i = { 0 };
	struct cpuinfo_cache l1d = { 0 };
	struct cpuinfo_cache l2 = { 0 };
	struct cpuinfo_cache l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a53, 4, UINT32_C(0x410FD034),
		&chipset, 0, 4,
		&l1i, &l1d, &l2, &l3);

	EXPECT_EQ(16 * 1024, l1i.size);
	EXPECT_EQ(16 * 1024, l1d.size);
	EXPECT_EQ(512 * 1024, l2.size);
	EXPECT_EQ(0, l3.size);
}

TEST(BROADCOM, bcm2711) {
	const struct cpuinfo_arm_chipset chipset = {
		.vendor = cpuinfo_arm_chipset_vendor_broadcom,
		.series = cpuinfo_arm_chipset_series_broadcom_bcm,
		.model = 2711,
	};

	struct cpuinfo_cache l1i = { 0 };
	struct cpuinfo_cache l1d = { 0 };
	struct cpuinfo_cache l2 = { 0 };
	struct cpuinfo_cache l3 = { 0 };
	cpuinfo_arm_decode_cache(
		cpuinfo_uarch_cortex_a72, 4, UINT32_C(0x410FD083),
		&chipset, 0, 4,
		&l1i, &l1d, &l2, &l3);

	EXPECT_EQ(48 * 1024, l1i.size);
	EXPECT_EQ(32 * 1024, l1d.size);
	EXPECT_EQ(1024 * 1024, l2.size);
	EXPECT_EQ(0, l3.size);
}
