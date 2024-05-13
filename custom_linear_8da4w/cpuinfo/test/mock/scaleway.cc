#include <gtest/gtest.h>

#include <cpuinfo.h>
#include <cpuinfo-mock.h>


TEST(PROCESSORS, count) {
	ASSERT_EQ(2, cpuinfo_processors_count);
}

TEST(PROCESSORS, non_null) {
	ASSERT_TRUE(cpuinfo_get_processors());
}

TEST(PROCESSORS, vendor) {
	for (uint32_t i = 0; i < cpuinfo_processors_count; i++) {
		ASSERT_EQ(cpuinfo_vendor_cavium, cpuinfo_get_processors()[i].vendor);
	}
}

TEST(PROCESSORS, uarch) {
	for (uint32_t i = 0; i < cpuinfo_processors_count; i++) {
		ASSERT_EQ(cpuinfo_uarch_thunderx, cpuinfo_get_processors()[i].uarch);
	}
}

TEST(ISA, thumb) {
	ASSERT_TRUE(cpuinfo_isa.thumb);
}

TEST(ISA, thumb2) {
	ASSERT_TRUE(cpuinfo_isa.thumb2);
}

TEST(ISA, thumbee) {
	ASSERT_FALSE(cpuinfo_isa.thumbee);
}

TEST(ISA, jazelle) {
	ASSERT_FALSE(cpuinfo_isa.jazelle);
}

TEST(ISA, armv5e) {
	ASSERT_TRUE(cpuinfo_isa.armv5e);
}

TEST(ISA, armv6) {
	ASSERT_TRUE(cpuinfo_isa.armv6);
}

TEST(ISA, armv6k) {
	ASSERT_TRUE(cpuinfo_isa.armv6k);
}

TEST(ISA, armv7) {
	ASSERT_TRUE(cpuinfo_isa.armv7);
}

TEST(ISA, armv7mp) {
	ASSERT_TRUE(cpuinfo_isa.armv7mp);
}

TEST(ISA, idiv) {
	ASSERT_TRUE(cpuinfo_isa.idiv);
}

TEST(ISA, vfpv2) {
	ASSERT_FALSE(cpuinfo_isa.vfpv2);
}

TEST(ISA, vfpv3) {
	ASSERT_TRUE(cpuinfo_isa.vfpv3);
}

TEST(ISA, d32) {
	ASSERT_TRUE(cpuinfo_isa.d32);
}

TEST(ISA, fp16) {
	ASSERT_TRUE(cpuinfo_isa.fp16);
}

TEST(ISA, fma) {
	ASSERT_TRUE(cpuinfo_isa.fma);
}

TEST(ISA, wmmx) {
	ASSERT_FALSE(cpuinfo_isa.wmmx);
}

TEST(ISA, wmmx2) {
	ASSERT_FALSE(cpuinfo_isa.wmmx2);
}

TEST(ISA, neon) {
	ASSERT_TRUE(cpuinfo_isa.neon);
}

TEST(ISA, aes) {
	ASSERT_TRUE(cpuinfo_isa.aes);
}

TEST(ISA, sha1) {
	ASSERT_TRUE(cpuinfo_isa.sha1);
}

TEST(ISA, sha2) {
	ASSERT_TRUE(cpuinfo_isa.sha2);
}

TEST(ISA, pmull) {
	ASSERT_TRUE(cpuinfo_isa.pmull);
}

TEST(ISA, crc32) {
	ASSERT_TRUE(cpuinfo_isa.crc32);
}

#if CPUINFO_ARCH_ARM64
TEST(ISA, atomics) {
	ASSERT_TRUE(cpuinfo_isa.atomics);
}

TEST(ISA, rdm) {
	ASSERT_FALSE(cpuinfo_isa.rdm);
}

TEST(ISA, fp16arith) {
	ASSERT_FALSE(cpuinfo_isa.fp16arith);
}

TEST(ISA, jscvt) {
	ASSERT_FALSE(cpuinfo_isa.jscvt);
}

TEST(ISA, fcma) {
	ASSERT_FALSE(cpuinfo_isa.fcma);
}
#endif /* CPUINFO_ARCH_ARM64 */

TEST(L1I, count) {
	ASSERT_EQ(2, cpuinfo_get_l1i_caches_count());
}

TEST(L1I, non_null) {
	ASSERT_TRUE(cpuinfo_get_l1i_caches());
}

TEST(L1I, size) {
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		ASSERT_EQ(78 * 1024, cpuinfo_get_l1i_cache(i)->size);
	}
}

TEST(L1I, associativity) {
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		ASSERT_EQ(4, cpuinfo_get_l1i_cache(i)->associativity);
	}
}

TEST(L1I, sets) {
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		ASSERT_EQ(312, cpuinfo_get_l1i_cache(i)->sets);
	}
}

TEST(L1I, partitions) {
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		ASSERT_EQ(1, cpuinfo_get_l1i_cache(i)->partitions);
	}
}

TEST(L1I, line_size) {
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		ASSERT_EQ(64, cpuinfo_get_l1i_cache(i)->line_size);
	}
}

TEST(L1I, flags) {
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		ASSERT_EQ(0, cpuinfo_get_l1i_cache(i)->flags);
	}
}

TEST(L1I, processors) {
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		ASSERT_EQ(i, cpuinfo_get_l1i_cache(i)->processor_start);
		ASSERT_EQ(1, cpuinfo_get_l1i_cache(i)->processor_count);
	}
}

TEST(L1D, count) {
	ASSERT_EQ(2, cpuinfo_get_l1d_caches_count());
}

TEST(L1D, non_null) {
	ASSERT_TRUE(cpuinfo_get_l1d_caches());
}

TEST(L1D, size) {
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		ASSERT_EQ(32 * 1024, cpuinfo_get_l1d_cache(i)->size);
	}
}

TEST(L1D, associativity) {
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		ASSERT_EQ(4, cpuinfo_get_l1d_cache(i)->associativity);
	}
}

TEST(L1D, sets) {
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		ASSERT_EQ(128, cpuinfo_get_l1d_cache(i)->sets);
	}
}

TEST(L1D, partitions) {
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		ASSERT_EQ(1, cpuinfo_get_l1d_cache(i)->partitions);
	}
}

TEST(L1D, line_size) {
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		ASSERT_EQ(64, cpuinfo_get_l1d_cache(i)->line_size);
	}
}

TEST(L1D, flags) {
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		ASSERT_EQ(0, cpuinfo_get_l1d_cache(i)->flags);
	}
}

TEST(L1D, processors) {
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		ASSERT_EQ(i, cpuinfo_get_l1d_cache(i)->processor_start);
		ASSERT_EQ(1, cpuinfo_get_l1d_cache(i)->processor_count);
	}
}

TEST(L2, count) {
	ASSERT_EQ(1, cpuinfo_get_l2_caches_count());
}

TEST(L2, non_null) {
	ASSERT_TRUE(cpuinfo_get_l2_caches());
}

TEST(L2, size) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(16 * 1024 * 1024, cpuinfo_get_l2_cache(i)->size);
	}
}

TEST(L2, associativity) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(8, cpuinfo_get_l2_cache(i)->associativity);
	}
}

TEST(L2, sets) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(32768, cpuinfo_get_l2_cache(i)->sets);
	}
}

TEST(L2, partitions) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(1, cpuinfo_get_l2_cache(i)->partitions);
	}
}

TEST(L2, line_size) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(64, cpuinfo_get_l2_cache(i)->line_size);
	}
}

TEST(L2, flags) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(0, cpuinfo_get_l2_cache(i)->flags);
	}
}

TEST(L2, processors) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(0, cpuinfo_get_l2_cache(i)->processor_start);
		ASSERT_EQ(2, cpuinfo_get_l2_cache(i)->processor_count);
	}
}

TEST(L3, none) {
	ASSERT_EQ(0, cpuinfo_get_l3_caches_count());
	ASSERT_FALSE(cpuinfo_get_l3_caches());
}

TEST(L4, none) {
	ASSERT_EQ(0, cpuinfo_get_l4_caches_count());
	ASSERT_FALSE(cpuinfo_get_l4_caches());
}

#include <scaleway.h>

int main(int argc, char* argv[]) {
	cpuinfo_mock_filesystem(filesystem);
	cpuinfo_initialize();
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}