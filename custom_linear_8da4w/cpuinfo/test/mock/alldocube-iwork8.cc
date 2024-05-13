#include <gtest/gtest.h>

#include <cpuinfo.h>
#include <cpuinfo-mock.h>


TEST(PROCESSORS, count) {
	ASSERT_EQ(4, cpuinfo_get_processors_count());
}

TEST(PROCESSORS, non_null) {
	ASSERT_TRUE(cpuinfo_get_processors());
}

TEST(PROCESSORS, smt_id) {
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		ASSERT_EQ(0, cpuinfo_get_processor(i)->smt_id);
	}
}

TEST(PROCESSORS, core) {
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		ASSERT_EQ(cpuinfo_get_core(i), cpuinfo_get_processor(i)->core);
	}
}

TEST(PROCESSORS, cluster) {
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		ASSERT_EQ(cpuinfo_get_cluster(i / 2), cpuinfo_get_processor(i)->cluster);
	}
}

TEST(PROCESSORS, package) {
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		ASSERT_EQ(cpuinfo_get_package(0), cpuinfo_get_processor(i)->package);
	}
}

TEST(PROCESSORS, linux_id) {
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		ASSERT_EQ(i, cpuinfo_get_processor(i)->linux_id);
	}
}

TEST(PROCESSORS, l1i) {
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		ASSERT_EQ(cpuinfo_get_l1i_cache(i), cpuinfo_get_processor(i)->cache.l1i);
	}
}

TEST(PROCESSORS, l1d) {
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		ASSERT_EQ(cpuinfo_get_l1d_cache(i), cpuinfo_get_processor(i)->cache.l1d);
	}
}

TEST(PROCESSORS, l2) {
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		ASSERT_EQ(cpuinfo_get_l2_cache(i / 2), cpuinfo_get_processor(i)->cache.l2);
	}
}

TEST(PROCESSORS, l3) {
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		ASSERT_FALSE(cpuinfo_get_processor(i)->cache.l3);
	}
}

TEST(PROCESSORS, l4) {
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		ASSERT_FALSE(cpuinfo_get_processor(i)->cache.l4);
	}
}

TEST(CORES, count) {
	ASSERT_EQ(4, cpuinfo_get_cores_count());
}

TEST(CORES, non_null) {
	ASSERT_TRUE(cpuinfo_get_cores());
}

TEST(CORES, processor_start) {
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		ASSERT_EQ(i, cpuinfo_get_core(i)->processor_start);
	}
}

TEST(CORES, processor_count) {
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		ASSERT_EQ(1, cpuinfo_get_core(i)->processor_count);
	}
}

TEST(CORES, core_id) {
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		ASSERT_EQ(i, cpuinfo_get_core(i)->core_id);
	}
}

TEST(CORES, cluster) {
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		ASSERT_EQ(cpuinfo_get_cluster(i / 2), cpuinfo_get_core(i)->cluster);
	}
}

TEST(CORES, package) {
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		ASSERT_EQ(cpuinfo_get_package(0), cpuinfo_get_core(i)->package);
	}
}

TEST(CORES, vendor) {
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		ASSERT_EQ(cpuinfo_vendor_intel, cpuinfo_get_core(i)->vendor);
	}
}

TEST(CORES, uarch) {
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		ASSERT_EQ(cpuinfo_uarch_airmont, cpuinfo_get_core(i)->uarch);
	}
}

TEST(CORES, cpuid) {
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		ASSERT_EQ(UINT32_C(0x000406C4), cpuinfo_get_core(i)->cpuid);
	}
}

TEST(CORES, DISABLED_frequency) {
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		ASSERT_EQ(UINT64_C(1920000000), cpuinfo_get_core(i)->frequency);
	}
}

TEST(CLUSTERS, count) {
	ASSERT_EQ(2, cpuinfo_get_clusters_count());
}

TEST(CLUSTERS, non_null) {
	ASSERT_TRUE(cpuinfo_get_clusters());
}

TEST(CLUSTERS, processor_start) {
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		ASSERT_EQ(i * 2, cpuinfo_get_cluster(i)->processor_start);
	}
}

TEST(CLUSTERS, processor_count) {
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		ASSERT_EQ(2, cpuinfo_get_cluster(i)->processor_count);
	}
}

TEST(CLUSTERS, core_start) {
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		ASSERT_EQ(i * 2, cpuinfo_get_cluster(i)->core_start);
	}
}

TEST(CLUSTERS, core_count) {
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		ASSERT_EQ(2, cpuinfo_get_cluster(i)->core_count);
	}
}

TEST(CLUSTERS, cluster_id) {
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		ASSERT_EQ(i, cpuinfo_get_cluster(i)->cluster_id);
	}
}

TEST(CLUSTERS, package) {
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		ASSERT_EQ(cpuinfo_get_package(0), cpuinfo_get_cluster(i)->package);
	}
}

TEST(CLUSTERS, vendor) {
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		ASSERT_EQ(cpuinfo_vendor_intel, cpuinfo_get_cluster(i)->vendor);
	}
}

TEST(CLUSTERS, uarch) {
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		ASSERT_EQ(cpuinfo_uarch_airmont, cpuinfo_get_cluster(i)->uarch);
	}
}

TEST(CLUSTERS, cpuid) {
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		ASSERT_EQ(UINT32_C(0x000406C4), cpuinfo_get_cluster(i)->cpuid);
	}
}

TEST(CLUSTERS, DISABLED_frequency) {
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		ASSERT_EQ(UINT64_C(1920000000), cpuinfo_get_cluster(i)->frequency);
	}
}

TEST(PACKAGES, count) {
	ASSERT_EQ(1, cpuinfo_get_packages_count());
}

TEST(PACKAGES, non_null) {
	ASSERT_TRUE(cpuinfo_get_packages());
}

TEST(PACKAGES, name) {
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		ASSERT_EQ("Intel Atom x5-Z8350",
			std::string(cpuinfo_get_package(i)->name,
				strnlen(cpuinfo_get_package(i)->name, CPUINFO_PACKAGE_NAME_MAX)));
	}
}

TEST(PACKAGES, processor_start) {
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		ASSERT_EQ(0, cpuinfo_get_package(i)->processor_start);
	}
}

TEST(PACKAGES, processor_count) {
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		ASSERT_EQ(4, cpuinfo_get_package(i)->processor_count);
	}
}

TEST(PACKAGES, core_start) {
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		ASSERT_EQ(0, cpuinfo_get_package(i)->core_start);
	}
}

TEST(PACKAGES, core_count) {
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		ASSERT_EQ(4, cpuinfo_get_package(i)->core_count);
	}
}

TEST(PACKAGES, cluster_start) {
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		ASSERT_EQ(0, cpuinfo_get_package(i)->cluster_start);
	}
}

TEST(PACKAGES, cluster_count) {
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		ASSERT_EQ(2, cpuinfo_get_package(i)->cluster_count);
	}
}

TEST(ISA, rdtsc) {
	ASSERT_TRUE(cpuinfo_has_x86_rdtsc());
}

TEST(ISA, rdtscp) {
	ASSERT_TRUE(cpuinfo_has_x86_rdtscp());
}

TEST(ISA, rdpid) {
	ASSERT_FALSE(cpuinfo_has_x86_rdpid());
}

TEST(ISA, clzero) {
	ASSERT_FALSE(cpuinfo_has_x86_clzero());
}

TEST(ISA, mwait) {
	ASSERT_TRUE(cpuinfo_has_x86_mwait());
}

TEST(ISA, mwaitx) {
	ASSERT_FALSE(cpuinfo_has_x86_mwaitx());
}

TEST(ISA, fxsave) {
	ASSERT_TRUE(cpuinfo_has_x86_fxsave());
}

TEST(ISA, xsave) {
	ASSERT_FALSE(cpuinfo_has_x86_xsave());
}

TEST(ISA, fpu) {
	ASSERT_TRUE(cpuinfo_has_x86_fpu());
}

TEST(ISA, mmx) {
	ASSERT_TRUE(cpuinfo_has_x86_mmx());
}

TEST(ISA, mmx_plus) {
	ASSERT_TRUE(cpuinfo_has_x86_mmx_plus());
}

TEST(ISA, three_d_now) {
	ASSERT_FALSE(cpuinfo_has_x86_3dnow());
}

TEST(ISA, three_d_now_plus) {
	ASSERT_FALSE(cpuinfo_has_x86_3dnow_plus());
}

TEST(ISA, three_d_now_geode) {
	ASSERT_FALSE(cpuinfo_has_x86_3dnow_geode());
}

TEST(ISA, prefetch) {
	ASSERT_FALSE(cpuinfo_has_x86_prefetch());
}

TEST(ISA, prefetchw) {
	ASSERT_TRUE(cpuinfo_has_x86_prefetchw());
}

TEST(ISA, prefetchwt1) {
	ASSERT_FALSE(cpuinfo_has_x86_prefetchwt1());
}

TEST(ISA, daz) {
	ASSERT_TRUE(cpuinfo_has_x86_daz());
}

TEST(ISA, sse) {
	ASSERT_TRUE(cpuinfo_has_x86_sse());
}

TEST(ISA, sse2) {
	ASSERT_TRUE(cpuinfo_has_x86_sse2());
}

TEST(ISA, sse3) {
	ASSERT_TRUE(cpuinfo_has_x86_sse3());
}

TEST(ISA, ssse3) {
	ASSERT_TRUE(cpuinfo_has_x86_ssse3());
}

TEST(ISA, sse4_1) {
	ASSERT_TRUE(cpuinfo_has_x86_sse4_1());
}

TEST(ISA, sse4_2) {
	ASSERT_TRUE(cpuinfo_has_x86_sse4_2());
}

TEST(ISA, sse4a) {
	ASSERT_FALSE(cpuinfo_has_x86_sse4a());
}

TEST(ISA, misaligned_sse) {
	ASSERT_FALSE(cpuinfo_has_x86_misaligned_sse());
}

TEST(ISA, avx) {
	ASSERT_FALSE(cpuinfo_has_x86_avx());
}

TEST(ISA, fma3) {
	ASSERT_FALSE(cpuinfo_has_x86_fma3());
}

TEST(ISA, fma4) {
	ASSERT_FALSE(cpuinfo_has_x86_fma4());
}

TEST(ISA, xop) {
	ASSERT_FALSE(cpuinfo_has_x86_xop());
}

TEST(ISA, f16c) {
	ASSERT_FALSE(cpuinfo_has_x86_f16c());
}

TEST(ISA, avx2) {
	ASSERT_FALSE(cpuinfo_has_x86_avx2());
}

TEST(ISA, avx512f) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512f());
}

TEST(ISA, avx512pf) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512pf());
}

TEST(ISA, avx512er) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512er());
}

TEST(ISA, avx512cd) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512cd());
}

TEST(ISA, avx512dq) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512dq());
}

TEST(ISA, avx512bw) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512bw());
}

TEST(ISA, avx512vl) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512vl());
}

TEST(ISA, avx512ifma) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512ifma());
}

TEST(ISA, avx512vbmi) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512vbmi());
}

TEST(ISA, avx512vpopcntdq) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512vpopcntdq());
}

TEST(ISA, avx512_4vnniw) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512_4vnniw());
}

TEST(ISA, avx512_4fmaps) {
	ASSERT_FALSE(cpuinfo_has_x86_avx512_4fmaps());
}

TEST(ISA, hle) {
	ASSERT_FALSE(cpuinfo_has_x86_hle());
}

TEST(ISA, rtm) {
	ASSERT_FALSE(cpuinfo_has_x86_rtm());
}

TEST(ISA, xtest) {
	ASSERT_FALSE(cpuinfo_has_x86_xtest());
}

TEST(ISA, mpx) {
	ASSERT_FALSE(cpuinfo_has_x86_mpx());
}

TEST(ISA, cmov) {
	ASSERT_TRUE(cpuinfo_has_x86_cmov());
}

TEST(ISA, cmpxchg8b) {
	ASSERT_TRUE(cpuinfo_has_x86_cmpxchg8b());
}

TEST(ISA, cmpxchg16b) {
	ASSERT_FALSE(cpuinfo_has_x86_cmpxchg16b());
}

TEST(ISA, clwb) {
	ASSERT_FALSE(cpuinfo_has_x86_clwb());
}

TEST(ISA, movbe) {
	ASSERT_TRUE(cpuinfo_has_x86_movbe());
}

TEST(ISA, lahf_sahf) {
	ASSERT_TRUE(cpuinfo_has_x86_lahf_sahf());
}

TEST(ISA, lzcnt) {
	ASSERT_FALSE(cpuinfo_has_x86_lzcnt());
}

TEST(ISA, popcnt) {
	ASSERT_TRUE(cpuinfo_has_x86_popcnt());
}

TEST(ISA, tbm) {
	ASSERT_FALSE(cpuinfo_has_x86_tbm());
}

TEST(ISA, bmi) {
	ASSERT_FALSE(cpuinfo_has_x86_bmi());
}

TEST(ISA, bmi2) {
	ASSERT_FALSE(cpuinfo_has_x86_bmi2());
}

TEST(ISA, adx) {
	ASSERT_FALSE(cpuinfo_has_x86_adx());
}

TEST(ISA, aes) {
	ASSERT_TRUE(cpuinfo_has_x86_aes());
}

TEST(ISA, pclmulqdq) {
	ASSERT_TRUE(cpuinfo_has_x86_pclmulqdq());
}

TEST(ISA, rdrand) {
	ASSERT_TRUE(cpuinfo_has_x86_rdrand());
}

TEST(ISA, rdseed) {
	ASSERT_FALSE(cpuinfo_has_x86_rdseed());
}

TEST(ISA, sha) {
	ASSERT_FALSE(cpuinfo_has_x86_sha());
}

TEST(L1I, count) {
	ASSERT_EQ(4, cpuinfo_get_l1i_caches_count());
}

TEST(L1I, non_null) {
	ASSERT_TRUE(cpuinfo_get_l1i_caches());
}

TEST(L1I, size) {
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		ASSERT_EQ(32 * 1024, cpuinfo_get_l1i_cache(i)->size);
	}
}

TEST(L1I, associativity) {
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		ASSERT_EQ(8, cpuinfo_get_l1i_cache(i)->associativity);
	}
}

TEST(L1I, sets) {
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		ASSERT_EQ(cpuinfo_get_l1i_cache(i)->size,
			cpuinfo_get_l1i_cache(i)->sets * cpuinfo_get_l1i_cache(i)->line_size * cpuinfo_get_l1i_cache(i)->partitions * cpuinfo_get_l1i_cache(i)->associativity);
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
	ASSERT_EQ(4, cpuinfo_get_l1d_caches_count());
}

TEST(L1D, non_null) {
	ASSERT_TRUE(cpuinfo_get_l1d_caches());
}

TEST(L1D, size) {
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		ASSERT_EQ(24 * 1024, cpuinfo_get_l1d_cache(i)->size);
	}
}

TEST(L1D, associativity) {
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		ASSERT_EQ(6, cpuinfo_get_l1d_cache(i)->associativity);
	}
}

TEST(L1D, sets) {
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		ASSERT_EQ(cpuinfo_get_l1d_cache(i)->size,
			cpuinfo_get_l1d_cache(i)->sets * cpuinfo_get_l1d_cache(i)->line_size * cpuinfo_get_l1d_cache(i)->partitions * cpuinfo_get_l1d_cache(i)->associativity);
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
	ASSERT_EQ(2, cpuinfo_get_l2_caches_count());
}

TEST(L2, non_null) {
	ASSERT_TRUE(cpuinfo_get_l2_caches());
}

TEST(L2, size) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(1024 * 1024, cpuinfo_get_l2_cache(i)->size);
	}
}

TEST(L2, associativity) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(16, cpuinfo_get_l2_cache(i)->associativity);
	}
}

TEST(L2, sets) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(cpuinfo_get_l2_cache(i)->size,
			cpuinfo_get_l2_cache(i)->sets * cpuinfo_get_l2_cache(i)->line_size * cpuinfo_get_l2_cache(i)->partitions * cpuinfo_get_l2_cache(i)->associativity);
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
		ASSERT_EQ(CPUINFO_CACHE_UNIFIED, cpuinfo_get_l2_cache(i)->flags);
	}
}

TEST(L2, processors) {
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		ASSERT_EQ(i * 2, cpuinfo_get_l2_cache(i)->processor_start);
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

#include <alldocube-iwork8.h>

int main(int argc, char* argv[]) {
	cpuinfo_mock_filesystem(filesystem);
	cpuinfo_mock_set_cpuid(cpuid_dump, sizeof(cpuid_dump) / sizeof(cpuinfo_mock_cpuid));
	cpuinfo_initialize();
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
