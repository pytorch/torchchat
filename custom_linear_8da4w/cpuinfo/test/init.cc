#include <gtest/gtest.h>

#include <cpuinfo.h>


TEST(PROCESSORS_COUNT, non_zero) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_NE(0, cpuinfo_get_processors_count());
	cpuinfo_deinitialize();
}

TEST(PROCESSORS, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_TRUE(cpuinfo_get_processors());
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		EXPECT_TRUE(cpuinfo_get_processor(i));
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, valid_smt_id) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);
		const cpuinfo_core* core = processor->core;
		ASSERT_TRUE(core);

		EXPECT_LT(processor->smt_id, core->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, valid_core) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);

		EXPECT_TRUE(processor->core);
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, consistent_core) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);
		const cpuinfo_core* core = processor->core;
		ASSERT_TRUE(core);

		EXPECT_GE(i, core->processor_start);
		EXPECT_LT(i, core->processor_start + core->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, valid_cluster) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);

		EXPECT_TRUE(processor->cluster);
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, consistent_cluster) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);
		const cpuinfo_cluster* cluster = processor->cluster;
		ASSERT_TRUE(cluster);

		EXPECT_GE(i, cluster->processor_start);
		EXPECT_LT(i, cluster->processor_start + cluster->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, valid_package) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);

		EXPECT_TRUE(processor->package);
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, consistent_package) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);
		const cpuinfo_package* package = processor->package;
		ASSERT_TRUE(package);

		EXPECT_GE(i, package->processor_start);
		EXPECT_LT(i, package->processor_start + package->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, consistent_l1i) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);
		const cpuinfo_cache* l1i = processor->cache.l1i;
		if (l1i != nullptr) {
			EXPECT_GE(i, l1i->processor_start);
			EXPECT_LT(i, l1i->processor_start + l1i->processor_count);
		}
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, consistent_l1d) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);
		const cpuinfo_cache* l1d = processor->cache.l1d;
		if (l1d != nullptr) {
			EXPECT_GE(i, l1d->processor_start);
			EXPECT_LT(i, l1d->processor_start + l1d->processor_count);
		}
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, consistent_l2) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);
		const cpuinfo_cache* l2 = processor->cache.l2;
		if (l2 != nullptr) {
			EXPECT_GE(i, l2->processor_start);
			EXPECT_LT(i, l2->processor_start + l2->processor_count);
		}
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, consistent_l3) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);
		const cpuinfo_cache* l3 = processor->cache.l3;
		if (l3 != nullptr) {
			EXPECT_GE(i, l3->processor_start);
			EXPECT_LT(i, l3->processor_start + l3->processor_count);
		}
	}
	cpuinfo_deinitialize();
}

TEST(PROCESSOR, consistent_l4) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
		const cpuinfo_processor* processor = cpuinfo_get_processor(i);
		ASSERT_TRUE(processor);
		const cpuinfo_cache* l4 = processor->cache.l4;
		if (l4 != nullptr) {
			EXPECT_GE(i, l4->processor_start);
			EXPECT_LT(i, l4->processor_start + l4->processor_count);
		}
	}
	cpuinfo_deinitialize();
}

TEST(CORES_COUNT, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_NE(0, cpuinfo_get_cores_count());
	EXPECT_LE(cpuinfo_get_cores_count(), cpuinfo_get_processors_count());
	EXPECT_GE(cpuinfo_get_cores_count(), cpuinfo_get_packages_count());
	cpuinfo_deinitialize();
}

TEST(CORES, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_TRUE(cpuinfo_get_cores());
	cpuinfo_deinitialize();
}

TEST(CORE, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		EXPECT_TRUE(cpuinfo_get_core(i));
	}
	cpuinfo_deinitialize();
}

TEST(CORE, non_zero_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		const cpuinfo_core* core = cpuinfo_get_core(i);
		ASSERT_TRUE(core);

		EXPECT_NE(0, core->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(CORE, consistent_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		const cpuinfo_core* core = cpuinfo_get_core(i);
		ASSERT_TRUE(core);

		for (uint32_t i = 0; i < core->processor_count; i++) {
			const cpuinfo_processor* processor = cpuinfo_get_processor(core->processor_start + i);
			ASSERT_TRUE(processor);

			EXPECT_EQ(core, processor->core);
		}
	}
	cpuinfo_deinitialize();
}

TEST(CORE, valid_core_id) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		const cpuinfo_core* core = cpuinfo_get_core(i);
		ASSERT_TRUE(core);
		const cpuinfo_package* package = core->package;
		ASSERT_TRUE(package);

		EXPECT_LT(core->core_id, package->core_count);
	}
	cpuinfo_deinitialize();
}

TEST(CORE, valid_cluster) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		const cpuinfo_core* core = cpuinfo_get_core(i);
		ASSERT_TRUE(core);

		EXPECT_TRUE(core->cluster);
	}
	cpuinfo_deinitialize();
}

TEST(CORE, consistent_cluster) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		const cpuinfo_core* core = cpuinfo_get_core(i);
		ASSERT_TRUE(core);
		const cpuinfo_cluster* cluster = core->cluster;
		ASSERT_TRUE(cluster);

		EXPECT_GE(i, cluster->core_start);
		EXPECT_LT(i, cluster->core_start + cluster->core_count);
	}
	cpuinfo_deinitialize();
}

TEST(CORE, valid_package) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		const cpuinfo_core* core = cpuinfo_get_core(i);
		ASSERT_TRUE(core);

		EXPECT_TRUE(core->package);
	}
	cpuinfo_deinitialize();
}

TEST(CORE, consistent_package) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		const cpuinfo_core* core = cpuinfo_get_core(i);
		ASSERT_TRUE(core);
		const cpuinfo_package* package = core->package;
		ASSERT_TRUE(package);

		EXPECT_GE(i, package->core_start);
		EXPECT_LT(i, package->core_start + package->core_count);
	}
	cpuinfo_deinitialize();
}

TEST(CORE, known_vendor) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		const cpuinfo_core* core = cpuinfo_get_core(i);
		ASSERT_TRUE(core);

		EXPECT_NE(cpuinfo_vendor_unknown, core->vendor);
	}
	cpuinfo_deinitialize();
}

TEST(CORE, known_uarch) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
		const cpuinfo_core* core = cpuinfo_get_core(i);
		ASSERT_TRUE(core);

		EXPECT_NE(cpuinfo_uarch_unknown, core->uarch);
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTERS_COUNT, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_NE(0, cpuinfo_get_clusters_count());
	EXPECT_LE(cpuinfo_get_clusters_count(), cpuinfo_get_cores_count());
	EXPECT_LE(cpuinfo_get_clusters_count(), cpuinfo_get_processors_count());
	EXPECT_GE(cpuinfo_get_clusters_count(), cpuinfo_get_packages_count());
	cpuinfo_deinitialize();
}

TEST(CLUSTERS, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_TRUE(cpuinfo_get_clusters());
	cpuinfo_deinitialize();
}

TEST(CLUSTER, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		EXPECT_TRUE(cpuinfo_get_cluster(i));
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, non_zero_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		EXPECT_NE(0, cluster->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, valid_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		EXPECT_LT(cluster->processor_start, cpuinfo_get_processors_count());
		EXPECT_LE(cluster->processor_start + cluster->processor_count, cpuinfo_get_processors_count());
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, consistent_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		for (uint32_t j = 0; j < cluster->processor_count; j++) {
			const cpuinfo_processor* processor = cpuinfo_get_processor(cluster->processor_start + j);
			EXPECT_EQ(cluster, processor->cluster);
		}
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, non_zero_cores) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		EXPECT_NE(0, cluster->core_count);
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, valid_cores) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		EXPECT_LT(cluster->core_start, cpuinfo_get_cores_count());
		EXPECT_LE(cluster->core_start + cluster->core_count, cpuinfo_get_cores_count());
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, consistent_cores) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		for (uint32_t j = 0; j < cluster->core_count; j++) {
			const cpuinfo_core* core = cpuinfo_get_core(cluster->core_start + j);
			ASSERT_TRUE(core);

			EXPECT_EQ(cluster, core->cluster);
		}
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, valid_cluster_id) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		for (uint32_t j = 0; j < cluster->core_count; j++) {
			const cpuinfo_package* package = cluster->package;
			ASSERT_TRUE(package);

			EXPECT_LT(cluster->cluster_id, package->cluster_count);
		}
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, valid_package) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		EXPECT_TRUE(cluster->package);
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, consistent_package) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);
		const cpuinfo_package* package = cluster->package;
		ASSERT_TRUE(package);

		EXPECT_GE(i, package->cluster_start);
		EXPECT_LT(i, package->cluster_start + package->cluster_count);
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, consistent_vendor) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		for (uint32_t j = 0; j < cluster->core_count; j++) {
			const cpuinfo_core* core = cpuinfo_get_core(cluster->core_start + j);
			ASSERT_TRUE(core);

			EXPECT_EQ(cluster->vendor, core->vendor);
		}
	}
	cpuinfo_deinitialize();
}

TEST(CLUSTER, consistent_uarch) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		for (uint32_t j = 0; j < cluster->core_count; j++) {
			const cpuinfo_core* core = cpuinfo_get_core(cluster->core_start + j);
			ASSERT_TRUE(core);

			EXPECT_EQ(cluster->uarch, core->uarch);
		}
	}
	cpuinfo_deinitialize();
}

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(CLUSTER, consistent_cpuid) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		for (uint32_t j = 0; j < cluster->core_count; j++) {
			const cpuinfo_core* core = cpuinfo_get_core(cluster->core_start + j);
			ASSERT_TRUE(core);

			EXPECT_EQ(cluster->cpuid, core->cpuid);
		}
	}
	cpuinfo_deinitialize();
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(CLUSTER, consistent_midr) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		for (uint32_t j = 0; j < cluster->core_count; j++) {
			const cpuinfo_core* core = cpuinfo_get_core(cluster->core_start + j);
			ASSERT_TRUE(core);

			EXPECT_EQ(cluster->midr, core->midr);
		}
	}
	cpuinfo_deinitialize();
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

TEST(CLUSTER, consistent_frequency) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_clusters_count(); i++) {
		const cpuinfo_cluster* cluster = cpuinfo_get_cluster(i);
		ASSERT_TRUE(cluster);

		for (uint32_t j = 0; j < cluster->core_count; j++) {
			const cpuinfo_core* core = cpuinfo_get_core(cluster->core_start + j);
			ASSERT_TRUE(core);

			EXPECT_EQ(cluster->frequency, core->frequency);
		}
	}
	cpuinfo_deinitialize();
}

TEST(PACKAGES_COUNT, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_NE(0, cpuinfo_get_packages_count());
	EXPECT_LE(cpuinfo_get_packages_count(), cpuinfo_get_cores_count());
	EXPECT_LE(cpuinfo_get_packages_count(), cpuinfo_get_processors_count());
	cpuinfo_deinitialize();
}

TEST(PACKAGES, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_TRUE(cpuinfo_get_packages());
	cpuinfo_deinitialize();
}

TEST(PACKAGE, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		EXPECT_TRUE(cpuinfo_get_package(i));
	}
	cpuinfo_deinitialize();
}

TEST(PACKAGE, non_zero_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		const cpuinfo_package* package = cpuinfo_get_package(i);
		ASSERT_TRUE(package);

		EXPECT_NE(0, package->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(PACKAGE, valid_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		const cpuinfo_package* package = cpuinfo_get_package(i);
		ASSERT_TRUE(package);

		EXPECT_LT(package->processor_start, cpuinfo_get_processors_count());
		EXPECT_LE(package->processor_start + package->processor_count, cpuinfo_get_processors_count());
	}
	cpuinfo_deinitialize();
}

TEST(PACKAGE, consistent_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		const cpuinfo_package* package = cpuinfo_get_package(i);
		ASSERT_TRUE(package);

		for (uint32_t j = 0; j < package->processor_count; j++) {
			const cpuinfo_processor* processor = cpuinfo_get_processor(package->processor_start + j);
			ASSERT_TRUE(processor);

			EXPECT_EQ(package, processor->package);
		}
	}
	cpuinfo_deinitialize();
}

TEST(PACKAGE, non_zero_cores) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		const cpuinfo_package* package = cpuinfo_get_package(i);
		ASSERT_TRUE(package);

		EXPECT_NE(0, package->core_count);
	}
	cpuinfo_deinitialize();
}

TEST(PACKAGE, valid_cores) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		const cpuinfo_package* package = cpuinfo_get_package(i);
		ASSERT_TRUE(package);

		EXPECT_LT(package->core_start, cpuinfo_get_cores_count());
		EXPECT_LE(package->core_start + package->core_count, cpuinfo_get_cores_count());
	}
	cpuinfo_deinitialize();
}

TEST(PACKAGE, consistent_cores) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		const cpuinfo_package* package = cpuinfo_get_package(i);
		ASSERT_TRUE(package);

		for (uint32_t j = 0; j < package->core_count; j++) {
			const cpuinfo_core* core = cpuinfo_get_core(package->core_start + j);
			ASSERT_TRUE(core);

			EXPECT_EQ(package, core->package);
		}
	}
	cpuinfo_deinitialize();
}

TEST(PACKAGE, non_zero_clusters) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		const cpuinfo_package* package = cpuinfo_get_package(i);
		ASSERT_TRUE(package);

		EXPECT_NE(0, package->cluster_count);
	}
	cpuinfo_deinitialize();
}

TEST(PACKAGE, valid_clusters) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		const cpuinfo_package* package = cpuinfo_get_package(i);
		ASSERT_TRUE(package);

		EXPECT_LT(package->cluster_start, cpuinfo_get_clusters_count());
		EXPECT_LE(package->cluster_start + package->cluster_count, cpuinfo_get_clusters_count());
	}
	cpuinfo_deinitialize();
}

TEST(PACKAGE, consistent_cluster) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_packages_count(); i++) {
		const cpuinfo_package* package = cpuinfo_get_package(i);
		ASSERT_TRUE(package);

		for (uint32_t j = 0; j < package->cluster_count; j++) {
			const cpuinfo_cluster* cluster = cpuinfo_get_cluster(package->cluster_start + j);
			ASSERT_TRUE(cluster);

			EXPECT_EQ(package, cluster->package);
		}
	}
	cpuinfo_deinitialize();
}

TEST(UARCHS_COUNT, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_NE(0, cpuinfo_get_uarchs_count());
	EXPECT_LE(cpuinfo_get_packages_count(), cpuinfo_get_cores_count());
	EXPECT_LE(cpuinfo_get_packages_count(), cpuinfo_get_processors_count());
	cpuinfo_deinitialize();
}

TEST(UARCHS, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_TRUE(cpuinfo_get_uarchs());
	cpuinfo_deinitialize();
}

TEST(UARCH, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_uarchs_count(); i++) {
		EXPECT_TRUE(cpuinfo_get_uarch(i));
	}
	cpuinfo_deinitialize();
}

TEST(UARCH, non_zero_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_uarchs_count(); i++) {
		const cpuinfo_uarch_info* uarch = cpuinfo_get_uarch(i);
		ASSERT_TRUE(uarch);

		EXPECT_NE(0, uarch->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(UARCH, valid_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_uarchs_count(); i++) {
		const cpuinfo_uarch_info* uarch = cpuinfo_get_uarch(i);
		ASSERT_TRUE(uarch);

		EXPECT_LE(uarch->processor_count, cpuinfo_get_processors_count());
	}
	cpuinfo_deinitialize();
}

TEST(UARCH, non_zero_cores) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_uarchs_count(); i++) {
		const cpuinfo_uarch_info* uarch = cpuinfo_get_uarch(i);
		ASSERT_TRUE(uarch);

		EXPECT_NE(0, uarch->core_count);
	}
	cpuinfo_deinitialize();
}

TEST(UARCH, valid_cores) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_uarchs_count(); i++) {
		const cpuinfo_uarch_info* uarch = cpuinfo_get_uarch(i);
		ASSERT_TRUE(uarch);

		EXPECT_LE(uarch->core_count, cpuinfo_get_cores_count());
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHES_COUNT, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_NE(0, cpuinfo_get_l1i_caches_count());
	EXPECT_LE(cpuinfo_get_l1i_caches_count(), cpuinfo_get_processors_count());
	cpuinfo_deinitialize();
}

TEST(L1I_CACHES, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_TRUE(cpuinfo_get_l1i_caches());
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		EXPECT_TRUE(cpuinfo_get_l1i_cache(i));
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, non_zero_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->size);
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, valid_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_EQ(cache->size,
			cache->associativity * cache->sets * cache->partitions * cache->line_size);
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, non_zero_associativity) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->associativity);
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, non_zero_partitions) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->partitions);
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, non_zero_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->line_size);
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, power_of_2_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		const uint32_t line_size = cache->line_size;
		EXPECT_NE(0, line_size);
		EXPECT_EQ(0, line_size & (line_size - 1));
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, reasonable_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_GE(cache->line_size, 16);
		EXPECT_LE(cache->line_size, 128);
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, valid_flags) {
	ASSERT_TRUE(cpuinfo_initialize());

	const uint32_t valid_flags = CPUINFO_CACHE_UNIFIED | CPUINFO_CACHE_INCLUSIVE | CPUINFO_CACHE_COMPLEX_INDEXING;
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_EQ(0, cache->flags & ~valid_flags);
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, non_inclusive) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(CPUINFO_CACHE_INCLUSIVE, cache->flags & CPUINFO_CACHE_INCLUSIVE);
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, non_zero_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, valid_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_LT(cache->processor_start, cpuinfo_get_processors_count());
		EXPECT_LE(cache->processor_start + cache->processor_count, cpuinfo_get_processors_count());
	}
	cpuinfo_deinitialize();
}

TEST(L1I_CACHE, consistent_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1i_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1i_cache(i);
		ASSERT_TRUE(cache);

		for (uint32_t j = 0; j < cache->processor_count; j++) {
			const cpuinfo_processor* processor = cpuinfo_get_processor(cache->processor_start + j);
			ASSERT_TRUE(processor);

			EXPECT_EQ(cache, processor->cache.l1i);
		}
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHES_COUNT, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_NE(0, cpuinfo_get_l1d_caches_count());
	EXPECT_LE(cpuinfo_get_l1d_caches_count(), cpuinfo_get_processors_count());
	cpuinfo_deinitialize();
}

TEST(L1D_CACHES, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_TRUE(cpuinfo_get_l1d_caches());
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		EXPECT_TRUE(cpuinfo_get_l1d_cache(i));
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, non_zero_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->size);
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, valid_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_EQ(cache->size,
			cache->associativity * cache->sets * cache->partitions * cache->line_size);
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, non_zero_associativity) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->associativity);
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, non_zero_partitions) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->partitions);
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, non_zero_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->line_size);
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, power_of_2_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		const uint32_t line_size = cache->line_size;
		EXPECT_NE(0, line_size);
		EXPECT_EQ(0, line_size & (line_size - 1));
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, reasonable_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_GE(cache->line_size, 16);
		EXPECT_LE(cache->line_size, 128);
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, valid_flags) {
	ASSERT_TRUE(cpuinfo_initialize());

	const uint32_t valid_flags = CPUINFO_CACHE_UNIFIED | CPUINFO_CACHE_INCLUSIVE | CPUINFO_CACHE_COMPLEX_INDEXING;
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_EQ(0, cache->flags & ~valid_flags);
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, non_inclusive) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(CPUINFO_CACHE_INCLUSIVE, cache->flags & CPUINFO_CACHE_INCLUSIVE);
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, non_zero_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, valid_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_LT(cache->processor_start, cpuinfo_get_processors_count());
		EXPECT_LE(cache->processor_start + cache->processor_count, cpuinfo_get_processors_count());
	}
	cpuinfo_deinitialize();
}

TEST(L1D_CACHE, consistent_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l1d_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l1d_cache(i);
		ASSERT_TRUE(cache);

		for (uint32_t j = 0; j < cache->processor_count; j++) {
			const cpuinfo_processor* processor = cpuinfo_get_processor(cache->processor_start + j);
			ASSERT_TRUE(processor);

			EXPECT_EQ(cache, processor->cache.l1d);
		}
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHES_COUNT, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_NE(0, cpuinfo_get_l2_caches_count());
	EXPECT_LE(cpuinfo_get_l2_caches_count(), cpuinfo_get_processors_count());
	EXPECT_LE(cpuinfo_get_l2_caches_count(), cpuinfo_get_l1d_caches_count());
	EXPECT_LE(cpuinfo_get_l2_caches_count(), cpuinfo_get_l1i_caches_count());
	cpuinfo_deinitialize();
}

TEST(L2_CACHES, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_TRUE(cpuinfo_get_l2_caches());
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		EXPECT_TRUE(cpuinfo_get_l2_cache(i));
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, non_zero_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->size);
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, valid_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_EQ(cache->size,
			cache->associativity * cache->sets * cache->partitions * cache->line_size);
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, non_zero_associativity) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->associativity);
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, non_zero_partitions) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->partitions);
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, non_zero_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->line_size);
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, power_of_2_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		const uint32_t line_size = cache->line_size;
		EXPECT_NE(0, line_size);
		EXPECT_EQ(0, line_size & (line_size - 1));
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, reasonable_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_GE(cache->line_size, 16);
		EXPECT_LE(cache->line_size, 128);
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, valid_flags) {
	ASSERT_TRUE(cpuinfo_initialize());

	const uint32_t valid_flags = CPUINFO_CACHE_UNIFIED | CPUINFO_CACHE_INCLUSIVE | CPUINFO_CACHE_COMPLEX_INDEXING;
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_EQ(0, cache->flags & ~valid_flags);
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, non_zero_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, valid_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_LT(cache->processor_start, cpuinfo_get_processors_count());
		EXPECT_LE(cache->processor_start + cache->processor_count, cpuinfo_get_processors_count());
	}
	cpuinfo_deinitialize();
}

TEST(L2_CACHE, consistent_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l2_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l2_cache(i);
		ASSERT_TRUE(cache);

		for (uint32_t j = 0; j < cache->processor_count; j++) {
			const cpuinfo_processor* processor = cpuinfo_get_processor(cache->processor_start + j);
			ASSERT_TRUE(processor);

			EXPECT_EQ(cache, processor->cache.l2);
		}
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHES_COUNT, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_LE(cpuinfo_get_l3_caches_count(), cpuinfo_get_processors_count());
	EXPECT_LE(cpuinfo_get_l3_caches_count(), cpuinfo_get_l2_caches_count());
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		EXPECT_TRUE(cpuinfo_get_l3_cache(i));
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, non_zero_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->size);
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, valid_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_EQ(cache->size,
			cache->associativity * cache->sets * cache->partitions * cache->line_size);
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, non_zero_associativity) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->associativity);
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, non_zero_partitions) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->partitions);
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, non_zero_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->line_size);
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, power_of_2_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		const uint32_t line_size = cache->line_size;
		EXPECT_NE(0, line_size);
		EXPECT_EQ(0, line_size & (line_size - 1));
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, reasonable_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_GE(cache->line_size, 16);
		EXPECT_LE(cache->line_size, 128);
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, valid_flags) {
	ASSERT_TRUE(cpuinfo_initialize());

	const uint32_t valid_flags = CPUINFO_CACHE_UNIFIED | CPUINFO_CACHE_INCLUSIVE | CPUINFO_CACHE_COMPLEX_INDEXING;
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_EQ(0, cache->flags & ~valid_flags);
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, non_zero_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, valid_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_LT(cache->processor_start, cpuinfo_get_processors_count());
		EXPECT_LE(cache->processor_start + cache->processor_count, cpuinfo_get_processors_count());
	}
	cpuinfo_deinitialize();
}

TEST(L3_CACHE, consistent_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l3_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l3_cache(i);
		ASSERT_TRUE(cache);

		for (uint32_t j = 0; j < cache->processor_count; j++) {
			const cpuinfo_processor* processor = cpuinfo_get_processor(cache->processor_start + j);
			ASSERT_TRUE(processor);

			EXPECT_EQ(cache, processor->cache.l3);
		}
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHES_COUNT, within_bounds) {
	ASSERT_TRUE(cpuinfo_initialize());
	EXPECT_LE(cpuinfo_get_l4_caches_count(), cpuinfo_get_processors_count());
	EXPECT_LE(cpuinfo_get_l4_caches_count(), cpuinfo_get_l3_caches_count());
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, non_null) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		EXPECT_TRUE(cpuinfo_get_l4_cache(i));
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, non_zero_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->size);
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, valid_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_EQ(cache->size,
			cache->associativity * cache->sets * cache->partitions * cache->line_size);
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, non_zero_associativity) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->associativity);
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, non_zero_partitions) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->partitions);
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, non_zero_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->line_size);
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, power_of_2_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		const uint32_t line_size = cache->line_size;
		EXPECT_NE(0, line_size);
		EXPECT_EQ(0, line_size & (line_size - 1));
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, reasonable_line_size) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_GE(cache->line_size, 16);
		EXPECT_LE(cache->line_size, 128);
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, valid_flags) {
	ASSERT_TRUE(cpuinfo_initialize());

	const uint32_t valid_flags = CPUINFO_CACHE_UNIFIED | CPUINFO_CACHE_INCLUSIVE | CPUINFO_CACHE_COMPLEX_INDEXING;
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_EQ(0, cache->flags & ~valid_flags);
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, non_zero_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_NE(0, cache->processor_count);
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, valid_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		EXPECT_LT(cache->processor_start, cpuinfo_get_processors_count());
		EXPECT_LE(cache->processor_start + cache->processor_count, cpuinfo_get_processors_count());
	}
	cpuinfo_deinitialize();
}

TEST(L4_CACHE, consistent_processors) {
	ASSERT_TRUE(cpuinfo_initialize());
	for (uint32_t i = 0; i < cpuinfo_get_l4_caches_count(); i++) {
		const cpuinfo_cache* cache = cpuinfo_get_l4_cache(i);
		ASSERT_TRUE(cache);

		for (uint32_t j = 0; j < cache->processor_count; j++) {
			const cpuinfo_processor* processor = cpuinfo_get_processor(cache->processor_start + j);
			ASSERT_TRUE(processor);

			EXPECT_EQ(cache, processor->cache.l4);
		}
	}
	cpuinfo_deinitialize();
}
