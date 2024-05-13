struct cpuinfo_mock_file filesystem[] = {
#if CPUINFO_ARCH_ARM64
	{
		.path = "/proc/cpuinfo",
		.size = 1440,
		.content =
			"processor\t: 0\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 4\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 5\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 6\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 7\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n",
	},
#elif CPUINFO_ARCH_ARM
	{
		.path = "/proc/cpuinfo",
		.size = 2231,
		.content =
			"processor\t: 0\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 1\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 2\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 3\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 4\n"
			"model name\t: ARMv8 Processor rev 2 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 5\n"
			"model name\t: ARMv8 Processor rev 2 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 6\n"
			"model name\t: ARMv8 Processor rev 2 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 7\n"
			"model name\t: ARMv8 Processor rev 2 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n",
	},
#endif
	{
		.path = "/sys/devices/system/cpu/isolated",
		.size = 1,
		.content = "\n",
	},
	{
		.path = "/sys/devices/system/cpu/kernel_max",
		.size = 2,
		.content = "7\n",
	},
	{
		.path = "/sys/devices/system/cpu/modalias",
		.size = 66,
		.content = "cpu:type:aarch64:feature:,0000,0001,0002,0003,0004,0005,0006,0007\n",
	},
	{
		.path = "/sys/devices/system/cpu/offline",
		.size = 1,
		.content = "\n",
	},
	{
		.path = "/sys/devices/system/cpu/online",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/possible",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/present",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_driver",
		.size = 25,
		.content = "hisi_little_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_governor_ro",
		.size = 5,
		.content = "menu\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpuidle/driver/name",
		.size = 25,
		.content = "hisi_little_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1844000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "509000 1018000 1210000 1402000 1556000 1690000 1844000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 89,
		.content =
			"509000 27173\n"
			"1018000 2729\n"
			"1210000 184\n"
			"1402000 1366\n"
			"1556000 446\n"
			"1690000 1044\n"
			"1844000 5229\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "655\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/trans_table",
		.size = 673,
		.content =
			"   From  :    To\n"
			"         :    509000   1018000   1210000   1402000   1556000   1690000   1844000 \n"
			"   509000:         0        47         0        67         0         2         0 \n"
			"  1018000:        22         0        28        67         0         2         0 \n"
			"  1210000:         3         5         0        24         0         0         0 \n"
			"  1402000:        32        36         1         0        59        40         0 \n"
			"  1556000:        18         3         0         1         0        42         0 \n"
			"  1690000:        14        13         2         1         0         0        63 \n"
			"  1844000:        28        15         1         8         5         6         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings",
		.size = 3,
		.content = "0f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/thread_siblings",
		.size = 3,
		.content = "01\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpuidle/driver/name",
		.size = 25,
		.content = "hisi_little_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1844000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "509000 1018000 1210000 1402000 1556000 1690000 1844000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 89,
		.content =
			"509000 27380\n"
			"1018000 2729\n"
			"1210000 184\n"
			"1402000 1366\n"
			"1556000 446\n"
			"1690000 1044\n"
			"1844000 5229\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 4,
		.content = "655\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/trans_table",
		.size = 673,
		.content =
			"   From  :    To\n"
			"         :    509000   1018000   1210000   1402000   1556000   1690000   1844000 \n"
			"   509000:         0        47         0        67         0         2         0 \n"
			"  1018000:        22         0        28        67         0         2         0 \n"
			"  1210000:         3         5         0        24         0         0         0 \n"
			"  1402000:        32        36         1         0        59        40         0 \n"
			"  1556000:        18         3         0         1         0        42         0 \n"
			"  1690000:        14        13         2         1         0         0        63 \n"
			"  1844000:        28        15         1         8         5         6         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_siblings",
		.size = 3,
		.content = "0f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_siblings_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/thread_siblings",
		.size = 3,
		.content = "02\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/thread_siblings_list",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpuidle/driver/name",
		.size = 25,
		.content = "hisi_little_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1844000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "509000 1018000 1210000 1402000 1556000 1690000 1844000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 89,
		.content =
			"509000 27589\n"
			"1018000 2729\n"
			"1210000 184\n"
			"1402000 1366\n"
			"1556000 446\n"
			"1690000 1044\n"
			"1844000 5229\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 4,
		.content = "655\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/trans_table",
		.size = 673,
		.content =
			"   From  :    To\n"
			"         :    509000   1018000   1210000   1402000   1556000   1690000   1844000 \n"
			"   509000:         0        47         0        67         0         2         0 \n"
			"  1018000:        22         0        28        67         0         2         0 \n"
			"  1210000:         3         5         0        24         0         0         0 \n"
			"  1402000:        32        36         1         0        59        40         0 \n"
			"  1556000:        18         3         0         1         0        42         0 \n"
			"  1690000:        14        13         2         1         0         0        63 \n"
			"  1844000:        28        15         1         8         5         6         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_id",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_siblings",
		.size = 3,
		.content = "0f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_siblings_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/thread_siblings",
		.size = 3,
		.content = "04\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/thread_siblings_list",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpuidle/driver/name",
		.size = 25,
		.content = "hisi_little_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1844000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "509000 1018000 1210000 1402000 1556000 1690000 1844000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "509000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 89,
		.content =
			"509000 27803\n"
			"1018000 2729\n"
			"1210000 184\n"
			"1402000 1366\n"
			"1556000 446\n"
			"1690000 1044\n"
			"1844000 5229\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "655\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/trans_table",
		.size = 673,
		.content =
			"   From  :    To\n"
			"         :    509000   1018000   1210000   1402000   1556000   1690000   1844000 \n"
			"   509000:         0        47         0        67         0         2         0 \n"
			"  1018000:        22         0        28        67         0         2         0 \n"
			"  1210000:         3         5         0        24         0         0         0 \n"
			"  1402000:        32        36         1         0        59        40         0 \n"
			"  1556000:        18         3         0         1         0        42         0 \n"
			"  1690000:        14        13         2         1         0         0        63 \n"
			"  1844000:        28        15         1         8         5         6         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_id",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_siblings",
		.size = 3,
		.content = "0f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_siblings_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/thread_siblings",
		.size = 3,
		.content = "08\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/thread_siblings_list",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpuidle/driver/name",
		.size = 22,
		.content = "hisi_big_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2362000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies",
		.size = 72,
		.content = "682000 1018000 1210000 1364000 1498000 1652000 1863000 2093000 2362000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2362000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 109,
		.content =
			"682000 33445\n"
			"1018000 417\n"
			"1210000 89\n"
			"1364000 777\n"
			"1498000 359\n"
			"1652000 380\n"
			"1863000 313\n"
			"2093000 376\n"
			"2362000 2856\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 4,
		.content = "397\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/trans_table",
		.size = 1037,
		.content =
			"   From  :    To\n"
			"         :    682000   1018000   1210000   1364000   1498000   1652000   1863000   2093000   2362000 \n"
			"   682000:         0        18         0        70         0         0         1         5         2 \n"
			"  1018000:        14         0         6        13         0         0         0         0         3 \n"
			"  1210000:         2         2         0         3         0         0         0         1         0 \n"
			"  1364000:        33         4         0         0        21        29         0         0         1 \n"
			"  1498000:        11         1         1         1         0        11         0         0         0 \n"
			"  1652000:        11         4         0         0         1         0        31         0         0 \n"
			"  1863000:         5         0         1         0         1         2         0        24         0 \n"
			"  2093000:         5         2         0         1         0         2         0         0        24 \n"
			"  2362000:        16         5         0         0         2         3         1         3         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/core_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/core_siblings",
		.size = 3,
		.content = "f0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/core_siblings_list",
		.size = 4,
		.content = "4-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/physical_package_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/thread_siblings",
		.size = 3,
		.content = "10\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/thread_siblings_list",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpuidle/driver/name",
		.size = 22,
		.content = "hisi_big_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2362000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_frequencies",
		.size = 72,
		.content = "682000 1018000 1210000 1364000 1498000 1652000 1863000 2093000 2362000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2362000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 109,
		.content =
			"682000 33660\n"
			"1018000 417\n"
			"1210000 89\n"
			"1364000 777\n"
			"1498000 359\n"
			"1652000 380\n"
			"1863000 313\n"
			"2093000 376\n"
			"2362000 2856\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 4,
		.content = "397\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/trans_table",
		.size = 1037,
		.content =
			"   From  :    To\n"
			"         :    682000   1018000   1210000   1364000   1498000   1652000   1863000   2093000   2362000 \n"
			"   682000:         0        18         0        70         0         0         1         5         2 \n"
			"  1018000:        14         0         6        13         0         0         0         0         3 \n"
			"  1210000:         2         2         0         3         0         0         0         1         0 \n"
			"  1364000:        33         4         0         0        21        29         0         0         1 \n"
			"  1498000:        11         1         1         1         0        11         0         0         0 \n"
			"  1652000:        11         4         0         0         1         0        31         0         0 \n"
			"  1863000:         5         0         1         0         1         2         0        24         0 \n"
			"  2093000:         5         2         0         1         0         2         0         0        24 \n"
			"  2362000:        16         5         0         0         2         3         1         3         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/core_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/core_siblings",
		.size = 3,
		.content = "f0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/core_siblings_list",
		.size = 4,
		.content = "4-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/physical_package_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/thread_siblings",
		.size = 3,
		.content = "20\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/thread_siblings_list",
		.size = 2,
		.content = "5\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpuidle/driver/name",
		.size = 22,
		.content = "hisi_big_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2362000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_frequencies",
		.size = 72,
		.content = "682000 1018000 1210000 1364000 1498000 1652000 1863000 2093000 2362000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2362000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 109,
		.content =
			"682000 33874\n"
			"1018000 417\n"
			"1210000 89\n"
			"1364000 777\n"
			"1498000 359\n"
			"1652000 380\n"
			"1863000 313\n"
			"2093000 376\n"
			"2362000 2856\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 4,
		.content = "397\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/trans_table",
		.size = 1037,
		.content =
			"   From  :    To\n"
			"         :    682000   1018000   1210000   1364000   1498000   1652000   1863000   2093000   2362000 \n"
			"   682000:         0        18         0        70         0         0         1         5         2 \n"
			"  1018000:        14         0         6        13         0         0         0         0         3 \n"
			"  1210000:         2         2         0         3         0         0         0         1         0 \n"
			"  1364000:        33         4         0         0        21        29         0         0         1 \n"
			"  1498000:        11         1         1         1         0        11         0         0         0 \n"
			"  1652000:        11         4         0         0         1         0        31         0         0 \n"
			"  1863000:         5         0         1         0         1         2         0        24         0 \n"
			"  2093000:         5         2         0         1         0         2         0         0        24 \n"
			"  2362000:        16         5         0         0         2         3         1         3         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/core_id",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/core_siblings",
		.size = 3,
		.content = "f0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/core_siblings_list",
		.size = 4,
		.content = "4-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/physical_package_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/thread_siblings",
		.size = 3,
		.content = "40\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/thread_siblings_list",
		.size = 2,
		.content = "6\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpuidle/driver/name",
		.size = 22,
		.content = "hisi_big_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2362000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_frequencies",
		.size = 72,
		.content = "682000 1018000 1210000 1364000 1498000 1652000 1863000 2093000 2362000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2362000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "682000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 109,
		.content =
			"682000 34092\n"
			"1018000 417\n"
			"1210000 89\n"
			"1364000 777\n"
			"1498000 359\n"
			"1652000 380\n"
			"1863000 313\n"
			"2093000 376\n"
			"2362000 2856\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 4,
		.content = "397\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/trans_table",
		.size = 1037,
		.content =
			"   From  :    To\n"
			"         :    682000   1018000   1210000   1364000   1498000   1652000   1863000   2093000   2362000 \n"
			"   682000:         0        18         0        70         0         0         1         5         2 \n"
			"  1018000:        14         0         6        13         0         0         0         0         3 \n"
			"  1210000:         2         2         0         3         0         0         0         1         0 \n"
			"  1364000:        33         4         0         0        21        29         0         0         1 \n"
			"  1498000:        11         1         1         1         0        11         0         0         0 \n"
			"  1652000:        11         4         0         0         1         0        31         0         0 \n"
			"  1863000:         5         0         1         0         1         2         0        24         0 \n"
			"  2093000:         5         2         0         1         0         2         0         0        24 \n"
			"  2362000:        16         5         0         0         2         3         1         3         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/core_id",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/core_siblings",
		.size = 3,
		.content = "f0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/core_siblings_list",
		.size = 4,
		.content = "4-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/physical_package_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/thread_siblings",
		.size = 3,
		.content = "80\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/thread_siblings_list",
		.size = 2,
		.content = "7\n",
	},
	{ NULL },
};
#ifdef __ANDROID__
struct cpuinfo_mock_property properties[] = {
	{
		.key = "ArmNN.float32Performance.execTime",
		.value = "0.72",
	},
	{
		.key = "ArmNN.float32Performance.powerUsage",
		.value = "0.74",
	},
	{
		.key = "ArmNN.quantized8Performance.execTime",
		.value = "1.0",
	},
	{
		.key = "ArmNN.quantized8Performance.powerUsage",
		.value = "1.0",
	},
	{
		.key = "audio.high.resolution.enable",
		.value = "true",
	},
	{
		.key = "bastet.service.enable",
		.value = "true",
	},
	{
		.key = "bg_fsck.pgid",
		.value = "393",
	},
	{
		.key = "bt.dpbap.enable",
		.value = "1",
	},
	{
		.key = "bt.max.hfpclient.connections",
		.value = "2",
	},
	{
		.key = "camera.dis.flag",
		.value = "2",
	},
	{
		.key = "camera.tnr.flag",
		.value = "1",
	},
	{
		.key = "config.disable_consumerir",
		.value = "false",
	},
	{
		.key = "dalvik.vm.appimageformat",
		.value = "lz4",
	},
	{
		.key = "dalvik.vm.boot-dex2oat-threads",
		.value = "4",
	},
	{
		.key = "dalvik.vm.checkjni",
		.value = "false",
	},
	{
		.key = "dalvik.vm.dex2oat-Xms",
		.value = "64m",
	},
	{
		.key = "dalvik.vm.dex2oat-Xmx",
		.value = "512m",
	},
	{
		.key = "dalvik.vm.dex2oat-threads",
		.value = "4",
	},
	{
		.key = "dalvik.vm.dexopt.secondary",
		.value = "true",
	},
	{
		.key = "dalvik.vm.heapgrowthlimit",
		.value = "384m",
	},
	{
		.key = "dalvik.vm.heapmaxfree",
		.value = "8m",
	},
	{
		.key = "dalvik.vm.heapminfree",
		.value = "2m",
	},
	{
		.key = "dalvik.vm.heapsize",
		.value = "512m",
	},
	{
		.key = "dalvik.vm.heapstartsize",
		.value = "8m",
	},
	{
		.key = "dalvik.vm.heaptargetutilization",
		.value = "0.75",
	},
	{
		.key = "dalvik.vm.image-dex2oat-Xms",
		.value = "64m",
	},
	{
		.key = "dalvik.vm.image-dex2oat-Xmx",
		.value = "64m",
	},
	{
		.key = "dalvik.vm.image-dex2oat-threads",
		.value = "4",
	},
	{
		.key = "dalvik.vm.isa.arm.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.arm.variant",
		.value = "cortex-a15",
	},
	{
		.key = "dalvik.vm.isa.arm64.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.arm64.variant",
		.value = "generic",
	},
	{
		.key = "dalvik.vm.stack-trace-dir",
		.value = "/data/anr",
	},
	{
		.key = "dalvik.vm.stack-trace-file",
		.value = "/data/anr/traces.txt",
	},
	{
		.key = "dalvik.vm.usejit",
		.value = "true",
	},
	{
		.key = "dalvik.vm.usejitprofiles",
		.value = "true",
	},
	{
		.key = "debug.aps.current_battery",
		.value = "84",
	},
	{
		.key = "debug.aps.enable",
		.value = "0",
	},
	{
		.key = "debug.aps.identify.pid",
		.value = "0",
	},
	{
		.key = "debug.aps.lcd_fps_scence",
		.value = "60",
	},
	{
		.key = "debug.aps.process.name",
		.value = "",
	},
	{
		.key = "debug.aps.scene_num",
		.value = "5",
	},
	{
		.key = "debug.atrace.tags.enableflags",
		.value = "0",
	},
	{
		.key = "debug.force_rtl",
		.value = "0",
	},
	{
		.key = "debug.hwui.render_dirty_regions",
		.value = "false",
	},
	{
		.key = "debug.sf.disable_backpressure",
		.value = "1",
	},
	{
		.key = "debug.sf.latch_unsignaled",
		.value = "1",
	},
	{
		.key = "dev.action_boot_completed",
		.value = "true",
	},
	{
		.key = "dev.bootcomplete",
		.value = "1",
	},
	{
		.key = "drm.service.enabled",
		.value = "true",
	},
	{
		.key = "fw.max_users",
		.value = "4",
	},
	{
		.key = "fw.show_multiuserui",
		.value = "1",
	},
	{
		.key = "gsm.check_is_single_pdp_sub1",
		.value = "false",
	},
	{
		.key = "gsm.current.phone-type",
		.value = "1,1",
	},
	{
		.key = "gsm.dualcards.switch",
		.value = "false",
	},
	{
		.key = "gsm.fastdormancy.mode",
		.value = "1",
	},
	{
		.key = "gsm.hw.fdn.activated1",
		.value = "false",
	},
	{
		.key = "gsm.hw.fdn.activated2",
		.value = "false",
	},
	{
		.key = "gsm.hw.operator.iso-country",
		.value = "",
	},
	{
		.key = "gsm.hw.operator.isroaming",
		.value = "false",
	},
	{
		.key = "gsm.hw.operator.numeric",
		.value = "",
	},
	{
		.key = "gsm.network.type",
		.value = "LTE,Unknown",
	},
	{
		.key = "gsm.nvcfg.resetrild",
		.value = "0",
	},
	{
		.key = "gsm.nvcfg.rildrestarting",
		.value = "0",
	},
	{
		.key = "gsm.operator.alpha",
		.value = "",
	},
	{
		.key = "gsm.operator.iso-country",
		.value = "",
	},
	{
		.key = "gsm.operator.isroaming",
		.value = "false,false",
	},
	{
		.key = "gsm.operator.numeric",
		.value = "",
	},
	{
		.key = "gsm.sigcust.configured",
		.value = "true",
	},
	{
		.key = "gsm.sim.c_card.plmn",
		.value = "",
	},
	{
		.key = "gsm.sim.hw_atr",
		.value = "null",
	},
	{
		.key = "gsm.sim.hw_atr1",
		.value = "null",
	},
	{
		.key = "gsm.sim.operator.alpha",
		.value = ",",
	},
	{
		.key = "gsm.sim.operator.iso-country",
		.value = ",",
	},
	{
		.key = "gsm.sim.operator.numeric",
		.value = ",",
	},
	{
		.key = "gsm.sim.state",
		.value = "ABSENT,ABSENT",
	},
	{
		.key = "gsm.sim1.type",
		.value = "-1",
	},
	{
		.key = "gsm.sim2.type",
		.value = "-1",
	},
	{
		.key = "gsm.version.baseband",
		.value = "21C20B353S000C000,21C20B353S000C000",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "android infineon balong-ril 1.0",
	},
	{
		.key = "hw.display.acl_support",
		.value = "true",
	},
	{
		.key = "hw.lcd.density",
		.value = "480",
	},
	{
		.key = "hw.wifipro.dns_fail_count",
		.value = "38",
	},
	{
		.key = "hwouc.hwpatch.version",
		.value = "",
	},
	{
		.key = "hwservicemanager.ready",
		.value = "true",
	},
	{
		.key = "init.svc.CameraDaemon",
		.value = "running",
	},
	{
		.key = "init.svc.activityrecognition_1_0",
		.value = "running",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.applogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.aptouch",
		.value = "running",
	},
	{
		.key = "init.svc.audio-ext-hal-2-0",
		.value = "running",
	},
	{
		.key = "init.svc.audioserver",
		.value = "running",
	},
	{
		.key = "init.svc.bastetd",
		.value = "running",
	},
	{
		.key = "init.svc.bluetooth-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.bootanim",
		.value = "stopped",
	},
	{
		.key = "init.svc.cameraserver",
		.value = "running",
	},
	{
		.key = "init.svc.cas-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.chargelogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.chargemonitor",
		.value = "running",
	},
	{
		.key = "init.svc.check_root",
		.value = "stopped",
	},
	{
		.key = "init.svc.configstore-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.cust_from_init",
		.value = "stopped",
	},
	{
		.key = "init.svc.display-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.displayeffect-1-1",
		.value = "running",
	},
	{
		.key = "init.svc.displayengine-hal-1-1",
		.value = "running",
	},
	{
		.key = "init.svc.dms-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.dpeservice",
		.value = "running",
	},
	{
		.key = "init.svc.drm",
		.value = "running",
	},
	{
		.key = "init.svc.drm-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.drm-widevine-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.dubaid",
		.value = "running",
	},
	{
		.key = "init.svc.emcomd",
		.value = "running",
	},
	{
		.key = "init.svc.eventslogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.face_hal",
		.value = "running",
	},
	{
		.key = "init.svc.fps_hal_ext",
		.value = "running",
	},
	{
		.key = "init.svc.fs_oeminfo_nv_start",
		.value = "stopped",
	},
	{
		.key = "init.svc.fs_oeminfo_nv_vmode",
		.value = "stopped",
	},
	{
		.key = "init.svc.fusd",
		.value = "running",
	},
	{
		.key = "init.svc.gatekeeper-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.gatekeeperd",
		.value = "running",
	},
	{
		.key = "init.svc.gnss_service",
		.value = "running",
	},
	{
		.key = "init.svc.gpsd_4774",
		.value = "running",
	},
	{
		.key = "init.svc.gpsdaemon",
		.value = "stopped",
	},
	{
		.key = "init.svc.gpuassistant",
		.value = "running",
	},
	{
		.key = "init.svc.gralloc-2-0",
		.value = "running",
	},
	{
		.key = "init.svc.hal_gnss_service_1",
		.value = "running",
	},
	{
		.key = "init.svc.hdbd",
		.value = "stopped",
	},
	{
		.key = "init.svc.health-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.healthd",
		.value = "running",
	},
	{
		.key = "init.svc.hiaiserver",
		.value = "running",
	},
	{
		.key = "init.svc.hidl_memory",
		.value = "running",
	},
	{
		.key = "init.svc.hilogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.hinetmanager",
		.value = "running",
	},
	{
		.key = "init.svc.hisupl_service",
		.value = "running",
	},
	{
		.key = "init.svc.hiview",
		.value = "running",
	},
	{
		.key = "init.svc.hivrar-hal-1-1",
		.value = "running",
	},
	{
		.key = "init.svc.hivrarserver",
		.value = "running",
	},
	{
		.key = "init.svc.hivwserver",
		.value = "running",
	},
	{
		.key = "init.svc.hostapd",
		.value = "stopped",
	},
	{
		.key = "init.svc.huaweiantitheft-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.huaweisigntool-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.hw_ueventd",
		.value = "running",
	},
	{
		.key = "init.svc.hwcomposer-2-1",
		.value = "running",
	},
	{
		.key = "init.svc.hwemerffu",
		.value = "stopped",
	},
	{
		.key = "init.svc.hwfactoryinterface-hal-1-1",
		.value = "running",
	},
	{
		.key = "init.svc.hwfs-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.hwhfd",
		.value = "stopped",
	},
	{
		.key = "init.svc.hwnffserver",
		.value = "running",
	},
	{
		.key = "init.svc.hwpged",
		.value = "running",
	},
	{
		.key = "init.svc.hwsched-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.hwservicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.iked",
		.value = "running",
	},
	{
		.key = "init.svc.inputlogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.installd",
		.value = "running",
	},
	{
		.key = "init.svc.ir-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.irqbalance",
		.value = "running",
	},
	{
		.key = "init.svc.irsl-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.isplogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.jpegdec-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.keymaster-3-0",
		.value = "running",
	},
	{
		.key = "init.svc.keystore",
		.value = "running",
	},
	{
		.key = "init.svc.kmsglogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.lhd_4774",
		.value = "running",
	},
	{
		.key = "init.svc.light-ext-hal-2-0",
		.value = "running",
	},
	{
		.key = "init.svc.lmkd",
		.value = "running",
	},
	{
		.key = "init.svc.logcat_service",
		.value = "stopped",
	},
	{
		.key = "init.svc.logctl_service",
		.value = "stopped",
	},
	{
		.key = "init.svc.logd",
		.value = "running",
	},
	{
		.key = "init.svc.logd-reinit",
		.value = "stopped",
	},
	{
		.key = "init.svc.macaddr",
		.value = "stopped",
	},
	{
		.key = "init.svc.media",
		.value = "running",
	},
	{
		.key = "init.svc.mediacodec",
		.value = "running",
	},
	{
		.key = "init.svc.mediacomm@2.0-service",
		.value = "running",
	},
	{
		.key = "init.svc.mediadrm",
		.value = "running",
	},
	{
		.key = "init.svc.mediaextractor",
		.value = "running",
	},
	{
		.key = "init.svc.mediametrics",
		.value = "running",
	},
	{
		.key = "init.svc.memtrack-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.modemchr_service",
		.value = "running",
	},
	{
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.nfc_hal_ext_service",
		.value = "running",
	},
	{
		.key = "init.svc.oeminfo_nvm",
		.value = "running",
	},
	{
		.key = "init.svc.otherdevices-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.perfgenius-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.pmom",
		.value = "running",
	},
	{
		.key = "init.svc.power-hw-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.powerlogd",
		.value = "running",
	},
	{
		.key = "init.svc.restart_logcat_service",
		.value = "stopped",
	},
	{
		.key = "init.svc.ril-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.rillogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.sensors-hal-1-0_hw",
		.value = "running",
	},
	{
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.shex",
		.value = "stopped",
	},
	{
		.key = "init.svc.shlogd",
		.value = "stopped",
	},
	{
		.key = "init.svc.sleeplogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.storage_info",
		.value = "running",
	},
	{
		.key = "init.svc.storaged",
		.value = "running",
	},
	{
		.key = "init.svc.surfaceflinger",
		.value = "running",
	},
	{
		.key = "init.svc.system_teecd",
		.value = "running",
	},
	{
		.key = "init.svc.teecd",
		.value = "running",
	},
	{
		.key = "init.svc.teelogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.thermal-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.thermal-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.thermalservice",
		.value = "running",
	},
	{
		.key = "init.svc.thermshex",
		.value = "stopped",
	},
	{
		.key = "init.svc.tombstoned",
		.value = "running",
	},
	{
		.key = "init.svc.tp-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.ueventd",
		.value = "running",
	},
	{
		.key = "init.svc.uniperf-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.unrmd",
		.value = "running",
	},
	{
		.key = "init.svc.usb-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.vibrator-HW-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.vndservicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.vold",
		.value = "running",
	},
	{
		.key = "init.svc.vr-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.webview_zygote32",
		.value = "running",
	},
	{
		.key = "init.svc.wifi_ext",
		.value = "running",
	},
	{
		.key = "init.svc.wificond",
		.value = "running",
	},
	{
		.key = "init.svc.wifidrvload",
		.value = "stopped",
	},
	{
		.key = "init.svc.wpa_supplicant",
		.value = "running",
	},
	{
		.key = "init.svc.zygote",
		.value = "running",
	},
	{
		.key = "init.svc.zygote_secondary",
		.value = "running",
	},
	{
		.key = "keyguard.no_require_sim",
		.value = "true",
	},
	{
		.key = "net.bt.name",
		.value = "Android",
	},
	{
		.key = "net.hostname",
		.value = "HUAWEI_P20_Pro-a9614bd7c2",
	},
	{
		.key = "net.lte.ims.data.enabled",
		.value = "true",
	},
	{
		.key = "net.portal.background",
		.value = "false",
	},
	{
		.key = "net.qtaguid_enabled",
		.value = "1",
	},
	{
		.key = "net.tcp.default_init_rwnd",
		.value = "60",
	},
	{
		.key = "nfc.node",
		.value = "/dev/pn544",
	},
	{
		.key = "partition.cust.verified",
		.value = "2",
	},
	{
		.key = "partition.odm.verified",
		.value = "2",
	},
	{
		.key = "partition.product.verified",
		.value = "2",
	},
	{
		.key = "partition.system.verified",
		.value = "2",
	},
	{
		.key = "partition.vendor.verified",
		.value = "2",
	},
	{
		.key = "partition.version.verified",
		.value = "2",
	},
	{
		.key = "persist.ai.timedebug.enable",
		.value = "false",
	},
	{
		.key = "persist.alloc_buffer_sync",
		.value = "true",
	},
	{
		.key = "persist.bt.max.a2dp.connections",
		.value = "2",
	},
	{
		.key = "persist.dsds.enabled",
		.value = "true",
	},
	{
		.key = "persist.egl.support_vr",
		.value = "1",
	},
	{
		.key = "persist.enable_task_snapshots",
		.value = "true",
	},
	{
		.key = "persist.fw.force_adoptable",
		.value = "true",
	},
	{
		.key = "persist.irqbalance.enable",
		.value = "true",
	},
	{
		.key = "persist.jank.gameskip",
		.value = "true",
	},
	{
		.key = "persist.media.lowlatency.enable",
		.value = "false",
	},
	{
		.key = "persist.media.offload.enable",
		.value = "true",
	},
	{
		.key = "persist.media.usbvoice.enable",
		.value = "true",
	},
	{
		.key = "persist.media.usbvoice.name",
		.value = "USB-Audio - HUAWEI GLASS",
	},
	{
		.key = "persist.partial_update_support",
		.value = "1",
	},
	{
		.key = "persist.radio.activemodem",
		.value = "1",
	},
	{
		.key = "persist.radio.airmode_sim0",
		.value = "false",
	},
	{
		.key = "persist.radio.airmode_sim1",
		.value = "true",
	},
	{
		.key = "persist.radio.apm_sim_not_pwdn",
		.value = "1",
	},
	{
		.key = "persist.radio.commril_mode",
		.value = "HISI_CGUL_MODE",
	},
	{
		.key = "persist.radio.defdualltecap",
		.value = "0",
	},
	{
		.key = "persist.radio.dualltecap",
		.value = "0",
	},
	{
		.key = "persist.radio.fast_switch_step",
		.value = "0,0",
	},
	{
		.key = "persist.radio.findmyphone",
		.value = "0",
	},
	{
		.key = "persist.radio.m0_ps_allow",
		.value = "1",
	},
	{
		.key = "persist.radio.modem.cap",
		.value = "09B9D52",
	},
	{
		.key = "persist.radio.modem_cdma_roam",
		.value = "true",
	},
	{
		.key = "persist.radio.multisim.config",
		.value = "dsds",
	},
	{
		.key = "persist.radio.nv_bin_loaded_m0",
		.value = "false",
	},
	{
		.key = "persist.radio.nv_bin_loaded_m1",
		.value = "false",
	},
	{
		.key = "persist.radio.nv_match_by_card",
		.value = "2",
	},
	{
		.key = "persist.radio.overseas_mode",
		.value = "true",
	},
	{
		.key = "persist.radio.prefer_nw",
		.value = "0704030201",
	},
	{
		.key = "persist.radio.prefer_nw_modem1",
		.value = "0201",
	},
	{
		.key = "persist.radio.ps_reg",
		.value = "1",
	},
	{
		.key = "persist.radio.ps_tech",
		.value = "14",
	},
	{
		.key = "persist.radio.standby_mode",
		.value = "mode_ulu",
	},
	{
		.key = "persist.radio.sub_state_cfg",
		.value = "1,1,1",
	},
	{
		.key = "persist.radio.test_card_nvcfg",
		.value = "false",
	},
	{
		.key = "persist.rog_feature",
		.value = "1",
	},
	{
		.key = "persist.service.hdb.enable",
		.value = "true",
	},
	{
		.key = "persist.service.tm2.tofile",
		.value = "false",
	},
	{
		.key = "persist.smart_pool",
		.value = "1",
	},
	{
		.key = "persist.support_lte_modem1",
		.value = "true",
	},
	{
		.key = "persist.sys.appstart.enable",
		.value = "true",
	},
	{
		.key = "persist.sys.appstart.sync",
		.value = "false",
	},
	{
		.key = "persist.sys.aps.defaultWidth",
		.value = "1080",
	},
	{
		.key = "persist.sys.aps.firstboot",
		.value = "0",
	},
	{
		.key = "persist.sys.boost.byeachfling",
		.value = "true",
	},
	{
		.key = "persist.sys.boost.durationms",
		.value = "1000",
	},
	{
		.key = "persist.sys.boost.skipframe",
		.value = "3",
	},
	{
		.key = "persist.sys.cpuset.enable",
		.value = "1",
	},
	{
		.key = "persist.sys.cpuset.subswitch",
		.value = "77328",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.devsched.subswitch",
		.value = "31",
	},
	{
		.key = "persist.sys.dolby.rotation",
		.value = "0",
	},
	{
		.key = "persist.sys.dolby.state",
		.value = "on",
	},
	{
		.key = "persist.sys.dualcards",
		.value = "true",
	},
	{
		.key = "persist.sys.enable_iaware",
		.value = "true",
	},
	{
		.key = "persist.sys.fast_h_duration",
		.value = "2000",
	},
	{
		.key = "persist.sys.fast_h_max",
		.value = "50",
	},
	{
		.key = "persist.sys.fingerpressnavi",
		.value = "0",
	},
	{
		.key = "persist.sys.fingersense",
		.value = "1",
	},
	{
		.key = "persist.sys.hiview.onekeycaptur",
		.value = "0",
	},
	{
		.key = "persist.sys.huawei.debug.on",
		.value = "0",
	},
	{
		.key = "persist.sys.hwairplanestate",
		.value = "error",
	},
	{
		.key = "persist.sys.iaware.cpuenable",
		.value = "true",
	},
	{
		.key = "persist.sys.iaware.vsyncfirst",
		.value = "true",
	},
	{
		.key = "persist.sys.iaware_config_cust",
		.value = "iaware_cust_CLT-L29_US_8.1.0_106(C636).xml",
	},
	{
		.key = "persist.sys.iaware_config_ver",
		.value = "CLT-L29_US_iaware_config_1.0_rev.xml",
	},
	{
		.key = "persist.sys.jankdb",
		.value = "<10><400><400><400><400><400>",
	},
	{
		.key = "persist.sys.jankenable",
		.value = "true",
	},
	{
		.key = "persist.sys.kmemleak.debug",
		.value = "0",
	},
	{
		.key = "persist.sys.locale",
		.value = "en-GB",
	},
	{
		.key = "persist.sys.logsystem.coredump",
		.value = "off",
	},
	{
		.key = "persist.sys.logsystem.dataflow",
		.value = "0",
	},
	{
		.key = "persist.sys.logsystem.modem",
		.value = "0",
	},
	{
		.key = "persist.sys.logsystem.protohint",
		.value = "0",
	},
	{
		.key = "persist.sys.max_rdh_delay",
		.value = "0",
	},
	{
		.key = "persist.sys.mcc_match_fyrom",
		.value = ",",
	},
	{
		.key = "persist.sys.performance",
		.value = "true",
	},
	{
		.key = "persist.sys.powerup_reason",
		.value = "NORMAL",
	},
	{
		.key = "persist.sys.restart.mediadrm",
		.value = "false",
	},
	{
		.key = "persist.sys.root.status",
		.value = "0",
	},
	{
		.key = "persist.sys.sdcardfs.emulated",
		.value = "1",
	},
	{
		.key = "persist.sys.sdcardfs.public",
		.value = "1",
	},
	{
		.key = "persist.sys.shut_alarm",
		.value = "none",
	},
	{
		.key = "persist.sys.srms.enable",
		.value = "true",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "hisuite,mtp,mass_storage,adb",
	},
	{
		.key = "persist.sys.webview.vmsize",
		.value = "113195568",
	},
	{
		.key = "persist.texture_cache_opt",
		.value = "1",
	},
	{
		.key = "persist.touch_move_opt",
		.value = "1",
	},
	{
		.key = "persist.touch_vsync_opt",
		.value = "1",
	},
	{
		.key = "pm.dexopt.ab-ota",
		.value = "speed-profile",
	},
	{
		.key = "pm.dexopt.bg-dexopt",
		.value = "speed-profile",
	},
	{
		.key = "pm.dexopt.boot",
		.value = "verify",
	},
	{
		.key = "pm.dexopt.first-boot",
		.value = "quicken",
	},
	{
		.key = "pm.dexopt.inactive",
		.value = "verify",
	},
	{
		.key = "pm.dexopt.install",
		.value = "quicken",
	},
	{
		.key = "pm.dexopt.shared",
		.value = "speed",
	},
	{
		.key = "prop.quicker.enable",
		.value = "true",
	},
	{
		.key = "qemu.hw.mainkeys",
		.value = "0",
	},
	{
		.key = "ril.balong_cid",
		.value = "0",
	},
	{
		.key = "ril.ecclist",
		.value = "112,911,000,08,110,118,119,999,120,122",
	},
	{
		.key = "ril.ecclist1",
		.value = "112,911,000,08,110,118,119,999,112,911",
	},
	{
		.key = "ril.force_to_set_ecc",
		.value = "invalid",
	},
	{
		.key = "ril.hw_modem0.rssi",
		.value = "-1",
	},
	{
		.key = "ril.hw_modem1.rssi",
		.value = "-1",
	},
	{
		.key = "ril.hw_modem2.rssi",
		.value = "-1",
	},
	{
		.key = "ril.modem.balong_nvm_server",
		.value = "true",
	},
	{
		.key = "ril.operator.numeric",
		.value = "311480",
	},
	{
		.key = "rild.libargs",
		.value = "-m modem0",
	},
	{
		.key = "rild.libargs1",
		.value = "-m modem1",
	},
	{
		.key = "rild.libargs2",
		.value = "-m modem2",
	},
	{
		.key = "rild.libpath",
		.value = "/vendor/lib64/libbalong-ril.so",
	},
	{
		.key = "rild.libpath1",
		.value = "/vendor/lib64/libbalong-ril-1.so",
	},
	{
		.key = "rild.libpath2",
		.value = "/vendor/lib64/libbalong-ril-2.so",
	},
	{
		.key = "rild.rild1_ready_to_start",
		.value = "false",
	},
	{
		.key = "ro.adb.btstatus",
		.value = "valid",
	},
	{
		.key = "ro.adb.secure",
		.value = "1",
	},
	{
		.key = "ro.allow.mock.location",
		.value = "0",
	},
	{
		.key = "ro.audio.gamescene_volume",
		.value = "0.4",
	},
	{
		.key = "ro.audio.offload_wakelock",
		.value = "false",
	},
	{
		.key = "ro.baseband",
		.value = "unknown",
	},
	{
		.key = "ro.blight.exempt_app_type",
		.value = "-1,1,16,24",
	},
	{
		.key = "ro.board.boardid",
		.value = "6542",
	},
	{
		.key = "ro.board.boardname",
		.value = "CHARLOTTE_LX9_VN1",
	},
	{
		.key = "ro.board.modemid",
		.value = "39034400",
	},
	{
		.key = "ro.board.platform",
		.value = "kirin970",
	},
	{
		.key = "ro.booking.channel.path",
		.value = "/cust_spec/xml/.booking.data.aid",
	},
	{
		.key = "ro.boot.avb_version",
		.value = "0.0",
	},
	{
		.key = "ro.boot.ddrsize",
		.value = "6",
	},
	{
		.key = "ro.boot.flash.locked",
		.value = "1",
	},
	{
		.key = "ro.boot.hardware",
		.value = "kirin970",
	},
	{
		.key = "ro.boot.mode",
		.value = "normal",
	},
	{
		.key = "ro.boot.oemmode",
		.value = "user",
	},
	{
		.key = "ro.boot.product.hardware.sku",
		.value = "CLT-L29",
	},
	{
		.key = "ro.boot.selinux",
		.value = "enforcing",
	},
	{
		.key = "ro.boot.serialno",
		.value = "WCR0218404022633",
	},
	{
		.key = "ro.boot.slot_suffix",
		.value = "_a",
	},
	{
		.key = "ro.boot.vbmeta.avb_version",
		.value = "0.0",
	},
	{
		.key = "ro.boot.vbmeta.device_state",
		.value = "locked",
	},
	{
		.key = "ro.boot.vbmeta.digest",
		.value = "d1fbd8c13521575a2814b9d999d55a6ac344748c2cbe90f2d81cf38d3ebbe4c1",
	},
	{
		.key = "ro.boot.vbmeta.hash_alg",
		.value = "sha256",
	},
	{
		.key = "ro.boot.vbmeta.invalidate_on_error",
		.value = "yes",
	},
	{
		.key = "ro.boot.vbmeta.size",
		.value = "16832",
	},
	{
		.key = "ro.boot.verifiedbootstate",
		.value = "GREEN",
	},
	{
		.key = "ro.boot.veritymode",
		.value = "enforcing",
	},
	{
		.key = "ro.bootimage.build.date",
		.value = "Wed Mar 7 03:59:46 CST 2018",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1520366386",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "Huawei/generic_a15/generic_a15:8.1.0/OPM1.171019.011/root03070358:user/test-keys",
	},
	{
		.key = "ro.bootloader",
		.value = "unknown",
	},
	{
		.key = "ro.bootmode",
		.value = "normal",
	},
	{
		.key = "ro.build.characteristics",
		.value = "default",
	},
	{
		.key = "ro.build.date",
		.value = "Wed Mar  7 03:58:08 CST 2018",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1520366288",
	},
	{
		.key = "ro.build.description",
		.value = "CLT-L29-user 8.1.0 HUAWEICLT-L29 106(C636) release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "CLT-L29 8.1.0.106(C636)",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "HUAWEI/CLT-L29/HWCLT:8.1.0/HUAWEICLT-L29/106(C636):user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "kirin970-user",
	},
	{
		.key = "ro.build.hide",
		.value = "false",
	},
	{
		.key = "ro.build.hide.matchers",
		.value = "CLT;HW;HWCLT;kirin970;HUAWEI;Heimdall-MP12;8.1.0",
	},
	{
		.key = "ro.build.hide.replacements",
		.value = "YAS;unknown;unknown;unknown;unknown;unknown;5.0.1",
	},
	{
		.key = "ro.build.hide.settings",
		.value = "8;1.8 GHz;2.0GB;11.00 GB;16.00 GB;1920 x 1080;5.1;3.10.30;3.1",
	},
	{
		.key = "ro.build.host",
		.value = "957df477-817d-485e-bd20-23a91341c865",
	},
	{
		.key = "ro.build.hw_emui_api_level",
		.value = "15",
	},
	{
		.key = "ro.build.id",
		.value = "HUAWEICLT-L29",
	},
	{
		.key = "ro.build.product",
		.value = "CLT",
	},
	{
		.key = "ro.build.tags",
		.value = "release-keys",
	},
	{
		.key = "ro.build.type",
		.value = "user",
	},
	{
		.key = "ro.build.update_version",
		.value = "V1_2",
	},
	{
		.key = "ro.build.user",
		.value = "test",
	},
	{
		.key = "ro.build.version.all_codenames",
		.value = "REL",
	},
	{
		.key = "ro.build.version.base_os",
		.value = "",
	},
	{
		.key = "ro.build.version.codename",
		.value = "REL",
	},
	{
		.key = "ro.build.version.emui",
		.value = "EmotionUI_8.1.0",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "106(C636)",
	},
	{
		.key = "ro.build.version.preview_sdk",
		.value = "0",
	},
	{
		.key = "ro.build.version.release",
		.value = "8.1.0",
	},
	{
		.key = "ro.build.version.sdk",
		.value = "27",
	},
	{
		.key = "ro.build.version.security_patch",
		.value = "2018-03-01",
	},
	{
		.key = "ro.carrier",
		.value = "unknown",
	},
	{
		.key = "ro.cdma.home.operator.numeric",
		.value = "46003",
	},
	{
		.key = "ro.cellbroadcast.emergencyids",
		.value = "0-65534",
	},
	{
		.key = "ro.check.modem_network",
		.value = "true",
	},
	{
		.key = "ro.cofig.onlinemusic.enabled",
		.value = "false",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-huawei",
	},
	{
		.key = "ro.com.google.clientidbase.am",
		.value = "android-huawei",
	},
	{
		.key = "ro.com.google.clientidbase.ms",
		.value = "android-huawei-rev1",
	},
	{
		.key = "ro.com.google.clientidbase.wal",
		.value = "android-huawei",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "8.1_r2",
	},
	{
		.key = "ro.com.google.rlz_ap_whitelist",
		.value = "y0,y5,y6,y7",
	},
	{
		.key = "ro.com.google.rlzbrandcode",
		.value = "HWDA",
	},
	{
		.key = "ro.comp.chipset_version",
		.value = "Chipset-boston 8.1.0.001(00RR)",
	},
	{
		.key = "ro.comp.cust_version",
		.value = "Cust-636000 8.0.0.1(0000)",
	},
	{
		.key = "ro.comp.product_version",
		.value = "Product-CLT 8.1.0.5(0000)",
	},
	{
		.key = "ro.comp.sys_support_vndk",
		.value = "",
	},
	{
		.key = "ro.comp.system_version",
		.value = "System 8.1.0.60(004K)",
	},
	{
		.key = "ro.comp.version_version",
		.value = "Version-CLT-L29-636000 8.1.0.1(0000)",
	},
	{
		.key = "ro.confg.hw_bootversion",
		.value = "System 8.1.0.60(004K)_BOOT",
	},
	{
		.key = "ro.confg.hw_fastbootversion",
		.value = "Chipset-boston8.1.0.001(00RR)_FASTBOOT",
	},
	{
		.key = "ro.confg.hw_odmversion",
		.value = "Chipset-boston 8.1.0.001(00RR)_ODM_CHARLOTTE",
	},
	{
		.key = "ro.confg.hw_systemversion",
		.value = "System 8.1.0.60(004K)",
	},
	{
		.key = "ro.confg.hw_userdataversion",
		.value = "CLT-L29 8.1.0.106(C636)_DATA_CLT-L29_hw_spcseas",
	},
	{
		.key = "ro.config.CphsOnsEnabled",
		.value = "false",
	},
	{
		.key = "ro.config.SupportSdcard",
		.value = "false",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Aegean_Sea.ogg",
	},
	{
		.key = "ro.config.app_big_icon_size",
		.value = "170",
	},
	{
		.key = "ro.config.argesture_enable",
		.value = "1",
	},
	{
		.key = "ro.config.arobject_enable",
		.value = "1",
	},
	{
		.key = "ro.config.attach_apn_enabled",
		.value = "true",
	},
	{
		.key = "ro.config.attach_ip_type",
		.value = "IPV4V6",
	},
	{
		.key = "ro.config.auto_display_mode",
		.value = "true",
	},
	{
		.key = "ro.config.backcolor",
		.value = "black",
	},
	{
		.key = "ro.config.beta_sec_ctrl",
		.value = "false",
	},
	{
		.key = "ro.config.blight_power_curve",
		.value = "50,1;100,0.8;200,0.6;300,0.6;350,0.8;380,1",
	},
	{
		.key = "ro.config.ca_withoutcat",
		.value = "true",
	},
	{
		.key = "ro.config.calendar_event_order",
		.value = "true",
	},
	{
		.key = "ro.config.callinwifi",
		.value = "200,6",
	},
	{
		.key = "ro.config.camera.region",
		.value = "ntdny",
	},
	{
		.key = "ro.config.carkitmodenotif",
		.value = "true",
	},
	{
		.key = "ro.config.cdma_quiet",
		.value = "true",
	},
	{
		.key = "ro.config.client_number",
		.value = "5",
	},
	{
		.key = "ro.config.colorTemperature_3d",
		.value = "true",
	},
	{
		.key = "ro.config.colorTemperature_K3",
		.value = "true",
	},
	{
		.key = "ro.config.colormode",
		.value = "0",
	},
	{
		.key = "ro.config.data_preinstalled",
		.value = "true",
	},
	{
		.key = "ro.config.default_commril_mode",
		.value = "HISI_CGUL_MODE",
	},
	{
		.key = "ro.config.default_sms_app",
		.value = "com.google.android.apps.messaging",
	},
	{
		.key = "ro.config.detect_sd_disable",
		.value = "true",
	},
	{
		.key = "ro.config.devicecolor",
		.value = "black",
	},
	{
		.key = "ro.config.disable_operator_name",
		.value = "true",
	},
	{
		.key = "ro.config.dist_nosim_airplane",
		.value = "true",
	},
	{
		.key = "ro.config.dolby_dap",
		.value = "true",
	},
	{
		.key = "ro.config.dolby_ddp",
		.value = "true",
	},
	{
		.key = "ro.config.dolby_volume",
		.value = "-40.2",
	},
	{
		.key = "ro.config.dsds_mode",
		.value = "cdma_gsm",
	},
	{
		.key = "ro.config.dual_imsi_hplmn_cust",
		.value = "21421,26832,74807",
	},
	{
		.key = "ro.config.emergencycall_show",
		.value = "true",
	},
	{
		.key = "ro.config.enable_iaware",
		.value = "true",
	},
	{
		.key = "ro.config.enable_perfhub_fling",
		.value = "true",
	},
	{
		.key = "ro.config.enable_rcc",
		.value = "true",
	},
	{
		.key = "ro.config.enable_thermal_bdata",
		.value = "true",
	},
	{
		.key = "ro.config.enable_typec_earphone",
		.value = "true",
	},
	{
		.key = "ro.config.face_recognition",
		.value = "true",
	},
	{
		.key = "ro.config.fast_switch_simslot",
		.value = "true",
	},
	{
		.key = "ro.config.finger_joint",
		.value = "true",
	},
	{
		.key = "ro.config.fix_commril_mode",
		.value = "false",
	},
	{
		.key = "ro.config.fp_navigation",
		.value = "true",
	},
	{
		.key = "ro.config.fp_ultrathin_sensor",
		.value = "true",
	},
	{
		.key = "ro.config.full_network_support",
		.value = "true",
	},
	{
		.key = "ro.config.gallery_story_disable",
		.value = "false",
	},
	{
		.key = "ro.config.helix_enable",
		.value = "true",
	},
	{
		.key = "ro.config.hiaiversion",
		.value = "100.150.010.010",
	},
	{
		.key = "ro.config.hideContactWithoutNum",
		.value = "true",
	},
	{
		.key = "ro.config.hisi_cdma_supported",
		.value = "true",
	},
	{
		.key = "ro.config.hisi_net_ai_change",
		.value = "true",
	},
	{
		.key = "ro.config.hpx_m6m8_support",
		.value = "true",
	},
	{
		.key = "ro.config.hspap_hsdpa_open",
		.value = "1",
	},
	{
		.key = "ro.config.huawei_smallwindow",
		.value = "667,0,1080,2240",
	},
	{
		.key = "ro.config.hw.imeisv",
		.value = "true",
	},
	{
		.key = "ro.config.hw.security_volume",
		.value = "8",
	},
	{
		.key = "ro.config.hw_OptiDBConfig",
		.value = "true",
	},
	{
		.key = "ro.config.hw_ReduceSAR",
		.value = "false",
	},
	{
		.key = "ro.config.hw_RemindWifiToPdp",
		.value = "true",
	},
	{
		.key = "ro.config.hw_TW_emergencyNum",
		.value = "true",
	},
	{
		.key = "ro.config.hw_allow_rs_mms",
		.value = "true",
	},
	{
		.key = "ro.config.hw_always_allow_mms",
		.value = "0",
	},
	{
		.key = "ro.config.hw_board_ipa",
		.value = "true",
	},
	{
		.key = "ro.config.hw_booster",
		.value = "true",
	},
	{
		.key = "ro.config.hw_camera_nfc_switch",
		.value = "true",
	},
	{
		.key = "ro.config.hw_ch_alg",
		.value = "2",
	},
	{
		.key = "ro.config.hw_change_volte_icon",
		.value = "false",
	},
	{
		.key = "ro.config.hw_charge_frz",
		.value = "true",
	},
	{
		.key = "ro.config.hw_dsda",
		.value = "false",
	},
	{
		.key = "ro.config.hw_dsdspowerup",
		.value = "true",
	},
	{
		.key = "ro.config.hw_dts_settings",
		.value = "true",
	},
	{
		.key = "ro.config.hw_eapsim",
		.value = "true",
	},
	{
		.key = "ro.config.hw_eccNumUseRplmn",
		.value = "true",
	},
	{
		.key = "ro.config.hw_ecc_with_sim_card",
		.value = "true",
	},
	{
		.key = "ro.config.hw_em_solution_ver",
		.value = "B060",
	},
	{
		.key = "ro.config.hw_emcom",
		.value = "true",
	},
	{
		.key = "ro.config.hw_emui_desktop_mode",
		.value = "true",
	},
	{
		.key = "ro.config.hw_enable_merge",
		.value = "true",
	},
	{
		.key = "ro.config.hw_front_fp_navi",
		.value = "true",
	},
	{
		.key = "ro.config.hw_front_fp_trikey",
		.value = "0",
	},
	{
		.key = "ro.config.hw_globalEcc",
		.value = "true",
	},
	{
		.key = "ro.config.hw_glovemode_enabled",
		.value = "1",
	},
	{
		.key = "ro.config.hw_hideSimIcon",
		.value = "false",
	},
	{
		.key = "ro.config.hw_hidecallforward",
		.value = "false",
	},
	{
		.key = "ro.config.hw_higeo_fusion_ver",
		.value = "1.5",
	},
	{
		.key = "ro.config.hw_higeo_map_matching",
		.value = "1",
	},
	{
		.key = "ro.config.hw_higeo_nw_pos_db",
		.value = "HERE",
	},
	{
		.key = "ro.config.hw_higeo_pdrsupport",
		.value = "true",
	},
	{
		.key = "ro.config.hw_hotswap_on",
		.value = "true",
	},
	{
		.key = "ro.config.hw_hungtasklist",
		.value = "whitelist,system_server,SurfaceFlinger",
	},
	{
		.key = "ro.config.hw_icon_supprot_cut",
		.value = "false",
	},
	{
		.key = "ro.config.hw_imei_sv_enable",
		.value = "true",
	},
	{
		.key = "ro.config.hw_imei_sv_show_two",
		.value = "true",
	},
	{
		.key = "ro.config.hw_ims_as_normal",
		.value = "true",
	},
	{
		.key = "ro.config.hw_ipv6_support",
		.value = "true",
	},
	{
		.key = "ro.config.hw_low_ram",
		.value = "false",
	},
	{
		.key = "ro.config.hw_lte_release",
		.value = "true",
	},
	{
		.key = "ro.config.hw_lte_support",
		.value = "true",
	},
	{
		.key = "ro.config.hw_magne_bracket",
		.value = "true",
	},
	{
		.key = "ro.config.hw_media_flags",
		.value = "1",
	},
	{
		.key = "ro.config.hw_multiscreen",
		.value = "false",
	},
	{
		.key = "ro.config.hw_music_lp",
		.value = "true",
	},
	{
		.key = "ro.config.hw_navigationbar",
		.value = "true",
	},
	{
		.key = "ro.config.hw_nfc_def_route",
		.value = "2",
	},
	{
		.key = "ro.config.hw_nfc_on",
		.value = "true",
	},
	{
		.key = "ro.config.hw_not_modify_wifi",
		.value = "Singtel WIFI",
	},
	{
		.key = "ro.config.hw_notch_hot_space",
		.value = "3",
	},
	{
		.key = "ro.config.hw_notch_size",
		.value = "261,81,410,30",
	},
	{
		.key = "ro.config.hw_omacp",
		.value = "1",
	},
	{
		.key = "ro.config.hw_opta",
		.value = "636",
	},
	{
		.key = "ro.config.hw_optb",
		.value = "999",
	},
	{
		.key = "ro.config.hw_perfgenius",
		.value = "true",
	},
	{
		.key = "ro.config.hw_power_saving",
		.value = "true",
	},
	{
		.key = "ro.config.hw_rcm_cert",
		.value = "true",
	},
	{
		.key = "ro.config.hw_rcs_product",
		.value = "true",
	},
	{
		.key = "ro.config.hw_save_pin",
		.value = "true",
	},
	{
		.key = "ro.config.hw_screen_aspect",
		.value = "2240:1992:1080",
	},
	{
		.key = "ro.config.hw_sensorhub",
		.value = "true",
	},
	{
		.key = "ro.config.hw_showSimName",
		.value = "true",
	},
	{
		.key = "ro.config.hw_show_4G_Plus_icon",
		.value = "false",
	},
	{
		.key = "ro.config.hw_show_mmiError",
		.value = "true",
	},
	{
		.key = "ro.config.hw_show_network_icon",
		.value = "true",
	},
	{
		.key = "ro.config.hw_sim2airplane",
		.value = "true",
	},
	{
		.key = "ro.config.hw_simcard_pre_city",
		.value = "true",
	},
	{
		.key = "ro.config.hw_simlock_retries",
		.value = "true",
	},
	{
		.key = "ro.config.hw_simpleui_enable",
		.value = "1",
	},
	{
		.key = "ro.config.hw_singlehand",
		.value = "1",
	},
	{
		.key = "ro.config.hw_southeast_asia",
		.value = "true",
	},
	{
		.key = "ro.config.hw_srlte",
		.value = "true",
	},
	{
		.key = "ro.config.hw_support_clone_app",
		.value = "true",
	},
	{
		.key = "ro.config.hw_support_geofence",
		.value = "true",
	},
	{
		.key = "ro.config.hw_support_vm_ecc",
		.value = "true",
	},
	{
		.key = "ro.config.hw_switchdata_4G",
		.value = "true",
	},
	{
		.key = "ro.config.hw_ukey_version",
		.value = "2",
	},
	{
		.key = "ro.config.hw_useCtrlSocket",
		.value = "true",
	},
	{
		.key = "ro.config.hw_use_browser_ua",
		.value = "http://wap1.huawei.com/uaprof/HUAWEI_%s_UAProfile.xml",
	},
	{
		.key = "ro.config.hw_vcardBase64",
		.value = "true",
	},
	{
		.key = "ro.config.hw_voicemail_sim",
		.value = "true",
	},
	{
		.key = "ro.config.hw_volte_dyn",
		.value = "true",
	},
	{
		.key = "ro.config.hw_volte_icon_rule",
		.value = "2",
	},
	{
		.key = "ro.config.hw_volte_on",
		.value = "true",
	},
	{
		.key = "ro.config.hw_vowifi",
		.value = "true",
	},
	{
		.key = "ro.config.hw_vowifi_mmsut",
		.value = "true",
	},
	{
		.key = "ro.config.hw_wakeup_device",
		.value = "true",
	},
	{
		.key = "ro.config.hw_watermark",
		.value = "false",
	},
	{
		.key = "ro.config.hw_wifibridge",
		.value = "true",
	},
	{
		.key = "ro.config.hw_wifipro_enable",
		.value = "true",
	},
	{
		.key = "ro.config.hwsync_enabled",
		.value = "true",
	},
	{
		.key = "ro.config.hwtheme",
		.value = "1",
	},
	{
		.key = "ro.config.ipv4.mtu",
		.value = "1400",
	},
	{
		.key = "ro.config.is_start_commril",
		.value = "true",
	},
	{
		.key = "ro.config.keyguard_unusedata",
		.value = "false",
	},
	{
		.key = "ro.config.keypasstouser",
		.value = "true",
	},
	{
		.key = "ro.config.light_ratio_max",
		.value = "230",
	},
	{
		.key = "ro.config.limit_lcd_manual",
		.value = "true",
	},
	{
		.key = "ro.config.linkplus.liveupdate",
		.value = "true",
	},
	{
		.key = "ro.config.lockscreen_sound_off",
		.value = "true",
	},
	{
		.key = "ro.config.m1csimlteflwims",
		.value = "true",
	},
	{
		.key = "ro.config.marketing_name",
		.value = "HUAWEI P20 Pro",
	},
	{
		.key = "ro.config.mmu_en",
		.value = "1",
	},
	{
		.key = "ro.config.modem_number",
		.value = "3",
	},
	{
		.key = "ro.config.music_lp_vol",
		.value = "true",
	},
	{
		.key = "ro.config.music_region",
		.value = "normal",
	},
	{
		.key = "ro.config.myloc_show_first",
		.value = "true",
	},
	{
		.key = "ro.config.nfc_ce_transevt",
		.value = "true",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "Bongo.ogg",
	},
	{
		.key = "ro.config.peq_support",
		.value = "true",
	},
	{
		.key = "ro.config.plmn_to_settings",
		.value = "true",
	},
	{
		.key = "ro.config.report_cell_info_list",
		.value = "true",
	},
	{
		.key = "ro.config.ringtone",
		.value = "Huawei_Tune_Living.ogg",
	},
	{
		.key = "ro.config.ringtone2",
		.value = "Huawei_Tune_Clean.ogg",
	},
	{
		.key = "ro.config.safety_certification",
		.value = "CE,FCC,GCF",
	},
	{
		.key = "ro.config.scr_timeout_10sec",
		.value = "true",
	},
	{
		.key = "ro.config.show_always_mms_ui",
		.value = "true",
	},
	{
		.key = "ro.config.show_epg_menu",
		.value = "false",
	},
	{
		.key = "ro.config.show_full_month",
		.value = "true",
	},
	{
		.key = "ro.config.show_vmail_number",
		.value = "true",
	},
	{
		.key = "ro.config.small.previewpos",
		.value = "right",
	},
	{
		.key = "ro.config.small_cover_size",
		.value = "_401x1920",
	},
	{
		.key = "ro.config.spare_ntp_server",
		.value = "ntp.sjtu.edu.cn,time.windows.com,time.nist.gov,1.cn.pool.ntp.org",
	},
	{
		.key = "ro.config.sup_lte_high_speed",
		.value = "true",
	},
	{
		.key = "ro.config.sup_six_at_chan",
		.value = "true",
	},
	{
		.key = "ro.config.support_aod",
		.value = "1",
	},
	{
		.key = "ro.config.support_ca",
		.value = "true",
	},
	{
		.key = "ro.config.support_ccmode",
		.value = "true",
	},
	{
		.key = "ro.config.support_crrconn",
		.value = "true",
	},
	{
		.key = "ro.config.support_face_mode",
		.value = "1",
	},
	{
		.key = "ro.config.support_hwpki",
		.value = "true",
	},
	{
		.key = "ro.config.support_iudf",
		.value = "true",
	},
	{
		.key = "ro.config.support_one_time_hota",
		.value = "true",
	},
	{
		.key = "ro.config.support_privacyspace",
		.value = "true",
	},
	{
		.key = "ro.config.support_sdcard_crypt",
		.value = "false",
	},
	{
		.key = "ro.config.support_wcdma_modem1",
		.value = "true",
	},
	{
		.key = "ro.config.switchPrimaryVolume",
		.value = "true",
	},
	{
		.key = "ro.config.swsAlwaysActiveForSPK",
		.value = "true",
	},
	{
		.key = "ro.config.sws_apk_hptype",
		.value = "1",
	},
	{
		.key = "ro.config.theme_display_tw",
		.value = "true",
	},
	{
		.key = "ro.config.third_key_provider",
		.value = "kukong",
	},
	{
		.key = "ro.config.toolorder",
		.value = "0,2,1,3,4",
	},
	{
		.key = "ro.config.updatelocation",
		.value = "true",
	},
	{
		.key = "ro.config.vm_prioritymode",
		.value = "2",
	},
	{
		.key = "ro.config.voice_cfg",
		.value = "force",
	},
	{
		.key = "ro.config.vol_steps",
		.value = "15",
	},
	{
		.key = "ro.config.vowifi_pref_domestic",
		.value = "45400;45402;45416;45418;45419;45403;45404;45414;45412;45413;45406;45415;45417;",
	},
	{
		.key = "ro.config.vowifi_pref_roaming",
		.value = "45400;45402;45416;45418;45419;45403;45404;45414;45412;45413;45406;45415;45417;",
	},
	{
		.key = "ro.config.vowifi_pref_wifi_cell",
		.value = "46692;",
	},
	{
		.key = "ro.config.widevine_level3",
		.value = "true",
	},
	{
		.key = "ro.config.wifi_country_code",
		.value = "true",
	},
	{
		.key = "ro.config.wifi_fast_bss_enable",
		.value = "true",
	},
	{
		.key = "ro.connectivity.chiptype",
		.value = "bcm43xx",
	},
	{
		.key = "ro.connectivity.sub_chiptype",
		.value = "bcm4359",
	},
	{
		.key = "ro.control.sleeplog",
		.value = "true",
	},
	{
		.key = "ro.control_privapp_permissions",
		.value = "log",
	},
	{
		.key = "ro.crypto.fuse_sdcard",
		.value = "true",
	},
	{
		.key = "ro.crypto.state",
		.value = "encrypted",
	},
	{
		.key = "ro.crypto.type",
		.value = "file",
	},
	{
		.key = "ro.cust.cdrom",
		.value = "/data/hw_init/version/region_comm/oversea/cdrom/autorun.iso",
	},
	{
		.key = "ro.dalvik.vm.native.bridge",
		.value = "0",
	},
	{
		.key = "ro.debuggable",
		.value = "0",
	},
	{
		.key = "ro.device_owner",
		.value = "false",
	},
	{
		.key = "ro.dts.licensepath",
		.value = "/product/etc/dts/",
	},
	{
		.key = "ro.dual.sim.phone",
		.value = "true",
	},
	{
		.key = "ro.email.inline_as_att",
		.value = "true",
	},
	{
		.key = "ro.expect.recovery_id",
		.value = "0x8712ab492c519963018d54c928dafa8882e4a9bf000000000000000000000000",
	},
	{
		.key = "ro.facebook.partnerid",
		.value = "huawei:3ed03d0-8ce2-42fa-a449-b9443817d7b4",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/bootdevice/by-name/frp",
	},
	{
		.key = "ro.game.low_audio_effect",
		.value = "true",
	},
	{
		.key = "ro.hardware",
		.value = "kirin970",
	},
	{
		.key = "ro.hardware.alter",
		.value = "HiSilicon Kirin 970",
	},
	{
		.key = "ro.hardware.audio.primary",
		.value = "hisi",
	},
	{
		.key = "ro.hdmi.service",
		.value = "false",
	},
	{
		.key = "ro.huawei.cust.oma",
		.value = "false",
	},
	{
		.key = "ro.huawei.cust.oma_drm",
		.value = "false",
	},
	{
		.key = "ro.huawei.remount.check",
		.value = "verify_success",
	},
	{
		.key = "ro.hw.country",
		.value = "spcseas",
	},
	{
		.key = "ro.hw.custPath",
		.value = "/cust/hw/spcseas",
	},
	{
		.key = "ro.hw.mirrorlink.enable",
		.value = "true",
	},
	{
		.key = "ro.hw.oemName",
		.value = "CLT-L29",
	},
	{
		.key = "ro.hw.specialCustPath",
		.value = "/cust_spec",
	},
	{
		.key = "ro.hw.vendor",
		.value = "hw",
	},
	{
		.key = "ro.hwaft.tpfp.filter_area",
		.value = "720,200",
	},
	{
		.key = "ro.hwcamera.modesuggest_enable",
		.value = "true",
	},
	{
		.key = "ro.hwcamera.portrait_mode",
		.value = "off",
	},
	{
		.key = "ro.hwcamera.use.videosize.1080p",
		.value = "true",
	},
	{
		.key = "ro.hwui.drop_shadow_cache_size",
		.value = "6",
	},
	{
		.key = "ro.hwui.gradient_cache_size",
		.value = "1",
	},
	{
		.key = "ro.hwui.layer_cache_size",
		.value = "48",
	},
	{
		.key = "ro.hwui.path_cache_size",
		.value = "32",
	},
	{
		.key = "ro.hwui.r_buffer_cache_size",
		.value = "8",
	},
	{
		.key = "ro.hwui.shape_cache_size",
		.value = "2",
	},
	{
		.key = "ro.hwui.text_large_cache_height",
		.value = "1024",
	},
	{
		.key = "ro.hwui.text_large_cache_width",
		.value = "2048",
	},
	{
		.key = "ro.hwui.text_small_cache_height",
		.value = "1024",
	},
	{
		.key = "ro.hwui.text_small_cache_width",
		.value = "1024",
	},
	{
		.key = "ro.hwui.texture_cache_flushrate",
		.value = "0.4",
	},
	{
		.key = "ro.hwui.texture_cache_size",
		.value = "72",
	},
	{
		.key = "ro.image",
		.value = "bootimage",
	},
	{
		.key = "ro.logsystem.usertype",
		.value = "6",
	},
	{
		.key = "ro.magic.api.version",
		.value = "0.1",
	},
	{
		.key = "ro.multi.rild",
		.value = "false",
	},
	{
		.key = "ro.oba.version",
		.value = "20180307014248_OBA_VERSION",
	},
	{
		.key = "ro.oem_unlock_supported",
		.value = "1",
	},
	{
		.key = "ro.opa.eligible_device",
		.value = "true",
	},
	{
		.key = "ro.opengles.version",
		.value = "196610",
	},
	{
		.key = "ro.patch.baseline.version",
		.value = "2.0",
	},
	{
		.key = "ro.product.CustCVersion",
		.value = "C636",
	},
	{
		.key = "ro.product.board",
		.value = "CLT",
	},
	{
		.key = "ro.product.brand",
		.value = "HUAWEI",
	},
	{
		.key = "ro.product.cpu.abi",
		.value = "arm64-v8a",
	},
	{
		.key = "ro.product.cpu.abilist",
		.value = "arm64-v8a,armeabi-v7a,armeabi",
	},
	{
		.key = "ro.product.cpu.abilist32",
		.value = "armeabi-v7a,armeabi",
	},
	{
		.key = "ro.product.cpu.abilist64",
		.value = "arm64-v8a",
	},
	{
		.key = "ro.product.device",
		.value = "HWCLT",
	},
	{
		.key = "ro.product.fingerprintName",
		.value = "HUAWEI-Z109",
	},
	{
		.key = "ro.product.hardwareversion",
		.value = "HL1CLTM",
	},
	{
		.key = "ro.product.imeisv",
		.value = "03",
	},
	{
		.key = "ro.product.locale",
		.value = "en-GB",
	},
	{
		.key = "ro.product.locale.language",
		.value = "en",
	},
	{
		.key = "ro.product.locale.region",
		.value = "US",
	},
	{
		.key = "ro.product.manufacturer",
		.value = "HUAWEI",
	},
	{
		.key = "ro.product.model",
		.value = "CLT-L29",
	},
	{
		.key = "ro.product.name",
		.value = "CLT-L29",
	},
	{
		.key = "ro.product.platform",
		.value = "kirin970",
	},
	{
		.key = "ro.product.platform.pseudonym",
		.value = "1ARB9CV",
	},
	{
		.key = "ro.property_service.version",
		.value = "2",
	},
	{
		.key = "ro.pwroff_card_if_tray_out",
		.value = "true",
	},
	{
		.key = "ro.quick_broadcast_cardstatus",
		.value = "false",
	},
	{
		.key = "ro.revision",
		.value = "0",
	},
	{
		.key = "ro.ril.ecclist",
		.value = "112,911,#911,*911",
	},
	{
		.key = "ro.runmode",
		.value = "normal",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "WCR0218404022633",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "OPTIONAL",
	},
	{
		.key = "ro.setupwizard.wifi_on_exit",
		.value = "false",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "480",
	},
	{
		.key = "ro.sys.powerup_reason",
		.value = "NORMAL",
	},
	{
		.key = "ro.sys.sdcardfs",
		.value = "1",
	},
	{
		.key = "ro.sys.umsdirtyratio",
		.value = "2",
	},
	{
		.key = "ro.sysui.show.normal.layout",
		.value = "true",
	},
	{
		.key = "ro.telephony.default_network",
		.value = "9",
	},
	{
		.key = "ro.treble.enabled",
		.value = "true",
	},
	{
		.key = "ro.tui.service",
		.value = "true",
	},
	{
		.key = "ro.userlock",
		.value = "locked",
	},
	{
		.key = "ro.vendor.build.date",
		.value = "Wed Mar 7 01:43:51 CST 2018",
	},
	{
		.key = "ro.vendor.build.date.utc",
		.value = "1520358231",
	},
	{
		.key = "ro.vendor.build.fingerprint",
		.value = "HUAWEI/CLT-L29/HWCLT:8.1.0/HUAWEICLT-L29/106(C636):user/release-keys",
	},
	{
		.key = "ro.vendor.product.brand",
		.value = "HUAWEI",
	},
	{
		.key = "ro.vendor.product.device",
		.value = "HWCLT",
	},
	{
		.key = "ro.vendor.product.manufacturer",
		.value = "HUAWEI",
	},
	{
		.key = "ro.vendor.product.model",
		.value = "CLT-L29",
	},
	{
		.key = "ro.vendor.product.name",
		.value = "CLT-L29",
	},
	{
		.key = "ro.vendor.vndk.version",
		.value = "26.0.0",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
	},
	{
		.key = "ro.zygote",
		.value = "zygote64_32",
	},
	{
		.key = "security.perf_harden",
		.value = "1",
	},
	{
		.key = "service.bootanim.exit",
		.value = "1",
	},
	{
		.key = "service.sf.present_timestamp",
		.value = "1",
	},
	{
		.key = "sys.2dsdr.pkgname",
		.value = "*",
	},
	{
		.key = "sys.2dsdr.startratio",
		.value = "1.0",
	},
	{
		.key = "sys.aps.browserProcessName",
		.value = "",
	},
	{
		.key = "sys.aps.gameProcessName",
		.value = "",
	},
	{
		.key = "sys.aps.support",
		.value = "12566763",
	},
	{
		.key = "sys.aps.version",
		.value = "5.1.2-6.0.1.25",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
	},
	{
		.key = "sys.current_backlight",
		.value = "5058",
	},
	{
		.key = "sys.defaultapn.enabled",
		.value = "true",
	},
	{
		.key = "sys.fingerprint.deviceId",
		.value = "0",
	},
	{
		.key = "sys.fs.vmode",
		.value = "x02",
	},
	{
		.key = "sys.hisi.pmom.service.enable",
		.value = "false",
	},
	{
		.key = "sys.huawei.thermal.enable",
		.value = "true",
	},
	{
		.key = "sys.hw_boot_success",
		.value = "1",
	},
	{
		.key = "sys.hwsholder.count",
		.value = "0",
	},
	{
		.key = "sys.iaware.cpuset.screenoff.bg",
		.value = "0-3",
	},
	{
		.key = "sys.iaware.cpuset.screenoff.boost",
		.value = "0-7",
	},
	{
		.key = "sys.iaware.cpuset.screenoff.fg",
		.value = "0-7",
	},
	{
		.key = "sys.iaware.cpuset.screenoff.kbg",
		.value = "0-3",
	},
	{
		.key = "sys.iaware.cpuset.screenoff.sysbg",
		.value = "0-3",
	},
	{
		.key = "sys.iaware.cpuset.screenoff.taboost",
		.value = "",
	},
	{
		.key = "sys.iaware.cpuset.screenoff.topapp",
		.value = "0-7",
	},
	{
		.key = "sys.iaware.cpuset.screenon.bg",
		.value = "2-3",
	},
	{
		.key = "sys.iaware.cpuset.screenon.boost",
		.value = "4-7",
	},
	{
		.key = "sys.iaware.cpuset.screenon.fg",
		.value = "0-7",
	},
	{
		.key = "sys.iaware.cpuset.screenon.kbg",
		.value = "2-3,7",
	},
	{
		.key = "sys.iaware.cpuset.screenon.sysbg",
		.value = "0-3",
	},
	{
		.key = "sys.iaware.cpuset.screenon.taboost",
		.value = "4-7",
	},
	{
		.key = "sys.iaware.cpuset.screenon.topapp",
		.value = "0-7",
	},
	{
		.key = "sys.iaware.cpuset.vron.bg",
		.value = "0-2",
	},
	{
		.key = "sys.iaware.cpuset.vron.boost",
		.value = "4-7",
	},
	{
		.key = "sys.iaware.cpuset.vron.fg",
		.value = "0-2,4-7",
	},
	{
		.key = "sys.iaware.cpuset.vron.kbg",
		.value = "0-2",
	},
	{
		.key = "sys.iaware.cpuset.vron.sysbg",
		.value = "0-2",
	},
	{
		.key = "sys.iaware.cpuset.vron.taboost",
		.value = "",
	},
	{
		.key = "sys.iaware.cpuset.vron.topapp",
		.value = "0-7",
	},
	{
		.key = "sys.iaware.eas.on",
		.value = "true",
	},
	{
		.key = "sys.iaware.empty_app_percent",
		.value = "50",
	},
	{
		.key = "sys.iaware.nosave.h.freq",
		.value = "1364000",
	},
	{
		.key = "sys.iaware.nosave.h.load",
		.value = "90",
	},
	{
		.key = "sys.iaware.nosave.h.tload",
		.value = "80:1364000:75:1498000:85:1652000:95",
	},
	{
		.key = "sys.iaware.nosave.l.freq",
		.value = "1402000",
	},
	{
		.key = "sys.iaware.nosave.l.load",
		.value = "95",
	},
	{
		.key = "sys.iaware.nosave.l.tload",
		.value = "75:1018000:85:1402000:75:1556000:83",
	},
	{
		.key = "sys.iaware.save.h.freq",
		.value = "1364000",
	},
	{
		.key = "sys.iaware.save.h.load",
		.value = "90",
	},
	{
		.key = "sys.iaware.save.h.tload",
		.value = "80:1364000:75:1498000:85:1652000:95",
	},
	{
		.key = "sys.iaware.save.l.freq",
		.value = "1402000",
	},
	{
		.key = "sys.iaware.save.l.load",
		.value = "95",
	},
	{
		.key = "sys.iaware.save.l.tload",
		.value = "75:1018000:85:1402000:75:1556000:83",
	},
	{
		.key = "sys.iaware.set.h.freq",
		.value = "1364000",
	},
	{
		.key = "sys.iaware.set.h.load",
		.value = "90",
	},
	{
		.key = "sys.iaware.set.h.tload",
		.value = "80:1364000:75:1498000:85:1652000:95",
	},
	{
		.key = "sys.iaware.set.l.freq",
		.value = "1402000",
	},
	{
		.key = "sys.iaware.set.l.load",
		.value = "95",
	},
	{
		.key = "sys.iaware.set.l.tload",
		.value = "75:1018000:85:1402000:75:1556000:83",
	},
	{
		.key = "sys.iaware.supersave.h.freq",
		.value = "1364000",
	},
	{
		.key = "sys.iaware.supersave.h.load",
		.value = "90",
	},
	{
		.key = "sys.iaware.supersave.h.tload",
		.value = "80:1364000:75:1498000:85:1652000:95",
	},
	{
		.key = "sys.iaware.supersave.l.freq",
		.value = "1402000",
	},
	{
		.key = "sys.iaware.supersave.l.load",
		.value = "95",
	},
	{
		.key = "sys.iaware.supersave.l.tload",
		.value = "75:1018000:85:1402000:75:1556000:83",
	},
	{
		.key = "sys.iswifihotspoton",
		.value = "false",
	},
	{
		.key = "sys.logbootcomplete",
		.value = "1",
	},
	{
		.key = "sys.oem_unlock_allowed",
		.value = "0",
	},
	{
		.key = "sys.pg.pre_rescue_boot_count",
		.value = "1",
	},
	{
		.key = "sys.refresh.dirty",
		.value = "1",
	},
	{
		.key = "sys.rescue_boot_count",
		.value = "1",
	},
	{
		.key = "sys.resettype",
		.value = "normal:BR_PRESS_1S",
	},
	{
		.key = "sys.retaildemo.enabled",
		.value = "0",
	},
	{
		.key = "sys.show_google_nlp",
		.value = "true",
	},
	{
		.key = "sys.super_power_save",
		.value = "false",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "28350",
	},
	{
		.key = "sys.usb.config",
		.value = "hisuite,mtp,mass_storage,adb",
	},
	{
		.key = "sys.usb.configfs",
		.value = "1",
	},
	{
		.key = "sys.usb.controller",
		.value = "ff100000.dwc3",
	},
	{
		.key = "sys.usb.ffs.mtp.ready",
		.value = "1",
	},
	{
		.key = "sys.usb.ffs.ready",
		.value = "1",
	},
	{
		.key = "sys.usb.ffs_hdb.ready",
		.value = "0",
	},
	{
		.key = "sys.usb.state",
		.value = "hisuite,mtp,mass_storage,adb",
	},
	{
		.key = "sys.userdata_is_ready",
		.value = "1",
	},
	{
		.key = "system_init.hwextdeviceservice",
		.value = "1",
	},
	{
		.key = "trustedcore_sfs_property",
		.value = "1",
	},
	{
		.key = "use_sensorhub_labc",
		.value = "false",
	},
	{
		.key = "vha.nonVolte.waiting.tone",
		.value = "true",
	},
	{
		.key = "viatel.device.at",
		.value = "spi.10.ttySPI",
	},
	{
		.key = "viatel.device.data",
		.value = "spi.0.ttySPI",
	},
	{
		.key = "viatel.device.fls",
		.value = "spi.2.ttySPI",
	},
	{
		.key = "viatel.device.gps",
		.value = "spi.5.ttySPI",
	},
	{
		.key = "viatel.device.pcv",
		.value = "spi.4.ttySPI",
	},
	{
		.key = "vold.crypto_unencrypt_updatedir",
		.value = "/data/update",
	},
	{
		.key = "vold.has_adoptable",
		.value = "0",
	},
	{
		.key = "vold.has_quota",
		.value = "0",
	},
	{
		.key = "vold.post_fs_data_done",
		.value = "1",
	},
	{
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
