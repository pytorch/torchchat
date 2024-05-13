struct cpuinfo_mock_file filesystem[] = {
#if CPUINFO_ARCH_ARM64
	{
		.path = "/proc/cpuinfo",
		.size = 1515,
		.content =
			"Processor\t: AArch64 Processor rev 4 (aarch64)\n"
			"processor\t: 0\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 4\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 5\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 6\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 7\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"Hardware\t: MT6771V/C\n",
	},
#elif CPUINFO_ARCH_ARM
	{
		.path = "/proc/cpuinfo",
		.size = 2306,
		.content =
			"Processor\t: AArch64 Processor rev 4 (aarch64)\n"
			"processor\t: 0\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 1\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 2\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 3\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 4\n"
			"model name\t: ARMv8 Processor rev 2 (v8l)\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 5\n"
			"model name\t: ARMv8 Processor rev 2 (v8l)\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 6\n"
			"model name\t: ARMv8 Processor rev 2 (v8l)\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 7\n"
			"model name\t: ARMv8 Processor rev 2 (v8l)\n"
			"BogoMIPS\t: 26.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd09\n"
			"CPU revision\t: 2\n"
			"\n"
			"Hardware\t: MT6771V/C",
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
		.path = "/sys/devices/system/cpu/sched_isolated",
		.size = 4,
		.content = "4-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_driver",
		.size = 26,
		.content = "mt67xx_acao_cpuidle_set_0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_governor_ro",
		.size = 9,
		.content = "mtk_menu\n",
	},
	{
		.path = "/sys/devices/system/cpu/cputopo/cpus_per_cluster",
		.size = 25,
		.content =
			"cluster0: f\n"
			"cluster1: f0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cputopo/glbinfo",
		.size = 72,
		.content =
			"big/little arch: yes\n"
			"nr_cups: 8\n"
			"nr_clusters: 2\n"
			"cluster0: f\n"
			"cluster1: f0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cputopo/is_big_little",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cputopo/is_multi_cluster",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cputopo/nr_clusters",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpuidle/driver/name",
		.size = 26,
		.content = "mt67xx_acao_cpuidle_set_0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1989000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 5,
		.content = "1000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 127,
		.content = "1989000 1924000 1846000 1781000 1716000 1677000 1625000 1586000 1508000 1417000 1326000 1248000 1131000 1014000 910000 793000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 64,
		.content = "ondemand userspace powersave interactive performance schedplus \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 11,
		.content = "mt-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/sched/down_throttle_nsec",
		.size = 8,
		.content = "4000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/sched/up_throttle_nsec",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 187,
		.content =
			"1989000 5369\n"
			"1924000 29\n"
			"1846000 185\n"
			"1781000 15\n"
			"1716000 15\n"
			"1677000 122\n"
			"1625000 456\n"
			"1586000 64\n"
			"1508000 41\n"
			"1417000 66\n"
			"1326000 218\n"
			"1248000 125\n"
			"1131000 216\n"
			"1014000 203\n"
			"910000 235\n"
			"793000 92671\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "648\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/trans_table",
		.size = 2941,
		.content =
			"   From  :    To\n"
			"         :   1989000   1924000   1846000   1781000   1716000   1677000   1625000   1586000   1508000   1417000   1326000   1248000   1131000   1014000    910000    793000 \n"
			"  1989000:         0         0         6         0         0         8         2         3         0         0         4         2         0         0         3        10 \n"
			"  1924000:         2         0         0         0         0         1         0         1         0         0         0         0         0         0         0         3 \n"
			"  1846000:         5         1         0         0         1         1         1         1         4         1         5         1         0         0         0         0 \n"
			"  1781000:         0         0         0         0         1         1         1         0         0         1         1         0         1         1         0         0 \n"
			"  1716000:         2         0         0         0         0         0         1         2         1         0         1         0         0         0         0         0 \n"
			"  1677000:         5         1         2         1         2         0         0         0         0         0         1         0         0         0         1         1 \n"
			"  1625000:         1         0         2         0         0         0         0         3         2         4         2         0         2         1         1         6 \n"
			"  1586000:         2         1         0         3         0         1         4         0         3         2         1         2         3         0         1         6 \n"
			"  1508000:         0         0         1         0         1         1         5         4         0         2         2         1         1         2         0         0 \n"
			"  1417000:         1         0         0         1         2         0         4         6         2         0         5         1         2         2         1         4 \n"
			"  1326000:         3         0         2         0         0         0         3         1         5         6         0        12         5         7         3         5 \n"
			"  1248000:         2         0         0         0         0         0         0         2         0         7        10         0        20         4         5         4 \n"
			"  1131000:         0         0         0         1         0         1         0         2         2         4        13        17         0        18         9         9 \n"
			"  1014000:         1         0         0         0         0         0         0         0         1         0         4         7        26         0        24        14 \n"
			"   910000:         0         0         0         0         0         0         0         0         0         1         1         6        13        31         0        39 \n"
			"   793000:        14         4         8         1         0         0         3         4         0         3         2         5         2        11        43         0 \n",
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
		.size = 26,
		.content = "mt67xx_acao_cpuidle_set_0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1989000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_transition_latency",
		.size = 5,
		.content = "1000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_frequencies",
		.size = 127,
		.content = "1989000 1924000 1846000 1781000 1716000 1677000 1625000 1586000 1508000 1417000 1326000 1248000 1131000 1014000 910000 793000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 64,
		.content = "ondemand userspace powersave interactive performance schedplus \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
		.size = 11,
		.content = "mt-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/sched/down_throttle_nsec",
		.size = 8,
		.content = "4000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/sched/up_throttle_nsec",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 187,
		.content =
			"1989000 5369\n"
			"1924000 29\n"
			"1846000 185\n"
			"1781000 15\n"
			"1716000 15\n"
			"1677000 122\n"
			"1625000 456\n"
			"1586000 64\n"
			"1508000 41\n"
			"1417000 66\n"
			"1326000 218\n"
			"1248000 125\n"
			"1131000 216\n"
			"1014000 203\n"
			"910000 235\n"
			"793000 92870\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 4,
		.content = "648\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/trans_table",
		.size = 2941,
		.content =
			"   From  :    To\n"
			"         :   1989000   1924000   1846000   1781000   1716000   1677000   1625000   1586000   1508000   1417000   1326000   1248000   1131000   1014000    910000    793000 \n"
			"  1989000:         0         0         6         0         0         8         2         3         0         0         4         2         0         0         3        10 \n"
			"  1924000:         2         0         0         0         0         1         0         1         0         0         0         0         0         0         0         3 \n"
			"  1846000:         5         1         0         0         1         1         1         1         4         1         5         1         0         0         0         0 \n"
			"  1781000:         0         0         0         0         1         1         1         0         0         1         1         0         1         1         0         0 \n"
			"  1716000:         2         0         0         0         0         0         1         2         1         0         1         0         0         0         0         0 \n"
			"  1677000:         5         1         2         1         2         0         0         0         0         0         1         0         0         0         1         1 \n"
			"  1625000:         1         0         2         0         0         0         0         3         2         4         2         0         2         1         1         6 \n"
			"  1586000:         2         1         0         3         0         1         4         0         3         2         1         2         3         0         1         6 \n"
			"  1508000:         0         0         1         0         1         1         5         4         0         2         2         1         1         2         0         0 \n"
			"  1417000:         1         0         0         1         2         0         4         6         2         0         5         1         2         2         1         4 \n"
			"  1326000:         3         0         2         0         0         0         3         1         5         6         0        12         5         7         3         5 \n"
			"  1248000:         2         0         0         0         0         0         0         2         0         7        10         0        20         4         5         4 \n"
			"  1131000:         0         0         0         1         0         1         0         2         2         4        13        17         0        18         9         9 \n"
			"  1014000:         1         0         0         0         0         0         0         0         1         0         4         7        26         0        24        14 \n"
			"   910000:         0         0         0         0         0         0         0         0         0         1         1         6        13        31         0        39 \n"
			"   793000:        14         4         8         1         0         0         3         4         0         3         2         5         2        11        43         0 \n",
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
		.size = 26,
		.content = "mt67xx_acao_cpuidle_set_0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1989000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_transition_latency",
		.size = 5,
		.content = "1000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies",
		.size = 127,
		.content = "1989000 1924000 1846000 1781000 1716000 1677000 1625000 1586000 1508000 1417000 1326000 1248000 1131000 1014000 910000 793000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 64,
		.content = "ondemand userspace powersave interactive performance schedplus \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver",
		.size = 11,
		.content = "mt-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/sched/down_throttle_nsec",
		.size = 8,
		.content = "4000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/sched/up_throttle_nsec",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 187,
		.content =
			"1989000 5369\n"
			"1924000 29\n"
			"1846000 185\n"
			"1781000 15\n"
			"1716000 15\n"
			"1677000 122\n"
			"1625000 456\n"
			"1586000 64\n"
			"1508000 41\n"
			"1417000 66\n"
			"1326000 218\n"
			"1248000 125\n"
			"1131000 216\n"
			"1014000 203\n"
			"910000 235\n"
			"793000 93070\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 4,
		.content = "648\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/trans_table",
		.size = 2941,
		.content =
			"   From  :    To\n"
			"         :   1989000   1924000   1846000   1781000   1716000   1677000   1625000   1586000   1508000   1417000   1326000   1248000   1131000   1014000    910000    793000 \n"
			"  1989000:         0         0         6         0         0         8         2         3         0         0         4         2         0         0         3        10 \n"
			"  1924000:         2         0         0         0         0         1         0         1         0         0         0         0         0         0         0         3 \n"
			"  1846000:         5         1         0         0         1         1         1         1         4         1         5         1         0         0         0         0 \n"
			"  1781000:         0         0         0         0         1         1         1         0         0         1         1         0         1         1         0         0 \n"
			"  1716000:         2         0         0         0         0         0         1         2         1         0         1         0         0         0         0         0 \n"
			"  1677000:         5         1         2         1         2         0         0         0         0         0         1         0         0         0         1         1 \n"
			"  1625000:         1         0         2         0         0         0         0         3         2         4         2         0         2         1         1         6 \n"
			"  1586000:         2         1         0         3         0         1         4         0         3         2         1         2         3         0         1         6 \n"
			"  1508000:         0         0         1         0         1         1         5         4         0         2         2         1         1         2         0         0 \n"
			"  1417000:         1         0         0         1         2         0         4         6         2         0         5         1         2         2         1         4 \n"
			"  1326000:         3         0         2         0         0         0         3         1         5         6         0        12         5         7         3         5 \n"
			"  1248000:         2         0         0         0         0         0         0         2         0         7        10         0        20         4         5         4 \n"
			"  1131000:         0         0         0         1         0         1         0         2         2         4        13        17         0        18         9         9 \n"
			"  1014000:         1         0         0         0         0         0         0         0         1         0         4         7        26         0        24        14 \n"
			"   910000:         0         0         0         0         0         0         0         0         0         1         1         6        13        31         0        39 \n"
			"   793000:        14         4         8         1         0         0         3         4         0         3         2         5         2        11        43         0 \n",
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
		.size = 26,
		.content = "mt67xx_acao_cpuidle_set_0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1989000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_transition_latency",
		.size = 5,
		.content = "1000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 127,
		.content = "1989000 1924000 1846000 1781000 1716000 1677000 1625000 1586000 1508000 1417000 1326000 1248000 1131000 1014000 910000 793000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 64,
		.content = "ondemand userspace powersave interactive performance schedplus \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_driver",
		.size = 11,
		.content = "mt-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/sched/down_throttle_nsec",
		.size = 8,
		.content = "4000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/sched/up_throttle_nsec",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 187,
		.content =
			"1989000 5369\n"
			"1924000 29\n"
			"1846000 185\n"
			"1781000 15\n"
			"1716000 15\n"
			"1677000 122\n"
			"1625000 456\n"
			"1586000 64\n"
			"1508000 41\n"
			"1417000 66\n"
			"1326000 218\n"
			"1248000 125\n"
			"1131000 216\n"
			"1014000 203\n"
			"910000 235\n"
			"793000 93274\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "648\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/trans_table",
		.size = 2941,
		.content =
			"   From  :    To\n"
			"         :   1989000   1924000   1846000   1781000   1716000   1677000   1625000   1586000   1508000   1417000   1326000   1248000   1131000   1014000    910000    793000 \n"
			"  1989000:         0         0         6         0         0         8         2         3         0         0         4         2         0         0         3        10 \n"
			"  1924000:         2         0         0         0         0         1         0         1         0         0         0         0         0         0         0         3 \n"
			"  1846000:         5         1         0         0         1         1         1         1         4         1         5         1         0         0         0         0 \n"
			"  1781000:         0         0         0         0         1         1         1         0         0         1         1         0         1         1         0         0 \n"
			"  1716000:         2         0         0         0         0         0         1         2         1         0         1         0         0         0         0         0 \n"
			"  1677000:         5         1         2         1         2         0         0         0         0         0         1         0         0         0         1         1 \n"
			"  1625000:         1         0         2         0         0         0         0         3         2         4         2         0         2         1         1         6 \n"
			"  1586000:         2         1         0         3         0         1         4         0         3         2         1         2         3         0         1         6 \n"
			"  1508000:         0         0         1         0         1         1         5         4         0         2         2         1         1         2         0         0 \n"
			"  1417000:         1         0         0         1         2         0         4         6         2         0         5         1         2         2         1         4 \n"
			"  1326000:         3         0         2         0         0         0         3         1         5         6         0        12         5         7         3         5 \n"
			"  1248000:         2         0         0         0         0         0         0         2         0         7        10         0        20         4         5         4 \n"
			"  1131000:         0         0         0         1         0         1         0         2         2         4        13        17         0        18         9         9 \n"
			"  1014000:         1         0         0         0         0         0         0         0         1         0         4         7        26         0        24        14 \n"
			"   910000:         0         0         0         0         0         0         0         0         0         1         1         6        13        31         0        39 \n"
			"   793000:        14         4         8         1         0         0         3         4         0         3         2         5         2        11        43         0 \n",
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
		.size = 26,
		.content = "mt67xx_acao_cpuidle_set_1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1989000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_transition_latency",
		.size = 5,
		.content = "1000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies",
		.size = 127,
		.content = "1989000 1924000 1846000 1781000 1716000 1677000 1625000 1586000 1508000 1417000 1326000 1248000 1131000 1014000 910000 793000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors",
		.size = 64,
		.content = "ondemand userspace powersave interactive performance schedplus \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver",
		.size = 11,
		.content = "mt-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/sched/down_throttle_nsec",
		.size = 8,
		.content = "4000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/sched/up_throttle_nsec",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 178,
		.content =
			"1989000 5860\n"
			"1924000 14\n"
			"1846000 100\n"
			"1781000 2\n"
			"1716000 2\n"
			"1677000 0\n"
			"1625000 12\n"
			"1586000 10\n"
			"1508000 0\n"
			"1417000 12\n"
			"1326000 4\n"
			"1248000 51\n"
			"1131000 240\n"
			"1014000 138\n"
			"910000 180\n"
			"793000 94210\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 4,
		.content = "162\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/trans_table",
		.size = 2941,
		.content =
			"   From  :    To\n"
			"         :   1989000   1924000   1846000   1781000   1716000   1677000   1625000   1586000   1508000   1417000   1326000   1248000   1131000   1014000    910000    793000 \n"
			"  1989000:         0         4         5         1         0         0         1         1         0         3         1         0         1         0         2        14 \n"
			"  1924000:         4         0         0         0         1         0         0         0         0         0         0         0         0         0         0         1 \n"
			"  1846000:         9         0         0         0         0         0         0         0         0         0         0         0         0         2         4         1 \n"
			"  1781000:         0         0         0         0         0         0         1         0         0         0         0         0         0         0         0         0 \n"
			"  1716000:         0         0         0         0         0         0         1         0         0         0         0         0         0         0         0         0 \n"
			"  1677000:         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1625000:         2         0         0         0         0         0         0         1         0         0         0         1         0         1         0         0 \n"
			"  1586000:         2         1         0         0         0         0         1         0         0         0         0         0         0         0         0         1 \n"
			"  1508000:         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1417000:         2         0         0         0         0         0         0         1         0         0         0         1         0         0         0         0 \n"
			"  1326000:         0         0         0         0         0         0         0         0         0         0         0         1         1         0         0         0 \n"
			"  1248000:         1         1         0         0         0         0         0         1         0         1         1         0         3         0         0         2 \n"
			"  1131000:         3         0         0         0         0         0         0         0         0         0         0         3         0         2         0         8 \n"
			"  1014000:         2         0         2         0         0         0         0         0         0         0         0         0         5         0         1         2 \n"
			"   910000:         5         0         1         0         0         0         0         0         0         0         0         1         0         2         0         7 \n"
			"   793000:         3         0         8         0         0         0         1         1         0         0         0         2         6         5         9         0 \n",
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
		.size = 26,
		.content = "mt67xx_acao_cpuidle_set_1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1989000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_transition_latency",
		.size = 5,
		.content = "1000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_frequencies",
		.size = 127,
		.content = "1989000 1924000 1846000 1781000 1716000 1677000 1625000 1586000 1508000 1417000 1326000 1248000 1131000 1014000 910000 793000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_governors",
		.size = 64,
		.content = "ondemand userspace powersave interactive performance schedplus \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_driver",
		.size = 11,
		.content = "mt-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_max_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/sched/down_throttle_nsec",
		.size = 8,
		.content = "4000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/sched/up_throttle_nsec",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 178,
		.content =
			"1989000 5860\n"
			"1924000 14\n"
			"1846000 100\n"
			"1781000 2\n"
			"1716000 2\n"
			"1677000 0\n"
			"1625000 12\n"
			"1586000 10\n"
			"1508000 0\n"
			"1417000 12\n"
			"1326000 4\n"
			"1248000 51\n"
			"1131000 240\n"
			"1014000 138\n"
			"910000 180\n"
			"793000 94410\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 4,
		.content = "162\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/trans_table",
		.size = 2941,
		.content =
			"   From  :    To\n"
			"         :   1989000   1924000   1846000   1781000   1716000   1677000   1625000   1586000   1508000   1417000   1326000   1248000   1131000   1014000    910000    793000 \n"
			"  1989000:         0         4         5         1         0         0         1         1         0         3         1         0         1         0         2        14 \n"
			"  1924000:         4         0         0         0         1         0         0         0         0         0         0         0         0         0         0         1 \n"
			"  1846000:         9         0         0         0         0         0         0         0         0         0         0         0         0         2         4         1 \n"
			"  1781000:         0         0         0         0         0         0         1         0         0         0         0         0         0         0         0         0 \n"
			"  1716000:         0         0         0         0         0         0         1         0         0         0         0         0         0         0         0         0 \n"
			"  1677000:         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1625000:         2         0         0         0         0         0         0         1         0         0         0         1         0         1         0         0 \n"
			"  1586000:         2         1         0         0         0         0         1         0         0         0         0         0         0         0         0         1 \n"
			"  1508000:         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1417000:         2         0         0         0         0         0         0         1         0         0         0         1         0         0         0         0 \n"
			"  1326000:         0         0         0         0         0         0         0         0         0         0         0         1         1         0         0         0 \n"
			"  1248000:         1         1         0         0         0         0         0         1         0         1         1         0         3         0         0         2 \n"
			"  1131000:         3         0         0         0         0         0         0         0         0         0         0         3         0         2         0         8 \n"
			"  1014000:         2         0         2         0         0         0         0         0         0         0         0         0         5         0         1         2 \n"
			"   910000:         5         0         1         0         0         0         0         0         0         0         0         1         0         2         0         7 \n"
			"   793000:         3         0         8         0         0         0         1         1         0         0         0         2         6         5         9         0 \n",
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
		.size = 26,
		.content = "mt67xx_acao_cpuidle_set_1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1989000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_transition_latency",
		.size = 5,
		.content = "1000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_frequencies",
		.size = 127,
		.content = "1989000 1924000 1846000 1781000 1716000 1677000 1625000 1586000 1508000 1417000 1326000 1248000 1131000 1014000 910000 793000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_governors",
		.size = 64,
		.content = "ondemand userspace powersave interactive performance schedplus \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_driver",
		.size = 11,
		.content = "mt-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/sched/down_throttle_nsec",
		.size = 8,
		.content = "4000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/sched/up_throttle_nsec",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 178,
		.content =
			"1989000 5860\n"
			"1924000 14\n"
			"1846000 100\n"
			"1781000 2\n"
			"1716000 2\n"
			"1677000 0\n"
			"1625000 12\n"
			"1586000 10\n"
			"1508000 0\n"
			"1417000 12\n"
			"1326000 4\n"
			"1248000 51\n"
			"1131000 240\n"
			"1014000 138\n"
			"910000 180\n"
			"793000 94609\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 4,
		.content = "162\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/trans_table",
		.size = 2941,
		.content =
			"   From  :    To\n"
			"         :   1989000   1924000   1846000   1781000   1716000   1677000   1625000   1586000   1508000   1417000   1326000   1248000   1131000   1014000    910000    793000 \n"
			"  1989000:         0         4         5         1         0         0         1         1         0         3         1         0         1         0         2        14 \n"
			"  1924000:         4         0         0         0         1         0         0         0         0         0         0         0         0         0         0         1 \n"
			"  1846000:         9         0         0         0         0         0         0         0         0         0         0         0         0         2         4         1 \n"
			"  1781000:         0         0         0         0         0         0         1         0         0         0         0         0         0         0         0         0 \n"
			"  1716000:         0         0         0         0         0         0         1         0         0         0         0         0         0         0         0         0 \n"
			"  1677000:         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1625000:         2         0         0         0         0         0         0         1         0         0         0         1         0         1         0         0 \n"
			"  1586000:         2         1         0         0         0         0         1         0         0         0         0         0         0         0         0         1 \n"
			"  1508000:         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1417000:         2         0         0         0         0         0         0         1         0         0         0         1         0         0         0         0 \n"
			"  1326000:         0         0         0         0         0         0         0         0         0         0         0         1         1         0         0         0 \n"
			"  1248000:         1         1         0         0         0         0         0         1         0         1         1         0         3         0         0         2 \n"
			"  1131000:         3         0         0         0         0         0         0         0         0         0         0         3         0         2         0         8 \n"
			"  1014000:         2         0         2         0         0         0         0         0         0         0         0         0         5         0         1         2 \n"
			"   910000:         5         0         1         0         0         0         0         0         0         0         0         1         0         2         0         7 \n"
			"   793000:         3         0         8         0         0         0         1         1         0         0         0         2         6         5         9         0 \n",
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
		.size = 26,
		.content = "mt67xx_acao_cpuidle_set_1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1989000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_transition_latency",
		.size = 5,
		.content = "1000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_frequencies",
		.size = 127,
		.content = "1989000 1924000 1846000 1781000 1716000 1677000 1625000 1586000 1508000 1417000 1326000 1248000 1131000 1014000 910000 793000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_governors",
		.size = 64,
		.content = "ondemand userspace powersave interactive performance schedplus \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_driver",
		.size = 11,
		.content = "mt-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "793000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/sched/down_throttle_nsec",
		.size = 8,
		.content = "4000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/sched/up_throttle_nsec",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 178,
		.content =
			"1989000 5860\n"
			"1924000 14\n"
			"1846000 100\n"
			"1781000 2\n"
			"1716000 2\n"
			"1677000 0\n"
			"1625000 12\n"
			"1586000 10\n"
			"1508000 0\n"
			"1417000 12\n"
			"1326000 4\n"
			"1248000 51\n"
			"1131000 240\n"
			"1014000 138\n"
			"910000 180\n"
			"793000 94810\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 4,
		.content = "162\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/trans_table",
		.size = 2941,
		.content =
			"   From  :    To\n"
			"         :   1989000   1924000   1846000   1781000   1716000   1677000   1625000   1586000   1508000   1417000   1326000   1248000   1131000   1014000    910000    793000 \n"
			"  1989000:         0         4         5         1         0         0         1         1         0         3         1         0         1         0         2        14 \n"
			"  1924000:         4         0         0         0         1         0         0         0         0         0         0         0         0         0         0         1 \n"
			"  1846000:         9         0         0         0         0         0         0         0         0         0         0         0         0         2         4         1 \n"
			"  1781000:         0         0         0         0         0         0         1         0         0         0         0         0         0         0         0         0 \n"
			"  1716000:         0         0         0         0         0         0         1         0         0         0         0         0         0         0         0         0 \n"
			"  1677000:         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1625000:         2         0         0         0         0         0         0         1         0         0         0         1         0         1         0         0 \n"
			"  1586000:         2         1         0         0         0         0         1         0         0         0         0         0         0         0         0         1 \n"
			"  1508000:         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1417000:         2         0         0         0         0         0         0         1         0         0         0         1         0         0         0         0 \n"
			"  1326000:         0         0         0         0         0         0         0         0         0         0         0         1         1         0         0         0 \n"
			"  1248000:         1         1         0         0         0         0         0         1         0         1         1         0         3         0         0         2 \n"
			"  1131000:         3         0         0         0         0         0         0         0         0         0         0         3         0         2         0         8 \n"
			"  1014000:         2         0         2         0         0         0         0         0         0         0         0         0         5         0         1         2 \n"
			"   910000:         5         0         1         0         0         0         0         0         0         0         0         1         0         2         0         7 \n"
			"   793000:         3         0         8         0         0         0         1         1         0         0         0         2         6         5         9         0 \n",
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
		.key = "af.modem_1.epof",
		.value = "0",
	},
	{
		.key = "af.music.outputid",
		.value = "3",
	},
	{
		.key = "af.recovery.mic_mute_on",
		.value = "0",
	},
	{
		.key = "af.speech.shm_init",
		.value = "1",
	},
	{
		.key = "bgw.current3gband",
		.value = "0",
	},
	{
		.key = "bt.profiles.avrcp.multiPlayer.enable",
		.value = "0",
	},
	{
		.key = "camera.disable_zsl_mode",
		.value = "1",
	},
	{
		.key = "camera.mdp.cz.enable",
		.value = "1",
	},
	{
		.key = "camera.mdp.dre.enable",
		.value = "1",
	},
	{
		.key = "cdma.operator.sid",
		.value = "0",
	},
	{
		.key = "cdma.prl.version0",
		.value = "302",
	},
	{
		.key = "dalvik.vm.appimageformat",
		.value = "lz4",
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
		.key = "dalvik.vm.dexopt.secondary",
		.value = "true",
	},
	{
		.key = "dalvik.vm.heapgrowthlimit",
		.value = "384m",
	},
	{
		.key = "dalvik.vm.heapmaxfree",
		.value = "16m",
	},
	{
		.key = "dalvik.vm.heapminfree",
		.value = "4m",
	},
	{
		.key = "dalvik.vm.heapsize",
		.value = "512m",
	},
	{
		.key = "dalvik.vm.heapstartsize",
		.value = "16m",
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
		.key = "dalvik.vm.isa.arm.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.arm.variant",
		.value = "cortex-a53",
	},
	{
		.key = "dalvik.vm.isa.arm64.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.arm64.variant",
		.value = "cortex-a53",
	},
	{
		.key = "dalvik.vm.mtk-stack-trace-file",
		.value = "/data/anr/mtk_traces.txt",
	},
	{
		.key = "dalvik.vm.stack-trace-dir",
		.value = "/data/anr",
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
		.key = "debug.MB.running",
		.value = "0",
	},
	{
		.key = "debug.atrace.tags.enableflags",
		.value = "0",
	},
	{
		.key = "debug.choreographer.skipwarning",
		.value = "1",
	},
	{
		.key = "debug.force_rtl",
		.value = "0",
	},
	{
		.key = "debug.junk.process.name",
		.value = "com.android.settings",
	},
	{
		.key = "debug.junk.process.pid",
		.value = "4944",
	},
	{
		.key = "debug.mdl.EE.done",
		.value = "",
	},
	{
		.key = "debug.mdl.EE.folder",
		.value = "",
	},
	{
		.key = "debug.mdl.run.folder",
		.value = "",
	},
	{
		.key = "debug.mdlogger.Running",
		.value = "0",
	},
	{
		.key = "debug.mdlogger.log2sd.path",
		.value = "internal_sd",
	},
	{
		.key = "debug.met.running",
		.value = "0",
	},
	{
		.key = "debug.met_log_d.user",
		.value = "shell",
	},
	{
		.key = "debug.met_log_d.version",
		.value = "V6.0.0",
	},
	{
		.key = "debug.mtk.aee.status",
		.value = "free",
	},
	{
		.key = "debug.mtk.aee.status64",
		.value = "free",
	},
	{
		.key = "debug.mtk.aee.vstatus",
		.value = "free",
	},
	{
		.key = "debug.mtk.aee.vstatus64",
		.value = "free",
	},
	{
		.key = "debug.mtklog.netlog.Running",
		.value = "0",
	},
	{
		.key = "debug.oppo.morning.time",
		.value = "4 : 40",
	},
	{
		.key = "debug.pq.acaltm.dbg",
		.value = "0",
	},
	{
		.key = "debug.pq.adl.dbg",
		.value = "0",
	},
	{
		.key = "debug.pq.cz.isp.tuning",
		.value = "0",
	},
	{
		.key = "debug.pq.dre.dbg",
		.value = "0",
	},
	{
		.key = "debug.pq.dre.demowin.x",
		.value = "536805376",
	},
	{
		.key = "debug.pq.dre.isp.tuning",
		.value = "0",
	},
	{
		.key = "debug.pq.dredriver.blk",
		.value = "0",
	},
	{
		.key = "debug.pq.dredriver.dbg",
		.value = "0",
	},
	{
		.key = "debug.pq.dshp.en",
		.value = "2",
	},
	{
		.key = "debug.pq.hdr.dbg",
		.value = "0",
	},
	{
		.key = "debug.pq.shp.en",
		.value = "2",
	},
	{
		.key = "debug.pullmdlog",
		.value = "",
	},
	{
		.key = "debug.screenoff.unlock",
		.value = "0",
	},
	{
		.key = "debug.sf.disable_backpressure",
		.value = "1",
	},
	{
		.key = "debug.sys.oppo.keytime",
		.value = "0",
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
		.key = "fmradio.driver.enable",
		.value = "1",
	},
	{
		.key = "fmradio_drv.ko",
		.value = "1",
	},
	{
		.key = "gps_drv.ko",
		.value = "1",
	},
	{
		.key = "gr.apk.number",
		.value = "5",
	},
	{
		.key = "gr.download.url",
		.value = "http://otafs.coloros.com/googles/7a3341ea988614321a209c55265d381f",
	},
	{
		.key = "gr.use.leader",
		.value = "true",
	},
	{
		.key = "gsm.baseband.capability",
		.value = "1023",
	},
	{
		.key = "gsm.current.phone-type",
		.value = "2,1",
	},
	{
		.key = "gsm.enable_hotswap",
		.value = "false",
	},
	{
		.key = "gsm.external.sim.timeout",
		.value = "13,13",
	},
	{
		.key = "gsm.gcf.testmode",
		.value = "0",
	},
	{
		.key = "gsm.ims.type0",
		.value = "",
	},
	{
		.key = "gsm.ims.type1",
		.value = "",
	},
	{
		.key = "gsm.lte.ca.support",
		.value = "1",
	},
	{
		.key = "gsm.modem.vsim.capability",
		.value = "2",
	},
	{
		.key = "gsm.network.type",
		.value = "Unknown,Unknown",
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
		.key = "gsm.oppo.oos0",
		.value = "0",
	},
	{
		.key = "gsm.oppo.oos1",
		.value = "0",
	},
	{
		.key = "gsm.project.baseband",
		.value = "OPPO6771_17197(LWCTG)",
	},
	{
		.key = "gsm.ril.ct3g",
		.value = "0",
	},
	{
		.key = "gsm.ril.ct3g.2",
		.value = "0",
	},
	{
		.key = "gsm.ril.eboot",
		.value = "0",
	},
	{
		.key = "gsm.ril.fulluicctype",
		.value = "",
	},
	{
		.key = "gsm.ril.fulluicctype.2",
		.value = "",
	},
	{
		.key = "gsm.ril.uicctype",
		.value = "",
	},
	{
		.key = "gsm.ril.uicctype.2",
		.value = "",
	},
	{
		.key = "gsm.serial",
		.value = "001719708329032600063245",
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
		.key = "gsm.sim.operator.spn",
		.value = "No service,No service",
	},
	{
		.key = "gsm.sim.ril.mcc.mnc",
		.value = "",
	},
	{
		.key = "gsm.sim.ril.mcc.mnc.2",
		.value = "",
	},
	{
		.key = "gsm.sim.state",
		.value = "ABSENT,ABSENT",
	},
	{
		.key = "gsm.version.baseband",
		.value = "M_V3_P10,M_V3_P10",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "android reference-ril 1.0",
	},
	{
		.key = "hwservicemanager.ready",
		.value = "true",
	},
	{
		.key = "init.svc.NvRAMAgent",
		.value = "running",
	},
	{
		.key = "init.svc.aal",
		.value = "running",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.aee-reinit",
		.value = "stopped",
	},
	{
		.key = "init.svc.aee_aed",
		.value = "running",
	},
	{
		.key = "init.svc.aee_aed64",
		.value = "running",
	},
	{
		.key = "init.svc.aee_aedv",
		.value = "running",
	},
	{
		.key = "init.svc.aee_aedv64",
		.value = "running",
	},
	{
		.key = "init.svc.aeev-reinit",
		.value = "stopped",
	},
	{
		.key = "init.svc.agpsd",
		.value = "running",
	},
	{
		.key = "init.svc.audio-hal-2-0",
		.value = "running",
	},
	{
		.key = "init.svc.audioserver",
		.value = "running",
	},
	{
		.key = "init.svc.batterywarning",
		.value = "running",
	},
	{
		.key = "init.svc.bip",
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
		.key = "init.svc.bootlogoupdater",
		.value = "stopped",
	},
	{
		.key = "init.svc.broadcastradio-hal",
		.value = "running",
	},
	{
		.key = "init.svc.bspCriticalLog",
		.value = "running",
	},
	{
		.key = "init.svc.bspFwUpdate",
		.value = "running",
	},
	{
		.key = "init.svc.camerahalserver",
		.value = "running",
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
		.key = "init.svc.ccci3_fsd",
		.value = "stopped",
	},
	{
		.key = "init.svc.ccci3_mdinit",
		.value = "stopped",
	},
	{
		.key = "init.svc.ccci_fsd",
		.value = "running",
	},
	{
		.key = "init.svc.ccci_mdinit",
		.value = "running",
	},
	{
		.key = "init.svc.ccci_rpcd",
		.value = "running",
	},
	{
		.key = "init.svc.configstore-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.criticallog",
		.value = "running",
	},
	{
		.key = "init.svc.datafree",
		.value = "stopped",
	},
	{
		.key = "init.svc.datarefresh",
		.value = "stopped",
	},
	{
		.key = "init.svc.dfps-1-0",
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
		.key = "init.svc.dumpLog_off",
		.value = "stopped",
	},
	{
		.key = "init.svc.emdlogger1",
		.value = "running",
	},
	{
		.key = "init.svc.emsvr_user",
		.value = "running",
	},
	{
		.key = "init.svc.engineer_native",
		.value = "running",
	},
	{
		.key = "init.svc.engineer_shell",
		.value = "stopped",
	},
	{
		.key = "init.svc.epdg_wod",
		.value = "running",
	},
	{
		.key = "init.svc.face_hal",
		.value = "running",
	},
	{
		.key = "init.svc.flash_recovery",
		.value = "stopped",
	},
	{
		.key = "init.svc.fpay_hal",
		.value = "running",
	},
	{
		.key = "init.svc.fps_hal",
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
		.key = "init.svc.ged_srv",
		.value = "running",
	},
	{
		.key = "init.svc.gnss_service",
		.value = "running",
	},
	{
		.key = "init.svc.gralloc-2-0",
		.value = "running",
	},
	{
		.key = "init.svc.gsm0710muxd",
		.value = "running",
	},
	{
		.key = "init.svc.hal_cryptoeng_oppo",
		.value = "running",
	},
	{
		.key = "init.svc.healthd",
		.value = "running",
	},
	{
		.key = "init.svc.hidl_memory",
		.value = "running",
	},
	{
		.key = "init.svc.hostapd",
		.value = "stopped",
	},
	{
		.key = "init.svc.hwcomposer-2-1",
		.value = "running",
	},
	{
		.key = "init.svc.hwservicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.hypnus_context",
		.value = "stopped",
	},
	{
		.key = "init.svc.hypnus_logging",
		.value = "stopped",
	},
	{
		.key = "init.svc.inittpdebug",
		.value = "stopped",
	},
	{
		.key = "init.svc.installd",
		.value = "running",
	},
	{
		.key = "init.svc.ipsec_mon",
		.value = "running",
	},
	{
		.key = "init.svc.junklog",
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
		.key = "init.svc.ktv-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.lbs_hidl_service",
		.value = "running",
	},
	{
		.key = "init.svc.light-hal-2-0",
		.value = "running",
	},
	{
		.key = "init.svc.lmkd",
		.value = "running",
	},
	{
		.key = "init.svc.logd",
		.value = "running",
	},
	{
		.key = "init.svc.logd-reconfig",
		.value = "stopped",
	},
	{
		.key = "init.svc.logd-reinit",
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
		.key = "init.svc.met_log_d",
		.value = "running",
	},
	{
		.key = "init.svc.mnld",
		.value = "running",
	},
	{
		.key = "init.svc.mobicore",
		.value = "running",
	},
	{
		.key = "init.svc.mobile_log_d",
		.value = "running",
	},
	{
		.key = "init.svc.mtk_advcamserver",
		.value = "running",
	},
	{
		.key = "init.svc.mtk_hal_wfo",
		.value = "running",
	},
	{
		.key = "init.svc.mtkcodecservice-1-1",
		.value = "running",
	},
	{
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.netdagent",
		.value = "running",
	},
	{
		.key = "init.svc.netdiag",
		.value = "running",
	},
	{
		.key = "init.svc.neuralnetworks_hal_service_apunn",
		.value = "running",
	},
	{
		.key = "init.svc.neuralnetworks_hal_service_armnn",
		.value = "running",
	},
	{
		.key = "init.svc.nvram_daemon",
		.value = "stopped",
	},
	{
		.key = "init.svc.oiface",
		.value = "running",
	},
	{
		.key = "init.svc.oppo_fingerprints_sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.oppo_kevents",
		.value = "running",
	},
	{
		.key = "init.svc.oppoalgo",
		.value = "running",
	},
	{
		.key = "init.svc.oppogift",
		.value = "running",
	},
	{
		.key = "init.svc.ousage",
		.value = "running",
	},
	{
		.key = "init.svc.power-hal-1-1",
		.value = "running",
	},
	{
		.key = "init.svc.powerlog",
		.value = "stopped",
	},
	{
		.key = "init.svc.pq-2-0",
		.value = "running",
	},
	{
		.key = "init.svc.program_binary",
		.value = "running",
	},
	{
		.key = "init.svc.recover_hang",
		.value = "stopped",
	},
	{
		.key = "init.svc.ril-daemon-mtk",
		.value = "running",
	},
	{
		.key = "init.svc.rutilsdaemon",
		.value = "stopped",
	},
	{
		.key = "init.svc.sensors-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.sn",
		.value = "stopped",
	},
	{
		.key = "init.svc.start_modem",
		.value = "stopped",
	},
	{
		.key = "init.svc.statusd",
		.value = "running",
	},
	{
		.key = "init.svc.storaged",
		.value = "running",
	},
	{
		.key = "init.svc.stp_dump",
		.value = "running",
	},
	{
		.key = "init.svc.surfaceflinger",
		.value = "running",
	},
	{
		.key = "init.svc.swap_enable_init",
		.value = "stopped",
	},
	{
		.key = "init.svc.sysenv_daemon",
		.value = "running",
	},
	{
		.key = "init.svc.thermal",
		.value = "running",
	},
	{
		.key = "init.svc.thermal-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.thermal_manager",
		.value = "stopped",
	},
	{
		.key = "init.svc.thermalloadalgod",
		.value = "running",
	},
	{
		.key = "init.svc.thermalservice",
		.value = "running",
	},
	{
		.key = "init.svc.tombstoned",
		.value = "running",
	},
	{
		.key = "init.svc.ueventd",
		.value = "running",
	},
	{
		.key = "init.svc.vibrator-1-0",
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
		.key = "init.svc.volte_imcb",
		.value = "running",
	},
	{
		.key = "init.svc.volte_imsm_93",
		.value = "running",
	},
	{
		.key = "init.svc.volte_stack",
		.value = "running",
	},
	{
		.key = "init.svc.volte_ua",
		.value = "running",
	},
	{
		.key = "init.svc.vtservice",
		.value = "running",
	},
	{
		.key = "init.svc.vtservice_hidl",
		.value = "running",
	},
	{
		.key = "init.svc.webview_zygote32",
		.value = "running",
	},
	{
		.key = "init.svc.wfca",
		.value = "running",
	},
	{
		.key = "init.svc.wifi2agps",
		.value = "running",
	},
	{
		.key = "init.svc.wifi_hal_legacy",
		.value = "running",
	},
	{
		.key = "init.svc.wificond",
		.value = "running",
	},
	{
		.key = "init.svc.wmt_launcher",
		.value = "running",
	},
	{
		.key = "init.svc.wmt_loader",
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
		.key = "is_can_remove_themespacelib",
		.value = "1",
	},
	{
		.key = "media.settings.xml",
		.value = "/vendor/etc/media_profiles.xml",
	},
	{
		.key = "media.wfd.portrait",
		.value = "0",
	},
	{
		.key = "media.wfd.video-format",
		.value = "7",
	},
	{
		.key = "mediatek.wlan.chip",
		.value = "CONSYS_MT6771",
	},
	{
		.key = "mediatek.wlan.ctia",
		.value = "0",
	},
	{
		.key = "mediatek.wlan.module.postfix",
		.value = "_consys_mt6771",
	},
	{
		.key = "mtk.eccci.c2k",
		.value = "enabled",
	},
	{
		.key = "mtk.md1.status",
		.value = "ready",
	},
	{
		.key = "mtk.vdec.waitkeyframeforplay",
		.value = "1",
	},
	{
		.key = "net.bt.name",
		.value = "Android",
	},
	{
		.key = "net.ims.ipsec.version",
		.value = "2.0",
	},
	{
		.key = "net.perf.internal.cpu.core",
		.value = "4,4,0,0",
	},
	{
		.key = "net.perf.internal.cpu.freq",
		.value = "-1,-1,-1,-1",
	},
	{
		.key = "net.perf.internal.rps",
		.value = "0f",
	},
	{
		.key = "net.perf.rps.default",
		.value = "0f",
	},
	{
		.key = "net.perf.tether.cpu.core",
		.value = "4,4,0,0",
	},
	{
		.key = "net.perf.tether.cpu.freq",
		.value = "1183000,1638000,0,0",
	},
	{
		.key = "net.perf.tether.rps",
		.value = "0f",
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
		.key = "oppo.camera.packname",
		.value = "",
	},
	{
		.key = "oppo.clear.running",
		.value = "0",
	},
	{
		.key = "oppo.device.firstboot",
		.value = "0",
	},
	{
		.key = "oppo.dex.front.package",
		.value = "com.android.settings",
	},
	{
		.key = "oppo.fpc.sw.version",
		.value = "23",
	},
	{
		.key = "oppo.hide.navigationbar",
		.value = "1",
	},
	{
		.key = "oppo.rutils.used.count",
		.value = "0",
	},
	{
		.key = "oppo.sau.modem.deletebkp",
		.value = "1",
	},
	{
		.key = "oppo.service.datafree.enable",
		.value = "0",
	},
	{
		.key = "oppo.service.rutils.enable",
		.value = "0",
	},
	{
		.key = "oppo.simsettings.boot.completed",
		.value = "true",
	},
	{
		.key = "persist.aee.core.direct",
		.value = "disable",
	},
	{
		.key = "persist.aee.core.dump",
		.value = "disable",
	},
	{
		.key = "persist.aee.db.count",
		.value = "4",
	},
	{
		.key = "persist.aee.fatal_db.count",
		.value = "4",
	},
	{
		.key = "persist.anr.dumpthr",
		.value = "1",
	},
	{
		.key = "persist.anr.enhancement",
		.value = "0",
	},
	{
		.key = "persist.datashaping.alarmgroup",
		.value = "1",
	},
	{
		.key = "persist.duraspeed.support",
		.value = "1",
	},
	{
		.key = "persist.log.size.main",
		.value = "",
	},
	{
		.key = "persist.log.tag.AT",
		.value = "I",
	},
	{
		.key = "persist.log.tag.AdnRecord",
		.value = "I",
	},
	{
		.key = "persist.log.tag.AdnRecordCache",
		.value = "I",
	},
	{
		.key = "persist.log.tag.AdnRecordLoader",
		.value = "I",
	},
	{
		.key = "persist.log.tag.AirplaneHandler",
		.value = "I",
	},
	{
		.key = "persist.log.tag.AutoRegSmsFwk",
		.value = "I",
	},
	{
		.key = "persist.log.tag.C2K_AT",
		.value = "I",
	},
	{
		.key = "persist.log.tag.C2K_ATConfig",
		.value = "I",
	},
	{
		.key = "persist.log.tag.C2K_RIL-DATA",
		.value = "I",
	},
	{
		.key = "persist.log.tag.C2K_RILC",
		.value = "I",
	},
	{
		.key = "persist.log.tag.CapaSwitch",
		.value = "I",
	},
	{
		.key = "persist.log.tag.CdmaMoSms",
		.value = "I",
	},
	{
		.key = "persist.log.tag.CdmaMtSms",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ConSmsFwkExt",
		.value = "I",
	},
	{
		.key = "persist.log.tag.CountryDetector",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DC-1",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DC-2",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DCT",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DSSelector",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DSSelectorOP01",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DSSelectorOP02",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DSSelectorOP09",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DSSelectorOP18",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DSSelectorOm",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DSSelectorUtil",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DataDispatcher",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DataOnlySmsFwk",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DcFcMgr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.DupSmsFilterExt",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ECCCallHelper",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ECCNumUtils",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ECCRetryHandler",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ECCRuleHandler",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ECCSwitchPhone",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ExternalSimMgr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.GbaApp",
		.value = "I",
	},
	{
		.key = "persist.log.tag.GbaBsfProcedure",
		.value = "I",
	},
	{
		.key = "persist.log.tag.GbaBsfResponse",
		.value = "I",
	},
	{
		.key = "persist.log.tag.GbaDebugParam",
		.value = "I",
	},
	{
		.key = "persist.log.tag.GbaService",
		.value = "I",
	},
	{
		.key = "persist.log.tag.GsmCallTkrHlpr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.GsmCdmaConn",
		.value = "I",
	},
	{
		.key = "persist.log.tag.GsmCdmaPhone",
		.value = "I",
	},
	{
		.key = "persist.log.tag.GsmConnection",
		.value = "I",
	},
	{
		.key = "persist.log.tag.GsmMmiCode",
		.value = "I",
	},
	{
		.key = "persist.log.tag.IMSRILRequest",
		.value = "I",
	},
	{
		.key = "persist.log.tag.IMS_RILA",
		.value = "I",
	},
	{
		.key = "persist.log.tag.IccCardProxy",
		.value = "I",
	},
	{
		.key = "persist.log.tag.IccPhoneBookIM",
		.value = "I",
	},
	{
		.key = "persist.log.tag.IccProvider",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsApp",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsBaseCommands",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsCall",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsCallProfile",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsCallSession",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsEcbm",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsEcbmProxy",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsManager",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsPhone",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsPhoneBase",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsPhoneCall",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsService",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsUt",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsUtService",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ImsVTProvider",
		.value = "I",
	},
	{
		.key = "persist.log.tag.IsimFileHandler",
		.value = "I",
	},
	{
		.key = "persist.log.tag.IsimRecords",
		.value = "I",
	},
	{
		.key = "persist.log.tag.LIBC2K_RIL",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MGsmSMSDisp",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MSimSmsIStatus",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MSmsStorageMtr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MSmsUsageMtr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MTKSST",
		.value = "D",
	},
	{
		.key = "persist.log.tag.MtkAdnRecord",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkConSmsFwk",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkCsimFH",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkDCT",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkDupSmsFilter",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkFactory",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkGsmCdmaConn",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkIccCardProxy",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkIccPHBIM",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkIccProvider",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkIccSmsIntMgr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkImsManager",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkImsService",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkIsimFH",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkPhoneNotifr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkRecordLoader",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkRetryManager",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkRuimFH",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkSIMFH",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkSIMRecords",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkSmsCbHeader",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkSmsManager",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkSmsMessage",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkSpnOverride",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkSubCtrl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkUiccCard",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkUiccCardApp",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkUiccCtrl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.MtkUsimFH",
		.value = "I",
	},
	{
		.key = "persist.log.tag.Mtk_RIL_ImsSms",
		.value = "I",
	},
	{
		.key = "persist.log.tag.NetAgentService",
		.value = "I",
	},
	{
		.key = "persist.log.tag.NetLnkEventHdlr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.NetworkPolicy",
		.value = "I",
	},
	{
		.key = "persist.log.tag.NetworkStats",
		.value = "I",
	},
	{
		.key = "persist.log.tag.OperatorUtils",
		.value = "I",
	},
	{
		.key = "persist.log.tag.Phone",
		.value = "I",
	},
	{
		.key = "persist.log.tag.PhoneFactory",
		.value = "I",
	},
	{
		.key = "persist.log.tag.ProxyController",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RFX",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL-CC",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL-DATA",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL-Fusion",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL-OEM",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL-PHB",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL-RP",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL-SIM",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL-SMS",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL-SS",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RILC",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RILC-MTK",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RILC-RP",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RILD",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RILMD2-SS",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RILMUXD",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL_Mux",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RIL_UIM_SOCKET",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RP_DAC",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RP_IMS",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RTC_DAC",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RadioManager",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RetryManager",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxAction",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxChannelMgr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxCloneMgr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxContFactory",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxController",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxDT",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxDebugInfo",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxDefDestUtils",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxDisThread",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxFragEnc",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxHandlerMgr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxIdToMsgId",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxIdToStr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxMainThread",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxMclDisThread",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxMclMessenger",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxMclStatusMgr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxMessage",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxObject",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxOpUtils",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxRilAdapter",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxRilUtils",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxRoot",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxSM",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxSocketSM",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxStatusMgr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxTimer",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RfxTransUtils",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RilClient",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RilMalClient",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcCapa",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcCdmaSimUrc",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcCommSimOpReq",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcCommSimReq",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcDcCommon",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcDcDefault",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcDcPdnManager",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcDcReqHandler",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcDcUtility",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcImsCtlReqHdl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcImsCtlUrcHdl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcNwHdlr",
		.value = "D",
	},
	{
		.key = "persist.log.tag.RmcNwReqHdlr",
		.value = "D",
	},
	{
		.key = "persist.log.tag.RmcOpRadioReq",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcPhbReq",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcPhbUrc",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcRadioReq",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RmcRatSwHdlr",
		.value = "D",
	},
	{
		.key = "persist.log.tag.RmcWp",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpAudioControl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpCallControl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpCdmaOemCtrl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpCdmaRadioCtrl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpFOUtils",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpMDCtrl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpMalController",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpModemMessage",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpPhbController",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpRadioCtrl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpRadioMessage",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpRilClientCtrl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpSimController",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RpSsController",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RtcCapa",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RtcDC",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RtcIms",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RtcNwCtrl",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RtcPhb",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RtcRadioCont",
		.value = "I",
	},
	{
		.key = "persist.log.tag.RtcRatSwCtrl",
		.value = "D",
	},
	{
		.key = "persist.log.tag.RtcWp",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SIMRecords",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SSDecisonMaker",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SimSwitchOP01",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SimSwitchOP02",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SimSwitchOP18",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SimservType",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SimservsTest",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SlotQueueEntry",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SmsPlusCode",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SpnOverride",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SresResponse",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SuppMsgMgr",
		.value = "I",
	},
	{
		.key = "persist.log.tag.SuppSrvConfig",
		.value = "I",
	},
	{
		.key = "persist.log.tag.TeleConfCtrler",
		.value = "I",
	},
	{
		.key = "persist.log.tag.TeleConnService",
		.value = "I",
	},
	{
		.key = "persist.log.tag.TelephonyConf",
		.value = "I",
	},
	{
		.key = "persist.log.tag.TelephonyConn",
		.value = "I",
	},
	{
		.key = "persist.log.tag.UiccCard",
		.value = "I",
	},
	{
		.key = "persist.log.tag.UiccController",
		.value = "I",
	},
	{
		.key = "persist.log.tag.VT",
		.value = "I",
	},
	{
		.key = "persist.log.tag.VsimAdaptor",
		.value = "I",
	},
	{
		.key = "persist.log.tag.WORLDMODE",
		.value = "I",
	},
	{
		.key = "persist.log.tag.WfoApp",
		.value = "I",
	},
	{
		.key = "persist.log.tag.tel_log_ctrl",
		.value = "1",
	},
	{
		.key = "persist.logd.size",
		.value = "16777216",
	},
	{
		.key = "persist.meta.dumpdata",
		.value = "0",
	},
	{
		.key = "persist.mtk.aee.mode",
		.value = "4",
	},
	{
		.key = "persist.mtk.connsys.poweron.ctl",
		.value = "0",
	},
	{
		.key = "persist.mtk.datashaping.support",
		.value = "1",
	},
	{
		.key = "persist.mtk.volte.enable",
		.value = "1",
	},
	{
		.key = "persist.mtk.wcn.combo.chipid",
		.value = "0x6771",
	},
	{
		.key = "persist.mtk.wcn.coredump.mode",
		.value = "2",
	},
	{
		.key = "persist.mtk.wcn.dynamic.dump",
		.value = "0",
	},
	{
		.key = "persist.mtk.wcn.patch.version",
		.value = "20180502111328a",
	},
	{
		.key = "persist.mtk_ct_volte_support",
		.value = "1",
	},
	{
		.key = "persist.mtk_dynamic_ims_switch",
		.value = "1",
	},
	{
		.key = "persist.mtk_epdg_support",
		.value = "1",
	},
	{
		.key = "persist.mtk_ims_support",
		.value = "1",
	},
	{
		.key = "persist.mtk_mims_support",
		.value = "2",
	},
	{
		.key = "persist.mtk_ussi_support",
		.value = "1",
	},
	{
		.key = "persist.mtk_vilte_support",
		.value = "1",
	},
	{
		.key = "persist.mtk_viwifi_support",
		.value = "1",
	},
	{
		.key = "persist.mtk_volte_support",
		.value = "1",
	},
	{
		.key = "persist.mtk_wfc_support",
		.value = "1",
	},
	{
		.key = "persist.power.useautobrightadj",
		.value = "true",
	},
	{
		.key = "persist.radio.airplane.mode.on",
		.value = "false",
	},
	{
		.key = "persist.radio.c_capability_slot",
		.value = "1",
	},
	{
		.key = "persist.radio.data.iccid",
		.value = "",
	},
	{
		.key = "persist.radio.default.sim",
		.value = "0",
	},
	{
		.key = "persist.radio.erlvt.on",
		.value = "1",
	},
	{
		.key = "persist.radio.fd.counter",
		.value = "150",
	},
	{
		.key = "persist.radio.fd.off.counter",
		.value = "50",
	},
	{
		.key = "persist.radio.fd.off.r8.counter",
		.value = "50",
	},
	{
		.key = "persist.radio.fd.r8.counter",
		.value = "150",
	},
	{
		.key = "persist.radio.flashless.fsm",
		.value = "0",
	},
	{
		.key = "persist.radio.flashless.fsm_cst",
		.value = "0",
	},
	{
		.key = "persist.radio.flashless.fsm_rw",
		.value = "0",
	},
	{
		.key = "persist.radio.lastsim1_iccid",
		.value = "null",
	},
	{
		.key = "persist.radio.lastsim2_iccid",
		.value = "null",
	},
	{
		.key = "persist.radio.lte.chip",
		.value = "0",
	},
	{
		.key = "persist.radio.mobile.data",
		.value = "0,0",
	},
	{
		.key = "persist.radio.mtk_dsbp_support",
		.value = "1",
	},
	{
		.key = "persist.radio.mtk_ps2_rat",
		.value = "L/W/G",
	},
	{
		.key = "persist.radio.mtk_ps3_rat",
		.value = "G",
	},
	{
		.key = "persist.radio.multisim.config",
		.value = "dsds",
	},
	{
		.key = "persist.radio.nitz_oper_code",
		.value = ",",
	},
	{
		.key = "persist.radio.nitz_oper_lname",
		.value = ",",
	},
	{
		.key = "persist.radio.nitz_oper_sname",
		.value = ",",
	},
	{
		.key = "persist.radio.raf1",
		.value = "225786",
	},
	{
		.key = "persist.radio.raf2",
		.value = "81928",
	},
	{
		.key = "persist.radio.reset_on_switch",
		.value = "false",
	},
	{
		.key = "persist.radio.rilj_nw_type1",
		.value = "-1",
	},
	{
		.key = "persist.radio.rilj_nw_type2",
		.value = "-1",
	},
	{
		.key = "persist.radio.sim.mode",
		.value = "3",
	},
	{
		.key = "persist.radio.sim.opid",
		.value = "0",
	},
	{
		.key = "persist.radio.sim.opid_1",
		.value = "0",
	},
	{
		.key = "persist.radio.simswitch",
		.value = "1",
	},
	{
		.key = "persist.radio.smart.data.switch",
		.value = "1",
	},
	{
		.key = "persist.radio.volte_state",
		.value = "1",
	},
	{
		.key = "persist.service.acm.enable",
		.value = "0",
	},
	{
		.key = "persist.service.stk.shutdown",
		.value = "0",
	},
	{
		.key = "persist.sys.allcommode",
		.value = "true",
	},
	{
		.key = "persist.sys.assert.enable",
		.value = "false",
	},
	{
		.key = "persist.sys.assert.enabletip",
		.value = "0",
	},
	{
		.key = "persist.sys.assert.panic",
		.value = "false",
	},
	{
		.key = "persist.sys.assert.state",
		.value = "false",
	},
	{
		.key = "persist.sys.bluelight.default",
		.value = "128",
	},
	{
		.key = "persist.sys.cfu_auto",
		.value = "1",
	},
	{
		.key = "persist.sys.close_engneer_ui",
		.value = "1",
	},
	{
		.key = "persist.sys.customize.forbcap",
		.value = "false",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.debug.color_temp",
		.value = "0",
	},
	{
		.key = "persist.sys.device_first_boot",
		.value = "0",
	},
	{
		.key = "persist.sys.disable_rescue",
		.value = "true",
	},
	{
		.key = "persist.sys.enable.hypnus",
		.value = "1",
	},
	{
		.key = "persist.sys.errmsg",
		.value = "-1",
	},
	{
		.key = "persist.sys.feedback.rooted",
		.value = "false",
	},
	{
		.key = "persist.sys.hardcoder.name",
		.value = "oiface",
	},
	{
		.key = "persist.sys.hdcp_checking",
		.value = "false",
	},
	{
		.key = "persist.sys.hw_status",
		.value = "1",
	},
	{
		.key = "persist.sys.lasttime",
		.value = "1526733385000",
	},
	{
		.key = "persist.sys.locale",
		.value = "en-US",
	},
	{
		.key = "persist.sys.media.use-awesome",
		.value = "false",
	},
	{
		.key = "persist.sys.modem.path",
		.value = "",
	},
	{
		.key = "persist.sys.mute.state",
		.value = "2",
	},
	{
		.key = "persist.sys.nlp.enabled",
		.value = "0",
	},
	{
		.key = "persist.sys.nw_lab_test",
		.value = "0",
	},
	{
		.key = "persist.sys.nw_mbn_icon",
		.value = "0",
	},
	{
		.key = "persist.sys.oem_smooth",
		.value = "1",
	},
	{
		.key = "persist.sys.oiface.enable",
		.value = "2",
	},
	{
		.key = "persist.sys.oiface.feature",
		.value = "oiface:1f,oifaceim:ffffffff",
	},
	{
		.key = "persist.sys.oppo.displaymetrics",
		.value = "1080,2280",
	},
	{
		.key = "persist.sys.oppo.dragstate",
		.value = "0",
	},
	{
		.key = "persist.sys.oppo.fatal",
		.value = "",
	},
	{
		.key = "persist.sys.oppo.fb_upgraded",
		.value = "1",
	},
	{
		.key = "persist.sys.oppo.fp_psensor",
		.value = "true",
	},
	{
		.key = "persist.sys.oppo.fp_tpprotecet",
		.value = "true",
	},
	{
		.key = "persist.sys.oppo.junklog",
		.value = "false",
	},
	{
		.key = "persist.sys.oppo.junkmonitor",
		.value = "true",
	},
	{
		.key = "persist.sys.oppo.log.config",
		.value = "0",
	},
	{
		.key = "persist.sys.oppo.longpwk",
		.value = "",
	},
	{
		.key = "persist.sys.oppo.reboot",
		.value = "",
	},
	{
		.key = "persist.sys.oppo.region",
		.value = "CN",
	},
	{
		.key = "persist.sys.oppo.screendrag",
		.value = "0,0,0,0",
	},
	{
		.key = "persist.sys.oppo.silence",
		.value = "0",
	},
	{
		.key = "persist.sys.oppodebug.tpcatcher",
		.value = "14",
	},
	{
		.key = "persist.sys.opponetwake.enable",
		.value = "true",
	},
	{
		.key = "persist.sys.oppopcm.enable",
		.value = "true",
	},
	{
		.key = "persist.sys.oppopm.enable",
		.value = "false",
	},
	{
		.key = "persist.sys.ota.downloaded",
		.value = "false",
	},
	{
		.key = "persist.sys.ota.last_screenoff",
		.value = "1531538953665",
	},
	{
		.key = "persist.sys.panictime",
		.value = "0",
	},
	{
		.key = "persist.sys.permission.enable",
		.value = "true",
	},
	{
		.key = "persist.sys.pms_sys_removable",
		.value = "1",
	},
	{
		.key = "persist.sys.poweroffsound",
		.value = "1",
	},
	{
		.key = "persist.sys.poweronsound",
		.value = "1",
	},
	{
		.key = "persist.sys.pq.adl.idx",
		.value = "0",
	},
	{
		.key = "persist.sys.pq.hdr.en",
		.value = "1",
	},
	{
		.key = "persist.sys.pq.iso.shp.en",
		.value = "2",
	},
	{
		.key = "persist.sys.pq.log.en",
		.value = "0",
	},
	{
		.key = "persist.sys.pq.mdp.ccorr.en",
		.value = "2",
	},
	{
		.key = "persist.sys.pq.mdp.color.dbg",
		.value = "1",
	},
	{
		.key = "persist.sys.pq.mdp.color.idx",
		.value = "0",
	},
	{
		.key = "persist.sys.pq.mdp.dre.en",
		.value = "2",
	},
	{
		.key = "persist.sys.pq.shp.idx",
		.value = "2",
	},
	{
		.key = "persist.sys.pq.ultrares.en",
		.value = "2",
	},
	{
		.key = "persist.sys.procmon_enable",
		.value = "1",
	},
	{
		.key = "persist.sys.sau.launchcheck",
		.value = "2",
	},
	{
		.key = "persist.sys.strictmode.visual",
		.value = "",
	},
	{
		.key = "persist.sys.themeflag",
		.value = "3",
	},
	{
		.key = "persist.sys.thermal.enable",
		.value = "false",
	},
	{
		.key = "persist.sys.timezone",
		.value = "America/Los_Angeles",
	},
	{
		.key = "persist.sys.ui.hw",
		.value = "false",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "adb",
	},
	{
		.key = "persist.sys.webview.vmsize",
		.value = "127064880",
	},
	{
		.key = "persist.sys.wipemedia",
		.value = "0",
	},
	{
		.key = "persist.version.confidential",
		.value = "false",
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
		.key = "pm.dexopt.core-app",
		.value = "speed",
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
		.key = "qemu.hw.mainkeys",
		.value = "0",
	},
	{
		.key = "ril.active.md",
		.value = "12",
	},
	{
		.key = "ril.apc.support",
		.value = "1",
	},
	{
		.key = "ril.cdma.card.omh",
		.value = "-1",
	},
	{
		.key = "ril.cdma.card.omh.1",
		.value = "-1",
	},
	{
		.key = "ril.cdma.card.type.1",
		.value = "255",
	},
	{
		.key = "ril.cdma.card.type.2",
		.value = "255",
	},
	{
		.key = "ril.cdma.ecclist",
		.value = "",
	},
	{
		.key = "ril.cdma.ecclist1",
		.value = "",
	},
	{
		.key = "ril.ecc.service.category.list",
		.value = "",
	},
	{
		.key = "ril.ecc.service.category.list.1",
		.value = "",
	},
	{
		.key = "ril.ecclist",
		.value = "",
	},
	{
		.key = "ril.ecclist1",
		.value = "",
	},
	{
		.key = "ril.epdg.interface.ctrl",
		.value = "1",
	},
	{
		.key = "ril.external.md",
		.value = "0",
	},
	{
		.key = "ril.fd.mode",
		.value = "1",
	},
	{
		.key = "ril.first.md",
		.value = "1",
	},
	{
		.key = "ril.flightmode.poweroffMD",
		.value = "0",
	},
	{
		.key = "ril.imsi.status.sim1",
		.value = "0",
	},
	{
		.key = "ril.imsi.status.sim2",
		.value = "0",
	},
	{
		.key = "ril.ipo.radiooff",
		.value = "0",
	},
	{
		.key = "ril.md_changed_apn_class.iccid0",
		.value = "",
	},
	{
		.key = "ril.md_changed_apn_class.iccid1",
		.value = "",
	},
	{
		.key = "ril.md_changed_apn_class0",
		.value = "",
	},
	{
		.key = "ril.md_changed_apn_class1",
		.value = "",
	},
	{
		.key = "ril.mtk",
		.value = "1",
	},
	{
		.key = "ril.mux.ee.md1",
		.value = "0",
	},
	{
		.key = "ril.mux.report.case",
		.value = "0",
	},
	{
		.key = "ril.muxreport.run",
		.value = "0",
	},
	{
		.key = "ril.nw.signalstrength.lte.1",
		.value = "-75,26",
	},
	{
		.key = "ril.nw.signalstrength.lte.2",
		.value = "2147483647,214748364",
	},
	{
		.key = "ril.nw.worldmode.activemode",
		.value = "1",
	},
	{
		.key = "ril.nw.worldmode.keep_3g_mode",
		.value = "0",
	},
	{
		.key = "ril.radiooff.poweroffMD",
		.value = "0",
	},
	{
		.key = "ril.read.imsi",
		.value = "1",
	},
	{
		.key = "ril.simswitch.no_reset_support",
		.value = "1",
	},
	{
		.key = "ril.simswitch.tpluswsupport",
		.value = "1",
	},
	{
		.key = "ril.specific.sm_cause",
		.value = "0",
	},
	{
		.key = "ril.telephony.mode",
		.value = "0",
	},
	{
		.key = "rild.libargs",
		.value = "-d /dev/ttyC0",
	},
	{
		.key = "rild.libpath",
		.value = "mtk-ril.so",
	},
	{
		.key = "rild.mark_switchuser",
		.value = "0",
	},
	{
		.key = "ro.adb.secure",
		.value = "1",
	},
	{
		.key = "ro.aee.enforcing",
		.value = "no",
	},
	{
		.key = "ro.aee.enperf",
		.value = "off",
	},
	{
		.key = "ro.allow.mock.location",
		.value = "0",
	},
	{
		.key = "ro.ap_info_monitor",
		.value = "0",
	},
	{
		.key = "ro.audio.silent",
		.value = "0",
	},
	{
		.key = "ro.audio.usb.period_us",
		.value = "16000",
	},
	{
		.key = "ro.baseband",
		.value = "unknown",
	},
	{
		.key = "ro.board.platform",
		.value = "mt6771",
	},
	{
		.key = "ro.boot.atm",
		.value = "disabled",
	},
	{
		.key = "ro.boot.bootreason",
		.value = "power_key",
	},
	{
		.key = "ro.boot.flash.locked",
		.value = "1",
	},
	{
		.key = "ro.boot.gauge",
		.value = "no",
	},
	{
		.key = "ro.boot.hardware",
		.value = "mt6771",
	},
	{
		.key = "ro.boot.meta_log_disable",
		.value = "0",
	},
	{
		.key = "ro.boot.mode",
		.value = "normal",
	},
	{
		.key = "ro.boot.opt_c2k_lte_mode",
		.value = "2",
	},
	{
		.key = "ro.boot.opt_c2k_support",
		.value = "1",
	},
	{
		.key = "ro.boot.opt_eccci_c2k",
		.value = "1",
	},
	{
		.key = "ro.boot.opt_lte_support",
		.value = "1",
	},
	{
		.key = "ro.boot.opt_md1_support",
		.value = "12",
	},
	{
		.key = "ro.boot.opt_md3_support",
		.value = "0",
	},
	{
		.key = "ro.boot.opt_ps1_rat",
		.value = "C/Lf/Lt/W/T/G",
	},
	{
		.key = "ro.boot.opt_using_default",
		.value = "0",
	},
	{
		.key = "ro.boot.serialno",
		.value = "9P4SUSOBEI7HIJJR",
	},
	{
		.key = "ro.boot.verifiedbootstate",
		.value = "green",
	},
	{
		.key = "ro.boot.veritymode",
		.value = "enforcing",
	},
	{
		.key = "ro.bootimage.build.date",
		.value = "Wed May 9 23:01:52 CST 2018",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1525878112",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "OPPO/PACM00/PACM00:8.1.0/O11019/1523979512:user/release-keys",
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
		.value = "Wed May  9 23:05:20 CST 2018",
	},
	{
		.key = "ro.build.date.Ymd",
		.value = "180509",
	},
	{
		.key = "ro.build.date.YmdHM",
		.value = "201805092153",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1525878320",
	},
	{
		.key = "ro.build.date.ymd",
		.value = "180509",
	},
	{
		.key = "ro.build.description",
		.value = "full_oppo6771_17197-user 8.1.0 O11019 1525878118 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "PACM00_11_A.15_180509",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "OPPO/PACM00/PACM00:8.1.0/O11019/1523979512:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "full_oppo6771_17197-user",
	},
	{
		.key = "ro.build.host",
		.value = "ubuntu-121-152",
	},
	{
		.key = "ro.build.id",
		.value = "O11019",
	},
	{
		.key = "ro.build.kernel.id",
		.value = "4.4.95-G201805092153",
	},
	{
		.key = "ro.build.master.date",
		.value = "201805092153",
	},
	{
		.key = "ro.build.product",
		.value = "PACM00",
	},
	{
		.key = "ro.build.release_type",
		.value = "true",
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
		.key = "ro.build.user",
		.value = "root",
	},
	{
		.key = "ro.build.version.all_codenames",
		.value = "REL",
	},
	{
		.key = "ro.build.version.base_os",
		.value = "OPPO/PACM00/PACM00:8.1.0/O11019/1522676105:user/release-keys",
	},
	{
		.key = "ro.build.version.codename",
		.value = "REL",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "1525878320",
	},
	{
		.key = "ro.build.version.opporom",
		.value = "V5.0",
	},
	{
		.key = "ro.build.version.ota",
		.value = "PACM00_11.A.15_0150_201805092153",
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
		.value = "2018-04-05",
	},
	{
		.key = "ro.camera.hfr.enable",
		.value = "1",
	},
	{
		.key = "ro.camera.sound.forced",
		.value = "0",
	},
	{
		.key = "ro.camera.temperature.limit",
		.value = "460",
	},
	{
		.key = "ro.camera.videoeis.enable",
		.value = "1",
	},
	{
		.key = "ro.carrier",
		.value = "unknown",
	},
	{
		.key = "ro.cdma.cfall.disable",
		.value = "*730",
	},
	{
		.key = "ro.cdma.cfb.disable",
		.value = "*900",
	},
	{
		.key = "ro.cdma.cfb.enable",
		.value = "*90",
	},
	{
		.key = "ro.cdma.cfdf.disable",
		.value = "*680",
	},
	{
		.key = "ro.cdma.cfdf.enable",
		.value = "*68",
	},
	{
		.key = "ro.cdma.cfnr.disable",
		.value = "*920",
	},
	{
		.key = "ro.cdma.cfnr.enable",
		.value = "*92",
	},
	{
		.key = "ro.cdma.cfu.disable",
		.value = "*720",
	},
	{
		.key = "ro.cdma.cfu.enable",
		.value = "*72",
	},
	{
		.key = "ro.cdma.cw.disable",
		.value = "*740",
	},
	{
		.key = "ro.cdma.cw.enable",
		.value = "*74",
	},
	{
		.key = "ro.com.android.mobiledata",
		.value = "true",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-oppo",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "ringtone_008.ogg",
	},
	{
		.key = "ro.config.calendar_sound",
		.value = "notification_003.ogg",
	},
	{
		.key = "ro.config.notification_sim2",
		.value = "notification_001.ogg",
	},
	{
		.key = "ro.config.notification_sms",
		.value = "notification_001.ogg",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "notification_008.ogg",
	},
	{
		.key = "ro.config.ringtone",
		.value = "ringtone_001.ogg",
	},
	{
		.key = "ro.config.ringtone_sim2",
		.value = "ringtone_001.ogg",
	},
	{
		.key = "ro.control_privapp_permissions",
		.value = "log",
	},
	{
		.key = "ro.crypto.fs_crypto_blkdev",
		.value = "/dev/block/dm-0",
	},
	{
		.key = "ro.crypto.state",
		.value = "encrypted",
	},
	{
		.key = "ro.crypto.type",
		.value = "block",
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
		.key = "ro.email_support_ucs2",
		.value = "0",
	},
	{
		.key = "ro.expect.recovery_id",
		.value = "",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/platform/bootdevice/by-name/frp",
	},
	{
		.key = "ro.gauge",
		.value = "no",
	},
	{
		.key = "ro.hardware",
		.value = "mt6771",
	},
	{
		.key = "ro.hardware.kmsetkey",
		.value = "trustonic",
	},
	{
		.key = "ro.have_aacencode_feature",
		.value = "1",
	},
	{
		.key = "ro.have_aee_feature",
		.value = "1",
	},
	{
		.key = "ro.hw.phone.color",
		.value = "FFC5C5C5",
	},
	{
		.key = "ro.kernel.zio",
		.value = "38,108,105,16",
	},
	{
		.key = "ro.ksc5601_write",
		.value = "0",
	},
	{
		.key = "ro.lcd.backlight.config_boe",
		.value = "11,816,5,246,397,665,912,1177,1473,1627,1890,2047",
	},
	{
		.key = "ro.lcd.backlight.config_dsjm",
		.value = "11,816,5,246,397,665,912,1177,1473,1627,1890,2047",
	},
	{
		.key = "ro.lcd.backlight.config_jdi",
		.value = "11,851,7,280,435,699,943,1191,1465,1605,1847,2047",
	},
	{
		.key = "ro.lcd.backlight.config_tianma",
		.value = "11,816,5,246,397,665,912,1177,1473,1627,1890,2047",
	},
	{
		.key = "ro.lcd.backlight.config_truly",
		.value = "11,816,5,246,397,665,912,1177,1473,1627,1890,2047",
	},
	{
		.key = "ro.lcd.backlight.samsung_tenbit",
		.value = "10,176,9,45,70,130,214,345,546,686,1023",
	},
	{
		.key = "ro.md_apps.support",
		.value = "1",
	},
	{
		.key = "ro.md_auto_setup_ims",
		.value = "1",
	},
	{
		.key = "ro.mediatek.chip_ver",
		.value = "S01",
	},
	{
		.key = "ro.mediatek.platform",
		.value = "MT6771",
	},
	{
		.key = "ro.mediatek.version.branch",
		.value = "alps-mp-o1.mp1.tc16sp",
	},
	{
		.key = "ro.mediatek.version.release",
		.value = "PACM00_11_A.15_180509",
	},
	{
		.key = "ro.mediatek.version.sdk",
		.value = "4",
	},
	{
		.key = "ro.mediatek.wlan.p2p",
		.value = "1",
	},
	{
		.key = "ro.mediatek.wlan.wsc",
		.value = "1",
	},
	{
		.key = "ro.mount.fs",
		.value = "EXT4",
	},
	{
		.key = "ro.mtk_aal_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_afw_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_agps_app",
		.value = "1",
	},
	{
		.key = "ro.mtk_aod_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_audio_alac_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_audio_ape_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_audio_tuning_tool_ver",
		.value = "V2.2",
	},
	{
		.key = "ro.mtk_bg_power_saving_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_bg_power_saving_ui",
		.value = "1",
	},
	{
		.key = "ro.mtk_bip_scws",
		.value = "1",
	},
	{
		.key = "ro.mtk_blulight_def_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_bsp_package",
		.value = "1",
	},
	{
		.key = "ro.mtk_bt_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_c2k_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_cam_lomo_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_cam_mfb_support",
		.value = "3",
	},
	{
		.key = "ro.mtk_cam_stereo_camera_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_camera_app_version",
		.value = "3",
	},
	{
		.key = "ro.mtk_config_max_dram_size",
		.value = "0x800000000",
	},
	{
		.key = "ro.mtk_cta_drm_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_cta_set",
		.value = "1",
	},
	{
		.key = "ro.mtk_data_config",
		.value = "1",
	},
	{
		.key = "ro.mtk_deinterlace_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_dhcpv6c_wifi",
		.value = "1",
	},
	{
		.key = "ro.mtk_dual_mic_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_eap_sim_aka",
		.value = "1",
	},
	{
		.key = "ro.mtk_emmc_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_exchange_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_external_sim_only_slots",
		.value = "0",
	},
	{
		.key = "ro.mtk_external_sim_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_f2fs_enable",
		.value = "0",
	},
	{
		.key = "ro.mtk_fd_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_flv_playback_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_fm_50khz_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_gps_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_is_tablet",
		.value = "0",
	},
	{
		.key = "ro.mtk_log_hide_gps",
		.value = "0",
	},
	{
		.key = "ro.mtk_lte_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_md_world_mode_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_modem_monitor_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_oma_drm_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_omacp_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_perf_fast_start_win",
		.value = "1",
	},
	{
		.key = "ro.mtk_perf_response_time",
		.value = "1",
	},
	{
		.key = "ro.mtk_perf_simple_start_win",
		.value = "1",
	},
	{
		.key = "ro.mtk_pow_perf_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_pq_color_mode",
		.value = "1",
	},
	{
		.key = "ro.mtk_pq_support",
		.value = "2",
	},
	{
		.key = "ro.mtk_protocol1_rat_config",
		.value = "C/Lf/Lt/W/T/G",
	},
	{
		.key = "ro.mtk_ril_mode",
		.value = "c6m_1rild",
	},
	{
		.key = "ro.mtk_rild_read_imsi",
		.value = "1",
	},
	{
		.key = "ro.mtk_search_db_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_send_rr_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_shared_sdcard",
		.value = "1",
	},
	{
		.key = "ro.mtk_sim_hot_swap",
		.value = "1",
	},
	{
		.key = "ro.mtk_sim_hot_swap_common_slot",
		.value = "1",
	},
	{
		.key = "ro.mtk_single_bin_modem_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_slow_motion_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_soter_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_tee_gp_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_tetheringipv6_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_trustonic_tee_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_wapi_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_wappush_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_wfd_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_widevine_drm_l3_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_wlan_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_wmv_playback_support",
		.value = "1",
	},
	{
		.key = "ro.mtk_world_phone_policy",
		.value = "0",
	},
	{
		.key = "ro.mtk_zsdhdr_support",
		.value = "1",
	},
	{
		.key = "ro.mtkrc.path",
		.value = "/vendor/etc/init/hw/",
	},
	{
		.key = "ro.num_md_protocol",
		.value = "2",
	},
	{
		.key = "ro.oem_unlock_supported",
		.value = "1",
	},
	{
		.key = "ro.opengles.version",
		.value = "196610",
	},
	{
		.key = "ro.oppo.market.name",
		.value = "OPPO R15",
	},
	{
		.key = "ro.oppo.screen.heteromorphism",
		.value = "378,0:702,80",
	},
	{
		.key = "ro.oppo.theme.version",
		.value = "805",
	},
	{
		.key = "ro.oppo.version",
		.value = "",
	},
	{
		.key = "ro.product.authentication",
		.value = "2018CP0915",
	},
	{
		.key = "ro.product.board",
		.value = "oppo6771_17197",
	},
	{
		.key = "ro.product.brand",
		.value = "OPPO",
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
		.value = "PACM00",
	},
	{
		.key = "ro.product.first_api_level",
		.value = "27",
	},
	{
		.key = "ro.product.hw",
		.value = "DB091",
	},
	{
		.key = "ro.product.locale",
		.value = "zh-CN",
	},
	{
		.key = "ro.product.manufacturer",
		.value = "OPPO",
	},
	{
		.key = "ro.product.model",
		.value = "PACM00",
	},
	{
		.key = "ro.product.name",
		.value = "PACM00",
	},
	{
		.key = "ro.product.sar",
		.value = "1.7",
	},
	{
		.key = "ro.property_service.version",
		.value = "2",
	},
	{
		.key = "ro.radio.simcount",
		.value = "2",
	},
	{
		.key = "ro.recovery_id",
		.value = "0x3d563eab71489464541dbb8210b4c7933e61bc7a000000000000000000000000",
	},
	{
		.key = "ro.reserve1.get",
		.value = "/dev/block/platform/bootdevice/by-name/reserve1",
	},
	{
		.key = "ro.revision",
		.value = "0",
	},
	{
		.key = "ro.rom.featrue",
		.value = "allnet",
	},
	{
		.key = "ro.script.version",
		.value = "1.0",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.separate.soft",
		.value = "17197",
	},
	{
		.key = "ro.serialno",
		.value = "9P4SUSOBEI7HIJJR",
	},
	{
		.key = "ro.sf.hwrotation",
		.value = "0",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "480",
	},
	{
		.key = "ro.sim_me_lock_mode",
		.value = "0",
	},
	{
		.key = "ro.sim_refresh_reset_by_modem",
		.value = "1",
	},
	{
		.key = "ro.sys.sdcardfs",
		.value = "1",
	},
	{
		.key = "ro.sys.usb.bicr",
		.value = "no",
	},
	{
		.key = "ro.sys.usb.charging.only",
		.value = "yes",
	},
	{
		.key = "ro.sys.usb.mtp.whql.enable",
		.value = "0",
	},
	{
		.key = "ro.sys.usb.storage.type",
		.value = "mtp",
	},
	{
		.key = "ro.telephony.default_network",
		.value = "10,10",
	},
	{
		.key = "ro.telephony.sim.count",
		.value = "2",
	},
	{
		.key = "ro.treble.enabled",
		.value = "true",
	},
	{
		.key = "ro.ussd_ksc5601",
		.value = "0",
	},
	{
		.key = "ro.vendor.build.date",
		.value = "Wed May 9 23:01:52 CST 2018",
	},
	{
		.key = "ro.vendor.build.date.utc",
		.value = "1525878112",
	},
	{
		.key = "ro.vendor.build.fingerprint",
		.value = "OPPO/PACM00/PACM00:8.1.0/O11019/1523979512:user/release-keys",
	},
	{
		.key = "ro.vendor.product.brand",
		.value = "OPPO",
	},
	{
		.key = "ro.vendor.product.device",
		.value = "PACM00",
	},
	{
		.key = "ro.vendor.product.manufacturer",
		.value = "OPPO",
	},
	{
		.key = "ro.vendor.product.model",
		.value = "PACM00",
	},
	{
		.key = "ro.vendor.product.name",
		.value = "PACM00",
	},
	{
		.key = "ro.vendor.product.oem",
		.value = "PACM00",
	},
	{
		.key = "ro.vold.serialno",
		.value = "9P4SUSOBEI7HIJJR",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
	},
	{
		.key = "ro.wlan.gen",
		.value = "gen3",
	},
	{
		.key = "ro.wlan.mtk.wifi.5g",
		.value = "1",
	},
	{
		.key = "ro.xxversion",
		.value = "V0.5",
	},
	{
		.key = "ro.zygote",
		.value = "zygote64_32",
	},
	{
		.key = "ro.zygote.preload.enable",
		.value = "0",
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
		.key = "service.nvram_init",
		.value = "Ready",
	},
	{
		.key = "service.sf.present_timestamp",
		.value = "1",
	},
	{
		.key = "service.wcn.driver.ready",
		.value = "yes",
	},
	{
		.key = "service.wcn.formeta.ready",
		.value = "yes",
	},
	{
		.key = "sys.app_freeze_timeout",
		.value = "0",
	},
	{
		.key = "sys.boot.reason",
		.value = "0",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
	},
	{
		.key = "sys.build.display.id",
		.value = "PACM00_11_A.15_180509_a7d06fc5",
	},
	{
		.key = "sys.fb_parent_pid",
		.value = "595",
	},
	{
		.key = "sys.fw_boot",
		.value = "progress_done",
	},
	{
		.key = "sys.ipo.disable",
		.value = "1",
	},
	{
		.key = "sys.ipo.pwrdncap",
		.value = "2",
	},
	{
		.key = "sys.ipowin.done",
		.value = "1",
	},
	{
		.key = "sys.logbootcomplete",
		.value = "1",
	},
	{
		.key = "sys.loglimit.enabled",
		.value = "true",
	},
	{
		.key = "sys.mediatek.version.release",
		.value = "PACM00_11_A.15_180509_a7d06fc5",
	},
	{
		.key = "sys.oem_unlock_allowed",
		.value = "0",
	},
	{
		.key = "sys.oppo.boot_completed",
		.value = "1",
	},
	{
		.key = "sys.oppo.call_mode",
		.value = "false",
	},
	{
		.key = "sys.oppo.gesturewakeup",
		.value = "0",
	},
	{
		.key = "sys.oppo.gift",
		.value = "1",
	},
	{
		.key = "sys.oppo.multibrightness",
		.value = "1023",
	},
	{
		.key = "sys.oppo.nw.hongbao",
		.value = "1",
	},
	{
		.key = "sys.oppo.reboot",
		.value = "0",
	},
	{
		.key = "sys.oppo.recheck_finish",
		.value = "true",
	},
	{
		.key = "sys.oppo.screenshot",
		.value = "0",
	},
	{
		.key = "sys.power.screenoff.reason",
		.value = "2",
	},
	{
		.key = "sys.power_off_alarm",
		.value = "0",
	},
	{
		.key = "sys.retaildemo.enabled",
		.value = "0",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "43284",
	},
	{
		.key = "sys.usb.acm_cnt",
		.value = "0",
	},
	{
		.key = "sys.usb.acm_enable",
		.value = "0",
	},
	{
		.key = "sys.usb.acm_port0",
		.value = "",
	},
	{
		.key = "sys.usb.acm_port1",
		.value = "",
	},
	{
		.key = "sys.usb.clear",
		.value = "boot",
	},
	{
		.key = "sys.usb.config",
		.value = "adb",
	},
	{
		.key = "sys.usb.configfs",
		.value = "1",
	},
	{
		.key = "sys.usb.controller",
		.value = "musb-hdrc",
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
		.key = "sys.usb.pid",
		.value = "0x2769",
	},
	{
		.key = "sys.usb.state",
		.value = "adb",
	},
	{
		.key = "sys.usb.temp",
		.value = "",
	},
	{
		.key = "sys.usb.vid",
		.value = "0x22d9",
	},
	{
		.key = "sys.user.0.ce_available",
		.value = "true",
	},
	{
		.key = "sys.wifitracing.started",
		.value = "1",
	},
	{
		.key = "telephony.lteOnCdmaDevice",
		.value = "1",
	},
	{
		.key = "vold.decrypt",
		.value = "trigger_restart_framework",
	},
	{
		.key = "vold.encryption.type",
		.value = "default",
	},
	{
		.key = "vold.has_adoptable",
		.value = "0",
	},
	{
		.key = "vold.has_quota",
		.value = "1",
	},
	{
		.key = "vold.post_fs_data_done",
		.value = "1",
	},
	{
		.key = "wfd.dummy.enable",
		.value = "1",
	},
	{
		.key = "wfd.iframesize.level",
		.value = "0",
	},
	{
		.key = "wifi.direct.interface",
		.value = "p2p0",
	},
	{
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "wifi.tethering.interface",
		.value = "ap0",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
