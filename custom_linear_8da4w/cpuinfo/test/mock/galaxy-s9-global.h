struct cpuinfo_mock_file filesystem[] = {
#if CPUINFO_ARCH_ARM64
	{
		.path = "/proc/cpuinfo",
		.size = 1616,
		.content =
			"processor\t: 0\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 4\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x53\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0x002\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 5\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x53\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0x002\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 6\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x53\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0x002\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 7\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x53\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0x002\n"
			"CPU revision\t: 0\n"
			"\n",
	},
#elif CPUINFO_ARCH_ARM
	{
		.path = "/proc/cpuinfo",
		.size = 2240,
		.content =
			"processor\t: 0\n"
			"model name\t: ARMv8 Processor rev 1 (v8l)\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 1\n"
			"model name\t: ARMv8 Processor rev 1 (v8l)\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 2\n"
			"model name\t: ARMv8 Processor rev 1 (v8l)\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 3\n"
			"model name\t: ARMv8 Processor rev 1 (v8l)\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 4\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x53\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0x002\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 5\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x53\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0x002\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 6\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x53\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0x002\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 7\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 52.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x53\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0x002\n"
			"CPU revision\t: 0\n"
			"\n",
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
		.size = 81,
		.content = "cpu:type:aarch64:feature:,0000,0001,0002,0003,0004,0005,0006,0007,0008,0009,000A\n",
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
		.size = 12,
		.content = "exynos_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_governor_ro",
		.size = 5,
		.content = "menu\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpuidle/driver/name",
		.size = 12,
		.content = "exynos_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "455000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 76,
		.content = "1794000 1690000 1456000 1248000 1053000 949000 832000 715000 598000 455000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 11,
		.content = "schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1248000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 15,
		.content = "exynos_cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "455000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 131,
		.content =
			"1794000 28706\n"
			"1690000 2524\n"
			"1456000 6223\n"
			"1248000 3526\n"
			"1053000 41990\n"
			"949000 3768\n"
			"832000 8121\n"
			"715000 12524\n"
			"598000 23493\n"
			"455000 607015\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 6,
		.content = "46410\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000410fd051\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
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
		.size = 12,
		.content = "exynos_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "455000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_frequencies",
		.size = 76,
		.content = "1794000 1690000 1456000 1248000 1053000 949000 832000 715000 598000 455000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 11,
		.content = "schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1248000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
		.size = 15,
		.content = "exynos_cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "455000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 131,
		.content =
			"1794000 28706\n"
			"1690000 2524\n"
			"1456000 6255\n"
			"1248000 3570\n"
			"1053000 41990\n"
			"949000 3768\n"
			"832000 8121\n"
			"715000 12525\n"
			"598000 23510\n"
			"455000 607190\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 6,
		.content = "46636\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000410fd051\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
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
		.size = 12,
		.content = "exynos_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "455000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies",
		.size = 76,
		.content = "1794000 1690000 1456000 1248000 1053000 949000 832000 715000 598000 455000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 11,
		.content = "schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "598000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver",
		.size = 15,
		.content = "exynos_cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "455000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 131,
		.content =
			"1794000 28706\n"
			"1690000 2526\n"
			"1456000 6285\n"
			"1248000 3619\n"
			"1053000 41997\n"
			"949000 3770\n"
			"832000 8122\n"
			"715000 12529\n"
			"598000 23533\n"
			"455000 607355\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 6,
		.content = "46873\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000410fd051\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
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
		.size = 12,
		.content = "exynos_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "455000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 76,
		.content = "1794000 1690000 1456000 1248000 1053000 949000 832000 715000 598000 455000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 11,
		.content = "schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1248000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_driver",
		.size = 15,
		.content = "exynos_cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "455000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 131,
		.content =
			"1794000 28706\n"
			"1690000 2529\n"
			"1456000 6307\n"
			"1248000 3668\n"
			"1053000 41998\n"
			"949000 3770\n"
			"832000 8122\n"
			"715000 12534\n"
			"598000 23559\n"
			"455000 607541\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 6,
		.content = "47107\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000410fd051\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
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
		.size = 12,
		.content = "exynos_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2704000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies",
		.size = 141,
		.content = "2704000 2652000 2496000 2314000 2106000 2002000 1924000 1794000 1690000 1586000 1469000 1261000 1170000 1066000 962000 858000 741000 650000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors",
		.size = 11,
		.content = "schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver",
		.size = 15,
		.content = "exynos_cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 224,
		.content =
			"2704000 8469\n"
			"2652000 44\n"
			"2496000 4623\n"
			"2314000 1315\n"
			"2106000 620\n"
			"2002000 16\n"
			"1924000 13\n"
			"1794000 10070\n"
			"1690000 465\n"
			"1586000 1211\n"
			"1469000 2430\n"
			"1261000 676\n"
			"1170000 677\n"
			"1066000 1472\n"
			"962000 1353\n"
			"858000 3174\n"
			"741000 41680\n"
			"650000 660716\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 6,
		.content = "11731\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000531f0020\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
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
		.size = 12,
		.content = "exynos_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2704000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_frequencies",
		.size = 141,
		.content = "2704000 2652000 2496000 2314000 2106000 2002000 1924000 1794000 1690000 1586000 1469000 1261000 1170000 1066000 962000 858000 741000 650000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_governors",
		.size = 11,
		.content = "schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_driver",
		.size = 15,
		.content = "exynos_cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 224,
		.content =
			"2704000 8469\n"
			"2652000 44\n"
			"2496000 4623\n"
			"2314000 1315\n"
			"2106000 620\n"
			"2002000 16\n"
			"1924000 13\n"
			"1794000 10070\n"
			"1690000 465\n"
			"1586000 1211\n"
			"1469000 2430\n"
			"1261000 676\n"
			"1170000 677\n"
			"1066000 1472\n"
			"962000 1353\n"
			"858000 3174\n"
			"741000 41680\n"
			"650000 660975\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 6,
		.content = "11731\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000531f0020\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
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
		.size = 12,
		.content = "exynos_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2704000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_frequencies",
		.size = 141,
		.content = "2704000 2652000 2496000 2314000 2106000 2002000 1924000 1794000 1690000 1586000 1469000 1261000 1170000 1066000 962000 858000 741000 650000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_governors",
		.size = 11,
		.content = "schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_driver",
		.size = 15,
		.content = "exynos_cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 224,
		.content =
			"2704000 8469\n"
			"2652000 44\n"
			"2496000 4623\n"
			"2314000 1315\n"
			"2106000 620\n"
			"2002000 16\n"
			"1924000 13\n"
			"1794000 10070\n"
			"1690000 465\n"
			"1586000 1211\n"
			"1469000 2430\n"
			"1261000 676\n"
			"1170000 677\n"
			"1066000 1472\n"
			"962000 1353\n"
			"858000 3174\n"
			"741000 41680\n"
			"650000 661245\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 6,
		.content = "11731\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000531f0020\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
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
		.size = 12,
		.content = "exynos_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2704000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_frequencies",
		.size = 141,
		.content = "2704000 2652000 2496000 2314000 2106000 2002000 1924000 1794000 1690000 1586000 1469000 1261000 1170000 1066000 962000 858000 741000 650000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_governors",
		.size = 11,
		.content = "schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_driver",
		.size = 15,
		.content = "exynos_cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1794000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "650000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 224,
		.content =
			"2704000 8469\n"
			"2652000 44\n"
			"2496000 4623\n"
			"2314000 1315\n"
			"2106000 620\n"
			"2002000 16\n"
			"1924000 13\n"
			"1794000 10070\n"
			"1690000 465\n"
			"1586000 1211\n"
			"1469000 2430\n"
			"1261000 676\n"
			"1170000 677\n"
			"1066000 1472\n"
			"962000 1353\n"
			"858000 3174\n"
			"741000 41681\n"
			"650000 661544\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 6,
		.content = "11733\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000531f0020\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
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
		.key = "af.fast_period_size",
		.value = "192",
	},
	{
		.key = "af.fast_track_multiplier",
		.value = "1",
	},
	{
		.key = "audioflinger.bootsnd",
		.value = "0",
	},
	{
		.key = "camera.disable_treble",
		.value = "1",
	},
	{
		.key = "config.disable_consumerir",
		.value = "true",
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
		.value = "256m",
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
		.value = "exynos-m3",
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
		.key = "ddk.set.afbc",
		.value = "1",
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
		.key = "debug.gpuwatch.api",
		.value = "1",
	},
	{
		.key = "debug.hwc.winupdate",
		.value = "1",
	},
	{
		.key = "debug.sf.disable_backpressure",
		.value = "1",
	},
	{
		.key = "debug.sf.layerdump",
		.value = "0",
	},
	{
		.key = "debug.slsi_platform",
		.value = "1",
	},
	{
		.key = "dev.bootcomplete",
		.value = "1",
	},
	{
		.key = "dev.mtp.opensession",
		.value = "1",
	},
	{
		.key = "dev.ssrm.app.install.standby",
		.value = "-1",
	},
	{
		.key = "dev.ssrm.appsync3p",
		.value = "true",
	},
	{
		.key = "dev.ssrm.gamelevel",
		.value = "-10,6,-2,3",
	},
	{
		.key = "dev.ssrm.init",
		.value = "1",
	},
	{
		.key = "dev.ssrm.mode",
		.value = "dm;",
	},
	{
		.key = "dev.ssrm.pst",
		.value = "256",
	},
	{
		.key = "dev.ssrm.smart_switch",
		.value = "true",
	},
	{
		.key = "dev.usbsetting.embedded",
		.value = "on",
	},
	{
		.key = "gsm.current.phone-type",
		.value = "1,1",
	},
	{
		.key = "gsm.network.type",
		.value = "Unknown,Unknown",
	},
	{
		.key = "gsm.operator.alpha",
		.value = ",",
	},
	{
		.key = "gsm.operator.iso-country",
		.value = ",",
	},
	{
		.key = "gsm.operator.ispsroaming",
		.value = "false,false",
	},
	{
		.key = "gsm.operator.isroaming",
		.value = "false,false",
	},
	{
		.key = "gsm.operator.numeric",
		.value = ",",
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
		.key = "gsm.version.baseband",
		.value = "G960FXXU1ARCC",
	},
	{
		.key = "gsm.version.baseband1",
		.value = "G960FXXU1ARCC",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "Samsung RIL v4.0",
	},
	{
		.key = "hwservicemanager.ready",
		.value = "true",
	},
	{
		.key = "init.svc.BCS-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.DIAG-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.DR-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.ExynosHWCServiceTW",
		.value = "running",
	},
	{
		.key = "init.svc.SMD-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.apaservice",
		.value = "running",
	},
	{
		.key = "init.svc.argos-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.at_distributor",
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
		.key = "init.svc.auditd",
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
		.key = "init.svc.bootchecker",
		.value = "stopped",
	},
	{
		.key = "init.svc.bsd",
		.value = "running",
	},
	{
		.key = "init.svc.cameraserver",
		.value = "running",
	},
	{
		.key = "init.svc.configstore-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.cpboot-daemon",
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
		.key = "init.svc.epmlogd",
		.value = "stopped",
	},
	{
		.key = "init.svc.esecomm",
		.value = "running",
	},
	{
		.key = "init.svc.faced",
		.value = "running",
	},
	{
		.key = "init.svc.flash_recovery",
		.value = "stopped",
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
		.key = "init.svc.gpsd",
		.value = "running",
	},
	{
		.key = "init.svc.gralloc-2-0",
		.value = "running",
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
		.key = "init.svc.icd",
		.value = "stopped",
	},
	{
		.key = "init.svc.imsd",
		.value = "running",
	},
	{
		.key = "init.svc.installd",
		.value = "running",
	},
	{
		.key = "init.svc.iod",
		.value = "running",
	},
	{
		.key = "init.svc.irisd",
		.value = "running",
	},
	{
		.key = "init.svc.jackservice",
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
		.key = "init.svc.lhd",
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
		.key = "init.svc.logd-reinit",
		.value = "stopped",
	},
	{
		.key = "init.svc.macloader",
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
		.key = "init.svc.mobicore",
		.value = "running",
	},
	{
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.power-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.powersnd",
		.value = "stopped",
	},
	{
		.key = "init.svc.prepare_param",
		.value = "stopped",
	},
	{
		.key = "init.svc.proca",
		.value = "running",
	},
	{
		.key = "init.svc.remotedisplay",
		.value = "running",
	},
	{
		.key = "init.svc.resetreason",
		.value = "stopped",
	},
	{
		.key = "init.svc.ril-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.ril-daemon1",
		.value = "running",
	},
	{
		.key = "init.svc.scs",
		.value = "stopped",
	},
	{
		.key = "init.svc.sdp_cryptod",
		.value = "running",
	},
	{
		.key = "init.svc.sec-camera-provider-2-4",
		.value = "running",
	},
	{
		.key = "init.svc.sec-miscpower-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.sec-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.sec-vibrator-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.sec_fps_hal",
		.value = "running",
	},
	{
		.key = "init.svc.sec_gnss_service",
		.value = "running",
	},
	{
		.key = "init.svc.sec_nfc_service",
		.value = "running",
	},
	{
		.key = "init.svc.secure_storage",
		.value = "running",
	},
	{
		.key = "init.svc.sem_daemon",
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
		.key = "init.svc.storaged",
		.value = "running",
	},
	{
		.key = "init.svc.surfaceflinger",
		.value = "running",
	},
	{
		.key = "init.svc.swapon",
		.value = "stopped",
	},
	{
		.key = "init.svc.thermal-hal-1-0",
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
		.key = "init.svc.vaultkeeperd",
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
		.key = "init.svc.watchdogd",
		.value = "running",
	},
	{
		.key = "init.svc.webview_zygote32",
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
		.key = "init.svc.wlandutservice",
		.value = "running",
	},
	{
		.key = "init.svc.wpa_supplicant",
		.value = "running",
	},
	{
		.key = "init.svc.wsmd",
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
		.key = "net.iptype",
		.value = "504:v4v6v6",
	},
	{
		.key = "net.knoxscep.version",
		.value = "2.2.0",
	},
	{
		.key = "net.knoxvpn.version",
		.value = "2.2.0",
	},
	{
		.key = "net.qtaguid_enabled",
		.value = "1",
	},
	{
		.key = "net.smart_switch.disabled",
		.value = "1",
	},
	{
		.key = "net.tcp.default_init_rwnd",
		.value = "60",
	},
	{
		.key = "nfc.boot.reason",
		.value = "1",
	},
	{
		.key = "nfc.cover.state",
		.value = "0",
	},
	{
		.key = "nfc.fw.dfl_areacode",
		.value = "DEF",
	},
	{
		.key = "nfc.fw.downloadmode_force",
		.value = "0",
	},
	{
		.key = "nfc.fw.rfreg_display_ver",
		.value = "2",
	},
	{
		.key = "nfc.fw.rfreg_ver",
		.value = "17/12/21/17.5.59",
	},
	{
		.key = "nfc.fw.ver",
		.value = "S.LSI 4.5.7",
	},
	{
		.key = "nfc.product.support.ese",
		.value = "1",
	},
	{
		.key = "nfc.product.support.uicc",
		.value = "1",
	},
	{
		.key = "persist.audio.a2dp_avc",
		.value = "1",
	},
	{
		.key = "persist.audio.allsoundmute",
		.value = "0",
	},
	{
		.key = "persist.audio.corefx",
		.value = "1",
	},
	{
		.key = "persist.audio.effectcpufreq",
		.value = "350000",
	},
	{
		.key = "persist.audio.finemediavolume",
		.value = "1",
	},
	{
		.key = "persist.audio.globaleffect",
		.value = "1",
	},
	{
		.key = "persist.audio.headsetsysvolume",
		.value = "9",
	},
	{
		.key = "persist.audio.hphonesysvolume",
		.value = "9",
	},
	{
		.key = "persist.audio.k2hd",
		.value = "1",
	},
	{
		.key = "persist.audio.mpseek",
		.value = "0",
	},
	{
		.key = "persist.audio.mysound",
		.value = "1",
	},
	{
		.key = "persist.audio.nxp_lvvil",
		.value = "0",
	},
	{
		.key = "persist.audio.pcmdump",
		.value = "0",
	},
	{
		.key = "persist.audio.ringermode",
		.value = "1",
	},
	{
		.key = "persist.audio.sales_code",
		.value = "MM1",
	},
	{
		.key = "persist.audio.soundalivefxsec",
		.value = "1",
	},
	{
		.key = "persist.audio.stereospeaker",
		.value = "1",
	},
	{
		.key = "persist.audio.sysvolume",
		.value = "9",
	},
	{
		.key = "persist.audio.uhqa",
		.value = "1",
	},
	{
		.key = "persist.audio.voipcpufreq",
		.value = "0",
	},
	{
		.key = "persist.demo.hdmirotationlock",
		.value = "false",
	},
	{
		.key = "persist.nfc.log.index",
		.value = "5",
	},
	{
		.key = "persist.radio.embms.support",
		.value = "false",
	},
	{
		.key = "persist.radio.latest-modeltype",
		.value = "2,2",
	},
	{
		.key = "persist.radio.multisim.config",
		.value = "dsds",
	},
	{
		.key = "persist.radio.sib16_support",
		.value = "1",
	},
	{
		.key = "persist.radio.ss.voiceclass_1",
		.value = "false",
	},
	{
		.key = "persist.ril.ims.eutranParam",
		.value = "3",
	},
	{
		.key = "persist.ril.ims.org.eutranParam",
		.value = "3",
	},
	{
		.key = "persist.ril.ims.org.utranParam",
		.value = "0",
	},
	{
		.key = "persist.ril.ims.utranParam",
		.value = "0",
	},
	{
		.key = "persist.ril.modem.board",
		.value = "SHANNON360",
	},
	{
		.key = "persist.ril.modem.board2",
		.value = "SHANNON360",
	},
	{
		.key = "persist.sys.ccm.date",
		.value = "Thu Mar 22 20:54:22 KST 2018",
	},
	{
		.key = "persist.sys.clssprld1",
		.value = "920",
	},
	{
		.key = "persist.sys.clssprld2",
		.value = "308",
	},
	{
		.key = "persist.sys.cpboot",
		.value = "unknown",
	},
	{
		.key = "persist.sys.csc_status",
		.value = "normal",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.display_density",
		.value = "480",
	},
	{
		.key = "persist.sys.dualapp.prop",
		.value = "1",
	},
	{
		.key = "persist.sys.enablehomekey",
		.value = "false",
	},
	{
		.key = "persist.sys.gps.lpp",
		.value = "",
	},
	{
		.key = "persist.sys.ims.supportmmtel1",
		.value = "0",
	},
	{
		.key = "persist.sys.ims.supportmmtel2",
		.value = "0",
	},
	{
		.key = "persist.sys.knox.device_owner",
		.value = "false",
	},
	{
		.key = "persist.sys.knox.userinfo",
		.value = "",
	},
	{
		.key = "persist.sys.locale",
		.value = "en-GB",
	},
	{
		.key = "persist.sys.localedefault",
		.value = "en-GB",
	},
	{
		.key = "persist.sys.members.cp_support",
		.value = "on",
	},
	{
		.key = "persist.sys.omc_etcpath",
		.value = "/odm/omc/single/MM1/etc",
	},
	{
		.key = "persist.sys.omc_path",
		.value = "/odm/omc/single/MM1/conf",
	},
	{
		.key = "persist.sys.omc_respath",
		.value = "/omr/res",
	},
	{
		.key = "persist.sys.omc_support",
		.value = "true",
	},
	{
		.key = "persist.sys.omcnw_path",
		.value = "/odm/omc/single/MM1/conf",
	},
	{
		.key = "persist.sys.omcnw_path2",
		.value = "/odm/omc/single/MM1/conf",
	},
	{
		.key = "persist.sys.ppr",
		.value = "true",
	},
	{
		.key = "persist.sys.preloads.file_cache_expired",
		.value = "1",
	},
	{
		.key = "persist.sys.prev_omcnwcode",
		.value = "MM1",
	},
	{
		.key = "persist.sys.prev_omcnwcode2",
		.value = "MM1",
	},
	{
		.key = "persist.sys.prev_salescode",
		.value = "MM1",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
	},
	{
		.key = "persist.sys.sb.setting.enabled",
		.value = "false",
	},
	{
		.key = "persist.sys.setupwizard",
		.value = "FINISH",
	},
	{
		.key = "persist.sys.silent",
		.value = "1",
	},
	{
		.key = "persist.sys.storage_preload",
		.value = "2",
	},
	{
		.key = "persist.sys.tcpOptimizer.on",
		.value = "1",
	},
	{
		.key = "persist.sys.timezone",
		.value = "Asia/Singapore",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "mtp,adb",
	},
	{
		.key = "persist.sys.usb.dualrole",
		.value = "true",
	},
	{
		.key = "persist.sys.vold.firstboot",
		.value = "1",
	},
	{
		.key = "persist.sys.vzw_wifi_running",
		.value = "false",
	},
	{
		.key = "persist.sys.webview.vmsize",
		.value = "114925168",
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
		.key = "pm.dexopt.install",
		.value = "quicken",
	},
	{
		.key = "ril.CompleteMsg",
		.value = "OK",
	},
	{
		.key = "ril.ICC_TYPE",
		.value = "0,0",
	},
	{
		.key = "ril.ICC_TYPE0",
		.value = "0",
	},
	{
		.key = "ril.ICC_TYPE1",
		.value = "0",
	},
	{
		.key = "ril.NwNmId",
		.value = "",
	},
	{
		.key = "ril.RildInit",
		.value = "1,1",
	},
	{
		.key = "ril.airplane.mode",
		.value = "1",
	},
	{
		.key = "ril.approved_codever",
		.value = "none",
	},
	{
		.key = "ril.approved_cscver",
		.value = "none",
	},
	{
		.key = "ril.approved_modemver",
		.value = "none",
	},
	{
		.key = "ril.atd_status",
		.value = "1_1_0",
	},
	{
		.key = "ril.backoffstate",
		.value = "1024",
	},
	{
		.key = "ril.callcount",
		.value = "0",
	},
	{
		.key = "ril.cbd.boot_done",
		.value = "1",
	},
	{
		.key = "ril.cbd.dt_revision",
		.value = "026",
	},
	{
		.key = "ril.cbd.first_xmit_done",
		.value = "1",
	},
	{
		.key = "ril.cbd.rfs_check_done",
		.value = "1",
	},
	{
		.key = "ril.cdma.esn",
		.value = "",
	},
	{
		.key = "ril.cs_svc",
		.value = "1",
	},
	{
		.key = "ril.data.intfprefix",
		.value = "rmnet",
	},
	{
		.key = "ril.debug.modemfactory",
		.value = "CSC Feature State: IMS ON, EPDG ON",
	},
	{
		.key = "ril.debug.ntc",
		.value = "M:EUR, S:EUR, T:GSM, C:EUR",
	},
	{
		.key = "ril.ecclist00",
		.value = "112,911,999,000,08,110,118,119",
	},
	{
		.key = "ril.ecclist10",
		.value = "112,911,999,000,08,110,118,119",
	},
	{
		.key = "ril.hasisim",
		.value = "0,0",
	},
	{
		.key = "ril.hw_ver",
		.value = "MP 0.900",
	},
	{
		.key = "ril.hw_ver2",
		.value = "MP 0.900",
	},
	{
		.key = "ril.ims.ecsupport",
		.value = "2,2",
	},
	{
		.key = "ril.ims.ltevoicesupport",
		.value = "2,2",
	},
	{
		.key = "ril.initPB",
		.value = "0",
	},
	{
		.key = "ril.initPB2",
		.value = "0",
	},
	{
		.key = "ril.lte_ps_only",
		.value = "0,0",
	},
	{
		.key = "ril.model_id",
		.value = "QB9270665",
	},
	{
		.key = "ril.model_id2",
		.value = "QB9270665",
	},
	{
		.key = "ril.modem.board",
		.value = "SHANNON360",
	},
	{
		.key = "ril.modem.board2",
		.value = "SHANNON360",
	},
	{
		.key = "ril.official_cscver",
		.value = "G960FOXM1ARCA",
	},
	{
		.key = "ril.product_code",
		.value = "SM-G960FZKDMM1",
	},
	{
		.key = "ril.product_code2",
		.value = "SM-G960FZKDMM1",
	},
	{
		.key = "ril.radiostate",
		.value = "0",
	},
	{
		.key = "ril.region_props",
		.value = "MM1.SINGAPORE.SG.MM1",
	},
	{
		.key = "ril.rfcal_date",
		.value = "20180223",
	},
	{
		.key = "ril.rfcal_date2",
		.value = "20180223",
	},
	{
		.key = "ril.serialnumber",
		.value = "R58K235S8VT",
	},
	{
		.key = "ril.servicestate",
		.value = "3,3",
	},
	{
		.key = "ril.simoperator",
		.value = ",",
	},
	{
		.key = "ril.ss.routing",
		.value = "0,0",
	},
	{
		.key = "ril.subinfo",
		.value = "0:-2,1:-3",
	},
	{
		.key = "ril.sw_ver",
		.value = "G960FXXU1ARCC",
	},
	{
		.key = "ril.sw_ver2",
		.value = "G960FXXU1ARCC",
	},
	{
		.key = "ril.twwan911Timer",
		.value = "0",
	},
	{
		.key = "rild.libargs",
		.value = "-d /dev/umts_ipc0",
	},
	{
		.key = "rild.libpath",
		.value = "/system/lib64/libsec-ril.so",
	},
	{
		.key = "ro.adb.qemud",
		.value = "1",
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
		.key = "ro.arch",
		.value = "exynos9810",
	},
	{
		.key = "ro.astcenc.hwencoder",
		.value = "1",
	},
	{
		.key = "ro.baseband",
		.value = "unknown",
	},
	{
		.key = "ro.board.platform",
		.value = "exynos5",
	},
	{
		.key = "ro.boot.ap_serial",
		.value = "0x030D8C948B32",
	},
	{
		.key = "ro.boot.bootloader",
		.value = "G960FXXU1ARCC",
	},
	{
		.key = "ro.boot.carrierid",
		.value = "XSP",
	},
	{
		.key = "ro.boot.carrierid_offset",
		.value = "7340608",
	},
	{
		.key = "ro.boot.debug_level",
		.value = "0x4f4c",
	},
	{
		.key = "ro.boot.dram_info",
		.value = "01,06,01,4G",
	},
	{
		.key = "ro.boot.em.did",
		.value = "20030d8c948b3211",
	},
	{
		.key = "ro.boot.em.model",
		.value = "SM-G960F",
	},
	{
		.key = "ro.boot.em.status",
		.value = "0x0",
	},
	{
		.key = "ro.boot.emmc_checksum",
		.value = "3",
	},
	{
		.key = "ro.boot.flash.locked",
		.value = "1",
	},
	{
		.key = "ro.boot.fmp_config",
		.value = "1",
	},
	{
		.key = "ro.boot.hardware",
		.value = "samsungexynos9810",
	},
	{
		.key = "ro.boot.hmac_mismatch",
		.value = "0",
	},
	{
		.key = "ro.boot.hw_rev",
		.value = "26",
	},
	{
		.key = "ro.boot.odin_download",
		.value = "1",
	},
	{
		.key = "ro.boot.prototype.param.offset",
		.value = "7351040",
	},
	{
		.key = "ro.boot.recovery_offset",
		.value = "7355136",
	},
	{
		.key = "ro.boot.sales.param.offset",
		.value = "7340572",
	},
	{
		.key = "ro.boot.sec_atd.tty",
		.value = "/dev/ttySAC0",
	},
	{
		.key = "ro.boot.security_mode",
		.value = "1526595585",
	},
	{
		.key = "ro.boot.selinux",
		.value = "enforcing",
	},
	{
		.key = "ro.boot.serialno",
		.value = "1cb2240629047ece",
	},
	{
		.key = "ro.boot.smsn_offset",
		.value = "7351040",
	},
	{
		.key = "ro.boot.ucs_mode",
		.value = "0",
	},
	{
		.key = "ro.boot.verifiedbootstate",
		.value = "green",
	},
	{
		.key = "ro.boot.warranty_bit",
		.value = "0",
	},
	{
		.key = "ro.boot_recovery",
		.value = "unknown",
	},
	{
		.key = "ro.bootimage.build.date",
		.value = "Thu Mar 22 20:54:22 KST 2018",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1521719662",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "samsung/starltexx/starlte:8.0.0/R16NW/G960FXXU1ARCC:user/release-keys",
	},
	{
		.key = "ro.bootloader",
		.value = "G960FXXU1ARCC",
	},
	{
		.key = "ro.bootmode",
		.value = "unknown",
	},
	{
		.key = "ro.build.PDA",
		.value = "G960FXXU1ARCC",
	},
	{
		.key = "ro.build.changelist",
		.value = "13138374",
	},
	{
		.key = "ro.build.characteristics",
		.value = "phone,emulator",
	},
	{
		.key = "ro.build.date",
		.value = "Thu Mar 22 20:54:22 KST 2018",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1521719662",
	},
	{
		.key = "ro.build.description",
		.value = "starltexx-user 8.0.0 R16NW G960FXXU1ARCC release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "R16NW.G960FXXU1ARCC",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "samsung/starltexx/starlte:8.0.0/R16NW/G960FXXU1ARCC:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "starltexx-user",
	},
	{
		.key = "ro.build.host",
		.value = "SWDD6507",
	},
	{
		.key = "ro.build.id",
		.value = "R16NW",
	},
	{
		.key = "ro.build.official.release",
		.value = "true",
	},
	{
		.key = "ro.build.product",
		.value = "starlte",
	},
	{
		.key = "ro.build.scafe.version",
		.value = "2018A",
	},
	{
		.key = "ro.build.selinux",
		.value = "1",
	},
	{
		.key = "ro.build.selinux.enforce",
		.value = "1",
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
		.value = "dpi",
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
		.key = "ro.build.version.incremental",
		.value = "G960FXXU1ARCC",
	},
	{
		.key = "ro.build.version.preview_sdk",
		.value = "0",
	},
	{
		.key = "ro.build.version.release",
		.value = "8.0.0",
	},
	{
		.key = "ro.build.version.sdk",
		.value = "26",
	},
	{
		.key = "ro.build.version.security_index",
		.value = "1",
	},
	{
		.key = "ro.build.version.security_patch",
		.value = "2018-03-01",
	},
	{
		.key = "ro.build.version.sem",
		.value = "2601",
	},
	{
		.key = "ro.build.version.sep",
		.value = "90000",
	},
	{
		.key = "ro.carrier",
		.value = "unknown",
	},
	{
		.key = "ro.carrierid",
		.value = "XSP",
	},
	{
		.key = "ro.carrierid.param.offset",
		.value = "7340608",
	},
	{
		.key = "ro.cfg.dha_cached_max",
		.value = "24",
	},
	{
		.key = "ro.chipname",
		.value = "exynos9810",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-samsung-ss",
	},
	{
		.key = "ro.com.google.clientidbase.ms",
		.value = "android-samsung-gs-rev1",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "8.0_r6",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Morning_Glory.ogg",
	},
	{
		.key = "ro.config.dha_cached_max",
		.value = "19",
	},
	{
		.key = "ro.config.dha_cached_min",
		.value = "6",
	},
	{
		.key = "ro.config.dha_empty_init",
		.value = "24",
	},
	{
		.key = "ro.config.dha_empty_max",
		.value = "24",
	},
	{
		.key = "ro.config.dha_empty_min",
		.value = "8",
	},
	{
		.key = "ro.config.dha_lmk_scale",
		.value = "1.0",
	},
	{
		.key = "ro.config.dha_pwhitelist_enable",
		.value = "1",
	},
	{
		.key = "ro.config.dha_pwhl_key",
		.value = "512",
	},
	{
		.key = "ro.config.dha_th_rate",
		.value = "2.0",
	},
	{
		.key = "ro.config.dmverity",
		.value = "true",
	},
	{
		.key = "ro.config.fall_prevent_enable",
		.value = "true",
	},
	{
		.key = "ro.config.iccc_version",
		.value = "3.0",
	},
	{
		.key = "ro.config.kap",
		.value = "true",
	},
	{
		.key = "ro.config.kap_default_on",
		.value = "true",
	},
	{
		.key = "ro.config.knox",
		.value = "v30",
	},
	{
		.key = "ro.config.media_sound",
		.value = "Media_preview_Touch_the_light.ogg",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "Skyline.ogg",
	},
	{
		.key = "ro.config.notification_sound_2",
		.value = "S_Charming_Bell.ogg",
	},
	{
		.key = "ro.config.ringtone",
		.value = "Over_the_Horizon.ogg",
	},
	{
		.key = "ro.config.ringtone_2",
		.value = "Basic_Bell.ogg",
	},
	{
		.key = "ro.config.rm_preload_enabled",
		.value = "1",
	},
	{
		.key = "ro.config.systemaudiodebug",
		.value = "abox&codecdsp",
	},
	{
		.key = "ro.config.tima",
		.value = "1",
	},
	{
		.key = "ro.config.timaversion",
		.value = "3.0",
	},
	{
		.key = "ro.config.vc_call_vol_steps",
		.value = "5",
	},
	{
		.key = "ro.control_privapp_permissions",
		.value = "log",
	},
	{
		.key = "ro.cp_debug_level",
		.value = "unknown",
	},
	{
		.key = "ro.crypto.fs_crypto_blkdev",
		.value = "/dev/block/dm-3",
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
		.value = "block",
	},
	{
		.key = "ro.csc.country_code",
		.value = "SINGAPORE",
	},
	{
		.key = "ro.csc.countryiso_code",
		.value = "SG",
	},
	{
		.key = "ro.csc.facebook.partnerid",
		.value = "samsung:dec1cc9c-1497-4aab-b953-cee702c2a481",
	},
	{
		.key = "ro.csc.omcnw_code",
		.value = "MM1",
	},
	{
		.key = "ro.csc.omcnw_code2",
		.value = "MM1",
	},
	{
		.key = "ro.csc.sales_code",
		.value = "MM1",
	},
	{
		.key = "ro.dalvik.vm.native.bridge",
		.value = "0",
	},
	{
		.key = "ro.debug_level",
		.value = "0x4f4c",
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
		.key = "ro.em.version",
		.value = "20",
	},
	{
		.key = "ro.emmc_checksum",
		.value = "3",
	},
	{
		.key = "ro.error.receiver.default",
		.value = "com.samsung.receiver.error",
	},
	{
		.key = "ro.expect.recovery_id",
		.value = "0x391347e6a649be99a22ebf04c41b03f07aad1993000000000000000000000000",
	},
	{
		.key = "ro.fmp_config",
		.value = "1",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/persistent",
	},
	{
		.key = "ro.gfx.driver.0",
		.value = "com.samsung.gpudriver.S9MaliG72_80",
	},
	{
		.key = "ro.hardware",
		.value = "samsungexynos9810",
	},
	{
		.key = "ro.hardware.egl",
		.value = "mali",
	},
	{
		.key = "ro.hardware.keystore",
		.value = "mdfpp",
	},
	{
		.key = "ro.hdcp2.rx",
		.value = "tz",
	},
	{
		.key = "ro.hmac_mismatch",
		.value = "0",
	},
	{
		.key = "ro.hwui.drop_shadow_cache_size",
		.value = "6",
	},
	{
		.key = "ro.hwui.gradient_cache_size",
		.value = "2",
	},
	{
		.key = "ro.hwui.layer_cache_size",
		.value = "58",
	},
	{
		.key = "ro.hwui.path_cache_size",
		.value = "16",
	},
	{
		.key = "ro.hwui.r_buffer_cache_size",
		.value = "8",
	},
	{
		.key = "ro.hwui.shape_cache_size",
		.value = "4",
	},
	{
		.key = "ro.hwui.text_large_cache_height",
		.value = "2048",
	},
	{
		.key = "ro.hwui.text_large_cache_width",
		.value = "4096",
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
		.value = "88",
	},
	{
		.key = "ro.im.param.offset",
		.value = "unknown",
	},
	{
		.key = "ro.kernel.qemu",
		.value = "0",
	},
	{
		.key = "ro.kernel.qemu.gles",
		.value = "2",
	},
	{
		.key = "ro.knox.enhance.zygote.aslr",
		.value = "0",
	},
	{
		.key = "ro.logd.auditd",
		.value = "false",
	},
	{
		.key = "ro.me.param.offset",
		.value = "unknown",
	},
	{
		.key = "ro.multisim.simslotcount",
		.value = "2",
	},
	{
		.key = "ro.oem.key1",
		.value = "MM1",
	},
	{
		.key = "ro.oem_unlock_supported",
		.value = "1",
	},
	{
		.key = "ro.omc.build.id",
		.value = "17395495",
	},
	{
		.key = "ro.omc.build.version",
		.value = "G960FOXM1ARCA",
	},
	{
		.key = "ro.omc.changetype",
		.value = "DATA_RESET_ON,TRUE",
	},
	{
		.key = "ro.omc.disabler",
		.value = "FALSE",
	},
	{
		.key = "ro.omc.img_mount",
		.value = "0",
	},
	{
		.key = "ro.omcnw.param.offset",
		.value = "unknown",
	},
	{
		.key = "ro.opengles.version",
		.value = "196610",
	},
	{
		.key = "ro.pr.param.offset",
		.value = "unknown",
	},
	{
		.key = "ro.product.board",
		.value = "universal9810",
	},
	{
		.key = "ro.product.brand",
		.value = "samsung",
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
		.value = "starlte",
	},
	{
		.key = "ro.product.first_api_level",
		.value = "26",
	},
	{
		.key = "ro.product.locale",
		.value = "en-GB",
	},
	{
		.key = "ro.product.manufacturer",
		.value = "samsung",
	},
	{
		.key = "ro.product.model",
		.value = "SM-G960F",
	},
	{
		.key = "ro.product.name",
		.value = "starltexx",
	},
	{
		.key = "ro.product_ship",
		.value = "true",
	},
	{
		.key = "ro.property_service.version",
		.value = "2",
	},
	{
		.key = "ro.radio.noril",
		.value = "no",
	},
	{
		.key = "ro.revision",
		.value = "26",
	},
	{
		.key = "ro.ril.gprsclass",
		.value = "10",
	},
	{
		.key = "ro.ril.hsxpa",
		.value = "1",
	},
	{
		.key = "ro.rtn_config",
		.value = "unknown",
	},
	{
		.key = "ro.sales.param.offset",
		.value = "7340572",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.security.ese.cosname",
		.value = "UT5.0_0100000B",
	},
	{
		.key = "ro.security.esest",
		.value = "per0201exi",
	},
	{
		.key = "ro.security.fips.ux",
		.value = "Enabled",
	},
	{
		.key = "ro.security.fips_bssl.ver",
		.value = "1.2",
	},
	{
		.key = "ro.security.fips_fmp.ver",
		.value = "1.4",
	},
	{
		.key = "ro.security.fips_scrypto.ver",
		.value = "2.2",
	},
	{
		.key = "ro.security.fips_skc.ver",
		.value = "1.9",
	},
	{
		.key = "ro.security.icd.flagmode",
		.value = "multi",
	},
	{
		.key = "ro.security.keystore.keytype",
		.value = "sak,gak",
	},
	{
		.key = "ro.security.mdf.release",
		.value = "1",
	},
	{
		.key = "ro.security.mdf.ux",
		.value = "Enabled",
	},
	{
		.key = "ro.security.mdf.ver",
		.value = "3.1",
	},
	{
		.key = "ro.security.reactive.version",
		.value = "2.0.11",
	},
	{
		.key = "ro.security.vaultkeeper.feature",
		.value = "1",
	},
	{
		.key = "ro.security.vpnpp.release",
		.value = "1.0",
	},
	{
		.key = "ro.security.vpnpp.ver",
		.value = "2.1",
	},
	{
		.key = "ro.security.wlan.release",
		.value = "1",
	},
	{
		.key = "ro.security.wlan.ver",
		.value = "1.0",
	},
	{
		.key = "ro.security_mode",
		.value = "1526595585",
	},
	{
		.key = "ro.serialno",
		.value = "1cb2240629047ece",
	},
	{
		.key = "ro.sf.init.lcd_density",
		.value = "640",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "480",
	},
	{
		.key = "ro.simbased.changetype",
		.value = "NO_DFLT_CSC,OMC",
	},
	{
		.key = "ro.sku.param.offset",
		.value = "unknown",
	},
	{
		.key = "ro.sn.param.offset",
		.value = "unknown",
	},
	{
		.key = "ro.supportmodel.mptcp",
		.value = "1",
	},
	{
		.key = "ro.telephony.default_cdma_sub",
		.value = "0",
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
		.key = "ro.userpartflashed",
		.value = "unknown",
	},
	{
		.key = "ro.vendor.build.date",
		.value = "Thu Mar 22 20:54:22 KST 2018",
	},
	{
		.key = "ro.vendor.build.date.utc",
		.value = "1521719662",
	},
	{
		.key = "ro.vendor.build.fingerprint",
		.value = "samsung/starltexx/starlte:8.0.0/R16NW/G960FXXU1ARCC:user/release-keys",
	},
	{
		.key = "ro.warranty_bit",
		.value = "0",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
	},
	{
		.key = "ro.wsmd.enable",
		.value = "true",
	},
	{
		.key = "ro.zygote",
		.value = "zygote64_32",
	},
	{
		.key = "ro.zygote.disable_gl_preload",
		.value = "true",
	},
	{
		.key = "secmm.codecsolution.ready",
		.value = "1",
	},
	{
		.key = "secmm.player.uhqamode",
		.value = "True",
	},
	{
		.key = "security.ASKS.policy_version",
		.value = "180123",
	},
	{
		.key = "security.ASKS.version",
		.value = "1.4",
	},
	{
		.key = "security.mdf",
		.value = "Ready",
	},
	{
		.key = "security.mdf.result",
		.value = "None",
	},
	{
		.key = "security.perf_harden",
		.value = "1",
	},
	{
		.key = "security.semdaemonfinish",
		.value = "1",
	},
	{
		.key = "service.browser.homepage",
		.value = "https://www.m1.com.sg",
	},
	{
		.key = "service.media.powersnd",
		.value = "1",
	},
	{
		.key = "service.sf.present_timestamp",
		.value = "1",
	},
	{
		.key = "storage.support.sdcard",
		.value = "1",
	},
	{
		.key = "storage.support.usb",
		.value = "1",
	},
	{
		.key = "sys.aasservice.aason",
		.value = "true",
	},
	{
		.key = "sys.bartender.batterystats.ver",
		.value = "17",
	},
	{
		.key = "sys.bluetooth.tty",
		.value = "ttySAC1",
	},
	{
		.key = "sys.boot.end_package",
		.value = "1",
	},
	{
		.key = "sys.boot.loop_forever",
		.value = "1",
	},
	{
		.key = "sys.boot.start_preload",
		.value = "1",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
	},
	{
		.key = "sys.cameramode.cam_binning",
		.value = "0",
	},
	{
		.key = "sys.cameramode.cam_fps",
		.value = "-1",
	},
	{
		.key = "sys.config.activelaunch_enable",
		.value = "true",
	},
	{
		.key = "sys.config.amp_perf_enable",
		.value = "true",
	},
	{
		.key = "sys.config.mars_version",
		.value = "2.10",
	},
	{
		.key = "sys.daydream.connected",
		.value = "0",
	},
	{
		.key = "sys.dockstate",
		.value = "0",
	},
	{
		.key = "sys.dualapp.profile_id",
		.value = "",
	},
	{
		.key = "sys.enterprise.billing.dualsim",
		.value = "true",
	},
	{
		.key = "sys.enterprise.billing.version",
		.value = "1.3.0",
	},
	{
		.key = "sys.gps.chipinfo",
		.value = "BCM4775",
	},
	{
		.key = "sys.gps.chipvendor",
		.value = "Broadcom",
	},
	{
		.key = "sys.gps.swversion",
		.value = "318732",
	},
	{
		.key = "sys.is_members",
		.value = "exist",
	},
	{
		.key = "sys.isdumpstaterunning",
		.value = "0",
	},
	{
		.key = "sys.logbootcomplete",
		.value = "1",
	},
	{
		.key = "sys.mdniecontrolservice.mscon",
		.value = "true",
	},
	{
		.key = "sys.members.installed",
		.value = "true",
	},
	{
		.key = "sys.nfc.support",
		.value = "1",
	},
	{
		.key = "sys.oem_unlock_allowed",
		.value = "0",
	},
	{
		.key = "sys.sbf.mnoname0",
		.value = "Mobileone_SG",
	},
	{
		.key = "sys.sbf.mnoname1",
		.value = "Mobileone_SG",
	},
	{
		.key = "sys.siop.level",
		.value = "-3",
	},
	{
		.key = "sys.skip_lockscreen",
		.value = "0",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "28096",
	},
	{
		.key = "sys.sysctl.tcp_def_init_rwnd",
		.value = "60",
	},
	{
		.key = "sys.usb.config",
		.value = "mtp,adb",
	},
	{
		.key = "sys.usb.configfs",
		.value = "1",
	},
	{
		.key = "sys.usb.controller",
		.value = "10c00000.dwc3",
	},
	{
		.key = "sys.usb.ffs.ready",
		.value = "1",
	},
	{
		.key = "sys.usb.state",
		.value = "mtp,adb",
	},
	{
		.key = "sys.use_fifo_ui",
		.value = "0",
	},
	{
		.key = "sys.vs.display",
		.value = "",
	},
	{
		.key = "sys.vs.mode",
		.value = "false",
	},
	{
		.key = "sys.vs.visible",
		.value = "false",
	},
	{
		.key = "sys.wifitracing.started",
		.value = "1",
	},
	{
		.key = "telephony.lteOnCdmaDevice",
		.value = "0",
	},
	{
		.key = "vendor.ril.debug.sales_code",
		.value = "EUR",
	},
	{
		.key = "vendor.sec.rild.libpath",
		.value = "/vendor/lib64/libsec-ril.so",
	},
	{
		.key = "vendor.sec.rild.libpath2",
		.value = "/vendor/lib64/libsec-ril-dsds.so",
	},
	{
		.key = "vold.crypt.type",
		.value = "default",
	},
	{
		.key = "vold.decrypt",
		.value = "trigger_restart_framework",
	},
	{
		.key = "vold.has_adoptable",
		.value = "0",
	},
	{
		.key = "vold.post_fs_data_done",
		.value = "1",
	},
	{
		.key = "vzw.os.rooted",
		.value = "false",
	},
	{
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
