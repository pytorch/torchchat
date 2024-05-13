struct cpuinfo_mock_file filesystem[] = {
#if CPUINFO_ARCH_ARM64
	{
		.path = "/proc/cpuinfo",
		.size = 1608,
		.content =
			"processor\t: 0\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 4\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x48\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd40\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 5\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x48\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd40\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 6\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x48\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd40\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 7\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x48\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd40\n"
			"CPU revision\t: 0\n"
			"\n",
	},
#elif CPUINFO_ARCH_ARM
	{
		.path = "/proc/cpuinfo",
		.size = 2231,
		.content =
			"processor\t: 0\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 1\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 2\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 3\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd05\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 4\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x48\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd40\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 5\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x48\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd40\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 6\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x48\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd40\n"
			"CPU revision\t: 0\n"
			"\n"
			"processor\t: 7\n"
			"model name\t: ARMv8 Processor rev 0 (v8l)\n"
			"BogoMIPS\t: 3.84\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x48\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd40\n"
			"CPU revision\t: 0\n",
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
		.size = 25,
		.content = "hisi_middle_cluster_idle\n",
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
		.content = "1805000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "830000\n",
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
		.size = 87,
		.content = "830000 980000 1056000 1152000 1248000 1325000 1421000 1517000 1613000 1709000 1805000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 77,
		.content = "interactive conservative ondemand userspace powersave performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "830000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 139,
		.content =
			"830000 210230\n"
			"980000 5490\n"
			"1056000 388\n"
			"1152000 777\n"
			"1248000 3390\n"
			"1325000 800\n"
			"1421000 1754\n"
			"1517000 673\n"
			"1613000 6759\n"
			"1709000 1696\n"
			"1805000 5590\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 5,
		.content = "9006\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/trans_table",
		.size = 1481,
		.content =
			"   From  :    To\n"
			"         :    830000    980000   1056000   1152000   1248000   1325000   1421000   1517000   1613000   1709000   1805000 \n"
			"   830000:         0       756         0         0       506        65        78        25       374        38         0 \n"
			"   980000:       610         0        82        89       211        31       105        12       248        15         0 \n"
			"  1056000:        37        33         0        18        55        11        19         5        47         3         0 \n"
			"  1152000:        65        56        13         0        96        23        48        14       108         5         0 \n"
			"  1248000:       310       133        34        60         0        85       202        18       226        11         0 \n"
			"  1325000:        96        49        11        34        24         0        72        20       137        13         0 \n"
			"  1421000:       154        74        21        72        41        43         0       107       221        14         0 \n"
			"  1517000:        54        21         4        21         6        21        30         0       159         4         0 \n"
			"  1613000:       517       281        63       134       140       177       193       119         0       131        19 \n"
			"  1709000:         0         0         0         0         0         0         0         0       178         0       266 \n"
			"  1805000:         0         0         0         0         0         0         0         0        75       210         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000411fd050\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x000000000000000f\n",
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
		.content = "1805000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "830000\n",
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
		.size = 87,
		.content = "830000 980000 1056000 1152000 1248000 1325000 1421000 1517000 1613000 1709000 1805000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 77,
		.content = "interactive conservative ondemand userspace powersave performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "830000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 139,
		.content =
			"830000 210358\n"
			"980000 5510\n"
			"1056000 388\n"
			"1152000 777\n"
			"1248000 3404\n"
			"1325000 800\n"
			"1421000 1760\n"
			"1517000 673\n"
			"1613000 6759\n"
			"1709000 1696\n"
			"1805000 5590\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 5,
		.content = "9017\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/trans_table",
		.size = 1481,
		.content =
			"   From  :    To\n"
			"         :    830000    980000   1056000   1152000   1248000   1325000   1421000   1517000   1613000   1709000   1805000 \n"
			"   830000:         0       758         0         0       509        65        78        25       374        38         0 \n"
			"   980000:       612         0        82        89       211        31       105        12       248        15         0 \n"
			"  1056000:        37        33         0        18        55        11        19         5        47         3         0 \n"
			"  1152000:        65        56        13         0        96        23        48        14       108         5         0 \n"
			"  1248000:       312       133        34        60         0        85       203        18       226        11         0 \n"
			"  1325000:        96        49        11        34        24         0        72        20       137        13         0 \n"
			"  1421000:       155        74        21        72        41        43         0       107       221        14         0 \n"
			"  1517000:        54        21         4        21         6        21        30         0       159         4         0 \n"
			"  1613000:       517       281        63       134       140       177       193       119         0       131        19 \n"
			"  1709000:         0         0         0         0         0         0         0         0       178         0       266 \n"
			"  1805000:         0         0         0         0         0         0         0         0        75       210         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000411fd050\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x000000000000000f\n",
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
		.content = "1805000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "830000\n",
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
		.size = 87,
		.content = "830000 980000 1056000 1152000 1248000 1325000 1421000 1517000 1613000 1709000 1805000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 77,
		.content = "interactive conservative ondemand userspace powersave performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1421000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 139,
		.content =
			"830000 210488\n"
			"980000 5536\n"
			"1056000 394\n"
			"1152000 777\n"
			"1248000 3410\n"
			"1325000 800\n"
			"1421000 1764\n"
			"1517000 673\n"
			"1613000 6759\n"
			"1709000 1696\n"
			"1805000 5590\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 5,
		.content = "9030\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/trans_table",
		.size = 1481,
		.content =
			"   From  :    To\n"
			"         :    830000    980000   1056000   1152000   1248000   1325000   1421000   1517000   1613000   1709000   1805000 \n"
			"   830000:         0       762         0         0       510        65        79        25       374        38         0 \n"
			"   980000:       614         0        83        89       212        31       105        12       248        15         0 \n"
			"  1056000:        38        33         0        18        55        11        19         5        47         3         0 \n"
			"  1152000:        65        56        13         0        96        23        48        14       108         5         0 \n"
			"  1248000:       313       133        34        60         0        85       203        18       226        11         0 \n"
			"  1325000:        96        49        11        34        24         0        72        20       137        13         0 \n"
			"  1421000:       156        74        21        72        41        43         0       107       221        14         0 \n"
			"  1517000:        54        21         4        21         6        21        30         0       159         4         0 \n"
			"  1613000:       517       281        63       134       140       177       193       119         0       131        19 \n"
			"  1709000:         0         0         0         0         0         0         0         0       178         0       266 \n"
			"  1805000:         0         0         0         0         0         0         0         0        75       210         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000411fd050\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x000000000000000f\n",
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
		.content = "1805000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "830000\n",
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
		.size = 87,
		.content = "830000 980000 1056000 1152000 1248000 1325000 1421000 1517000 1613000 1709000 1805000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 77,
		.content = "interactive conservative ondemand userspace powersave performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "980000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 139,
		.content =
			"830000 210606\n"
			"980000 5556\n"
			"1056000 394\n"
			"1152000 785\n"
			"1248000 3428\n"
			"1325000 800\n"
			"1421000 1764\n"
			"1517000 673\n"
			"1613000 6765\n"
			"1709000 1696\n"
			"1805000 5590\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 5,
		.content = "9047\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/trans_table",
		.size = 1481,
		.content =
			"   From  :    To\n"
			"         :    830000    980000   1056000   1152000   1248000   1325000   1421000   1517000   1613000   1709000   1805000 \n"
			"   830000:         0       767         0         0       511        65        79        25       375        38         0 \n"
			"   980000:       616         0        83        89       213        31       106        12       248        15         0 \n"
			"  1056000:        38        33         0        18        55        11        19         5        47         3         0 \n"
			"  1152000:        66        56        13         0        96        23        48        14       108         5         0 \n"
			"  1248000:       316       133        34        60         0        85       203        18       226        11         0 \n"
			"  1325000:        96        49        11        34        24         0        72        20       137        13         0 \n"
			"  1421000:       156        74        21        73        41        43         0       107       221        14         0 \n"
			"  1517000:        54        21         4        21         6        21        30         0       159         4         0 \n"
			"  1613000:       518       281        63       134       140       177       193       119         0       131        19 \n"
			"  1709000:         0         0         0         0         0         0         0         0       178         0       266 \n"
			"  1805000:         0         0         0         0         0         0         0         0        75       210         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000411fd050\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x000000000000000f\n",
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
		.size = 25,
		.content = "hisi_middle_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/affected_cpus",
		.size = 4,
		.content = "4 5\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1901000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "826000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/related_cpus",
		.size = 4,
		.content = "4 5\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies",
		.size = 95,
		.content = "826000 903000 1018000 1114000 1210000 1306000 1402000 1517000 1594000 1671000 1805000 1901000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors",
		.size = 77,
		.content = "interactive conservative ondemand userspace powersave performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "826000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 149,
		.content =
			"826000 217030\n"
			"903000 393\n"
			"1018000 728\n"
			"1114000 316\n"
			"1210000 200\n"
			"1306000 1820\n"
			"1402000 386\n"
			"1517000 784\n"
			"1594000 159\n"
			"1671000 9174\n"
			"1805000 1695\n"
			"1901000 5547\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 5,
		.content = "5719\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/trans_table",
		.size = 1733,
		.content =
			"   From  :    To\n"
			"         :    826000    903000   1018000   1114000   1210000   1306000   1402000   1517000   1594000   1671000   1805000   1901000 \n"
			"   826000:         0       133       201         0         2       517         8        41         9       238        58         0 \n"
			"   903000:       111         0        44        14         0        97        17         9         2        17         2         0 \n"
			"  1018000:       189        40         0        41         7       146        18         7         4        62         2         0 \n"
			"  1114000:        60        10        32         0        15       110        10         6         0        26         2         0 \n"
			"  1210000:        32         9        16         6         0        68         5         9         0        19         3         0 \n"
			"  1306000:       287        73       125       121        56         0        55       250        11        84        17         0 \n"
			"  1402000:        39        10        18        26        16        29         0        16        25        38         8         0 \n"
			"  1517000:        80         6        18        18        29        16        20         0         6       195        11         0 \n"
			"  1594000:        27         6         8         0         6        10         5         7         0        27         2         0 \n"
			"  1671000:       383        26        54        45        36        86        87        54        41         0       116        19 \n"
			"  1805000:         0         0         0         0         0         0         0         0         0       182         0       148 \n"
			"  1901000:         0         0         0         0         0         0         0         0         0        58       109         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000481fd400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000001\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/core_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/core_siblings",
		.size = 3,
		.content = "30\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/core_siblings_list",
		.size = 4,
		.content = "4-5\n",
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
		.size = 25,
		.content = "hisi_middle_cluster_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/affected_cpus",
		.size = 4,
		.content = "4 5\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1901000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "826000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/related_cpus",
		.size = 4,
		.content = "4 5\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_frequencies",
		.size = 95,
		.content = "826000 903000 1018000 1114000 1210000 1306000 1402000 1517000 1594000 1671000 1805000 1901000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_governors",
		.size = 77,
		.content = "interactive conservative ondemand userspace powersave performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "826000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 149,
		.content =
			"826000 217210\n"
			"903000 393\n"
			"1018000 728\n"
			"1114000 316\n"
			"1210000 200\n"
			"1306000 1820\n"
			"1402000 386\n"
			"1517000 784\n"
			"1594000 159\n"
			"1671000 9174\n"
			"1805000 1695\n"
			"1901000 5547\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 5,
		.content = "5719\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/trans_table",
		.size = 1733,
		.content =
			"   From  :    To\n"
			"         :    826000    903000   1018000   1114000   1210000   1306000   1402000   1517000   1594000   1671000   1805000   1901000 \n"
			"   826000:         0       133       201         0         2       517         8        41         9       238        58         0 \n"
			"   903000:       111         0        44        14         0        97        17         9         2        17         2         0 \n"
			"  1018000:       189        40         0        41         7       146        18         7         4        62         2         0 \n"
			"  1114000:        60        10        32         0        15       110        10         6         0        26         2         0 \n"
			"  1210000:        32         9        16         6         0        68         5         9         0        19         3         0 \n"
			"  1306000:       287        73       125       121        56         0        55       250        11        84        17         0 \n"
			"  1402000:        39        10        18        26        16        29         0        16        25        38         8         0 \n"
			"  1517000:        80         6        18        18        29        16        20         0         6       195        11         0 \n"
			"  1594000:        27         6         8         0         6        10         5         7         0        27         2         0 \n"
			"  1671000:       383        26        54        45        36        86        87        54        41         0       116        19 \n"
			"  1805000:         0         0         0         0         0         0         0         0         0       182         0       148 \n"
			"  1901000:         0         0         0         0         0         0         0         0         0        58       109         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000481fd400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000001\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/core_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/core_siblings",
		.size = 3,
		.content = "30\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/core_siblings_list",
		.size = 4,
		.content = "4-5\n",
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
		.size = 4,
		.content = "6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2600000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_min_freq",
		.size = 8,
		.content = "1460000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/related_cpus",
		.size = 4,
		.content = "6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_frequencies",
		.size = 105,
		.content = "1460000 1594000 1671000 1767000 1863000 1959000 2036000 2112000 2208000 2304000 2420000 2496000 2600000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_governors",
		.size = 77,
		.content = "interactive conservative ondemand userspace powersave performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1460000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 162,
		.content =
			"1460000 224532\n"
			"1594000 224\n"
			"1671000 266\n"
			"1767000 2223\n"
			"1863000 566\n"
			"1959000 1390\n"
			"2036000 544\n"
			"2112000 476\n"
			"2208000 427\n"
			"2304000 636\n"
			"2420000 312\n"
			"2496000 189\n"
			"2600000 6810\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 5,
		.content = "2998\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/trans_table",
		.size = 2005,
		.content =
			"   From  :    To\n"
			"         :   1460000   1594000   1671000   1767000   1863000   1959000   2036000   2112000   2208000   2304000   2420000   2496000   2600000 \n"
			"  1460000:         0        84        30       352        17        32         4         4         4        18         1        11        26 \n"
			"  1594000:        62         0        16        74         1         6         3         3         2         1         1         2         4 \n"
			"  1671000:        57        16         0        95         0         2         0         2         0         4         0         3         7 \n"
			"  1767000:       171        32        89         0        34       209         5        14         4        13        10        13        35 \n"
			"  1863000:        31         7         4        12         0        32         3         0         2         0         2         4         8 \n"
			"  1959000:       104        18        23        39        29         0        61        71         3         7         5         9        23 \n"
			"  2036000:        39         2         9        15         5        32         0        17        20         2         1         3        11 \n"
			"  2112000:         8         2         2         7         5        13        25         0        54         4         1         0        16 \n"
			"  2208000:        14         2         6         5         3        10         4        10         0        50         5         3         6 \n"
			"  2304000:        21         5         3         4         3        15         8         7        13         0        40         4         8 \n"
			"  2420000:         5         2         0         7         4         8         8         1         2         7         0         7        43 \n"
			"  2496000:        10         3         0         3         1        13         8         5         2         5         7         0        24 \n"
			"  2600000:        62         2         4        16         3        20        27         3        12        19        21        22         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000481fd400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000001\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/core_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/core_siblings",
		.size = 3,
		.content = "c0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/core_siblings_list",
		.size = 4,
		.content = "6-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/physical_package_id",
		.size = 2,
		.content = "2\n",
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
		.size = 4,
		.content = "6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2600000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_min_freq",
		.size = 8,
		.content = "1460000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_transition_latency",
		.size = 8,
		.content = "2000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/related_cpus",
		.size = 4,
		.content = "6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_frequencies",
		.size = 105,
		.content = "1460000 1594000 1671000 1767000 1863000 1959000 2036000 2112000 2208000 2304000 2420000 2496000 2600000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_governors",
		.size = 77,
		.content = "interactive conservative ondemand userspace powersave performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1460000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_driver",
		.size = 11,
		.content = "cpufreq-dt\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 162,
		.content =
			"1460000 224720\n"
			"1594000 224\n"
			"1671000 266\n"
			"1767000 2223\n"
			"1863000 566\n"
			"1959000 1390\n"
			"2036000 544\n"
			"2112000 476\n"
			"2208000 427\n"
			"2304000 636\n"
			"2420000 312\n"
			"2496000 189\n"
			"2600000 6810\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 5,
		.content = "2998\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/trans_table",
		.size = 2005,
		.content =
			"   From  :    To\n"
			"         :   1460000   1594000   1671000   1767000   1863000   1959000   2036000   2112000   2208000   2304000   2420000   2496000   2600000 \n"
			"  1460000:         0        84        30       352        17        32         4         4         4        18         1        11        26 \n"
			"  1594000:        62         0        16        74         1         6         3         3         2         1         1         2         4 \n"
			"  1671000:        57        16         0        95         0         2         0         2         0         4         0         3         7 \n"
			"  1767000:       171        32        89         0        34       209         5        14         4        13        10        13        35 \n"
			"  1863000:        31         7         4        12         0        32         3         0         2         0         2         4         8 \n"
			"  1959000:       104        18        23        39        29         0        61        71         3         7         5         9        23 \n"
			"  2036000:        39         2         9        15         5        32         0        17        20         2         1         3        11 \n"
			"  2112000:         8         2         2         7         5        13        25         0        54         4         1         0        16 \n"
			"  2208000:        14         2         6         5         3        10         4        10         0        50         5         3         6 \n"
			"  2304000:        21         5         3         4         3        15         8         7        13         0        40         4         8 \n"
			"  2420000:         5         2         0         7         4         8         8         1         2         7         0         7        43 \n"
			"  2496000:        10         3         0         3         1        13         8         5         2         5         7         0        24 \n"
			"  2600000:        62         2         4        16         3        20        27         3        12        19        21        22         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000481fd400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000001\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/core_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/core_siblings",
		.size = 3,
		.content = "c0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/core_siblings_list",
		.size = 4,
		.content = "6-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/physical_package_id",
		.size = 2,
		.content = "2\n",
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
		.key = "aaudio.mmap_exclusive_policy",
		.value = "2",
	},
	{
		.key = "aaudio.mmap_policy",
		.value = "2",
	},
	{
		.key = "bastet.service.enable",
		.value = "true",
	},
	{
		.key = "bg_fsck.pgid",
		.value = "438",
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
		.key = "dalvik.vm.dex2oat-minidebuginfo",
		.value = "true",
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
		.value = "100",
	},
	{
		.key = "debug.aps.enable",
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
		.value = "0",
	},
	{
		.key = "gsm.fastdormancy.screen",
		.value = "2",
	},
	{
		.key = "gsm.fastdormancy.time_scroff",
		.value = "4000",
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
		.value = "Unknown,Unknown",
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
		.value = ",",
	},
	{
		.key = "gsm.operator.iso-country",
		.value = ",",
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
		.key = "gsm.sim.preiccid_0",
		.value = "",
	},
	{
		.key = "gsm.sim.preiccid_1",
		.value = "",
	},
	{
		.key = "gsm.sim.state",
		.value = "ABSENT,ABSENT",
	},
	{
		.key = "gsm.sim.updatenitz",
		.value = "-1",
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
		.value = "21C10B675S000C000,21C10B675S000C000",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "android infineon balong-ril 1.0",
	},
	{
		.key = "hw.hicure.dns_fail_count",
		.value = "64",
	},
	{
		.key = "hw.lcd.density",
		.value = "480",
	},
	{
		.key = "hw.wifi.dns_stat",
		.value = "125,25,119878,1,384874",
	},
	{
		.key = "hw.wifipro.uid_dns_fail_count",
		.value = "1000-2/10026-6/10090-5/10011-18/10089-19/10084-2/10071-8/10074-4",
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
		.key = "init.svc.ITouchservice",
		.value = "running",
	},
	{
		.key = "init.svc.activityrecognition_1_1",
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
		.key = "init.svc.audioserver",
		.value = "running",
	},
	{
		.key = "init.svc.bastetd",
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
		.key = "init.svc.chargelogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.chargemonitor",
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
		.key = "init.svc.displayeffect-1-2",
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
		.key = "init.svc.dubai-hal-1-1",
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
		.key = "init.svc.fm-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.fps_hal_ext",
		.value = "running",
	},
	{
		.key = "init.svc.fusd",
		.value = "running",
	},
	{
		.key = "init.svc.gatekeeperd",
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
		.key = "init.svc.hal_gnss_service_1-1",
		.value = "running",
	},
	{
		.key = "init.svc.hbslogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.hdbd",
		.value = "stopped",
	},
	{
		.key = "init.svc.health-hal-2-0",
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
		.key = "init.svc.hignss",
		.value = "running",
	},
	{
		.key = "init.svc.hinetmanager",
		.value = "running",
	},
	{
		.key = "init.svc.hisecd",
		.value = "running",
	},
	{
		.key = "init.svc.hisi_bfg",
		.value = "stopped",
	},
	{
		.key = "init.svc.hiview",
		.value = "running",
	},
	{
		.key = "init.svc.hivrar-hal-1-2",
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
		.key = "init.svc.hw_ueventd",
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
		.key = "init.svc.hwhiview-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.hwlogd",
		.value = "stopped",
	},
	{
		.key = "init.svc.hwnffearly",
		.value = "stopped",
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
		.key = "init.svc.hwsecurity-hal",
		.value = "running",
	},
	{
		.key = "init.svc.hwservicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.iGraphicsservice",
		.value = "running",
	},
	{
		.key = "init.svc.igraphicslogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.iked",
		.value = "running",
	},
	{
		.key = "init.svc.incidentd",
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
		.key = "init.svc.keystore",
		.value = "running",
	},
	{
		.key = "init.svc.kmsglogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.libteec-2-0",
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
		.key = "init.svc.maplelogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.media",
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
		.key = "init.svc.modem_driver",
		.value = "stopped",
	},
	{
		.key = "init.svc.modemchr_service",
		.value = "running",
	},
	{
		.key = "init.svc.motion-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.msdplogcat",
		.value = "stopped",
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
		.key = "init.svc.oam_hisi",
		.value = "running",
	},
	{
		.key = "init.svc.octty",
		.value = "running",
	},
	{
		.key = "init.svc.odmf-data-chgrp",
		.value = "stopped",
	},
	{
		.key = "init.svc.odmflogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.oeminfo_nvm",
		.value = "running",
	},
	{
		.key = "init.svc.perfgenius-hal-2-0",
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
		.key = "init.svc.powerct",
		.value = "stopped",
	},
	{
		.key = "init.svc.powerlogd",
		.value = "running",
	},
	{
		.key = "init.svc.restart_xlogcat_service",
		.value = "stopped",
	},
	{
		.key = "init.svc.rillogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.secure_element_hal_service",
		.value = "running",
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
		.key = "init.svc.shlogd",
		.value = "stopped",
	},
	{
		.key = "init.svc.sleeplogcat",
		.value = "stopped",
	},
	{
		.key = "init.svc.statsd",
		.value = "running",
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
		.key = "init.svc.teeauth",
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
		.key = "init.svc.thermalservice",
		.value = "running",
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
		.key = "init.svc.usbd",
		.value = "stopped",
	},
	{
		.key = "init.svc.vdecoder_1_0",
		.value = "running",
	},
	{
		.key = "init.svc.vibrator-HW-1-1",
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
		.key = "init.svc.watchlssd",
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
		.key = "init.svc.wpa_supplicant",
		.value = "running",
	},
	{
		.key = "init.svc.xlogcat_service",
		.value = "stopped",
	},
	{
		.key = "init.svc.xlogctl_service",
		.value = "stopped",
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
		.key = "log.tag.stats_log",
		.value = "I",
	},
	{
		.key = "net.bt.name",
		.value = "Android",
	},
	{
		.key = "net.hostname",
		.value = "HUAWEI_Mate_20-94bb0551fb",
	},
	{
		.key = "net.ntp.time",
		.value = "1545183032075",
	},
	{
		.key = "net.ntp.timereference",
		.value = "1309896",
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
		.key = "nfc.initialized",
		.value = "true",
	},
	{
		.key = "nfc.node",
		.value = "/dev/pn544",
	},
	{
		.key = "odm.drm.stop",
		.value = "false",
	},
	{
		.key = "partition.odm.verified",
		.value = "2",
	},
	{
		.key = "partition.vendor.verified",
		.value = "2",
	},
	{
		.key = "persist.bt.max.a2dp.connections",
		.value = "2",
	},
	{
		.key = "persist.jank.gameskip",
		.value = "true",
	},
	{
		.key = "persist.kirin.alloc_buffer_sync",
		.value = "true",
	},
	{
		.key = "persist.kirin.texture_cache_opt",
		.value = "1",
	},
	{
		.key = "persist.kirin.touch_move_opt",
		.value = "1",
	},
	{
		.key = "persist.kirin.touch_vsync_opt",
		.value = "1",
	},
	{
		.key = "persist.kirin.touchevent_opt",
		.value = "1",
	},
	{
		.key = "persist.media.lowlatency.enable",
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
		.key = "persist.radio.findmyphone",
		.value = "0",
	},
	{
		.key = "persist.radio.modem.cap",
		.value = "09B9D52",
	},
	{
		.key = "persist.radio.multisim.config",
		.value = "dsds",
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
		.key = "persist.sys.adoptable",
		.value = "force_on",
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
		.key = "persist.sys.boot.reason",
		.value = "shutdown,",
	},
	{
		.key = "persist.sys.cpuset.enable",
		.value = "1",
	},
	{
		.key = "persist.sys.cpuset.subswitch",
		.value = "863760",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.devsched.subswitch",
		.value = "255",
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
		.key = "persist.sys.getvolumelist.cache",
		.value = "true",
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
		.key = "persist.sys.iaware.appboost.click_duration",
		.value = "1000",
	},
	{
		.key = "persist.sys.iaware.appboost.click_times",
		.value = "3",
	},
	{
		.key = "persist.sys.iaware.appboost.slide_duration",
		.value = "5000",
	},
	{
		.key = "persist.sys.iaware.appboost.slide_times",
		.value = "16",
	},
	{
		.key = "persist.sys.iaware.appboost.switch",
		.value = "true",
	},
	{
		.key = "persist.sys.iaware.blitparallel",
		.value = "true",
	},
	{
		.key = "persist.sys.iaware.cpuenable",
		.value = "true",
	},
	{
		.key = "persist.sys.iaware.jpg_sample_adapt",
		.value = "3",
	},
	{
		.key = "persist.sys.iaware.size.BitmapDeocodeCache",
		.value = "2048",
	},
	{
		.key = "persist.sys.iaware.switch.BitmapDeocodeCache",
		.value = "true",
	},
	{
		.key = "persist.sys.iaware.vsyncfirst",
		.value = "true",
	},
	{
		.key = "persist.sys.iaware_config_cust",
		.value = "iaware_cust_HMA-L29_US_9_108C605R1.xml",
	},
	{
		.key = "persist.sys.iaware_config_ver",
		.value = "iaware_config_HMA-L29_US_9_108C605R1.xml",
	},
	{
		.key = "persist.sys.jankenable",
		.value = "true",
	},
	{
		.key = "persist.sys.locale",
		.value = "en-US",
	},
	{
		.key = "persist.sys.logsystem.coredump",
		.value = "off",
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
		.key = "persist.sys.opkey0",
		.value = "",
	},
	{
		.key = "persist.sys.opkey1",
		.value = "",
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
		.key = "persist.sys.rog.height",
		.value = "2244",
	},
	{
		.key = "persist.sys.rog.width",
		.value = "1080",
	},
	{
		.key = "persist.sys.root.status",
		.value = "0",
	},
	{
		.key = "persist.sys.sdencryption.enable",
		.value = "true",
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
		.key = "persist.sys.timezone",
		.value = "America/Los_Angeles",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "hisuite,mtp,mass_storage,adb",
	},
	{
		.key = "persist.sys.webview.vmsize",
		.value = "110500960",
	},
	{
		.key = "persist.sys.zen_mode",
		.value = "0",
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
		.value = "speed-profile",
	},
	{
		.key = "pm.dexopt.priv-apps-oob",
		.value = "false",
	},
	{
		.key = "pm.dexopt.priv-apps-oob-list",
		.value = "ALL",
	},
	{
		.key = "pm.dexopt.shared",
		.value = "speed",
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
		.key = "ril.operator.numeric",
		.value = "311480",
	},
	{
		.key = "ro.actionable_compatible_property.enabled",
		.value = "true",
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
		.key = "ro.appsflyer.preinstall.path",
		.value = "/preload/HMA-L29/hw/la/xml/pre_install.appsflyer",
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
		.value = "8413",
	},
	{
		.key = "ro.board.boardname",
		.value = "HMA_LX9_VE",
	},
	{
		.key = "ro.board.chiptype",
		.value = "kirin980_cs",
	},
	{
		.key = "ro.board.modemid",
		.value = "37003000",
	},
	{
		.key = "ro.board.platform",
		.value = "kirin980",
	},
	{
		.key = "ro.booking.channel.path",
		.value = "preload/HMA-L29/hw/la/xml",
	},
	{
		.key = "ro.boot.avb_version",
		.value = "0.0",
	},
	{
		.key = "ro.boot.boot_devices",
		.value = "ff3c0000.ufs",
	},
	{
		.key = "ro.boot.bootreason",
		.value = "shutdown,",
	},
	{
		.key = "ro.boot.dtbo_idx",
		.value = "0",
	},
	{
		.key = "ro.boot.flash.locked",
		.value = "1",
	},
	{
		.key = "ro.boot.hardware",
		.value = "kirin980",
	},
	{
		.key = "ro.boot.mode",
		.value = "normal",
	},
	{
		.key = "ro.boot.product.hardware.sku",
		.value = "HMA-L29",
	},
	{
		.key = "ro.boot.selinux",
		.value = "enforcing",
	},
	{
		.key = "ro.boot.serialno",
		.value = "MUN0218A29000006",
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
		.value = "b0ce84c9dbcc63c1ed36713293f89f69eba25f594796d8789f5f7dc3ca1dbb95",
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
		.value = "23296",
	},
	{
		.key = "ro.boot.vercnt1",
		.value = "1",
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
		.value = "Thu Sep 13 21:40:01 CST 2018",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1536846001",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "Huawei/generic_a15/generic_a15:9/PPR1.180610.011/jenkins09132137:user/test-keys",
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
		.key = "ro.build.backtargetmode",
		.value = "V1.0",
	},
	{
		.key = "ro.build.characteristics",
		.value = "default",
	},
	{
		.key = "ro.build.date",
		.value = "Thu Sep 13 21:37:24 CST 2018",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1536845844",
	},
	{
		.key = "ro.build.description",
		.value = "HMA-L29-user 9.0.0 HUAWEIHMA-L29 108-OVS-LGRP2 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "HMA-L29 9.0.0.108(C605E10R1P16)",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "HUAWEI/HMA-L29/HWHMA:9/HUAWEIHMA-L29/108C605R1:user/release-keys",
	},
	{
		.key = "ro.build.hide",
		.value = "false",
	},
	{
		.key = "ro.build.hide.matchers",
		.value = "HMA;HW;HWHMA;kirin980;Heimdall-MP12;9.0.0",
	},
	{
		.key = "ro.build.hide.replacements",
		.value = "PAN;unknown;unknown;unknown;unknown;5.0.1",
	},
	{
		.key = "ro.build.hide.settings",
		.value = "8;1.8 GHz;2.0GB;11.00 GB;16.00 GB;1920 x 1080;5.1;3.10.30;3.1",
	},
	{
		.key = "ro.build.host",
		.value = "szvjk004cna",
	},
	{
		.key = "ro.build.hw_emui_api_level",
		.value = "17",
	},
	{
		.key = "ro.build.hw_emui_lite.enable",
		.value = "false",
	},
	{
		.key = "ro.build.id",
		.value = "HUAWEIHMA-L29",
	},
	{
		.key = "ro.build.product",
		.value = "HMA",
	},
	{
		.key = "ro.build.system_root_image",
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
		.value = "EmotionUI_9.0.0",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "108C605R1",
	},
	{
		.key = "ro.build.version.min_supported_target_sdk",
		.value = "17",
	},
	{
		.key = "ro.build.version.preview_sdk",
		.value = "0",
	},
	{
		.key = "ro.build.version.release",
		.value = "9",
	},
	{
		.key = "ro.build.version.sdk",
		.value = "28",
	},
	{
		.key = "ro.build.version.security_patch",
		.value = "2018-09-01",
	},
	{
		.key = "ro.camera.cos_ttpic_supported",
		.value = "false",
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
		.key = "ro.com.google.clientidbase.ms",
		.value = "android-huawei-rev1",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "9_201808",
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
		.key = "ro.comp.cust_version",
		.value = "Cust-OVS 9.0.0.1(0000)",
	},
	{
		.key = "ro.comp.hl.product_base_version",
		.value = "HMA-LGRP2-OVS 9.0.0.108",
	},
	{
		.key = "ro.comp.hl.product_cust_version",
		.value = "HMA-L29-CUST 9.0.0.10(C605)",
	},
	{
		.key = "ro.comp.hl.product_preload_version",
		.value = "HMA-L29-PRELOAD 9.0.0.16(C605R1)",
	},
	{
		.key = "ro.comp.product_version",
		.value = "Product-HMA 9.0.0(0000)",
	},
	{
		.key = "ro.comp.sys_support_vndk",
		.value = "",
	},
	{
		.key = "ro.comp.system_version",
		.value = "System 9.0.0.18(004F)",
	},
	{
		.key = "ro.comp.version_version",
		.value = "Version-HMA-L29-605000 9.0.0(000J)",
	},
	{
		.key = "ro.confg.hw_base_userdataversion",
		.value = "BASE_DATA",
	},
	{
		.key = "ro.confg.hw_systemversion",
		.value = "System 9.0.0.18(004F)",
	},
	{
		.key = "ro.config.CphsOnsEnabled",
		.value = "true",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Forest_Melody.ogg",
	},
	{
		.key = "ro.config.aperture_zoom_custom",
		.value = "1",
	},
	{
		.key = "ro.config.app_big_icon_size",
		.value = "160",
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
		.value = "IPV4V6PCSCF",
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
		.value = "50,1;100,0.9;150,0.8;400,0.8;450,0.9;470,1",
	},
	{
		.key = "ro.config.carkitmodenotif",
		.value = "true",
	},
	{
		.key = "ro.config.cbs_del_2B",
		.value = "true",
	},
	{
		.key = "ro.config.cdma_quiet",
		.value = "true",
	},
	{
		.key = "ro.config.cl_volte_autoswitch",
		.value = "true",
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
		.key = "ro.config.data_preinstalled",
		.value = "true",
	},
	{
		.key = "ro.config.default_sms_app",
		.value = "com.google.android.apps.messaging",
	},
	{
		.key = "ro.config.delay_updatename",
		.value = "true",
	},
	{
		.key = "ro.config.delete.preferapn",
		.value = "true",
	},
	{
		.key = "ro.config.demo_allow_pwd",
		.value = "false",
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
		.key = "ro.config.disable_reset_by_mdm",
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
		.key = "ro.config.dolby_game_mode",
		.value = "false",
	},
	{
		.key = "ro.config.dolby_volume",
		.value = "-40;-50",
	},
	{
		.key = "ro.config.dsds_mode",
		.value = "cdma_gsm",
	},
	{
		.key = "ro.config.empty.package",
		.value = "true",
	},
	{
		.key = "ro.config.enable_iaware",
		.value = "true",
	},
	{
		.key = "ro.config.enable_partition_move_update",
		.value = "1",
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
		.key = "ro.config.filterservice",
		.value = "false",
	},
	{
		.key = "ro.config.finger_joint",
		.value = "true",
	},
	{
		.key = "ro.config.fp_navigation",
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
		.key = "ro.config.gameassist",
		.value = "1",
	},
	{
		.key = "ro.config.gameassist.peripherals",
		.value = "1",
	},
	{
		.key = "ro.config.gameassist_booster",
		.value = "1",
	},
	{
		.key = "ro.config.gameassist_soundtovibrate",
		.value = "0",
	},
	{
		.key = "ro.config.helix_enable",
		.value = "true",
	},
	{
		.key = "ro.config.hisi_cdma_supported",
		.value = "true",
	},
	{
		.key = "ro.config.hpx_m6m8_support",
		.value = "true",
	},
	{
		.key = "ro.config.huawei_smallwindow",
		.value = "294,120,1080,1504",
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
		.key = "ro.config.hw_agps_adpt_sim",
		.value = "true",
	},
	{
		.key = "ro.config.hw_allow_rs_mms",
		.value = "true",
	},
	{
		.key = "ro.config.hw_always_allow_mms",
		.value = "6",
	},
	{
		.key = "ro.config.hw_camera_nfc_switch",
		.value = "true",
	},
	{
		.key = "ro.config.hw_cbs_mcc",
		.value = "730",
	},
	{
		.key = "ro.config.hw_charge_frz",
		.value = "true",
	},
	{
		.key = "ro.config.hw_codec_support",
		.value = "0.180410",
	},
	{
		.key = "ro.config.hw_cota",
		.value = "true",
	},
	{
		.key = "ro.config.hw_custverdisplay",
		.value = "true",
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
		.value = "B018",
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
		.key = "ro.config.hw_emui_dp_pc_mode",
		.value = "true",
	},
	{
		.key = "ro.config.hw_emui_wfd_pc_mode",
		.value = "true",
	},
	{
		.key = "ro.config.hw_enable_merge",
		.value = "true",
	},
	{
		.key = "ro.config.hw_front_camera_support_zoom",
		.value = "false",
	},
	{
		.key = "ro.config.hw_front_fp_navi",
		.value = "false",
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
		.value = "2",
	},
	{
		.key = "ro.config.hw_multiscreen",
		.value = "false",
	},
	{
		.key = "ro.config.hw_multiscreen_optimize",
		.value = "true",
	},
	{
		.key = "ro.config.hw_navigationbar",
		.value = "true",
	},
	{
		.key = "ro.config.hw_nfc_on",
		.value = "true",
	},
	{
		.key = "ro.config.hw_not_modify_wifi",
		.value = "WIFI ETB,WIFI ETB2",
	},
	{
		.key = "ro.config.hw_notch_size",
		.value = "226,81,427,54",
	},
	{
		.key = "ro.config.hw_omacp",
		.value = "1",
	},
	{
		.key = "ro.config.hw_opta",
		.value = "605",
	},
	{
		.key = "ro.config.hw_optb",
		.value = "999",
	},
	{
		.key = "ro.config.hw_power_saving",
		.value = "true",
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
		.key = "ro.config.hw_show_mmiError",
		.value = "true",
	},
	{
		.key = "ro.config.hw_sim2airplane",
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
		.key = "ro.config.hw_support_long_vmNum",
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
		.key = "ro.config.hw_updateCotaPara",
		.value = "true",
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
		.value = "0",
	},
	{
		.key = "ro.config.hw_volte_on",
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
		.key = "ro.config.hw_wfd_optimize",
		.value = "true",
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
		.key = "ro.config.keyguard_unusedata",
		.value = "false",
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
		.key = "ro.config.marketing_name",
		.value = "HUAWEI Mate 20",
	},
	{
		.key = "ro.config.new_hw_screen_aspect",
		.value = "2244:2134:1080",
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
		.key = "ro.config.pg_camera_cabc",
		.value = "true",
	},
	{
		.key = "ro.config.plmn_to_settings",
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
		.value = "FCC,CE",
	},
	{
		.key = "ro.config.screenon_turnoff_led",
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
		.key = "ro.config.simlang",
		.value = "true",
	},
	{
		.key = "ro.config.small_cover_size",
		.value = "_1048x1912",
	},
	{
		.key = "ro.config.sn_main_page",
		.value = "true",
	},
	{
		.key = "ro.config.soft_single_navi",
		.value = "false",
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
		.key = "ro.config.support_ca",
		.value = "true",
	},
	{
		.key = "ro.config.support_ccmode",
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
		.key = "ro.config.support_wcdma_modem1",
		.value = "true",
	},
	{
		.key = "ro.config.switchPrimaryVolume",
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
		.key = "ro.config.widevine_level3",
		.value = "true",
	},
	{
		.key = "ro.config.wifi_fast_bss_enable",
		.value = "true",
	},
	{
		.key = "ro.connectivity.chiptype",
		.value = "hisi",
	},
	{
		.key = "ro.connectivity.sub_chiptype",
		.value = "hi1103",
	},
	{
		.key = "ro.control.sleeplog",
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
		.value = "/product/region_comm/oversea/cdrom/autorun.iso",
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
		.key = "ro.dual.sim.phone",
		.value = "true",
	},
	{
		.key = "ro.expect.recovery_id",
		.value = "0x9fa8e68d6518086e768455a638403afc84a17d88000000000000000000000000",
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
		.key = "ro.gpu_turbo",
		.value = "",
	},
	{
		.key = "ro.hardware",
		.value = "kirin980",
	},
	{
		.key = "ro.hardware.audio.primary",
		.value = "hisi",
	},
	{
		.key = "ro.huawei.cust.drm.fl_only",
		.value = "false",
	},
	{
		.key = "ro.huawei.cust.oma",
		.value = "false",
	},
	{
		.key = "ro.huawei.cust.oma_drm",
		.value = "true",
	},
	{
		.key = "ro.huawei.remount.check",
		.value = "verify_success",
	},
	{
		.key = "ro.hw.base_all_groupversion",
		.value = "G1.0",
	},
	{
		.key = "ro.hw.country",
		.value = "la",
	},
	{
		.key = "ro.hw.custPath",
		.value = "/cust/hw/la",
	},
	{
		.key = "ro.hw.cust_all_groupversion",
		.value = "NA",
	},
	{
		.key = "ro.hw.hota.is_hwinit_cust_exists",
		.value = "false",
	},
	{
		.key = "ro.hw.hota.is_hwinit_exists",
		.value = "false",
	},
	{
		.key = "ro.hw.hota.is_hwinit_preload_exists",
		.value = "false",
	},
	{
		.key = "ro.hw.mirrorlink.enable",
		.value = "true",
	},
	{
		.key = "ro.hw.oemName",
		.value = "HMA-L29",
	},
	{
		.key = "ro.hw.preload_all_groupversion",
		.value = "G1.0",
	},
	{
		.key = "ro.hw.vendor",
		.value = "hw",
	},
	{
		.key = "ro.hwcamera.aimovie_enable",
		.value = "0",
	},
	{
		.key = "ro.hwcamera.frontzoom_enable",
		.value = "0",
	},
	{
		.key = "ro.hwcamera.modesuggest_enable",
		.value = "true",
	},
	{
		.key = "ro.hwcamera.smartzoom_enable",
		.value = "false",
	},
	{
		.key = "ro.hwcamera.use.videosize.1080p",
		.value = "true",
	},
	{
		.key = "ro.hwtracking.com.booking",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.facebook.appmanager",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.facebook.katana",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.facebook.orca",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.facebook.services",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.facebook.system",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.gameloft.android.GloftANPH",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.gameloft.android.GloftDBMF",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.gameloft.android.GloftDMKF",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.gameloft.android.GloftPDMF",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.gameloft.android.GloftSMIF",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.google.android.apps.docs.editors.docs",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.google.android.apps.docs.editors.sheets",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.google.android.apps.docs.editors.slides",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.huawei.autoinstallapkfrommcc",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.igg.android.lordsmobile",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.netflix.mediaclient",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.hwtracking.com.netflix.partner.activation",
		.value = "huawei_preload_default",
	},
	{
		.key = "ro.logd.size.stats",
		.value = "64K",
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
		.key = "ro.oba.version",
		.value = "NO_OBA_VERSION",
	},
	{
		.key = "ro.odm.build.fingerprint",
		.value = "Huawei/Atlanta/Atlanta_HMA-L29:9/PPR1.180610.011/20180913215553:user/release-keys",
	},
	{
		.key = "ro.odm.ca_product_version",
		.value = "HMA-L29",
	},
	{
		.key = "ro.odm.radio.nvcfg_normalization",
		.value = "true",
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
		.value = "C605",
	},
	{
		.key = "ro.product.CustDVersion",
		.value = "D1",
	},
	{
		.key = "ro.product.board",
		.value = "HMA",
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
		.value = "HWHMA",
	},
	{
		.key = "ro.product.fingerprintName",
		.value = "HUAWEI-Z114",
	},
	{
		.key = "ro.product.first_api_level",
		.value = "28",
	},
	{
		.key = "ro.product.hardwareversion",
		.value = "HL1HIMAM",
	},
	{
		.key = "ro.product.imeisv",
		.value = "02",
	},
	{
		.key = "ro.product.locale",
		.value = "es-AR",
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
		.value = "HMA-L29",
	},
	{
		.key = "ro.product.name",
		.value = "HMA-L29",
	},
	{
		.key = "ro.product.odm.brand",
		.value = "Huawei",
	},
	{
		.key = "ro.product.odm.device",
		.value = "Atlanta",
	},
	{
		.key = "ro.product.odm.name",
		.value = "Atlanta",
	},
	{
		.key = "ro.product.vendor.brand",
		.value = "kirin980",
	},
	{
		.key = "ro.product.vendor.device",
		.value = "kirin980",
	},
	{
		.key = "ro.product.vendor.manufacturer",
		.value = "HUAWEI",
	},
	{
		.key = "ro.product.vendor.model",
		.value = "kirin980",
	},
	{
		.key = "ro.product.vendor.name",
		.value = "kirin980",
	},
	{
		.key = "ro.prop.hwkeychain_switch",
		.value = "true",
	},
	{
		.key = "ro.property_service.version",
		.value = "2",
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
		.value = "MUN0218A29000006",
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
		.key = "ro.sf.enable_backpressure_opt",
		.value = "0",
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
		.key = "ro.syssvccallrecord.enable",
		.value = "true",
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
		.key = "ro.vendor.build.date",
		.value = "Thu Sep 13 21:57:23 CST 2018",
	},
	{
		.key = "ro.vendor.build.date.utc",
		.value = "1536847043",
	},
	{
		.key = "ro.vendor.build.fingerprint",
		.value = "kirin980/kirin980/kirin980:9/PPR1.180610.011/jenkins09132155:user/release-keys",
	},
	{
		.key = "ro.vendor.build.security_patch",
		.value = "2018-06-19",
	},
	{
		.key = "ro.vndk.version",
		.value = "28",
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
		.key = "selinux.restorecon_recursive",
		.value = "/data/misc_ce/0",
	},
	{
		.key = "service.bootanim.exit",
		.value = "1",
	},
	{
		.key = "service.bootanim.stop",
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
		.key = "sys.aps.maxFlingVelocity",
		.value = "75",
	},
	{
		.key = "sys.aps.support",
		.value = "34177027",
	},
	{
		.key = "sys.aps.version",
		.value = "5.1.2-9.0.0.14",
	},
	{
		.key = "sys.boot.reason",
		.value = "shutdown,",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
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
		.value = "2-3,5",
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
		.value = "67",
	},
	{
		.key = "sys.iaware.switch_set_success",
		.value = "true",
	},
	{
		.key = "sys.iaware.type",
		.value = "255",
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
		.key = "sys.rescue_boot_count",
		.value = "1",
	},
	{
		.key = "sys.resettype",
		.value = "others:BR_POWERON_CHARGE",
	},
	{
		.key = "sys.retaildemo.enabled",
		.value = "0",
	},
	{
		.key = "sys.runtime_data.hiddenapi.enable",
		.value = "true",
	},
	{
		.key = "sys.settingsprovider_ready",
		.value = "1",
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
		.value = "28400",
	},
	{
		.key = "sys.sysctl.tcp_def_init_rwnd",
		.value = "60",
	},
	{
		.key = "sys.thermal.vr_fps",
		.value = "0",
	},
	{
		.key = "sys.thermal.vr_ratio_height",
		.value = "0",
	},
	{
		.key = "sys.thermal.vr_ratio_width",
		.value = "0",
	},
	{
		.key = "sys.thermal.vr_warning_level",
		.value = "0",
	},
	{
		.key = "sys.uidcpupower",
		.value = "",
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
		.key = "sys.usb.ffs.ready",
		.value = "1",
	},
	{
		.key = "sys.usb.ffs_hdb.ready",
		.value = "0",
	},
	{
		.key = "sys.usb.mtp.device_type",
		.value = "3",
	},
	{
		.key = "sys.usb.state",
		.value = "hisuite,mtp,mass_storage,adb",
	},
	{
		.key = "sys.user.0.ce_available",
		.value = "true",
	},
	{
		.key = "system_init.hwextdeviceservice",
		.value = "1",
	},
	{
		.key = "vold.crypto_unencrypt_updatedir",
		.value = "/data/update",
	},
	{
		.key = "vold.cryptsd.keystate",
		.value = "lock",
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
		.key = "vold.has_reserved",
		.value = "1",
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
