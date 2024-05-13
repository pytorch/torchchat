struct cpuinfo_mock_file filesystem[] = {
#if CPUINFO_ARCH_ARM64
	{
		.path = "/proc/cpuinfo",
		.size = 1030,
		.content =
			"processor\t: 0\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 3\n"
			"\n"
			"processor\t: 1\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 3\n"
			"\n"
			"processor\t: 2\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 3\n"
			"\n"
			"processor\t: 3\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 3\n"
			"\n"
			"processor\t: 4\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd07\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 5\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd07\n"
			"CPU revision\t: 2\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc MSM8992\n",
	},
#elif CPUINFO_ARCH_ARM
	{
		.path = "/proc/cpuinfo",
		.size = 1348,
		.content =
			"processor\t: 0\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 3\n"
			"\n"
			"processor\t: 1\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 3\n"
			"\n"
			"processor\t: 2\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 3\n"
			"\n"
			"processor\t: 3\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 3\n"
			"\n"
			"processor\t: 4\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd07\n"
			"CPU revision\t: 2\n"
			"\n"
			"processor\t: 5\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x1\n"
			"CPU part\t: 0xd07\n"
			"CPU revision\t: 2\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc MSM8992\n",
	},
#endif
	{
		.path = "/sys/class/kgsl/kgsl-3d0/default_pwrlevel",
		.size = 2,
		.content = "5\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/max_gpuclk",
		.size = 10,
		.content = "600000000\n",
	},
	{
		.path = "/sys/devices/soc0/accessory_chip",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/soc0/build_id",
		.size = 25,
		.content = "8992A-ESAAANAZA-40000000\n",
	},
	{
		.path = "/sys/devices/soc0/foundry_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/soc0/hw_platform",
		.size = 7,
		.content = "(null)\n",
	},
	{
		.path = "/sys/devices/soc0/image_crm_version",
		.size = 12,
		.content = "LGEARND7B14\n",
	},
	{
		.path = "/sys/devices/soc0/image_variant",
		.size = 10,
		.content = "ESAAANAZA\n",
	},
	{
		.path = "/sys/devices/soc0/image_version",
		.size = 21,
		.content = "00:BOOT.BF.2.3-00366\n",
	},
	{
		.path = "/sys/devices/soc0/machine",
		.size = 11,
		.content = "Snapdragon\n",
	},
	{
		.path = "/sys/devices/soc0/platform_subtype",
		.size = 8,
		.content = "Unknown\n",
	},
	{
		.path = "/sys/devices/soc0/platform_subtype_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/soc0/platform_version",
		.size = 3,
		.content = "11\n",
	},
	{
		.path = "/sys/devices/soc0/pmic_die_revision",
		.size = 7,
		.content = "131072\n",
	},
	{
		.path = "/sys/devices/soc0/pmic_model",
		.size = 6,
		.content = "65545\n",
	},
	{
		.path = "/sys/devices/soc0/raw_id",
		.size = 5,
		.content = "2409\n",
	},
	{
		.path = "/sys/devices/soc0/raw_version",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/soc0/revision",
		.size = 4,
		.content = "1.0\n",
	},
	{
		.path = "/sys/devices/soc0/select_image",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/soc0/soc_id",
		.size = 4,
		.content = "251\n",
	},
	{
		.path = "/sys/devices/soc0/vendor",
		.size = 9,
		.content = "Qualcomm\n",
	},
	{
		.path = "/sys/devices/system/cpu/kernel_max",
		.size = 2,
		.content = "5\n",
	},
	{
		.path = "/sys/devices/system/cpu/modalias",
		.size = 66,
		.content = "cpu:type:aarch64:feature:,0000,0001,0002,0003,0004,0005,0006,0007\n",
	},
	{
		.path = "/sys/devices/system/cpu/offline",
		.size = 4,
		.content = "4-5\n",
	},
	{
		.path = "/sys/devices/system/cpu/online",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/possible",
		.size = 4,
		.content = "0-5\n",
	},
	{
		.path = "/sys/devices/system/cpu/present",
		.size = 4,
		.content = "0-5\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/all_time_in_state",
		.size = 43,
		.content = "freq\t\tcpu0\t\tcpu1\t\tcpu2\t\tcpu3\t\tcpu4\t\tcpu5\t\t\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/current_in_state",
		.size = 882,
		.content =
			"CPU0:384000=18250 460800=24330 600000=26920 672000=34600 787200=38150 864000=46880 960000=55940 1248000=81740 1440000=105870 \n"
			"CPU1:384000=7665 460800=10219 600000=11306 672000=14532 787200=16023 864000=19690 960000=23495 1248000=34331 1440000=44465 \n"
			"CPU2:384000=7848 460800=10462 600000=11576 672000=14878 787200=16405 864000=20158 960000=24054 1248000=35148 1440000=45524 \n"
			"CPU3:384000=8760 460800=11678 600000=12922 672000=16608 787200=18312 864000=22502 960000=26851 1248000=39235 1440000=50818 \n"
			"CPU4:384000=67740 480000=82960 633600=105870 768000=133160 864000=150160 960000=167180 1248000=230040 1344000=261430 1440000=290460 1536000=317200 1632000=352870 1689600=374360 1824000=443880 \n"
			"CPU5:384000=49450 480000=60561 633600=77285 768000=97207 864000=109617 960000=122041 1248000=167929 1344000=190844 1440000=212036 1536000=231556 1632000=257595 1689600=273283 1824000=324032 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_driver",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_governor_ro",
		.size = 5,
		.content = "null\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1440000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 66,
		.content = "384000 460800 600000 672000 787200 864000 960000 1248000 1440000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 54,
		.content = "interactive ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1440000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 95,
		.content =
			"384000 192\n"
			"460800 44\n"
			"600000 18\n"
			"672000 5\n"
			"787200 95\n"
			"864000 42\n"
			"960000 103\n"
			"1248000 21\n"
			"1440000 4796\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 3,
		.content = "56\n",
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
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1440000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_frequencies",
		.size = 66,
		.content = "384000 460800 600000 672000 787200 864000 960000 1248000 1440000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 54,
		.content = "interactive ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 95,
		.content =
			"384000 283\n"
			"460800 51\n"
			"600000 25\n"
			"672000 5\n"
			"787200 99\n"
			"864000 42\n"
			"960000 127\n"
			"1248000 23\n"
			"1440000 4899\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 3,
		.content = "72\n",
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
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1440000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies",
		.size = 66,
		.content = "384000 460800 600000 672000 787200 864000 960000 1248000 1440000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 54,
		.content = "interactive ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 95,
		.content =
			"384000 492\n"
			"460800 59\n"
			"600000 34\n"
			"672000 5\n"
			"787200 99\n"
			"864000 42\n"
			"960000 142\n"
			"1248000 32\n"
			"1440000 4899\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 3,
		.content = "86\n",
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
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1440000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 66,
		.content = "384000 460800 600000 672000 787200 864000 960000 1248000 1440000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 54,
		.content = "interactive ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "384000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 95,
		.content =
			"384000 703\n"
			"460800 76\n"
			"600000 36\n"
			"672000 5\n"
			"787200 99\n"
			"864000 42\n"
			"960000 157\n"
			"1248000 40\n"
			"1440000 4922\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "102\n",
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
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{ NULL },
};
#ifdef __ANDROID__
struct cpuinfo_mock_property properties[] = {
	{
		.key = "af.fast_track_multiplier",
		.value = "1",
	},
	{
		.key = "audio_hal.period_size",
		.value = "192",
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
		.value = "192m",
	},
	{
		.key = "dalvik.vm.heapmaxfree",
		.value = "8m",
	},
	{
		.key = "dalvik.vm.heapminfree",
		.value = "512k",
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
		.value = "cortex-a53.a57",
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
		.key = "debug.atrace.tags.enableflags",
		.value = "0",
	},
	{
		.key = "debug.force_rtl",
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
		.key = "gsm.current.phone-type",
		.value = "2",
	},
	{
		.key = "gsm.network.type",
		.value = "Unknown",
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
		.value = "false",
	},
	{
		.key = "gsm.operator.numeric",
		.value = "00000",
	},
	{
		.key = "gsm.sim.operator.alpha",
		.value = "",
	},
	{
		.key = "gsm.sim.operator.iso-country",
		.value = "",
	},
	{
		.key = "gsm.sim.operator.numeric",
		.value = "",
	},
	{
		.key = "gsm.sim.state",
		.value = "ABSENT",
	},
	{
		.key = "gsm.version.baseband",
		.value = "M8994F-2.6.39.3.03",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "Qualcomm RIL 1.0",
	},
	{
		.key = "hwservicemanager.ready",
		.value = "true",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.atfwd",
		.value = "running",
	},
	{
		.key = "init.svc.audioserver",
		.value = "running",
	},
	{
		.key = "init.svc.bootanim",
		.value = "stopped",
	},
	{
		.key = "init.svc.bullhead-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.cameraserver",
		.value = "running",
	},
	{
		.key = "init.svc.cnd",
		.value = "running",
	},
	{
		.key = "init.svc.cnss-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.configstore-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.devstart_sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.drm",
		.value = "running",
	},
	{
		.key = "init.svc.dumpstate-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.flash-nanohub-fw",
		.value = "stopped",
	},
	{
		.key = "init.svc.flash_recovery",
		.value = "stopped",
	},
	{
		.key = "init.svc.fps_hal",
		.value = "running",
	},
	{
		.key = "init.svc.gatekeeperd",
		.value = "running",
	},
	{
		.key = "init.svc.gralloc-2-0",
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
		.key = "init.svc.hwservicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.imsdatadaemon",
		.value = "running",
	},
	{
		.key = "init.svc.imsqmidaemon",
		.value = "running",
	},
	{
		.key = "init.svc.installd",
		.value = "running",
	},
	{
		.key = "init.svc.irsc_util",
		.value = "stopped",
	},
	{
		.key = "init.svc.keystore",
		.value = "running",
	},
	{
		.key = "init.svc.lmkd",
		.value = "running",
	},
	{
		.key = "init.svc.loc_launcher",
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
		.key = "init.svc.msm_irqbalance",
		.value = "running",
	},
	{
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.netmgrd",
		.value = "running",
	},
	{
		.key = "init.svc.per_mgr",
		.value = "running",
	},
	{
		.key = "init.svc.per_proxy",
		.value = "running",
	},
	{
		.key = "init.svc.perfd",
		.value = "running",
	},
	{
		.key = "init.svc.qcamerasvr",
		.value = "running",
	},
	{
		.key = "init.svc.qmuxd",
		.value = "running",
	},
	{
		.key = "init.svc.qseecomd",
		.value = "running",
	},
	{
		.key = "init.svc.qti",
		.value = "running",
	},
	{
		.key = "init.svc.ril-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.rmt_storage",
		.value = "running",
	},
	{
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.start_hci_filter",
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
		.key = "init.svc.thermal-engine",
		.value = "running",
	},
	{
		.key = "init.svc.time_daemon",
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
		.key = "init.svc.usb-hal-1-0",
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
		.key = "media.aac_51_output_enabled",
		.value = "true",
	},
	{
		.key = "media.recorder.show_manufacturer_and_model",
		.value = "true",
	},
	{
		.key = "net.bt.name",
		.value = "Android",
	},
	{
		.key = "net.lte.ims.data.enabled",
		.value = "true",
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
		.key = "partition.system.verified",
		.value = "2",
	},
	{
		.key = "partition.vendor.verified",
		.value = "2",
	},
	{
		.key = "persist.audio.fluence.speaker",
		.value = "true",
	},
	{
		.key = "persist.audio.fluence.voicecall",
		.value = "true",
	},
	{
		.key = "persist.audio.fluence.voicecomm",
		.value = "true",
	},
	{
		.key = "persist.audio.fluence.voicerec",
		.value = "false",
	},
	{
		.key = "persist.camera.tnr.preview",
		.value = "0",
	},
	{
		.key = "persist.camera.tnr.video",
		.value = "0",
	},
	{
		.key = "persist.data.iwlan.enable",
		.value = "true",
	},
	{
		.key = "persist.hwc.mdpcomp.enable",
		.value = "true",
	},
	{
		.key = "persist.media.treble_omx",
		.value = "false",
	},
	{
		.key = "persist.qcril.disable_retry",
		.value = "true",
	},
	{
		.key = "persist.radio.adb_log_on",
		.value = "0",
	},
	{
		.key = "persist.radio.always_send_plmn",
		.value = "true",
	},
	{
		.key = "persist.radio.apm_sim_not_pwdn",
		.value = "1",
	},
	{
		.key = "persist.radio.custom_ecc",
		.value = "1",
	},
	{
		.key = "persist.radio.data_con_rprt",
		.value = "true",
	},
	{
		.key = "persist.radio.data_no_toggle",
		.value = "1",
	},
	{
		.key = "persist.radio.eons.enabled",
		.value = "false",
	},
	{
		.key = "persist.radio.eri64_as_home",
		.value = "1",
	},
	{
		.key = "persist.radio.mode_pref_nv10",
		.value = "1",
	},
	{
		.key = "persist.radio.nitz_lons_0_0",
		.value = "AT&T",
	},
	{
		.key = "persist.radio.nitz_lons_1_0",
		.value = "",
	},
	{
		.key = "persist.radio.nitz_lons_2_0",
		.value = "",
	},
	{
		.key = "persist.radio.nitz_lons_3_0",
		.value = "",
	},
	{
		.key = "persist.radio.nitz_plmn_0",
		.value = "310 410",
	},
	{
		.key = "persist.radio.nitz_sons_0_0",
		.value = "AT&T",
	},
	{
		.key = "persist.radio.nitz_sons_1_0",
		.value = "",
	},
	{
		.key = "persist.radio.nitz_sons_2_0",
		.value = "",
	},
	{
		.key = "persist.radio.nitz_sons_3_0",
		.value = "",
	},
	{
		.key = "persist.radio.process_sups_ind",
		.value = "1",
	},
	{
		.key = "persist.radio.redir_party_num",
		.value = "0",
	},
	{
		.key = "persist.radio.ril_payload_on",
		.value = "0",
	},
	{
		.key = "persist.radio.snapshot_enabled",
		.value = "1",
	},
	{
		.key = "persist.radio.snapshot_timer",
		.value = "10",
	},
	{
		.key = "persist.radio.use_cc_names",
		.value = "true",
	},
	{
		.key = "persist.speaker.prot.enable",
		.value = "true",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.gps.lpp",
		.value = "",
	},
	{
		.key = "persist.sys.locale",
		.value = "en-US",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
	},
	{
		.key = "persist.sys.timezone",
		.value = "America/Los_Angeles",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "adb",
	},
	{
		.key = "persist.sys.webview.vmsize",
		.value = "112887376",
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
		.key = "qcom.bluetooth.soc",
		.value = "rome",
	},
	{
		.key = "ril.ecclist",
		.value = "911,*911,#911,112,000,08,110,999,118,119,111,113,117,122,125",
	},
	{
		.key = "ril.nosim.ecc_list_1",
		.value = "111,113,117,122,125",
	},
	{
		.key = "ril.nosim.ecc_list_count",
		.value = "1",
	},
	{
		.key = "ril.qcril_pre_init_lock_held",
		.value = "0",
	},
	{
		.key = "rild.libpath",
		.value = "/vendor/lib64/libril-qc-qmi-1.so",
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
		.key = "ro.atrace.core.services",
		.value = "com.google.android.gms,com.google.android.gms.ui,com.google.android.gms.persistent",
	},
	{
		.key = "ro.audio.flinger_standbytime_ms",
		.value = "300",
	},
	{
		.key = "ro.baseband",
		.value = "msm",
	},
	{
		.key = "ro.board.platform",
		.value = "msm8992",
	},
	{
		.key = "ro.boot.authorized_kernel",
		.value = "true",
	},
	{
		.key = "ro.boot.baseband",
		.value = "msm",
	},
	{
		.key = "ro.boot.bootloader",
		.value = "BHZ21c",
	},
	{
		.key = "ro.boot.bootreason",
		.value = "oem_powerkey",
	},
	{
		.key = "ro.boot.dlcomplete",
		.value = "0",
	},
	{
		.key = "ro.boot.emmc",
		.value = "true",
	},
	{
		.key = "ro.boot.flash.locked",
		.value = "1",
	},
	{
		.key = "ro.boot.hardware",
		.value = "bullhead",
	},
	{
		.key = "ro.boot.hardware.sku",
		.value = "LGH790",
	},
	{
		.key = "ro.boot.revision",
		.value = "rev_1.0",
	},
	{
		.key = "ro.boot.serialno",
		.value = "0105b6567ac4ced5",
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
		.key = "ro.boot.wificountrycode",
		.value = "US",
	},
	{
		.key = "ro.bootimage.build.date",
		.value = "Mon Aug 28 18:57:48 UTC 2017",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1503946668",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "google/bullhead/bullhead:8.0.0/OPR4.170623.009/4302492:user/release-keys",
	},
	{
		.key = "ro.bootloader",
		.value = "BHZ21c",
	},
	{
		.key = "ro.bootmode",
		.value = "unknown",
	},
	{
		.key = "ro.build.characteristics",
		.value = "nosdcard",
	},
	{
		.key = "ro.build.date",
		.value = "Mon Aug 28 18:57:48 UTC 2017",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1503946668",
	},
	{
		.key = "ro.build.description",
		.value = "bullhead-user 8.0.0 OPR4.170623.009 4302492 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "OPR4.170623.009",
	},
	{
		.key = "ro.build.expect.baseband",
		.value = "M8994F-2.6.39.3.03",
	},
	{
		.key = "ro.build.expect.bootloader",
		.value = "BHZ21c",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "google/bullhead/bullhead:8.0.0/OPR4.170623.009/4302492:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "bullhead-user",
	},
	{
		.key = "ro.build.host",
		.value = "wpiu13.hot.corp.google.com",
	},
	{
		.key = "ro.build.id",
		.value = "OPR4.170623.009",
	},
	{
		.key = "ro.build.product",
		.value = "bullhead",
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
		.value = "android-build",
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
		.value = "4302492",
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
		.key = "ro.build.version.security_patch",
		.value = "2017-10-05",
	},
	{
		.key = "ro.camera.notify_nfc",
		.value = "1",
	},
	{
		.key = "ro.carrier",
		.value = "unknown",
	},
	{
		.key = "ro.com.android.dataroaming",
		.value = "false",
	},
	{
		.key = "ro.com.android.prov_mobiledata",
		.value = "false",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-google",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Oxygen.ogg",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "Tethys.ogg",
	},
	{
		.key = "ro.config.ringtone",
		.value = "Titania.ogg",
	},
	{
		.key = "ro.config.vc_call_vol_steps",
		.value = "7",
	},
	{
		.key = "ro.control_privapp_permissions",
		.value = "enforce",
	},
	{
		.key = "ro.crypto.fs_crypto_blkdev",
		.value = "/dev/block/dm-2",
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
		.key = "ro.error.receiver.system.apps",
		.value = "com.google.android.gms",
	},
	{
		.key = "ro.expect.recovery_id",
		.value = "0xd24d616d72d2e4db78f2d45a9fddcdb17d50e3fd000000000000000000000000",
	},
	{
		.key = "ro.facelock.black_timeout",
		.value = "700",
	},
	{
		.key = "ro.facelock.det_timeout",
		.value = "2500",
	},
	{
		.key = "ro.facelock.est_max_time",
		.value = "600",
	},
	{
		.key = "ro.facelock.rec_timeout",
		.value = "3500",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/platform/soc.0/f9824900.sdhci/by-name/persistent",
	},
	{
		.key = "ro.hardware",
		.value = "bullhead",
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
		.value = "32",
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
		.value = "56",
	},
	{
		.key = "ro.min_freq_0",
		.value = "384000",
	},
	{
		.key = "ro.min_freq_4",
		.value = "384000",
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
		.key = "ro.product.board",
		.value = "bullhead",
	},
	{
		.key = "ro.product.brand",
		.value = "google",
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
		.value = "bullhead",
	},
	{
		.key = "ro.product.first_api_level",
		.value = "23",
	},
	{
		.key = "ro.product.locale",
		.value = "en-US",
	},
	{
		.key = "ro.product.manufacturer",
		.value = "LGE",
	},
	{
		.key = "ro.product.model",
		.value = "Nexus 5X",
	},
	{
		.key = "ro.product.name",
		.value = "bullhead",
	},
	{
		.key = "ro.property_service.version",
		.value = "2",
	},
	{
		.key = "ro.qc.sdk.audio.fluencetype",
		.value = "fluencepro",
	},
	{
		.key = "ro.recovery_id",
		.value = "0x952a5168adfd47b78c6ee8f3ea86ebba59d843cb000000000000000000000000",
	},
	{
		.key = "ro.retaildemo.video_path",
		.value = "/data/preloads/demo/retail_demo.mp4",
	},
	{
		.key = "ro.revision",
		.value = "rev_1.0",
	},
	{
		.key = "ro.ril.svdo",
		.value = "false",
	},
	{
		.key = "ro.ril.svlte1x",
		.value = "false",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "0105b6567ac4ced5",
	},
	{
		.key = "ro.setupwizard.enterprise_mode",
		.value = "1",
	},
	{
		.key = "ro.setupwizard.rotation_locked",
		.value = "true",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "420",
	},
	{
		.key = "ro.telephony.call_ring.multiple",
		.value = "0",
	},
	{
		.key = "ro.telephony.default_cdma_sub",
		.value = "0",
	},
	{
		.key = "ro.telephony.default_network",
		.value = "22",
	},
	{
		.key = "ro.treble.enabled",
		.value = "false",
	},
	{
		.key = "ro.url.legal",
		.value = "http://www.google.com/intl/%s/mobile/android/basic/phone-legal.html",
	},
	{
		.key = "ro.url.legal.android_privacy",
		.value = "http://www.google.com/intl/%s/mobile/android/basic/privacy.html",
	},
	{
		.key = "ro.vendor.build.date",
		.value = "Mon Aug 28 18:57:48 UTC 2017",
	},
	{
		.key = "ro.vendor.build.date.utc",
		.value = "1503946668",
	},
	{
		.key = "ro.vendor.build.fingerprint",
		.value = "google/bullhead/bullhead:8.0.0/OPR4.170623.009/4302492:user/release-keys",
	},
	{
		.key = "ro.vendor.extension_library",
		.value = "libqti-perfd-client.so",
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
		.value = "0",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
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
		.key = "sys.qcom.devup",
		.value = "1",
	},
	{
		.key = "sys.rescue_boot_count",
		.value = "1",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "24300",
	},
	{
		.key = "sys.sysctl.tcp_def_init_rwnd",
		.value = "60",
	},
	{
		.key = "sys.usb.config",
		.value = "adb",
	},
	{
		.key = "sys.usb.configfs",
		.value = "0",
	},
	{
		.key = "sys.usb.controller",
		.value = "f9200000.dwc3",
	},
	{
		.key = "sys.usb.ffs.max_read",
		.value = "262144",
	},
	{
		.key = "sys.usb.ffs.max_write",
		.value = "262144",
	},
	{
		.key = "sys.usb.ffs.ready",
		.value = "1",
	},
	{
		.key = "sys.usb.mtp.device_type",
		.value = "3",
	},
	{
		.key = "sys.usb.state",
		.value = "adb",
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
		.key = "vidc.debug.perf.mode",
		.value = "2",
	},
	{
		.key = "vidc.enc.dcvs.extra-buff-count",
		.value = "2",
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
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "wifi.supplicant_scan_interval",
		.value = "15",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
