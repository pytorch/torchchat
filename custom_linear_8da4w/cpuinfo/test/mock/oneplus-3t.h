struct cpuinfo_mock_file filesystem[] = {
#if CPUINFO_ARCH_ARM64
	{
		.path = "/proc/cpuinfo",
		.size = 773,
		.content =
			"processor\t: 0\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x2\n"
			"CPU part\t: 0x201\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x2\n"
			"CPU part\t: 0x201\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x2\n"
			"CPU part\t: 0x205\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x2\n"
			"CPU part\t: 0x205\n"
			"CPU revision\t: 1\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc MSM8996pro\n",
	},
#elif CPUINFO_ARCH_ARM
	{
		.path = "/proc/cpuinfo",
		.size = 1005,
		.content =
			"processor\t: 0\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x2\n"
			"CPU part\t: 0x201\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x2\n"
			"CPU part\t: 0x201\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x2\n"
			"CPU part\t: 0x205\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x2\n"
			"CPU part\t: 0x205\n"
			"CPU revision\t: 1\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc MSM8996pro\n",
	},
#endif
	{
		.path = "/sys/devices/soc0/accessory_chip",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/soc0/build_id",
		.size = 1,
		.content = "\n",
	},
	{
		.path = "/sys/devices/soc0/family",
		.size = 11,
		.content = "Snapdragon\n",
	},
	{
		.path = "/sys/devices/soc0/foundry_id",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/soc0/hw_platform",
		.size = 4,
		.content = "MTP\n",
	},
	{
		.path = "/sys/devices/soc0/image_crm_version",
		.size = 5,
		.content =
			"REL\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/image_variant",
		.size = 15,
		.content =
			"OnePlus3-user\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/image_version",
		.size = 22,
		.content =
			"10:OPR6.170623.013:5\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/images",
		.size = 580,
		.content =
			"0:\n"
			"\tCRM:\t\t00:BOOT.XF.1.0-00316\n"
			"\tVariant:\tM8996LAB\n"
			"\tVersion:\tubuntu-142\n"
			"\n"
			"1:\n"
			"\tCRM:\t\t01:TZ.BF.4.0.1-123590\n"
			"\tVariant:\t\n"
			"\tVersion:\tCRM\n"
			"\n"
			"3:\n"
			"\tCRM:\t\t03:RPM.BF.1.6-00153\n"
			"\tVariant:\tAAAAANAAR\n"
			"\tVersion:\tubuntu-142\n"
			"\n"
			"10:\n"
			"\tCRM:\t\t10:OPR6.170623.013:5\n"
			"\n"
			"\tVariant:\tOnePlus3-user\n"
			"\n"
			"\tVersion:\tREL\n"
			"\n"
			"\n"
			"11:\n"
			"\tCRM:\t\t11:MPSS.TH.2.0.C1.9-119765\n"
			"\tVariant:\t8996.gen.prodQ\n"
			"\tVersion:\tubuntu-142\n"
			"\n"
			"12:\n"
			"\tCRM:\t\t12:ADSP.8996.2.7.1.C3-00009\n"
			"\tVariant:\tAAAAAAAAQ\n"
			"\tVersion:\tubuntu-142\n"
			"\n"
			"14:\n"
			"\tCRM:\t\t14:VIDEO.VE.4.2-00057\n"
			"\tVariant:\tPROD\n"
			"\tVersion:\t:CRM\n"
			"\n"
			"15:\n"
			"\tCRM:\t\t15:SLPI.HB.1.0-00311\n"
			"\tVariant:\tAAAAAAAAQ\n"
			"\tVersion:\tubuntu-142\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/machine",
		.size = 11,
		.content = "MSM8996pro\n",
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
		.size = 6,
		.content = "65536\n",
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
		.size = 3,
		.content = "95\n",
	},
	{
		.path = "/sys/devices/soc0/raw_version",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/soc0/revision",
		.size = 4,
		.content = "1.1\n",
	},
	{
		.path = "/sys/devices/soc0/select_image",
		.size = 3,
		.content = "10\n",
	},
	{
		.path = "/sys/devices/soc0/serial_number",
		.size = 11,
		.content = "3055758979\n",
	},
	{
		.path = "/sys/devices/soc0/soc_id",
		.size = 4,
		.content = "305\n",
	},
	{
		.path = "/sys/devices/soc0/vendor",
		.size = 9,
		.content = "Qualcomm\n",
	},
	{
		.path = "/sys/devices/system/cpu/kernel_max",
		.size = 2,
		.content = "3\n",
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
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/possible",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/present",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/all_time_in_state",
		.size = 31,
		.content = "freq\t\tcpu0\t\tcpu1\t\tcpu2\t\tcpu3\t\t\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/current_in_state",
		.size = 532,
		.content =
			"CPU2:307200=0 384000=0 460800=0 537600=0 614400=0 691200=0 748800=0 825600=0 902400=0 979200=0 1056000=0 1132800=0 1209600=0 1286400=0 1363200=0 1440000=0 1516800=0 1593600=0 1670400=0 1747200=0 1824000=0 1900800=0 1977600=0 2054400=0 2150400=0 2246400=0 2342400=0 \n"
			"CPU3:307200=0 384000=0 460800=0 537600=0 614400=0 691200=0 748800=0 825600=0 902400=0 979200=0 1056000=0 1132800=0 1209600=0 1286400=0 1363200=0 1440000=0 1516800=0 1593600=0 1670400=0 1747200=0 1824000=0 1900800=0 1977600=0 2054400=0 2150400=0 2246400=0 2342400=0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_driver",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_governor_ro",
		.size = 5,
		.content = "qcom\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/affected_cpus",
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2188800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "307200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 143,
		.content = "307200 384000 460800 537600 614400 691200 768000 844800 902400 979200 1056000 1132800 1209600 1286400 1363200 1440000 1516800 1593600 2188800 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1286400\n",
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
		.content = "307200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 204,
		.content =
			"307200 1791\n"
			"384000 60\n"
			"460800 25\n"
			"537600 18\n"
			"614400 15\n"
			"691200 10\n"
			"768000 17\n"
			"844800 23\n"
			"902400 2\n"
			"979200 221\n"
			"1056000 19\n"
			"1132800 14\n"
			"1209600 13\n"
			"1286400 28\n"
			"1363200 55\n"
			"1440000 12\n"
			"1516800 11\n"
			"1593600 3217\n"
			"2188800 568\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "408\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings_list",
		.size = 4,
		.content = "0-1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/thread_siblings",
		.size = 2,
		.content = "1\n",
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
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2188800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "307200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/related_cpus",
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_frequencies",
		.size = 143,
		.content = "307200 384000 460800 537600 614400 691200 768000 844800 902400 979200 1056000 1132800 1209600 1286400 1363200 1440000 1516800 1593600 2188800 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "307200\n",
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
		.content = "307200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 204,
		.content =
			"307200 1955\n"
			"384000 70\n"
			"460800 33\n"
			"537600 26\n"
			"614400 20\n"
			"691200 10\n"
			"768000 17\n"
			"844800 27\n"
			"902400 7\n"
			"979200 245\n"
			"1056000 19\n"
			"1132800 22\n"
			"1209600 13\n"
			"1286400 28\n"
			"1363200 60\n"
			"1440000 14\n"
			"1516800 13\n"
			"1593600 3240\n"
			"2188800 568\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 4,
		.content = "458\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_siblings",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_siblings_list",
		.size = 4,
		.content = "0-1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/thread_siblings",
		.size = 2,
		.content = "2\n",
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
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2342400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "307200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/related_cpus",
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies",
		.size = 207,
		.content = "307200 384000 460800 537600 614400 691200 748800 825600 902400 979200 1056000 1132800 1209600 1286400 1363200 1440000 1516800 1593600 1670400 1747200 1824000 1900800 1977600 2054400 2150400 2246400 2342400 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "307200\n",
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
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2342400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "307200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 285,
		.content =
			"307200 1945\n"
			"384000 40\n"
			"460800 23\n"
			"537600 23\n"
			"614400 25\n"
			"691200 21\n"
			"748800 9\n"
			"825600 16\n"
			"902400 18\n"
			"979200 17\n"
			"1056000 22\n"
			"1132800 31\n"
			"1209600 18\n"
			"1286400 321\n"
			"1363200 43\n"
			"1440000 14\n"
			"1516800 50\n"
			"1593600 11\n"
			"1670400 4\n"
			"1747200 4\n"
			"1824000 0\n"
			"1900800 0\n"
			"1977600 10\n"
			"2054400 0\n"
			"2150400 90\n"
			"2246400 6\n"
			"2342400 3885\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 4,
		.content = "514\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_siblings",
		.size = 2,
		.content = "c\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_siblings_list",
		.size = 4,
		.content = "2-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/physical_package_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/thread_siblings",
		.size = 2,
		.content = "4\n",
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
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2342400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "307200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/related_cpus",
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 207,
		.content = "307200 384000 460800 537600 614400 691200 748800 825600 902400 979200 1056000 1132800 1209600 1286400 1363200 1440000 1516800 1593600 1670400 1747200 1824000 1900800 1977600 2054400 2150400 2246400 2342400 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "307200\n",
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
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2342400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "307200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 285,
		.content =
			"307200 2186\n"
			"384000 42\n"
			"460800 23\n"
			"537600 25\n"
			"614400 25\n"
			"691200 21\n"
			"748800 9\n"
			"825600 16\n"
			"902400 18\n"
			"979200 17\n"
			"1056000 23\n"
			"1132800 31\n"
			"1209600 18\n"
			"1286400 329\n"
			"1363200 43\n"
			"1440000 14\n"
			"1516800 50\n"
			"1593600 11\n"
			"1670400 4\n"
			"1747200 4\n"
			"1824000 0\n"
			"1900800 0\n"
			"1977600 10\n"
			"2054400 0\n"
			"2150400 90\n"
			"2246400 6\n"
			"2342400 3889\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "520\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_siblings",
		.size = 2,
		.content = "c\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_siblings_list",
		.size = 4,
		.content = "2-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/physical_package_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/thread_siblings",
		.size = 2,
		.content = "8\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/thread_siblings_list",
		.size = 2,
		.content = "3\n",
	},
	{ NULL },
};
#ifdef __ANDROID__
struct cpuinfo_mock_property properties[] = {
	{
		.key = "Camera.no_navigation_bar",
		.value = "true",
	},
	{
		.key = "DEVICE_PROVISIONED",
		.value = "1",
	},
	{
		.key = "audio.deep_buffer.media",
		.value = "true",
	},
	{
		.key = "audio.offload.min.duration.secs",
		.value = "30",
	},
	{
		.key = "audio.offload.multiple.enabled",
		.value = "true",
	},
	{
		.key = "audio.offload.pcm.16bit.enable",
		.value = "false",
	},
	{
		.key = "audio.offload.video",
		.value = "true",
	},
	{
		.key = "audio.parser.ip.buffer.size",
		.value = "0",
	},
	{
		.key = "bt.max.hfpclient.connections",
		.value = "1",
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
		.value = "cortex-a9",
	},
	{
		.key = "dalvik.vm.isa.arm64.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.arm64.variant",
		.value = "kryo",
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
		.key = "debug.egl.hw",
		.value = "1",
	},
	{
		.key = "debug.enable.gamed",
		.value = "0",
	},
	{
		.key = "debug.force_rtl",
		.value = "0",
	},
	{
		.key = "debug.gralloc.enable_fb_ubwc",
		.value = "1",
	},
	{
		.key = "debug.gralloc.gfx_ubwc_disable",
		.value = "0",
	},
	{
		.key = "debug.sf.dump",
		.value = "0",
	},
	{
		.key = "debug.sf.dump.enable",
		.value = "true",
	},
	{
		.key = "debug.sf.dump.external",
		.value = "true",
	},
	{
		.key = "debug.sf.dump.primary",
		.value = "true",
	},
	{
		.key = "debug.sf.enable_hwc_vds",
		.value = "1",
	},
	{
		.key = "debug.sf.hw",
		.value = "1",
	},
	{
		.key = "debug.sf.latch_unsignaled",
		.value = "1",
	},
	{
		.key = "dev.bootcomplete",
		.value = "1",
	},
	{
		.key = "dev.pm.dyn_samplingrate",
		.value = "1",
	},
	{
		.key = "drm.service.enabled",
		.value = "true",
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
		.value = "MPSS.TH.2.0.c1.9-00102-M8996FAAAANAZM-1.99649.1.118107.1",
	},
	{
		.key = "gsm.version.baseband1",
		.value = "MPSS.TH.2.0.c1.9-00102-M8996FAAAANAZM-1.99649.1.118107.1",
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
		.key = "init.svc.OPNetlinkService",
		.value = "running",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.adsprpcd",
		.value = "running",
	},
	{
		.key = "init.svc.atfwd",
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
		.key = "init.svc.bluetooth-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.bootanim",
		.value = "stopped",
	},
	{
		.key = "init.svc.camera-provider-2-4",
		.value = "running",
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
		.key = "init.svc.configstore-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.dashd",
		.value = "running",
	},
	{
		.key = "init.svc.display-color-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.dpmQmiMgr",
		.value = "running",
	},
	{
		.key = "init.svc.dpmd",
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
		.key = "init.svc.energy-awareness",
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
		.key = "init.svc.gatekeeper-1-0",
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
		.key = "init.svc.hostapd_dual",
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
		.key = "init.svc.ifaadaemon",
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
		.key = "init.svc.iop-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.ipacm-diag",
		.value = "running",
	},
	{
		.key = "init.svc.irsc_util",
		.value = "stopped",
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
		.key = "init.svc.light-hal-2-0",
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
		.key = "init.svc.memtrack-hal-1-0",
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
		.key = "init.svc.oem_audio_device",
		.value = "stopped",
	},
	{
		.key = "init.svc.oemlogkit",
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
		.key = "init.svc.perf-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.port-bridge",
		.value = "running",
	},
	{
		.key = "init.svc.power-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.ppd",
		.value = "running",
	},
	{
		.key = "init.svc.qcamerasvr",
		.value = "running",
	},
	{
		.key = "init.svc.qcom-c_core-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.qcom-c_main-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.qcom-post-boot",
		.value = "stopped",
	},
	{
		.key = "init.svc.qcom-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.qcom-usb-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.qseecomd",
		.value = "running",
	},
	{
		.key = "init.svc.qteeconnector-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.qti",
		.value = "running",
	},
	{
		.key = "init.svc.qti_esepowermanager_service",
		.value = "running",
	},
	{
		.key = "init.svc.qti_gnss_service",
		.value = "running",
	},
	{
		.key = "init.svc.remosaic_deamon",
		.value = "running",
	},
	{
		.key = "init.svc.ril-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.ril-daemon2",
		.value = "running",
	},
	{
		.key = "init.svc.rmt_storage",
		.value = "running",
	},
	{
		.key = "init.svc.self-init",
		.value = "stopped",
	},
	{
		.key = "init.svc.sensor-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.sensors",
		.value = "running",
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
		.key = "init.svc.smartadjust",
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
		.key = "init.svc.tftp_server",
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
		.key = "init.svc.usf-post-boot",
		.value = "stopped",
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
		.key = "log.tag.BeamManager",
		.value = "INFO",
	},
	{
		.key = "log.tag.BeamShareActivity",
		.value = "INFO",
	},
	{
		.key = "log.tag.BluetoothHidDev",
		.value = "INFO",
	},
	{
		.key = "log.tag.BluetoothHidHost",
		.value = "INFO",
	},
	{
		.key = "log.tag.BluetoothMap",
		.value = "INFO",
	},
	{
		.key = "log.tag.BluetoothOpp",
		.value = "INFO",
	},
	{
		.key = "log.tag.BluetoothPan",
		.value = "INFO",
	},
	{
		.key = "log.tag.BluetoothPbap",
		.value = "INFO",
	},
	{
		.key = "log.tag.BluetoothPbapVcardManager",
		.value = "INFO",
	},
	{
		.key = "log.tag.BluetoothPeripheralHandover",
		.value = "INFO",
	},
	{
		.key = "log.tag.BluetoothSap",
		.value = "INFO",
	},
	{
		.key = "log.tag.BluetoothSocket",
		.value = "INFO",
	},
	{
		.key = "log.tag.DockService",
		.value = "INFO",
	},
	{
		.key = "log.tag.FlpHardwareProvider",
		.value = "INFO",
	},
	{
		.key = "log.tag.FlpServiceProvider",
		.value = "INFO",
	},
	{
		.key = "log.tag.ForegroundUtils",
		.value = "INFO",
	},
	{
		.key = "log.tag.FusedLocationProvider",
		.value = "INFO",
	},
	{
		.key = "log.tag.GNPProxy",
		.value = "INFO",
	},
	{
		.key = "log.tag.GeoFenceKeeper",
		.value = "INFO",
	},
	{
		.key = "log.tag.GeoFenceService",
		.value = "INFO",
	},
	{
		.key = "log.tag.GeofenceManager",
		.value = "INFO",
	},
	{
		.key = "log.tag.GeofenceServiceProvider",
		.value = "INFO",
	},
	{
		.key = "log.tag.GmsNetworkLocationProvi",
		.value = "INFO",
	},
	{
		.key = "log.tag.GnssLocationProvider",
		.value = "INFO",
	},
	{
		.key = "log.tag.GpsXtraDownloader",
		.value = "INFO",
	},
	{
		.key = "log.tag.GpsXtraDownloader-Q",
		.value = "INFO",
	},
	{
		.key = "log.tag.HandoverClient",
		.value = "INFO",
	},
	{
		.key = "log.tag.HandoverServer",
		.value = "INFO",
	},
	{
		.key = "log.tag.Handsfree",
		.value = "INFO",
	},
	{
		.key = "log.tag.IZatManager",
		.value = "INFO",
	},
	{
		.key = "log.tag.InPostcard",
		.value = "INFO",
	},
	{
		.key = "log.tag.IzatProviderBase",
		.value = "INFO",
	},
	{
		.key = "log.tag.IzatProviderEngine",
		.value = "INFO",
	},
	{
		.key = "log.tag.IzatService",
		.value = "INFO",
	},
	{
		.key = "log.tag.IzatServiceBase",
		.value = "INFO",
	},
	{
		.key = "log.tag.IzatSettingsInjector",
		.value = "INFO",
	},
	{
		.key = "log.tag.LBSSystemMonitorService",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_ApiV02",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_EngAdapter",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_LBSApiBase",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_LBSApiV02",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_LBSProxy",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_LocUlpProxy",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_NiA",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_NiH",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_afw",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_api_v02",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_eng",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_ext",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_java",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_jni",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_launcher",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocSvc_libulp",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocationManagerService",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocationServiceReceiver",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocationSettings",
		.value = "INFO",
	},
	{
		.key = "log.tag.LocationSettingsBase",
		.value = "INFO",
	},
	{
		.key = "log.tag.MQClient",
		.value = "INFO",
	},
	{
		.key = "log.tag.NFC",
		.value = "INFO",
	},
	{
		.key = "log.tag.NetworkLocationProvider",
		.value = "INFO",
	},
	{
		.key = "log.tag.NetworkLocationService",
		.value = "INFO",
	},
	{
		.key = "log.tag.NfcDispatcher",
		.value = "INFO",
	},
	{
		.key = "log.tag.NfcHandover",
		.value = "INFO",
	},
	{
		.key = "log.tag.NlpProxy",
		.value = "INFO",
	},
	{
		.key = "log.tag.NlpProxyProvider",
		.value = "INFO",
	},
	{
		.key = "log.tag.NpProxy",
		.value = "INFO",
	},
	{
		.key = "log.tag.OsAgent",
		.value = "INFO",
	},
	{
		.key = "log.tag.PosMgr",
		.value = "INFO",
	},
	{
		.key = "log.tag.QCALOG",
		.value = "INFO",
	},
	{
		.key = "log.tag.RilInfoMonitor",
		.value = "INFO",
	},
	{
		.key = "log.tag.SapRilReceiver",
		.value = "INFO",
	},
	{
		.key = "log.tag.SettingsInjector",
		.value = "INFO",
	},
	{
		.key = "log.tag.SnepClient",
		.value = "INFO",
	},
	{
		.key = "log.tag.SnepMessenger",
		.value = "INFO",
	},
	{
		.key = "log.tag.SnepServer",
		.value = "INFO",
	},
	{
		.key = "log.tag.UlpEngine",
		.value = "INFO",
	},
	{
		.key = "log.tag.UlpService",
		.value = "INFO",
	},
	{
		.key = "log.tag.UnifiedLocationProvider",
		.value = "INFO",
	},
	{
		.key = "log.tag.UnifiedLocationService",
		.value = "INFO",
	},
	{
		.key = "log.tag.Wiper",
		.value = "INFO",
	},
	{
		.key = "log.tag.XTActivity",
		.value = "INFO",
	},
	{
		.key = "log.tag.XTBroadcastReceiver",
		.value = "INFO",
	},
	{
		.key = "log.tag.XTSettingInjectorSrv",
		.value = "INFO",
	},
	{
		.key = "log.tag.XTSrv",
		.value = "INFO",
	},
	{
		.key = "log.tag.XTWiFiLP",
		.value = "INFO",
	},
	{
		.key = "log.tag.XTWiFiOS",
		.value = "INFO",
	},
	{
		.key = "log.tag.XTWiFiZpp",
		.value = "INFO",
	},
	{
		.key = "log.tag.gpsone_dmn",
		.value = "INFO",
	},
	{
		.key = "media.aac_51_output_enabled",
		.value = "true",
	},
	{
		.key = "media.settings.xml",
		.value = "/vendor/etc/media_profiles_vendor.xml",
	},
	{
		.key = "media.stagefright.enable-aac",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-http",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-player",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-qcp",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-scan",
		.value = "true",
	},
	{
		.key = "mm.enable.qcom_parser",
		.value = "4177919",
	},
	{
		.key = "mm.enable.smoothstreaming",
		.value = "true",
	},
	{
		.key = "mmp.enable.3g2",
		.value = "true",
	},
	{
		.key = "net.bt.name",
		.value = "Android",
	},
	{
		.key = "net.hostname",
		.value = "OnePlus_3T",
	},
	{
		.key = "net.lte.ims.data.enabled2147483643",
		.value = "true",
	},
	{
		.key = "net.lte.ims.data.enabled2147483644",
		.value = "true",
	},
	{
		.key = "net.qtaguid_enabled",
		.value = "1",
	},
	{
		.key = "net.tcp.2g_init_rwnd",
		.value = "10",
	},
	{
		.key = "net.tcp.buffersize.default",
		.value = "4096,87380,524288,4096,16384,110208",
	},
	{
		.key = "net.tcp.buffersize.edge",
		.value = "4093,26280,35040,4096,16384,35040",
	},
	{
		.key = "net.tcp.buffersize.evdo",
		.value = "4094,87380,524288,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.gprs",
		.value = "4092,8760,11680,4096,8760,11680",
	},
	{
		.key = "net.tcp.buffersize.hsdpa",
		.value = "4094,87380,1220608,4096,16384,1220608",
	},
	{
		.key = "net.tcp.buffersize.hspa",
		.value = "4094,87380,1220608,4096,16384,1220608",
	},
	{
		.key = "net.tcp.buffersize.hspap",
		.value = "4094,87380,1220608,4096,16384,1220608",
	},
	{
		.key = "net.tcp.buffersize.hsupa",
		.value = "4094,87380,1220608,4096,16384,1220608",
	},
	{
		.key = "net.tcp.buffersize.lte",
		.value = "2097152,4194304,8388608,262144,524288,1048576",
	},
	{
		.key = "net.tcp.buffersize.umts",
		.value = "4094,87380,110208,4096,16384,110208",
	},
	{
		.key = "net.tcp.buffersize.wifi",
		.value = "524288,2097152,4194304,262144,524288,1048576",
	},
	{
		.key = "net.tcp.default_init_rwnd",
		.value = "60",
	},
	{
		.key = "oem.device.imeicache",
		.value = "014836007344683",
	},
	{
		.key = "oplib.oneplus_sdk_utils",
		.value = "0.1.2",
	},
	{
		.key = "oplib.oneplus_sdk_wrapper",
		.value = "0.1.0",
	},
	{
		.key = "persist.cne.feature",
		.value = "1",
	},
	{
		.key = "persist.data.df.agg.dl_pkt",
		.value = "10",
	},
	{
		.key = "persist.data.df.agg.dl_size",
		.value = "4096",
	},
	{
		.key = "persist.data.df.dev_name",
		.value = "rmnet_usb0",
	},
	{
		.key = "persist.data.df.dl_mode",
		.value = "5",
	},
	{
		.key = "persist.data.df.iwlan_mux",
		.value = "9",
	},
	{
		.key = "persist.data.df.mux_count",
		.value = "8",
	},
	{
		.key = "persist.data.df.ul_mode",
		.value = "5",
	},
	{
		.key = "persist.data.iwlan.enable",
		.value = "true",
	},
	{
		.key = "persist.data.mode",
		.value = "concurrent",
	},
	{
		.key = "persist.data.netmgrd.qos.enable",
		.value = "true",
	},
	{
		.key = "persist.data.wda.enable",
		.value = "true",
	},
	{
		.key = "persist.debug.coresight.config",
		.value = "stm-events",
	},
	{
		.key = "persist.debug.wfd.enable",
		.value = "1",
	},
	{
		.key = "persist.demo.hdmirotationlock",
		.value = "false",
	},
	{
		.key = "persist.fuse_sdcard",
		.value = "true",
	},
	{
		.key = "persist.hwc.enable_vds",
		.value = "1",
	},
	{
		.key = "persist.mm.enable.prefetch",
		.value = "true",
	},
	{
		.key = "persist.oem.dump",
		.value = "0",
	},
	{
		.key = "persist.qfp",
		.value = "false",
	},
	{
		.key = "persist.radio.apm_sim_not_pwdn",
		.value = "1",
	},
	{
		.key = "persist.radio.apns_ver_xml",
		.value = "8",
	},
	{
		.key = "persist.radio.data_ltd_sys_ind",
		.value = "1",
	},
	{
		.key = "persist.radio.efssync",
		.value = "true",
	},
	{
		.key = "persist.radio.enhance_ecall",
		.value = "true",
	},
	{
		.key = "persist.radio.force_on_dc",
		.value = "true",
	},
	{
		.key = "persist.radio.hw_mbn_update",
		.value = "0",
	},
	{
		.key = "persist.radio.ignore_dom_time",
		.value = "10",
	},
	{
		.key = "persist.radio.multisim.config",
		.value = "dsds",
	},
	{
		.key = "persist.radio.rat_on",
		.value = "combine",
	},
	{
		.key = "persist.radio.serialno",
		.value = "5f846180",
	},
	{
		.key = "persist.radio.start_ota_daemon",
		.value = "1",
	},
	{
		.key = "persist.radio.sw_mbn_update",
		.value = "0",
	},
	{
		.key = "persist.rild.nitz_long_ons_0",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_long_ons_1",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_long_ons_2",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_long_ons_3",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_plmn",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_short_ons_0",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_short_ons_1",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_short_ons_2",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_short_ons_3",
		.value = "",
	},
	{
		.key = "persist.rmnet.data.enable",
		.value = "true",
	},
	{
		.key = "persist.sys.assert.enable",
		.value = "false",
	},
	{
		.key = "persist.sys.assert.panic",
		.value = "false",
	},
	{
		.key = "persist.sys.bootloader",
		.value = "yes",
	},
	{
		.key = "persist.sys.cfu_auto",
		.value = "1",
	},
	{
		.key = "persist.sys.crash",
		.value = "yes",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.device_first_boot",
		.value = "0",
	},
	{
		.key = "persist.sys.diag.max.size",
		.value = "200",
	},
	{
		.key = "persist.sys.event",
		.value = "yes",
	},
	{
		.key = "persist.sys.idle.soff",
		.value = "4,33492",
	},
	{
		.key = "persist.sys.kernel",
		.value = "yes",
	},
	{
		.key = "persist.sys.launcher.set",
		.value = "true",
	},
	{
		.key = "persist.sys.locale",
		.value = "en-US",
	},
	{
		.key = "persist.sys.main",
		.value = "yes",
	},
	{
		.key = "persist.sys.oem.region",
		.value = "OverSeas",
	},
	{
		.key = "persist.sys.oem_smooth",
		.value = "1",
	},
	{
		.key = "persist.sys.pd_enable",
		.value = "0",
	},
	{
		.key = "persist.sys.pre_bootloader",
		.value = "yes",
	},
	{
		.key = "persist.sys.preloads.file_cache_expired",
		.value = "1",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
	},
	{
		.key = "persist.sys.qsee",
		.value = "yes",
	},
	{
		.key = "persist.sys.qxdm",
		.value = "no",
	},
	{
		.key = "persist.sys.radio",
		.value = "yes",
	},
	{
		.key = "persist.sys.system",
		.value = "yes",
	},
	{
		.key = "persist.sys.theme_first_launch",
		.value = "false",
	},
	{
		.key = "persist.sys.theme_version",
		.value = "OnePlus3TOxygen_28.A.62_GLO_062_1712272248",
	},
	{
		.key = "persist.sys.timezone",
		.value = "America/New_York",
	},
	{
		.key = "persist.sys.tz",
		.value = "yes",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "adb",
	},
	{
		.key = "persist.sys.usb.config.extra",
		.value = "none",
	},
	{
		.key = "persist.sys.usb.ffbm-00.func",
		.value = "diag",
	},
	{
		.key = "persist.sys.usb.ffbm-01.func",
		.value = "diag",
	},
	{
		.key = "persist.sys.usb.ffbm-02.func",
		.value = "diag",
	},
	{
		.key = "persist.sys.version.lastota",
		.value = "",
	},
	{
		.key = "persist.sys.version.ota",
		.value = "OnePlus3TOxygen_28.A.62_GLO_062_1712272248",
	},
	{
		.key = "persist.sys.webview.vmsize",
		.value = "113626032",
	},
	{
		.key = "persist.sys.wfd.virtual",
		.value = "0",
	},
	{
		.key = "persist.timed.enable",
		.value = "true",
	},
	{
		.key = "persist.ts.postmakeup",
		.value = "false",
	},
	{
		.key = "persist.ts.rtmakeup",
		.value = "false",
	},
	{
		.key = "persist.vendor.audio.fluence.speaker",
		.value = "true",
	},
	{
		.key = "persist.vendor.audio.fluence.voicecall",
		.value = "true",
	},
	{
		.key = "persist.vendor.audio.fluence.voicerec",
		.value = "false",
	},
	{
		.key = "persist.vendor.bt.enable.splita2dp",
		.value = "false",
	},
	{
		.key = "persist.vendor.dpm.feature",
		.value = "11",
	},
	{
		.key = "persist.vendor.dpm.tcm",
		.value = "2",
	},
	{
		.key = "persist.vendor.qti.inputopts.enable",
		.value = "true",
	},
	{
		.key = "persist.vendor.qti.inputopts.movetouchslop",
		.value = "0.6",
	},
	{
		.key = "persist.vendor.radio.adb_log_on",
		.value = "0",
	},
	{
		.key = "persist.vendor.radio.custom_ecc",
		.value = "1",
	},
	{
		.key = "persist.vendor.radio.data_con_rprt",
		.value = "1",
	},
	{
		.key = "persist.vendor.radio.data_ltd_sys_ind",
		.value = "1",
	},
	{
		.key = "persist.vendor.radio.eons.enabled",
		.value = "false",
	},
	{
		.key = "persist.vendor.radio.hw_mbn_update",
		.value = "0",
	},
	{
		.key = "persist.vendor.radio.ignore_dom_time",
		.value = "10",
	},
	{
		.key = "persist.vendor.radio.msim.stackid_0",
		.value = "0",
	},
	{
		.key = "persist.vendor.radio.msim.stackid_1",
		.value = "1",
	},
	{
		.key = "persist.vendor.radio.rat_on",
		.value = "combine",
	},
	{
		.key = "persist.vendor.radio.ril_payload_on",
		.value = "0",
	},
	{
		.key = "persist.vendor.radio.sglte_target",
		.value = "0",
	},
	{
		.key = "persist.vendor.radio.sib16_support",
		.value = "1",
	},
	{
		.key = "persist.vendor.radio.stack_id_0",
		.value = "0",
	},
	{
		.key = "persist.vendor.radio.stack_id_1",
		.value = "1",
	},
	{
		.key = "persist.vendor.radio.start_ota_daemon",
		.value = "1",
	},
	{
		.key = "persist.vendor.radio.sw_mbn_loaded",
		.value = "1",
	},
	{
		.key = "persist.vendor.radio.sw_mbn_update",
		.value = "0",
	},
	{
		.key = "persist.volte_enalbed_by_hw",
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
		.key = "pm.dexopt.install",
		.value = "quicken",
	},
	{
		.key = "qcom.bluetooth.soc",
		.value = "rome",
	},
	{
		.key = "ril.ecclist",
		.value = "*911,#911,000,08,110,999,118,119,120,122,911,112",
	},
	{
		.key = "ril.ecclist1",
		.value = "*911,#911,000,08,110,999,118,119,120,122,911,112",
	},
	{
		.key = "ril.qcril_pre_init_lock_held",
		.value = "0",
	},
	{
		.key = "ril.subscription.types",
		.value = "NV,RUIM",
	},
	{
		.key = "rild.libpath",
		.value = "/system/vendor/lib64/libril-qc-qmi-1.so",
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
		.key = "ro.baseband",
		.value = "msm",
	},
	{
		.key = "ro.board.platform",
		.value = "msm8996",
	},
	{
		.key = "ro.boot.angela",
		.value = "disabled",
	},
	{
		.key = "ro.boot.baseband",
		.value = "msm",
	},
	{
		.key = "ro.boot.bootdevice",
		.value = "624000.ufshc",
	},
	{
		.key = "ro.boot.enable_dm_verity",
		.value = "1",
	},
	{
		.key = "ro.boot.flash.locked",
		.value = "0",
	},
	{
		.key = "ro.boot.hardware",
		.value = "qcom",
	},
	{
		.key = "ro.boot.hw_version",
		.value = "28",
	},
	{
		.key = "ro.boot.keymaster",
		.value = "1",
	},
	{
		.key = "ro.boot.mode",
		.value = "normal",
	},
	{
		.key = "ro.boot.nobatt",
		.value = "0",
	},
	{
		.key = "ro.boot.pcba_number",
		.value = "001581307213032000008067",
	},
	{
		.key = "ro.boot.project_name",
		.value = "15811",
	},
	{
		.key = "ro.boot.rf_version",
		.value = "32",
	},
	{
		.key = "ro.boot.rpmb_enable",
		.value = "true",
	},
	{
		.key = "ro.boot.secboot",
		.value = "enabled",
	},
	{
		.key = "ro.boot.serialno",
		.value = "5f846180",
	},
	{
		.key = "ro.boot.verifiedbootstate",
		.value = "orange",
	},
	{
		.key = "ro.boot.veritymode",
		.value = "enforcing",
	},
	{
		.key = "ro.bootimage.build.date",
		.value = "Wed Dec 27 22:43:32 CST 2017",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1514385812",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "OnePlus/OnePlus3/OnePlus3T:8.0.0/OPR6.170623.013/12041042:user/release-keys",
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
		.value = "nosdcard",
	},
	{
		.key = "ro.build.date",
		.value = "Wed Dec 27 22:43:32 CST 2017",
	},
	{
		.key = "ro.build.date.Ymd",
		.value = "171227",
	},
	{
		.key = "ro.build.date.YmdHM",
		.value = "201712272248",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1514385812",
	},
	{
		.key = "ro.build.date.ymd",
		.value = "171227",
	},
	{
		.key = "ro.build.description",
		.value = "OnePlus3-user 8.0.0 OPR6.170623.013 5 release-keys",
	},
	{
		.key = "ro.build.display.full_id",
		.value = "ONEPLUS A3000_28_A.62_171227",
	},
	{
		.key = "ro.build.display.id",
		.value = "ONEPLUS A3000_28_171227",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "OnePlus/OnePlus3/OnePlus3T:8.0.0/OPR6.170623.013/12041042:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "OnePlus3-user",
	},
	{
		.key = "ro.build.host",
		.value = "ubuntu-142",
	},
	{
		.key = "ro.build.id",
		.value = "OPR6.170623.013",
	},
	{
		.key = "ro.build.id.hardware",
		.value = "ONEPLUS A3000_28_",
	},
	{
		.key = "ro.build.kernel.id",
		.value = "3.18-G1712272248",
	},
	{
		.key = "ro.build.oemfingerprint",
		.value = "8.0.0/OPR6.170623.013/12041042:user/release-keys",
	},
	{
		.key = "ro.build.ota.versionname",
		.value = "OnePlus3TOxygen_28_1712272248",
	},
	{
		.key = "ro.build.product",
		.value = "OnePlus3",
	},
	{
		.key = "ro.build.release_type",
		.value = "release",
	},
	{
		.key = "ro.build.shutdown_timeout",
		.value = "0",
	},
	{
		.key = "ro.build.soft.majorversion",
		.value = "A",
	},
	{
		.key = "ro.build.soft.version",
		.value = "A.62",
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
		.value = "OnePlus",
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
		.value = "5",
	},
	{
		.key = "ro.build.version.ota",
		.value = "OnePlus3TOxygen_28.A.62_GLO_062_1712272248",
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
		.value = "2017-12-01",
	},
	{
		.key = "ro.carrier",
		.value = "unknown",
	},
	{
		.key = "ro.com.android.dataroaming",
		.value = "true",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "8.0_r3",
	},
	{
		.key = "ro.common.soft",
		.value = "OnePlus3",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "spring.ogg",
	},
	{
		.key = "ro.config.mms_notification",
		.value = "free.ogg",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "meet.ogg",
	},
	{
		.key = "ro.config.ringtone",
		.value = "oneplus_tune.ogg",
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
		.key = "ro.cutoff_voltage_mv",
		.value = "3250",
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
		.key = "ro.dirac.acs.controller",
		.value = "qem",
	},
	{
		.key = "ro.dirac.acs.storeSettings",
		.value = "1",
	},
	{
		.key = "ro.dirac.ignore_error",
		.value = "1",
	},
	{
		.key = "ro.display.series",
		.value = "OnePlus 3T",
	},
	{
		.key = "ro.expect.recovery_id",
		.value = "0x59c57b6a4324b30dc4f208307ed9ef923000c718000000000000000000000000",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/bootdevice/by-name/config",
	},
	{
		.key = "ro.hardware",
		.value = "qcom",
	},
	{
		.key = "ro.hardware.nfc_nci",
		.value = "nqx.default",
	},
	{
		.key = "ro.hwui.drop_shadow_cache_size",
		.value = "7",
	},
	{
		.key = "ro.hwui.gradient_cache_size",
		.value = "1",
	},
	{
		.key = "ro.hwui.layer_cache_size",
		.value = "64",
	},
	{
		.key = "ro.hwui.path_cache_size",
		.value = "39",
	},
	{
		.key = "ro.hwui.r_buffer_cache_size",
		.value = "12",
	},
	{
		.key = "ro.hwui.text_large_cache_height",
		.value = "2048",
	},
	{
		.key = "ro.hwui.text_large_cache_width",
		.value = "3072",
	},
	{
		.key = "ro.hwui.text_small_cache_height",
		.value = "2048",
	},
	{
		.key = "ro.hwui.text_small_cache_width",
		.value = "2048",
	},
	{
		.key = "ro.hwui.texture_cache_flushrate",
		.value = "0.4",
	},
	{
		.key = "ro.hwui.texture_cache_size",
		.value = "96",
	},
	{
		.key = "ro.logdumpd.enabled",
		.value = "0",
	},
	{
		.key = "ro.nfc.port",
		.value = "I2C",
	},
	{
		.key = "ro.oem_unlock_supported",
		.value = "true",
	},
	{
		.key = "ro.oxygen.version",
		.value = "5.0.1",
	},
	{
		.key = "ro.product.board",
		.value = "msm8996",
	},
	{
		.key = "ro.product.brand",
		.value = "OnePlus",
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
		.value = "OnePlus3T",
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
		.value = "OnePlus",
	},
	{
		.key = "ro.product.model",
		.value = "ONEPLUS A3000",
	},
	{
		.key = "ro.product.name",
		.value = "OnePlus3",
	},
	{
		.key = "ro.property_service.version",
		.value = "2",
	},
	{
		.key = "ro.qti.sensors.dev_ori",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.mot_detect",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.pmd",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.sta_detect",
		.value = "false",
	},
	{
		.key = "ro.qualcomm.cabl",
		.value = "0",
	},
	{
		.key = "ro.remount.time",
		.value = "1",
	},
	{
		.key = "ro.revision",
		.value = "0",
	},
	{
		.key = "ro.rf_version",
		.value = "TDD_FDD_Am",
	},
	{
		.key = "ro.ril.supportLTE",
		.value = "1",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "5f846180",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "OPTIONAL",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "420",
	},
	{
		.key = "ro.sys.sdcardfs",
		.value = "true",
	},
	{
		.key = "ro.telephony.call_ring.multiple",
		.value = "false",
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
		.key = "ro.use_data_netmgrd",
		.value = "true",
	},
	{
		.key = "ro.vendor.at_library",
		.value = "libqti-at.so",
	},
	{
		.key = "ro.vendor.audio.sdk.fluencetype",
		.value = "fluence",
	},
	{
		.key = "ro.vendor.audio.sdk.ssr",
		.value = "false",
	},
	{
		.key = "ro.vendor.extension_library",
		.value = "libqti-perfd-client.so",
	},
	{
		.key = "ro.vendor.gt_library",
		.value = "libqti-gt.so",
	},
	{
		.key = "ro.vendor.qti.am.reschedule_service",
		.value = "true",
	},
	{
		.key = "ro.vendor.qti.config.zram",
		.value = "true",
	},
	{
		.key = "ro.vendor.qti.sys.fw.bg_apps_limit",
		.value = "32",
	},
	{
		.key = "ro.vendor.qti.sys.fw.bservice_age",
		.value = "5000",
	},
	{
		.key = "ro.vendor.qti.sys.fw.bservice_enable",
		.value = "true",
	},
	{
		.key = "ro.vendor.qti.sys.fw.bservice_limit",
		.value = "5",
	},
	{
		.key = "ro.vendor.ril.svdo",
		.value = "false",
	},
	{
		.key = "ro.vendor.ril.svlte1x",
		.value = "false",
	},
	{
		.key = "ro.vendor.wl_library",
		.value = "libqti-wl.so",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
	},
	{
		.key = "ro.xxversion",
		.value = "v0.5",
	},
	{
		.key = "ro.zygote",
		.value = "zygote64_32",
	},
	{
		.key = "sdm.debug.disable_rotator_split",
		.value = "1",
	},
	{
		.key = "sdm.debug.disable_skip_validate",
		.value = "1",
	},
	{
		.key = "sdm.perf_hint_window",
		.value = "50",
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
		.key = "sys.automode",
		.value = "0",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
	},
	{
		.key = "sys.cgroup.active",
		.value = "0",
	},
	{
		.key = "sys.cgroup.version",
		.value = "v15112601",
	},
	{
		.key = "sys.dci3p",
		.value = "0",
	},
	{
		.key = "sys.default_mode",
		.value = "0",
	},
	{
		.key = "sys.games.gt.prof",
		.value = "1",
	},
	{
		.key = "sys.listeners.registered",
		.value = "true",
	},
	{
		.key = "sys.logbootcomplete",
		.value = "1",
	},
	{
		.key = "sys.night_mode",
		.value = "0",
	},
	{
		.key = "sys.oem_unlock_allowed",
		.value = "1",
	},
	{
		.key = "sys.post_boot.parsed",
		.value = "1",
	},
	{
		.key = "sys.radio.mcc",
		.value = "310",
	},
	{
		.key = "sys.rescue_boot_count",
		.value = "1",
	},
	{
		.key = "sys.srgb",
		.value = "0",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "24300",
	},
	{
		.key = "sys.usb.config",
		.value = "adb",
	},
	{
		.key = "sys.usb.ffs.ready",
		.value = "1",
	},
	{
		.key = "sys.usb.rmnet.func.name",
		.value = "rmnet_bam",
	},
	{
		.key = "sys.usb.rndis.func.name",
		.value = "rndis_bam",
	},
	{
		.key = "sys.usb.rps_mask",
		.value = "0",
	},
	{
		.key = "sys.usb.state",
		.value = "adb",
	},
	{
		.key = "sys.vendor.shutdown.waittime",
		.value = "500",
	},
	{
		.key = "sys.wifitracing.started",
		.value = "1",
	},
	{
		.key = "telephony.lteOnCdmaDevice",
		.value = "1,1",
	},
	{
		.key = "vendor.audio.flac.sw.decoder.24bit",
		.value = "true",
	},
	{
		.key = "vendor.audio.hw.aac.encoder",
		.value = "true",
	},
	{
		.key = "vendor.audio.offload.buffer.size.kb",
		.value = "64",
	},
	{
		.key = "vendor.audio.offload.gapless.enabled",
		.value = "true",
	},
	{
		.key = "vendor.audio.offload.multiaac.enable",
		.value = "true",
	},
	{
		.key = "vendor.audio.offload.multiple.enabled",
		.value = "true",
	},
	{
		.key = "vendor.audio.offload.passthrough",
		.value = "false",
	},
	{
		.key = "vendor.audio.offload.track.enable",
		.value = "false",
	},
	{
		.key = "vendor.audio.safx.pbe.enabled",
		.value = "true",
	},
	{
		.key = "vendor.audio.tunnel.encode",
		.value = "false",
	},
	{
		.key = "vendor.audio.use.sw.alac.decoder",
		.value = "true",
	},
	{
		.key = "vendor.audio.use.sw.ape.decoder",
		.value = "true",
	},
	{
		.key = "vendor.audio_hal.period_size",
		.value = "192",
	},
	{
		.key = "vendor.vidc.debug.perf.mode",
		.value = "2",
	},
	{
		.key = "vendor.vidc.enc.disable.pq",
		.value = "true",
	},
	{
		.key = "vendor.voice.path.for.pcm.voip",
		.value = "true",
	},
	{
		.key = "vidc.enc.dcvs.extra-buff-count",
		.value = "2",
	},
	{
		.key = "vold.datafs.type",
		.value = "F2FS",
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
		.key = "vold.internalSD.mount",
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
