struct cpuinfo_mock_file filesystem[] = {
	{
		.path = "/proc/cpuinfo",
		.size = 1540,
		.content =
			"Processor\t: AArch64 Processor rev 1 (aarch64)\n"
			"processor\t: 0\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 4\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 5\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 6\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 7\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc MSM8998\n",
	},
	{
		.path = "/sys/devices/system/cpu/kernel_max",
		.size = 2,
		.content = "7\n",
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
		.path = "/sys/devices/system/cpu/online",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/offline",
		.size = 1,
		.content = "\n",
	},
	{
		.path = "/sys/devices/system/cpu/modalias",
		.size = 66,
		.content = "cpu:type:aarch64:feature:,0000,0001,0002,0003,0004,0005,0006,0007\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/all_time_in_state",
		.size = 2793,
		.content =
			"freq\t\tcpu0\t\tcpu1\t\tcpu2\t\tcpu3\t\tcpu4\t\tcpu5\t\tcpu6\t\tcpu7\t\t\n"
			"300000\t\t622774\t\t622774\t\t622774\t\t622774\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"364800\t\t7579\t\t7579\t\t7579\t\t7579\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"441600\t\t7761\t\t7761\t\t7761\t\t7761\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"518400\t\t6959\t\t6959\t\t6959\t\t6959\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"595200\t\t1061\t\t1061\t\t1061\t\t1061\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"672000\t\t1038\t\t1038\t\t1038\t\t1038\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"748800\t\t1601\t\t1601\t\t1601\t\t1601\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"825600\t\t1238\t\t1238\t\t1238\t\t1238\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"883200\t\t476\t\t476\t\t476\t\t476\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"960000\t\t742\t\t742\t\t742\t\t742\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1036800\t\t482\t\t482\t\t482\t\t482\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1094400\t\t1064\t\t1064\t\t1064\t\t1064\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1171200\t\t1026\t\t1026\t\t1026\t\t1026\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1248000\t\t1072\t\t1072\t\t1072\t\t1072\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1324800\t\t485\t\t485\t\t485\t\t485\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1401600\t\t826\t\t826\t\t826\t\t826\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1478400\t\t410\t\t410\t\t410\t\t410\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1555200\t\t1405\t\t1405\t\t1405\t\t1405\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1670400\t\t997\t\t997\t\t997\t\t997\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1747200\t\t1330\t\t1330\t\t1330\t\t1330\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1824000\t\t816\t\t816\t\t816\t\t816\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1900800\t\t34020\t\t34020\t\t34020\t\t34020\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"300000\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t659914\t\t659914\t\t659914\t\t659914\t\t\n"
			"345600\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t1638\t\t1638\t\t1638\t\t1638\t\t\n"
			"422400\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t1418\t\t1418\t\t1418\t\t1418\t\t\n"
			"499200\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t1723\t\t1723\t\t1723\t\t1723\t\t\n"
			"576000\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t1435\t\t1435\t\t1435\t\t1435\t\t\n"
			"652800\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t1728\t\t1728\t\t1728\t\t1728\t\t\n"
			"729600\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t751\t\t751\t\t751\t\t751\t\t\n"
			"806400\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t1014\t\t1014\t\t1014\t\t1014\t\t\n"
			"902400\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t799\t\t799\t\t799\t\t799\t\t\n"
			"979200\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t536\t\t536\t\t536\t\t536\t\t\n"
			"1056000\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t704\t\t704\t\t704\t\t704\t\t\n"
			"1132800\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t1112\t\t1112\t\t1112\t\t1112\t\t\n"
			"1190400\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t463\t\t463\t\t463\t\t463\t\t\n"
			"1267200\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t381\t\t381\t\t381\t\t381\t\t\n"
			"1344000\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t391\t\t391\t\t391\t\t391\t\t\n"
			"1420800\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t568\t\t568\t\t568\t\t568\t\t\n"
			"1497600\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t290\t\t290\t\t290\t\t290\t\t\n"
			"1574400\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t261\t\t261\t\t261\t\t261\t\t\n"
			"1651200\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t422\t\t422\t\t422\t\t422\t\t\n"
			"1728000\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t669\t\t669\t\t669\t\t669\t\t\n"
			"1804800\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t337\t\t337\t\t337\t\t337\t\t\n"
			"1881600\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t229\t\t229\t\t229\t\t229\t\t\n"
			"1958400\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t225\t\t225\t\t225\t\t225\t\t\n"
			"2035200\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t237\t\t237\t\t237\t\t237\t\t\n"
			"2112000\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t357\t\t357\t\t357\t\t357\t\t\n"
			"2208000\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t297\t\t297\t\t297\t\t297\t\t\n"
			"2265600\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t162\t\t162\t\t162\t\t162\t\t\n"
			"2323200\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t141\t\t141\t\t141\t\t141\t\t\n"
			"2342400\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t72\t\t72\t\t72\t\t72\t\t\n"
			"2361600\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t68\t\t68\t\t68\t\t68\t\t\n"
			"2457600\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t16815\t\t16815\t\t16815\t\t16815\t\t\n",
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
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "300000\n",
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
		.size = 167,
		.content = "300000 364800 441600 518400 595200 672000 748800 825600 883200 960000 1036800 1094400 1171200 1248000 1324800 1401600 1478400 1555200 1670400 1747200 1824000 1900800 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 83,
		.content = "interactive conservative ondemand userspace powersave performance schedutil sched \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 271,
		.content =
			"300000 622775\n"
			"364800 7579\n"
			"441600 7761\n"
			"518400 6959\n"
			"595200 1061\n"
			"672000 1038\n"
			"748800 1601\n"
			"825600 1238\n"
			"883200 476\n"
			"960000 742\n"
			"1036800 482\n"
			"1094400 1064\n"
			"1171200 1026\n"
			"1248000 1072\n"
			"1324800 485\n"
			"1401600 826\n"
			"1478400 410\n"
			"1555200 1405\n"
			"1670400 997\n"
			"1747200 1330\n"
			"1824000 816\n"
			"1900800 34144\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 6,
		.content = "75504\n",
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
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/number_of_sets",
		.size = 4,
		.content = "128\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/shared_cpu_map",
		.size = 3,
		.content = "01\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/size",
		.size = 4,
		.content = "32K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/allocation_policy",
		.size = 13,
		.content = "ReadAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/number_of_sets",
		.size = 4,
		.content = "256\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/shared_cpu_map",
		.size = 3,
		.content = "01\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/size",
		.size = 4,
		.content = "32K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/type",
		.size = 12,
		.content = "Instruction\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/ways_of_associativity",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/level",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/number_of_sets",
		.size = 5,
		.content = "1024\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/shared_cpu_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "0f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/size",
		.size = 6,
		.content = "1024K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
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
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "300000\n",
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
		.size = 167,
		.content = "300000 364800 441600 518400 595200 672000 748800 825600 883200 960000 1036800 1094400 1171200 1248000 1324800 1401600 1478400 1555200 1670400 1747200 1824000 1900800 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 83,
		.content = "interactive conservative ondemand userspace powersave performance schedutil sched \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 271,
		.content =
			"300000 622778\n"
			"364800 7579\n"
			"441600 7761\n"
			"518400 6959\n"
			"595200 1061\n"
			"672000 1038\n"
			"748800 1601\n"
			"825600 1238\n"
			"883200 476\n"
			"960000 742\n"
			"1036800 482\n"
			"1094400 1064\n"
			"1171200 1026\n"
			"1248000 1072\n"
			"1324800 485\n"
			"1401600 826\n"
			"1478400 410\n"
			"1555200 1405\n"
			"1670400 997\n"
			"1747200 1330\n"
			"1824000 816\n"
			"1900800 34439\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 6,
		.content = "75584\n",
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
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/number_of_sets",
		.size = 4,
		.content = "128\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/shared_cpu_map",
		.size = 3,
		.content = "02\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/size",
		.size = 4,
		.content = "32K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/allocation_policy",
		.size = 13,
		.content = "ReadAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/number_of_sets",
		.size = 4,
		.content = "256\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/shared_cpu_map",
		.size = 3,
		.content = "02\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/size",
		.size = 4,
		.content = "32K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/type",
		.size = 12,
		.content = "Instruction\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/ways_of_associativity",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/level",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/number_of_sets",
		.size = 5,
		.content = "1024\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/shared_cpu_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "0f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/size",
		.size = 6,
		.content = "1024K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
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
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "300000\n",
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
		.size = 167,
		.content = "300000 364800 441600 518400 595200 672000 748800 825600 883200 960000 1036800 1094400 1171200 1248000 1324800 1401600 1478400 1555200 1670400 1747200 1824000 1900800 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 83,
		.content = "interactive conservative ondemand userspace powersave performance schedutil sched \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 271,
		.content =
			"300000 622780\n"
			"364800 7579\n"
			"441600 7761\n"
			"518400 6959\n"
			"595200 1061\n"
			"672000 1038\n"
			"748800 1601\n"
			"825600 1238\n"
			"883200 476\n"
			"960000 742\n"
			"1036800 482\n"
			"1094400 1064\n"
			"1171200 1026\n"
			"1248000 1072\n"
			"1324800 485\n"
			"1401600 826\n"
			"1478400 410\n"
			"1555200 1405\n"
			"1670400 997\n"
			"1747200 1330\n"
			"1824000 816\n"
			"1900800 34716\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 6,
		.content = "75632\n",
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
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/number_of_sets",
		.size = 4,
		.content = "128\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/shared_cpu_map",
		.size = 3,
		.content = "04\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/size",
		.size = 4,
		.content = "32K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/allocation_policy",
		.size = 13,
		.content = "ReadAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/number_of_sets",
		.size = 4,
		.content = "256\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/shared_cpu_map",
		.size = 3,
		.content = "04\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/size",
		.size = 4,
		.content = "32K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/type",
		.size = 12,
		.content = "Instruction\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/ways_of_associativity",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/level",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/number_of_sets",
		.size = 5,
		.content = "1024\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/shared_cpu_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "0f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/size",
		.size = 6,
		.content = "1024K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
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
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "300000\n",
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
		.size = 167,
		.content = "300000 364800 441600 518400 595200 672000 748800 825600 883200 960000 1036800 1094400 1171200 1248000 1324800 1401600 1478400 1555200 1670400 1747200 1824000 1900800 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 83,
		.content = "interactive conservative ondemand userspace powersave performance schedutil sched \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 8,
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 271,
		.content =
			"300000 622782\n"
			"364800 7579\n"
			"441600 7761\n"
			"518400 6959\n"
			"595200 1061\n"
			"672000 1038\n"
			"748800 1601\n"
			"825600 1238\n"
			"883200 476\n"
			"960000 742\n"
			"1036800 482\n"
			"1094400 1064\n"
			"1171200 1026\n"
			"1248000 1072\n"
			"1324800 485\n"
			"1401600 826\n"
			"1478400 410\n"
			"1555200 1405\n"
			"1670400 997\n"
			"1747200 1330\n"
			"1824000 816\n"
			"1900800 35016\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 6,
		.content = "75688\n",
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
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/number_of_sets",
		.size = 4,
		.content = "128\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/shared_cpu_map",
		.size = 3,
		.content = "08\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/size",
		.size = 4,
		.content = "32K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/allocation_policy",
		.size = 13,
		.content = "ReadAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/number_of_sets",
		.size = 4,
		.content = "256\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/shared_cpu_map",
		.size = 3,
		.content = "08\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/size",
		.size = 4,
		.content = "32K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/type",
		.size = 12,
		.content = "Instruction\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/ways_of_associativity",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/level",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/number_of_sets",
		.size = 5,
		.content = "1024\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/shared_cpu_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "0f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/size",
		.size = 6,
		.content = "1024K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2457600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies",
		.size = 239,
		.content = "300000 345600 422400 499200 576000 652800 729600 806400 902400 979200 1056000 1132800 1190400 1267200 1344000 1420800 1497600 1574400 1651200 1728000 1804800 1881600 1958400 2035200 2112000 2208000 2265600 2323200 2342400 2361600 2457600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors",
		.size = 83,
		.content = "interactive conservative ondemand userspace powersave performance schedutil sched \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2457600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 372,
		.content =
			"300000 661160\n"
			"345600 1638\n"
			"422400 1424\n"
			"499200 1728\n"
			"576000 1435\n"
			"652800 1733\n"
			"729600 752\n"
			"806400 1022\n"
			"902400 799\n"
			"979200 538\n"
			"1056000 706\n"
			"1132800 1112\n"
			"1190400 465\n"
			"1267200 384\n"
			"1344000 391\n"
			"1420800 570\n"
			"1497600 290\n"
			"1574400 263\n"
			"1651200 422\n"
			"1728000 671\n"
			"1804800 337\n"
			"1881600 229\n"
			"1958400 225\n"
			"2035200 239\n"
			"2112000 357\n"
			"2208000 299\n"
			"2265600 162\n"
			"2323200 141\n"
			"2342400 72\n"
			"2361600 68\n"
			"2457600 16817\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 6,
		.content = "45792\n",
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
		.path = "/sys/devices/system/cpu/cpu4/cache/index0/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index0/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index0/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index0/number_of_sets",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index0/shared_cpu_map",
		.size = 3,
		.content = "10\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index0/size",
		.size = 4,
		.content = "64K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index0/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index0/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index1/allocation_policy",
		.size = 13,
		.content = "ReadAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index1/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index1/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index1/number_of_sets",
		.size = 4,
		.content = "256\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index1/shared_cpu_map",
		.size = 3,
		.content = "10\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index1/size",
		.size = 4,
		.content = "64K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index1/type",
		.size = 12,
		.content = "Instruction\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index1/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/level",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/shared_cpu_list",
		.size = 4,
		.content = "4-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "f0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2457600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_frequencies",
		.size = 239,
		.content = "300000 345600 422400 499200 576000 652800 729600 806400 902400 979200 1056000 1132800 1190400 1267200 1344000 1420800 1497600 1574400 1651200 1728000 1804800 1881600 1958400 2035200 2112000 2208000 2265600 2323200 2342400 2361600 2457600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_governors",
		.size = 83,
		.content = "interactive conservative ondemand userspace powersave performance schedutil sched \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2457600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 372,
		.content =
			"300000 661459\n"
			"345600 1638\n"
			"422400 1424\n"
			"499200 1728\n"
			"576000 1435\n"
			"652800 1733\n"
			"729600 752\n"
			"806400 1022\n"
			"902400 799\n"
			"979200 538\n"
			"1056000 706\n"
			"1132800 1112\n"
			"1190400 465\n"
			"1267200 384\n"
			"1344000 391\n"
			"1420800 570\n"
			"1497600 290\n"
			"1574400 263\n"
			"1651200 422\n"
			"1728000 671\n"
			"1804800 337\n"
			"1881600 229\n"
			"1958400 225\n"
			"2035200 239\n"
			"2112000 357\n"
			"2208000 299\n"
			"2265600 162\n"
			"2323200 141\n"
			"2342400 72\n"
			"2361600 68\n"
			"2457600 16817\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 6,
		.content = "45792\n",
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
		.path = "/sys/devices/system/cpu/cpu5/cache/index0/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index0/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index0/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index0/number_of_sets",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "5\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index0/shared_cpu_map",
		.size = 3,
		.content = "20\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index0/size",
		.size = 4,
		.content = "64K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index0/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index0/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index1/allocation_policy",
		.size = 13,
		.content = "ReadAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index1/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index1/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index1/number_of_sets",
		.size = 4,
		.content = "256\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "5\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index1/shared_cpu_map",
		.size = 3,
		.content = "20\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index1/size",
		.size = 4,
		.content = "64K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index1/type",
		.size = 12,
		.content = "Instruction\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index1/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/level",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/shared_cpu_list",
		.size = 4,
		.content = "4-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "f0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2457600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_frequencies",
		.size = 239,
		.content = "300000 345600 422400 499200 576000 652800 729600 806400 902400 979200 1056000 1132800 1190400 1267200 1344000 1420800 1497600 1574400 1651200 1728000 1804800 1881600 1958400 2035200 2112000 2208000 2265600 2323200 2342400 2361600 2457600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_governors",
		.size = 83,
		.content = "interactive conservative ondemand userspace powersave performance schedutil sched \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2457600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 372,
		.content =
			"300000 661746\n"
			"345600 1638\n"
			"422400 1427\n"
			"499200 1731\n"
			"576000 1435\n"
			"652800 1733\n"
			"729600 752\n"
			"806400 1022\n"
			"902400 799\n"
			"979200 538\n"
			"1056000 706\n"
			"1132800 1112\n"
			"1190400 465\n"
			"1267200 384\n"
			"1344000 391\n"
			"1420800 570\n"
			"1497600 290\n"
			"1574400 263\n"
			"1651200 422\n"
			"1728000 671\n"
			"1804800 337\n"
			"1881600 229\n"
			"1958400 225\n"
			"2035200 239\n"
			"2112000 357\n"
			"2208000 299\n"
			"2265600 162\n"
			"2323200 141\n"
			"2342400 72\n"
			"2361600 68\n"
			"2457600 16817\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 6,
		.content = "45816\n",
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
		.path = "/sys/devices/system/cpu/cpu6/cache/index0/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index0/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index0/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index0/number_of_sets",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "6\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index0/shared_cpu_map",
		.size = 3,
		.content = "40\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index0/size",
		.size = 4,
		.content = "64K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index0/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index0/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index1/allocation_policy",
		.size = 13,
		.content = "ReadAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index1/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index1/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index1/number_of_sets",
		.size = 4,
		.content = "256\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "6\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index1/shared_cpu_map",
		.size = 3,
		.content = "40\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index1/size",
		.size = 4,
		.content = "64K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index1/type",
		.size = 12,
		.content = "Instruction\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index1/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/level",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/shared_cpu_list",
		.size = 4,
		.content = "4-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "f0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2457600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/related_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_frequencies",
		.size = 239,
		.content = "300000 345600 422400 499200 576000 652800 729600 806400 902400 979200 1056000 1132800 1190400 1267200 1344000 1420800 1497600 1574400 1651200 1728000 1804800 1881600 1958400 2035200 2112000 2208000 2265600 2323200 2342400 2361600 2457600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_governors",
		.size = 83,
		.content = "interactive conservative ondemand userspace powersave performance schedutil sched \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2457600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 372,
		.content =
			"300000 662030\n"
			"345600 1638\n"
			"422400 1427\n"
			"499200 1731\n"
			"576000 1437\n"
			"652800 1735\n"
			"729600 752\n"
			"806400 1022\n"
			"902400 799\n"
			"979200 538\n"
			"1056000 706\n"
			"1132800 1112\n"
			"1190400 465\n"
			"1267200 384\n"
			"1344000 391\n"
			"1420800 570\n"
			"1497600 290\n"
			"1574400 263\n"
			"1651200 422\n"
			"1728000 671\n"
			"1804800 337\n"
			"1881600 229\n"
			"1958400 225\n"
			"2035200 239\n"
			"2112000 357\n"
			"2208000 299\n"
			"2265600 162\n"
			"2323200 141\n"
			"2342400 72\n"
			"2361600 68\n"
			"2457600 16817\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 6,
		.content = "45832\n",
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
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index0/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index0/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index0/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index0/number_of_sets",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index0/shared_cpu_map",
		.size = 3,
		.content = "80\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index0/size",
		.size = 4,
		.content = "64K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index0/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index0/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index1/allocation_policy",
		.size = 13,
		.content = "ReadAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index1/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index1/level",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index1/number_of_sets",
		.size = 4,
		.content = "256\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index1/shared_cpu_map",
		.size = 3,
		.content = "80\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index1/size",
		.size = 4,
		.content = "64K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index1/type",
		.size = 12,
		.content = "Instruction\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index1/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/level",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/shared_cpu_list",
		.size = 4,
		.content = "4-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "f0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{ NULL },
};
#ifdef __ANDROID__
struct cpuinfo_mock_property properties[] = {
	{
		.key = "DEVICE_PROVISIONED",
		.value = "1",
	},
	{
		.key = "aaudio.hw_burst_min_usec",
		.value = "2000",
	},
	{
		.key = "aaudio.mmap_exclusive_policy",
		.value = "2",
	},
	{
		.key = "aaudio.mmap_policy",
		.value = "2",
	},
	{
		.key = "af.fast_track_multiplier",
		.value = "1",
	},
	{
		.key = "audio.adm.buffering.ms",
		.value = "3",
	},
	{
		.key = "audio.snd_card.open.retries",
		.value = "50",
	},
	{
		.key = "audio_hal.period_multiplier",
		.value = "2",
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
		.key = "dalvik.vm.isa.arm.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.arm.variant",
		.value = "cortex-a73",
	},
	{
		.key = "dalvik.vm.isa.arm64.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.arm64.variant",
		.value = "cortex-a73",
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
		.key = "debug.atrace.tags.enableflags",
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
		.key = "debug.sf.hw",
		.value = "1",
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
		.key = "fmas.hdph_sgain",
		.value = "0",
	},
	{
		.key = "gsm.current.phone-type",
		.value = "1",
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
		.value = "",
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
		.value = "g8998-00159-1709201454",
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
		.key = "init.svc.adsprpcd",
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
		.key = "init.svc.boot-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.bootanim",
		.value = "stopped",
	},
	{
		.key = "init.svc.bufferhubd",
		.value = "running",
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
		.key = "init.svc.cas-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.chre",
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
		.key = "init.svc.contexthub-hal-1-0",
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
		.key = "init.svc.drm-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.drm-widevine-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.dumpstate-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.ese_load",
		.value = "stopped",
	},
	{
		.key = "init.svc.esed",
		.value = "running",
	},
	{
		.key = "init.svc.folio_daemon",
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
		.key = "init.svc.gnss_service",
		.value = "running",
	},
	{
		.key = "init.svc.gralloc-2-0",
		.value = "running",
	},
	{
		.key = "init.svc.hci_filter",
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
		.key = "init.svc.imsdatadaemon",
		.value = "running",
	},
	{
		.key = "init.svc.imsqmidaemon",
		.value = "running",
	},
	{
		.key = "init.svc.init-elabel-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.init-radio-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.insmod_sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.installd",
		.value = "running",
	},
	{
		.key = "init.svc.ipacm",
		.value = "running",
	},
	{
		.key = "init.svc.ipastart_sh",
		.value = "stopped",
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
		.key = "init.svc.nfc_hal_service",
		.value = "running",
	},
	{
		.key = "init.svc.oemlock_bridge",
		.value = "running",
	},
	{
		.key = "init.svc.oemlock_hal",
		.value = "running",
	},
	{
		.key = "init.svc.offload-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.pd_mapper",
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
		.key = "init.svc.performanced",
		.value = "running",
	},
	{
		.key = "init.svc.port-bridge",
		.value = "running",
	},
	{
		.key = "init.svc.power-hal-1-1",
		.value = "running",
	},
	{
		.key = "init.svc.power_sh",
		.value = "stopped",
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
		.key = "init.svc.ramoops_sh",
		.value = "stopped",
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
		.key = "init.svc.ssr_setup",
		.value = "stopped",
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
		.key = "init.svc.thermalservice",
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
		.key = "init.svc.update_engine",
		.value = "running",
	},
	{
		.key = "init.svc.update_verifier_nonencrypted",
		.value = "stopped",
	},
	{
		.key = "init.svc.usb-hal-1-1",
		.value = "running",
	},
	{
		.key = "init.svc.vibrator-1-1",
		.value = "running",
	},
	{
		.key = "init.svc.virtual_touchpad",
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
		.key = "init.svc.vr-wahoo-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.vr_hwc",
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
		.key = "media.mediadrmservice.enable",
		.value = "true",
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
		.value = "13631487",
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
		.key = "net.lte.ims.data.enabled",
		.value = "true",
	},
	{
		.key = "net.qtaguid_enabled",
		.value = "1",
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
		.key = "nfc.initialized",
		.value = "true",
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
		.key = "persist.camera.debug.logfile",
		.value = "0",
	},
	{
		.key = "persist.camera.gcam.fd.ensemble",
		.value = "1",
	},
	{
		.key = "persist.camera.gyro.android",
		.value = "20",
	},
	{
		.key = "persist.camera.gzoom.at",
		.value = "0",
	},
	{
		.key = "persist.camera.is_type",
		.value = "5",
	},
	{
		.key = "persist.camera.llv.fuse",
		.value = "2",
	},
	{
		.key = "persist.camera.max.previewfps",
		.value = "60",
	},
	{
		.key = "persist.camera.perfd.enable",
		.value = "true",
	},
	{
		.key = "persist.camera.saturationext",
		.value = "1",
	},
	{
		.key = "persist.camera.sensor.hdr",
		.value = "2",
	},
	{
		.key = "persist.camera.tnr.video",
		.value = "1",
	},
	{
		.key = "persist.camera.tof.direct",
		.value = "1",
	},
	{
		.key = "persist.cne.feature",
		.value = "1",
	},
	{
		.key = "persist.config.calibration_fac",
		.value = "/persist/sensors/calibration/calibration.xml",
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
		.key = "persist.delta_time.enable",
		.value = "true",
	},
	{
		.key = "persist.fuse_sdcard",
		.value = "true",
	},
	{
		.key = "persist.logd.size",
		.value = "",
	},
	{
		.key = "persist.mm.enable.prefetch",
		.value = "true",
	},
	{
		.key = "persist.net.doxlat",
		.value = "true",
	},
	{
		.key = "persist.radio.RATE_ADAPT_ENABLE",
		.value = "1",
	},
	{
		.key = "persist.radio.ROTATION_ENABLE",
		.value = "1",
	},
	{
		.key = "persist.radio.VT_ENABLE",
		.value = "1",
	},
	{
		.key = "persist.radio.VT_HYBRID_ENABLE",
		.value = "1",
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
		.key = "persist.radio.ci_status",
		.value = "2",
	},
	{
		.key = "persist.radio.cnv.ver_info",
		.value = "mbn.v1.1_20170920_GITCL#a4f8c9f_ForT",
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
		.key = "persist.radio.data_ltd_sys_ind",
		.value = "1",
	},
	{
		.key = "persist.radio.eons.enabled",
		.value = "false",
	},
	{
		.key = "persist.radio.is_wps_enabled",
		.value = "true",
	},
	{
		.key = "persist.radio.ril_payload_on",
		.value = "0",
	},
	{
		.key = "persist.radio.sglte_target",
		.value = "0",
	},
	{
		.key = "persist.radio.sib16_support",
		.value = "1",
	},
	{
		.key = "persist.radio.snapshot_enabled",
		.value = "0",
	},
	{
		.key = "persist.radio.snapshot_timer",
		.value = "0",
	},
	{
		.key = "persist.radio.videopause.mode",
		.value = "1",
	},
	{
		.key = "persist.rcs.supported",
		.value = "1",
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
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
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
		.key = "persist.sys.sf.color_saturation",
		.value = "1.1",
	},
	{
		.key = "persist.sys.ssr.restart_level",
		.value = "modem,slpi,adsp",
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
		.value = "114785072",
	},
	{
		.key = "persist.timed.enable",
		.value = "true",
	},
	{
		.key = "persist.vendor.ims.dropset_feature",
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
		.value = "quicken",
	},
	{
		.key = "pm.dexopt.shared",
		.value = "speed",
	},
	{
		.key = "qcom.bluetooth.soc",
		.value = "cherokee",
	},
	{
		.key = "ril.ecclist",
		.value = "911,112,*911,#911,000,08,110,999,118,119",
	},
	{
		.key = "ril.power.backoff_suppressed",
		.value = "0",
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
		.key = "ro.audio.monitorRotation",
		.value = "true",
	},
	{
		.key = "ro.baseband",
		.value = "msm",
	},
	{
		.key = "ro.bluetooth.a4wp",
		.value = "false",
	},
	{
		.key = "ro.bluetooth.emb_wp_mode",
		.value = "false",
	},
	{
		.key = "ro.bluetooth.wipower",
		.value = "false",
	},
	{
		.key = "ro.board.platform",
		.value = "msm8998",
	},
	{
		.key = "ro.boot.avb_version",
		.value = "1.0",
	},
	{
		.key = "ro.boot.baseband",
		.value = "msm",
	},
	{
		.key = "ro.boot.bootdevice",
		.value = "1da4000.ufshc",
	},
	{
		.key = "ro.boot.bootloader",
		.value = "TMZ11f",
	},
	{
		.key = "ro.boot.bootreason",
		.value = "reboot",
	},
	{
		.key = "ro.boot.boottime",
		.value = "1BLL:132,1BLE:890,2BLL:29,2BLE:102,AVB:326,KL:0,KD:456,ODT:150,SW:0",
	},
	{
		.key = "ro.boot.cid",
		.value = "00000000",
	},
	{
		.key = "ro.boot.ddr_info",
		.value = "HYNIX",
	},
	{
		.key = "ro.boot.ddr_size",
		.value = "4096MB",
	},
	{
		.key = "ro.boot.flash.locked",
		.value = "1",
	},
	{
		.key = "ro.boot.hardware",
		.value = "taimen",
	},
	{
		.key = "ro.boot.hardware.color",
		.value = "VB",
	},
	{
		.key = "ro.boot.hardware.display",
		.value = "MP",
	},
	{
		.key = "ro.boot.hardware.mid",
		.value = "2",
	},
	{
		.key = "ro.boot.hardware.revision",
		.value = "rev_10",
	},
	{
		.key = "ro.boot.hardware.sku",
		.value = "G011C",
	},
	{
		.key = "ro.boot.hardware.ufs",
		.value = "64GB,SAMSUNG",
	},
	{
		.key = "ro.boot.hardware.variant",
		.value = "GA00125-US",
	},
	{
		.key = "ro.boot.keymaster",
		.value = "1",
	},
	{
		.key = "ro.boot.phandle",
		.value = "",
	},
	{
		.key = "ro.boot.ramdump_enable",
		.value = "0",
	},
	{
		.key = "ro.boot.revision",
		.value = "rev_10",
	},
	{
		.key = "ro.boot.serialno",
		.value = "710KPQJ0358188",
	},
	{
		.key = "ro.boot.slot",
		.value = "b",
	},
	{
		.key = "ro.boot.slot_suffix",
		.value = "_b",
	},
	{
		.key = "ro.boot.vbmeta.avb_version",
		.value = "1.0",
	},
	{
		.key = "ro.boot.vbmeta.device_state",
		.value = "locked",
	},
	{
		.key = "ro.boot.vbmeta.digest",
		.value = "19ba11417e57df5aa2f366828dcc45e4a06cb3ad46feb53a47e83f04c9e58a21",
	},
	{
		.key = "ro.boot.vbmeta.hash_alg",
		.value = "sha256",
	},
	{
		.key = "ro.boot.vbmeta.size",
		.value = "2496",
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
		.value = "00",
	},
	{
		.key = "ro.bootimage.build.date",
		.value = "Tue Oct 3 04:55:08 UTC 2017",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1507006508",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "google/taimen/taimen:8.1.0/OPP5.170921.005/4373449:user/release-keys",
	},
	{
		.key = "ro.bootloader",
		.value = "TMZ11f",
	},
	{
		.key = "ro.bootmode",
		.value = "unknown",
	},
	{
		.key = "ro.build.ab_update",
		.value = "true",
	},
	{
		.key = "ro.build.characteristics",
		.value = "nosdcard",
	},
	{
		.key = "ro.build.date",
		.value = "Tue Oct  3 04:55:08 UTC 2017",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1507006508",
	},
	{
		.key = "ro.build.description",
		.value = "taimen-user 8.1.0 OPP5.170921.005 4373449 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "OPP5.170921.005",
	},
	{
		.key = "ro.build.expect.baseband",
		.value = "g8998-00159-1709201454",
	},
	{
		.key = "ro.build.expect.bootloader",
		.value = "TMZ11f",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "google/taimen/taimen:8.1.0/OPP5.170921.005/4373449:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "taimen-user",
	},
	{
		.key = "ro.build.host",
		.value = "kpfk7.cbf.corp.google.com",
	},
	{
		.key = "ro.build.id",
		.value = "OPP5.170921.005",
	},
	{
		.key = "ro.build.product",
		.value = "taimen",
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
		.value = "4373449",
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
		.value = "2017-10-05",
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
		.key = "ro.com.google.ime.theme_id",
		.value = "5",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Bright_morning.ogg",
	},
	{
		.key = "ro.config.media_vol_steps",
		.value = "25",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "Popcorn.ogg",
	},
	{
		.key = "ro.config.ringtone",
		.value = "The_big_adventure.ogg",
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
		.key = "ro.cp_system_other_odex",
		.value = "1",
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
		.value = "/dev/block/platform/soc/1da4000.ufshc/by-name/frp",
	},
	{
		.key = "ro.gfx.driver.0",
		.value = "com.google.pixel.wahoo.gfxdrv",
	},
	{
		.key = "ro.hardware",
		.value = "taimen",
	},
	{
		.key = "ro.hardware.fingerprint",
		.value = "fpc",
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
		.value = "64",
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
		.value = "84",
	},
	{
		.key = "ro.oem_unlock.pst",
		.value = "/dev/block/platform/soc/1da4000.ufshc/by-name/misc",
	},
	{
		.key = "ro.oem_unlock.pst_offset",
		.value = "6144",
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
		.key = "ro.product.board",
		.value = "taimen",
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
		.value = "taimen",
	},
	{
		.key = "ro.product.first_api_level",
		.value = "26",
	},
	{
		.key = "ro.product.locale",
		.value = "en-US",
	},
	{
		.key = "ro.product.manufacturer",
		.value = "Google",
	},
	{
		.key = "ro.product.model",
		.value = "Pixel 2 XL",
	},
	{
		.key = "ro.product.name",
		.value = "taimen",
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
		.key = "ro.qti.sdk.sensors.gestures",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.amd",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.cmc",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.dev_ori",
		.value = "true",
	},
	{
		.key = "ro.qti.sensors.facing",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.pedometer",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.rmd",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.scrn_ortn",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.step_counter",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.step_detector",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.wu",
		.value = "false",
	},
	{
		.key = "ro.revision",
		.value = "rev_10",
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
		.value = "710KPQJ0358188",
	},
	{
		.key = "ro.setupwizard.enterprise_mode",
		.value = "1",
	},
	{
		.key = "ro.setupwizard.esim_cid_ignore",
		.value = "00000001",
	},
	{
		.key = "ro.setupwizard.rotation_locked",
		.value = "true",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "560",
	},
	{
		.key = "ro.storage_manager.enabled",
		.value = "true",
	},
	{
		.key = "ro.telephony.default_cdma_sub",
		.value = "0",
	},
	{
		.key = "ro.telephony.default_network",
		.value = "10",
	},
	{
		.key = "ro.treble.enabled",
		.value = "true",
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
		.value = "Tue Oct 3 04:55:08 UTC 2017",
	},
	{
		.key = "ro.vendor.build.date.utc",
		.value = "1507006508",
	},
	{
		.key = "ro.vendor.build.fingerprint",
		.value = "google/taimen/taimen:8.1.0/OPP5.170921.005/4373449:user/release-keys",
	},
	{
		.key = "ro.vendor.extension_library",
		.value = "libqti-perfd-client.so",
	},
	{
		.key = "ro.vendor.vndk.version",
		.value = "27.1.0",
	},
	{
		.key = "ro.vibrator.hal.click.duration",
		.value = "10",
	},
	{
		.key = "ro.vibrator.hal.tick.duration",
		.value = "4",
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
		.key = "service.sf.present_timestamp",
		.value = "1",
	},
	{
		.key = "setupwizard.enable_assist_gesture_training",
		.value = "true",
	},
	{
		.key = "setupwizard.theme",
		.value = "glif_v2_light",
	},
	{
		.key = "sys.all.modules.ready",
		.value = "1",
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
		.key = "sys.retaildemo.enabled",
		.value = "0",
	},
	{
		.key = "sys.slpi.firmware.version",
		.value = "msm8998-slpi_v2-g8b1164f-4324859 Fri Sep  8 21:06:35 UTC 2017",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "48600",
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
		.value = "a800000.dwc3",
	},
	{
		.key = "sys.usb.ffs.max_read",
		.value = "524288",
	},
	{
		.key = "sys.usb.ffs.max_write",
		.value = "524288",
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
		.key = "sys.usb.mtp.device_type",
		.value = "3",
	},
	{
		.key = "sys.usb.state",
		.value = "adb",
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
		.key = "vendor.audio.adm.buffering.ms",
		.value = "3",
	},
	{
		.key = "vendor.vidc.enc.dcvs.extra-buff-count",
		.value = "2",
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
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
