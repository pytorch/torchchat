struct cpuinfo_mock_file filesystem[] = {
#if CPUINFO_ARCH_ARM64
	{
		.path = "/proc/cpuinfo",
		.size = 1540,
		.content =
			"Processor\t: AArch64 Processor rev 4 (aarch64)\n"
			"processor\t: 0\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 4\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 5\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 6\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 7\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc MSM8998\n",
	},
#elif CPUINFO_ARCH_ARM
	{
		.path = "/proc/cpuinfo",
		.size = 2332,
		.content =
			"Processor\t: AArch64 Processor rev 4 (aarch64)\n"
			"processor\t: 0\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 1\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 2\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 3\n"
			"model name\t: ARMv8 Processor rev 4 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x801\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 4\n"
			"model name\t: ARMv8 Processor rev 1 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 5\n"
			"model name\t: ARMv8 Processor rev 1 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 6\n"
			"model name\t: ARMv8 Processor rev 1 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"processor\t: 7\n"
			"model name\t: ARMv8 Processor rev 1 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0xa\n"
			"CPU part\t: 0x800\n"
			"CPU revision\t: 1\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc MSM8998\n",
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
			"OnePlus5-user\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/image_version",
		.size = 24,
		.content =
			"10:OPR1.170623.032:119\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/images",
		.size = 637,
		.content =
			"0:\n"
			"\tCRM:\t\t00:BOOT.XF.1.2.2.c1-00021-M8998LZB-1\n"
			"\tVariant:\tMsm8998LA\n"
			"\tVersion:\t:ubuntu-23\n"
			"1:\n"
			"\tCRM:\t\t01:TZ.BF.4.0.6-00144\n"
			"\tVariant:\t \n"
			"\tVersion:\t:CRM\n"
			"3:\n"
			"\tCRM:\t\t03:RPM.BF.1.7-00128\n"
			"\tVariant:\tAAAAANAZR\n"
			"\tVersion:\t:ubuntu-23\n"
			"10:\n"
			"\tCRM:\t\t10:OPR1.170623.032:119\n"
			"\n"
			"\tVariant:\tOnePlus5-user\n"
			"\n"
			"\tVersion:\tREL\n"
			"\n"
			"11:\n"
			"\tCRM:\t\t11:MPSS.AT.2.0.c4.7-00070-8998_GEN_PACK-2.130961.1.131284.2\n"
			"\tVariant:\t8998.gen.prodQ\n"
			"\tVersion:\t:ubuntu-23\n"
			"12:\n"
			"\tCRM:\t\t12:ADSP.HT.3.0-00366-CB8998-1\n"
			"\tVariant:\tAAAAAAAAQ\n"
			"\tVersion:\t:ubuntu-23\n"
			"14:\n"
			"\tCRM:\t\t14:VIDEO.VE.4.4-00031\n"
			"\tVariant:\tPROD\n"
			"\tVersion:\t\n"
			"15:\n"
			"\tCRM:\t\t15:SLPI.HB.2.0.c3-00012-M8998AZL-1\n"
			"\tVariant:\tAAAAAAAAQ\n"
			"\tVersion:\t:ubuntu-23\n",
	},
	{
		.path = "/sys/devices/soc0/machine",
		.size = 8,
		.content = "MSM8998\n",
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
		.content = "65556\n",
	},
	{
		.path = "/sys/devices/soc0/raw_id",
		.size = 3,
		.content = "94\n",
	},
	{
		.path = "/sys/devices/soc0/raw_version",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/soc0/revision",
		.size = 4,
		.content = "2.1\n",
	},
	{
		.path = "/sys/devices/soc0/select_image",
		.size = 3,
		.content = "10\n",
	},
	{
		.path = "/sys/devices/soc0/serial_number",
		.size = 11,
		.content = "1620680061\n",
	},
	{
		.path = "/sys/devices/soc0/soc_id",
		.size = 4,
		.content = "292\n",
	},
	{
		.path = "/sys/devices/soc0/vendor",
		.size = 9,
		.content = "Qualcomm\n",
	},
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
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_governor_ro",
		.size = 5,
		.content = "qcom\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/active_cpus",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/busy_up_thres",
		.size = 9,
		.content = "0 0 0 0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/busy_down_thres",
		.size = 9,
		.content = "0 0 0 0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/enable",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/global_state",
		.size = 1336,
		.content =
			"CPU0\n"
			"\tCPU: 0\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 2\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 1\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU1\n"
			"\tCPU: 1\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 4\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 1\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU2\n"
			"\tCPU: 2\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 2\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 1\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU3\n"
			"\tCPU: 3\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 2\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 1\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU4\n"
			"\tCPU: 4\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNot preferred: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tNr isolated CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU5\n"
			"\tCPU: 5\n"
			"\tOnline: 1\n"
			"\tIsolated: 1\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNot preferred: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tNr isolated CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU6\n"
			"\tCPU: 6\n"
			"\tOnline: 1\n"
			"\tIsolated: 1\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNot preferred: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tNr isolated CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU7\n"
			"\tCPU: 7\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNot preferred: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tNr isolated CPUs: 2\n"
			"\tBoost: 0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/is_big_cluster",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/max_cpus",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/min_cpus",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/need_cpus",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/not_preferred",
		.size = 36,
		.content =
			"CPU#0: 0\n"
			"CPU#1: 0\n"
			"CPU#2: 0\n"
			"CPU#3: 0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/offline_delay_ms",
		.size = 4,
		.content = "100\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/task_thres",
		.size = 11,
		.content = "4294967295\n",
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
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "518400\n",
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
		.content = "518400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 234,
		.content =
			"300000 0\n"
			"364800 0\n"
			"441600 0\n"
			"518400 2291\n"
			"595200 25\n"
			"672000 32\n"
			"748800 31\n"
			"825600 31\n"
			"883200 23\n"
			"960000 20\n"
			"1036800 35\n"
			"1094400 4\n"
			"1171200 12\n"
			"1248000 548\n"
			"1324800 53\n"
			"1401600 14\n"
			"1478400 31\n"
			"1555200 60\n"
			"1670400 40\n"
			"1747200 20\n"
			"1824000 156\n"
			"1900800 4081\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "675\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/isolate",
		.size = 2,
		.content = "0\n",
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
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "518400\n",
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
		.content = "518400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 234,
		.content =
			"300000 0\n"
			"364800 0\n"
			"441600 0\n"
			"518400 2482\n"
			"595200 25\n"
			"672000 32\n"
			"748800 31\n"
			"825600 31\n"
			"883200 23\n"
			"960000 20\n"
			"1036800 35\n"
			"1094400 4\n"
			"1171200 12\n"
			"1248000 564\n"
			"1324800 53\n"
			"1401600 17\n"
			"1478400 31\n"
			"1555200 70\n"
			"1670400 48\n"
			"1747200 22\n"
			"1824000 176\n"
			"1900800 4115\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 4,
		.content = "709\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/isolate",
		.size = 2,
		.content = "0\n",
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
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "518400\n",
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
		.content = "518400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 234,
		.content =
			"300000 0\n"
			"364800 0\n"
			"441600 0\n"
			"518400 2769\n"
			"595200 25\n"
			"672000 32\n"
			"748800 33\n"
			"825600 31\n"
			"883200 23\n"
			"960000 20\n"
			"1036800 35\n"
			"1094400 4\n"
			"1171200 12\n"
			"1248000 596\n"
			"1324800 53\n"
			"1401600 17\n"
			"1478400 31\n"
			"1555200 70\n"
			"1670400 48\n"
			"1747200 22\n"
			"1824000 176\n"
			"1900800 4121\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 4,
		.content = "733\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/isolate",
		.size = 2,
		.content = "0\n",
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
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "518400\n",
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
		.content = "518400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 234,
		.content =
			"300000 0\n"
			"364800 0\n"
			"441600 0\n"
			"518400 3023\n"
			"595200 25\n"
			"672000 32\n"
			"748800 33\n"
			"825600 31\n"
			"883200 23\n"
			"960000 20\n"
			"1036800 35\n"
			"1094400 4\n"
			"1171200 16\n"
			"1248000 624\n"
			"1324800 53\n"
			"1401600 19\n"
			"1478400 31\n"
			"1555200 70\n"
			"1670400 48\n"
			"1747200 22\n"
			"1824000 176\n"
			"1900800 4133\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "759\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/isolate",
		.size = 2,
		.content = "0\n",
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
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/active_cpus",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/busy_up_thres",
		.size = 13,
		.content = "60 60 60 60 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/busy_down_thres",
		.size = 13,
		.content = "30 30 30 30 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/enable",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/global_state",
		.size = 1336,
		.content =
			"CPU0\n"
			"\tCPU: 0\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 1\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 1\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU1\n"
			"\tCPU: 1\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 1\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 1\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU2\n"
			"\tCPU: 2\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 0\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 1\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU3\n"
			"\tCPU: 3\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 4\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 1\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU4\n"
			"\tCPU: 4\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNot preferred: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tNr isolated CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU5\n"
			"\tCPU: 5\n"
			"\tOnline: 1\n"
			"\tIsolated: 1\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNot preferred: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tNr isolated CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU6\n"
			"\tCPU: 6\n"
			"\tOnline: 1\n"
			"\tIsolated: 1\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNot preferred: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tNr isolated CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU7\n"
			"\tCPU: 7\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNot preferred: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tNr isolated CPUs: 2\n"
			"\tBoost: 0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/is_big_cluster",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/max_cpus",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/min_cpus",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/need_cpus",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/not_preferred",
		.size = 36,
		.content =
			"CPU#4: 0\n"
			"CPU#5: 0\n"
			"CPU#6: 0\n"
			"CPU#7: 0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/offline_delay_ms",
		.size = 4,
		.content = "100\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/task_thres",
		.size = 2,
		.content = "4\n",
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
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "806400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2361600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "806400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 323,
		.content =
			"300000 0\n"
			"345600 0\n"
			"422400 0\n"
			"499200 0\n"
			"576000 0\n"
			"652800 0\n"
			"729600 0\n"
			"806400 5163\n"
			"902400 16\n"
			"979200 19\n"
			"1056000 14\n"
			"1132800 16\n"
			"1190400 5\n"
			"1267200 15\n"
			"1344000 10\n"
			"1420800 16\n"
			"1497600 12\n"
			"1574400 239\n"
			"1651200 14\n"
			"1728000 19\n"
			"1804800 9\n"
			"1881600 5\n"
			"1958400 31\n"
			"2035200 16\n"
			"2112000 9\n"
			"2208000 1\n"
			"2265600 4\n"
			"2323200 3\n"
			"2342400 1\n"
			"2361600 1990\n"
			"2457600 1104\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 4,
		.content = "333\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/isolate",
		.size = 2,
		.content = "0\n",
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
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "806400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2361600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "806400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 323,
		.content =
			"300000 0\n"
			"345600 0\n"
			"422400 0\n"
			"499200 0\n"
			"576000 0\n"
			"652800 0\n"
			"729600 0\n"
			"806400 5477\n"
			"902400 16\n"
			"979200 19\n"
			"1056000 14\n"
			"1132800 16\n"
			"1190400 5\n"
			"1267200 15\n"
			"1344000 10\n"
			"1420800 16\n"
			"1497600 12\n"
			"1574400 239\n"
			"1651200 14\n"
			"1728000 19\n"
			"1804800 9\n"
			"1881600 5\n"
			"1958400 31\n"
			"2035200 16\n"
			"2112000 9\n"
			"2208000 1\n"
			"2265600 4\n"
			"2323200 3\n"
			"2342400 1\n"
			"2361600 1990\n"
			"2457600 1104\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 4,
		.content = "333\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/isolate",
		.size = 2,
		.content = "1\n",
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
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "806400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2361600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "806400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 323,
		.content =
			"300000 0\n"
			"345600 0\n"
			"422400 0\n"
			"499200 0\n"
			"576000 0\n"
			"652800 0\n"
			"729600 0\n"
			"806400 5834\n"
			"902400 16\n"
			"979200 19\n"
			"1056000 14\n"
			"1132800 16\n"
			"1190400 5\n"
			"1267200 15\n"
			"1344000 10\n"
			"1420800 16\n"
			"1497600 12\n"
			"1574400 239\n"
			"1651200 14\n"
			"1728000 19\n"
			"1804800 9\n"
			"1881600 5\n"
			"1958400 31\n"
			"2035200 16\n"
			"2112000 9\n"
			"2208000 1\n"
			"2265600 4\n"
			"2323200 3\n"
			"2342400 1\n"
			"2361600 1990\n"
			"2457600 1104\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 4,
		.content = "333\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/isolate",
		.size = 2,
		.content = "1\n",
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
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "806400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2361600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "806400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 323,
		.content =
			"300000 0\n"
			"345600 0\n"
			"422400 0\n"
			"499200 0\n"
			"576000 0\n"
			"652800 0\n"
			"729600 0\n"
			"806400 6189\n"
			"902400 16\n"
			"979200 19\n"
			"1056000 14\n"
			"1132800 16\n"
			"1190400 5\n"
			"1267200 15\n"
			"1344000 10\n"
			"1420800 16\n"
			"1497600 12\n"
			"1574400 239\n"
			"1651200 14\n"
			"1728000 19\n"
			"1804800 9\n"
			"1881600 5\n"
			"1958400 31\n"
			"2035200 16\n"
			"2112000 9\n"
			"2208000 1\n"
			"2265600 4\n"
			"2323200 3\n"
			"2342400 1\n"
			"2361600 1990\n"
			"2457600 1104\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 4,
		.content = "333\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/isolate",
		.size = 2,
		.content = "0\n",
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
		.key = "af.fast_track_multiplier",
		.value = "2",
	},
	{
		.key = "audio.cts.media",
		.value = "false",
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
		.key = "audio.offload.video",
		.value = "true",
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
		.value = "cortex-a9",
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
		.value = "MPSS.AT.2.0.c4.7-00070-8998_GEN_PACK-2.130961.1.131284.2",
	},
	{
		.key = "gsm.version.baseband1",
		.value = "MPSS.AT.2.0.c4.7-00070-8998_GEN_PACK-2.130961.1.131284.2",
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
		.key = "init.svc.cnss-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.configstore-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.crashdata-sh",
		.value = "stopped",
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
		.key = "init.svc.drm-widevine-hal-1-0",
		.value = "running",
	},
	{
		.key = "init.svc.energy-awareness",
		.value = "stopped",
	},
	{
		.key = "init.svc.faceulnative",
		.value = "running",
	},
	{
		.key = "init.svc.filebuilderd",
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
		.key = "init.svc.hvdcp_opti",
		.value = "running",
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
		.key = "init.svc.ipacm",
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
		.key = "init.svc.mlid",
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
		.key = "init.svc.qdutils_disp-1-0",
		.value = "running",
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
		.key = "init.svc.qvop-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.qvrd",
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
		.key = "init.svc.thermal-hal-1-0",
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
		.key = "init.svc.tui_comm-1-0",
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
		.key = "init.svc.vendor.msm_irqbalance",
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
		.key = "init.svc.vr-1-0",
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
		.value = "16760831",
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
		.key = "nxpWechatDebugEnable",
		.value = "1",
	},
	{
		.key = "oem.device.imeicache0",
		.value = "864630032470279",
	},
	{
		.key = "oem.device.imeicache1",
		.value = "864630032470261",
	},
	{
		.key = "oem.drm.widevine.level",
		.value = "L3",
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
		.key = "persist.backup.ntpServer",
		.value = "\"0.pool.ntp.org\"",
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
		.value = "none",
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
		.key = "persist.dirac.acs.controller",
		.value = "qem",
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
		.key = "persist.nfc.smartcard.config",
		.value = "SIM1,eSE1",
	},
	{
		.key = "persist.oem.dump",
		.value = "0",
	},
	{
		.key = "persist.qua.op",
		.value = "3501500",
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
		.key = "persist.radio.multisim.config",
		.value = "dsds",
	},
	{
		.key = "persist.radio.serialno",
		.value = "54f801ef",
	},
	{
		.key = "persist.radio.start_ota_daemon",
		.value = "0",
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
		.key = "persist.sys.force_sw_gles",
		.value = "0",
	},
	{
		.key = "persist.sys.idle.soff",
		.value = "52,51304",
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
		.key = "persist.sys.public.type",
		.value = "daily",
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
		.value = "OnePlus5Oxygen_23.O.31_GLO_031_1802230119",
	},
	{
		.key = "persist.sys.timezone",
		.value = "America/New_York",
	},
	{
		.key = "persist.sys.ui.hw",
		.value = "true",
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
		.value = "OnePlus5Oxygen_23.O.31_GLO_031_1802230119",
	},
	{
		.key = "persist.sys.webview.vmsize",
		.value = "114785072",
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
		.key = "persist.ts.rtmakeup",
		.value = "false",
	},
	{
		.key = "persist.vendor.audio.aanc.enable",
		.value = "true",
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
		.value = "true",
	},
	{
		.key = "persist.vendor.audio.ras.enabled",
		.value = "false",
	},
	{
		.key = "persist.vendor.bt.a2dp_offload_cap",
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
		.key = "persist.vendor.radio.force_on_dc",
		.value = "true",
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
		.value = "cherokee",
	},
	{
		.key = "ril.ecclist",
		.value = "911,112,*911,#911,000,08,110,999,118,119",
	},
	{
		.key = "ril.ecclist1",
		.value = "911,112,*911,#911,000,08,110,999,118,119",
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
		.key = "ro.baseband",
		.value = "msm",
	},
	{
		.key = "ro.board.platform",
		.value = "msm8998",
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
		.key = "ro.boot.battery.absent",
		.value = "false",
	},
	{
		.key = "ro.boot.bootdevice",
		.value = "1da4000.ufshc",
	},
	{
		.key = "ro.boot.console",
		.value = "ttyMSM0",
	},
	{
		.key = "ro.boot.enable_dm_verity",
		.value = "1",
	},
	{
		.key = "ro.boot.flash.locked",
		.value = "1",
	},
	{
		.key = "ro.boot.hardware",
		.value = "qcom",
	},
	{
		.key = "ro.boot.hw_version",
		.value = "23",
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
		.key = "ro.boot.pcba_number",
		.value = "001685907522036000002439",
	},
	{
		.key = "ro.boot.project_name",
		.value = "16859",
	},
	{
		.key = "ro.boot.rf_version",
		.value = "53",
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
		.value = "54f801ef",
	},
	{
		.key = "ro.boot.startupmode",
		.value = "pwrkey",
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
		.value = "Fri Feb 23 01:13:37 CST 2018",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1519319617",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "OnePlus/OnePlus5/OnePlus5:8.0.0/OPR1.170623.032/01301703:user/release-keys",
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
		.value = "Fri Feb 23 01:13:37 CST 2018",
	},
	{
		.key = "ro.build.date.Ymd",
		.value = "180223",
	},
	{
		.key = "ro.build.date.YmdHM",
		.value = "201802230119",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1519319617",
	},
	{
		.key = "ro.build.date.ymd",
		.value = "180223",
	},
	{
		.key = "ro.build.description",
		.value = "OnePlus5-user 8.0.0 OPR1.170623.032 119 release-keys",
	},
	{
		.key = "ro.build.display.full_id",
		.value = "ONEPLUS A5000_23_O.31_180223",
	},
	{
		.key = "ro.build.display.id",
		.value = "ONEPLUS A5000_23_180223",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "OnePlus/OnePlus5/OnePlus5:8.0.0/OPR1.170623.032/01301703:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "OnePlus5-user",
	},
	{
		.key = "ro.build.host",
		.value = "ubuntu-23",
	},
	{
		.key = "ro.build.id",
		.value = "OPR1.170623.032",
	},
	{
		.key = "ro.build.id.hardware",
		.value = "ONEPLUS A5000_23_",
	},
	{
		.key = "ro.build.kernel.id",
		.value = "4.4-G1802230119",
	},
	{
		.key = "ro.build.oemfingerprint",
		.value = "8.0.0/OPR1.170623.032/01301703:user/release-keys",
	},
	{
		.key = "ro.build.ota.versionname",
		.value = "OnePlus5Oxygen_23_1802230119",
	},
	{
		.key = "ro.build.product",
		.value = "OnePlus5",
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
		.value = "O.31",
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
		.value = "119",
	},
	{
		.key = "ro.build.version.ota",
		.value = "OnePlus5Oxygen_23.O.31_GLO_031_1802230119",
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
		.value = "8.0_r4",
	},
	{
		.key = "ro.common.soft",
		.value = "OnePlus5",
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
		.key = "ro.dirac.acs.storeSettings",
		.value = "1",
	},
	{
		.key = "ro.dirac.ignore_error",
		.value = "1",
	},
	{
		.key = "ro.display.series",
		.value = "OnePlus 5",
	},
	{
		.key = "ro.expect.recovery_id",
		.value = "0xfee05c25867ac2fa97d3aee3537a9a8fbb7520b1000000000000000000000000",
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
		.key = "ro.hwui.text_large_cache_height",
		.value = "4096",
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
		.key = "ro.nfc.port",
		.value = "I2C",
	},
	{
		.key = "ro.oem_unlock_supported",
		.value = "true",
	},
	{
		.key = "ro.oxygen.version",
		.value = "5.0.4",
	},
	{
		.key = "ro.product.board",
		.value = "msm8998",
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
		.value = "OnePlus5",
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
		.value = "ONEPLUS A5000",
	},
	{
		.key = "ro.product.name",
		.value = "OnePlus5",
	},
	{
		.key = "ro.property_service.version",
		.value = "2",
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
		.value = "TDD_FDD_All",
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
		.value = "54f801ef",
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
		.value = "22,20",
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
		.value = "fluencepro",
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
		.key = "ro.vendor.qti.core_ctl_max_cpu",
		.value = "4",
	},
	{
		.key = "ro.vendor.qti.core_ctl_min_cpu",
		.value = "2",
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
		.key = "ro.vendor.sensors.dev_ori",
		.value = "true",
	},
	{
		.key = "ro.vendor.sensors.dpc",
		.value = "true",
	},
	{
		.key = "ro.vendor.sensors.mot_detect",
		.value = "true",
	},
	{
		.key = "ro.vendor.sensors.multishake",
		.value = "true",
	},
	{
		.key = "ro.vendor.sensors.pmd",
		.value = "true",
	},
	{
		.key = "ro.vendor.sensors.sta_detect",
		.value = "true",
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
		.key = "sched.colocate.enable",
		.value = "1",
	},
	{
		.key = "sdm.debug.disable_skip_validate",
		.value = "1",
	},
	{
		.key = "sdm.perf_hint_window",
		.value = "0",
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
		.value = "gsi",
	},
	{
		.key = "sys.usb.rndis.func.name",
		.value = "gsi",
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
		.key = "vendor.audio.dolby.ds2.enabled",
		.value = "false",
	},
	{
		.key = "vendor.audio.dolby.ds2.hardbypass",
		.value = "false",
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
		.key = "vendor.audio.noisy.broadcast.delay",
		.value = "600",
	},
	{
		.key = "vendor.audio.offload.buffer.size.kb",
		.value = "32",
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
		.value = "true",
	},
	{
		.key = "vendor.audio.parser.ip.buffer.size",
		.value = "0",
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
		.key = "vendor.camera.aux.packagelist",
		.value = "org.codeaurora.snapcam,com.oneplus.camera,com.oneplus.factorymode",
	},
	{
		.key = "vendor.display.enable_default_color_mode",
		.value = "0",
	},
	{
		.key = "vendor.fm.a2dp.conc.disabled",
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
		.value = "EXT4",
	},
	{
		.key = "vold.emulated.ready",
		.value = "1",
	},
	{
		.key = "vold.fbe.decrypted",
		.value = "1",
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
		.key = "vold.internalSD.startcopy",
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
