struct cpuinfo_mock_file filesystem[] = {
#if CPUINFO_ARCH_ARM64
	{
		.path = "/proc/cpuinfo",
		.size = 1716,
		.content =
			"Processor\t: AArch64 Processor rev 12 (aarch64)\n"
			"processor\t: 0\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x7\n"
			"CPU part\t: 0x803\n"
			"CPU revision\t: 12\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x7\n"
			"CPU part\t: 0x803\n"
			"CPU revision\t: 12\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x7\n"
			"CPU part\t: 0x803\n"
			"CPU revision\t: 12\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x7\n"
			"CPU part\t: 0x803\n"
			"CPU revision\t: 12\n"
			"\n"
			"processor\t: 4\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x6\n"
			"CPU part\t: 0x802\n"
			"CPU revision\t: 13\n"
			"\n"
			"processor\t: 5\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x6\n"
			"CPU part\t: 0x802\n"
			"CPU revision\t: 13\n"
			"\n"
			"processor\t: 6\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x6\n"
			"CPU part\t: 0x802\n"
			"CPU revision\t: 13\n"
			"\n"
			"processor\t: 7\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x6\n"
			"CPU part\t: 0x802\n"
			"CPU revision\t: 13\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc SDM845\n",
	},
#elif CPUINFO_ARCH_ARM
	{
		.path = "/proc/cpuinfo",
		.size = 2348,
		.content =
			"Processor\t: AArch64 Processor rev 13 (aarch64)\n"
			"processor\t: 0\n"
			"model name\t: ARMv8 Processor rev 12 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x7\n"
			"CPU part\t: 0x803\n"
			"CPU revision\t: 12\n"
			"\n"
			"processor\t: 1\n"
			"model name\t: ARMv8 Processor rev 12 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x7\n"
			"CPU part\t: 0x803\n"
			"CPU revision\t: 12\n"
			"\n"
			"processor\t: 2\n"
			"model name\t: ARMv8 Processor rev 12 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x7\n"
			"CPU part\t: 0x803\n"
			"CPU revision\t: 12\n"
			"\n"
			"processor\t: 3\n"
			"model name\t: ARMv8 Processor rev 12 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x7\n"
			"CPU part\t: 0x803\n"
			"CPU revision\t: 12\n"
			"\n"
			"processor\t: 4\n"
			"model name\t: ARMv8 Processor rev 13 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x6\n"
			"CPU part\t: 0x802\n"
			"CPU revision\t: 13\n"
			"\n"
			"processor\t: 5\n"
			"model name\t: ARMv8 Processor rev 13 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x6\n"
			"CPU part\t: 0x802\n"
			"CPU revision\t: 13\n"
			"\n"
			"processor\t: 6\n"
			"model name\t: ARMv8 Processor rev 13 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x6\n"
			"CPU part\t: 0x802\n"
			"CPU revision\t: 13\n"
			"\n"
			"processor\t: 7\n"
			"model name\t: ARMv8 Processor rev 13 (v8l)\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x51\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x6\n"
			"CPU part\t: 0x802\n"
			"CPU revision\t: 13\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc SDM845\n",
	},
#endif
	{
		.path = "/sys/class/kgsl/kgsl-3d0/bus_split",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/clock_mhz",
		.size = 4,
		.content = "257\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/default_pwrlevel",
		.size = 2,
		.content = "6\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/dev",
		.size = 6,
		.content = "234:0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/freq_table_mhz",
		.size = 29,
		.content = "710 675 596 520 414 342 257 \n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/ft_hang_intr_status",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/ft_long_ib_detect",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/ft_pagefault_policy",
		.size = 4,
		.content = "0x0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/ft_policy",
		.size = 5,
		.content = "0xC2\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_available_frequencies",
		.size = 71,
		.content = "710000000 675000000 596000000 520000000 414000000 342000000 257000000 \n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
		.size = 4,
		.content = "7 %\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_clock_stats",
		.size = 28,
		.content = "82129 3395 0 0 0 0 4684346 \n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_llc_slice_enable",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_model",
		.size = 12,
		.content = "Adreno630v2\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpubusy",
		.size = 16,
		.content = "      0       0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpuclk",
		.size = 10,
		.content = "257000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpuhtw_llc_slice_enable",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/hwcg",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/idle_timer",
		.size = 3,
		.content = "80\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/lm",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/max_gpuclk",
		.size = 10,
		.content = "710000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/max_pwrlevel",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/min_clock_mhz",
		.size = 4,
		.content = "257\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/min_pwrlevel",
		.size = 2,
		.content = "6\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/num_pwrlevels",
		.size = 2,
		.content = "7\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/pmqos_active_latency",
		.size = 4,
		.content = "460\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/popp",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/preempt_count",
		.size = 4,
		.content = "0x1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/preempt_level",
		.size = 4,
		.content = "0x1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/preemption",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/pwrscale",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/reset_count",
		.size = 4,
		.content = "325\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/skipsaverestore",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/sptp_pc",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/thermal_pwrlevel",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/throttling",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/usesgmem",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/wake_nice",
		.size = 3,
		.content = "-7\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/wake_timeout",
		.size = 4,
		.content = "100\n",
	},
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
		.path = "/sys/devices/soc0/chip_family",
		.size = 5,
		.content = "0x4f\n",
	},
	{
		.path = "/sys/devices/soc0/chip_name",
		.size = 7,
		.content = "SDM845\n",
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
		.size = 17,
		.content =
			"starqltesq-user\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/image_version",
		.size = 24,
		.content =
			"10:R16NW:G960USQU1ARB7\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/images",
		.size = 698,
		.content =
			"0:\n"
			"\tCRM:\t\t00:BOOT.XF.2.0-00340-SDM845LZB-1\n"
			"\tVariant:\tSDM845LA\n"
			"\tVersion:\tSWDG4715\n"
			"\n"
			"1:\n"
			"\tCRM:\t\t01:TZ.XF.5.0.1-132629-63\n"
			"\tVariant:\t \n"
			"\tVersion:\tCRM\n"
			"\n"
			"10:\n"
			"\tCRM:\t\t10:R16NW:G960USQU1ARB7\n"
			"\n"
			"\tVariant:\tstarqltesq-user\n"
			"\n"
			"\tVersion:\tREL\n"
			"\n"
			"\n"
			"11:\n"
			"\tCRM:\t\t11:MPSS.AT.4.0.c2.5-00045-SDM845_GEN_PACK-1.135721.1\n"
			"\tVariant:\tsdm845.gen.prodQ\n"
			"\tVersion:\tSWDG4510-VM01\n"
			"\n"
			"12:\n"
			"\tCRM:\t\t12:ADSP.HT.4.0.c2-00006-SDM845-1\n"
			"\tVariant:\t845.adsp.prodQ\n"
			"\tVersion:\tSWDG4510-VM01\n"
			"\n"
			"14:\n"
			"\tCRM:\t\t14:VIDEO.VE.5.0-00062-PROD-1536691\n"
			"\tVariant:\tPROD\n"
			"\tVersion:\t:HARV-MMUNDHRA\n"
			"\n"
			"15:\n"
			"\tCRM:\t\t15:SLPI.HY.1.0-00272-SDM845AZL-1.129244.1\n"
			"\tVariant:\tAAAAAAAAQ\n"
			"\tVersion:\tSWDG4715\n"
			"\n"
			"16:\n"
			"\tCRM:\t\t16:CDSP.HT.1.0-00428-SDM845-1\n"
			"\tVariant:\t845.cdsp.prodQ\n"
			"\tVersion:\tSWDG4715\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/machine",
		.size = 7,
		.content = "SDM845\n",
	},
	{
		.path = "/sys/devices/soc0/ncluster_array_offset",
		.size = 5,
		.content = "0xb0\n",
	},
	{
		.path = "/sys/devices/soc0/ndefective_parts_array_offset",
		.size = 5,
		.content = "0xb4\n",
	},
	{
		.path = "/sys/devices/soc0/nmodem_supported",
		.size = 5,
		.content = "0xff\n",
	},
	{
		.path = "/sys/devices/soc0/nproduct_id",
		.size = 6,
		.content = "0x3f4\n",
	},
	{
		.path = "/sys/devices/soc0/num_clusters",
		.size = 4,
		.content = "0x1\n",
	},
	{
		.path = "/sys/devices/soc0/num_defective_parts",
		.size = 4,
		.content = "0x6\n",
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
		.path = "/sys/devices/soc0/raw_device_family",
		.size = 4,
		.content = "0x6\n",
	},
	{
		.path = "/sys/devices/soc0/raw_device_number",
		.size = 4,
		.content = "0x0\n",
	},
	{
		.path = "/sys/devices/soc0/raw_id",
		.size = 4,
		.content = "139\n",
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
		.content = "3372745183\n",
	},
	{
		.path = "/sys/devices/soc0/soc_id",
		.size = 4,
		.content = "321\n",
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
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/big_cpu_num",
		.size = 4,
		.content = "4-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/big_max_freq",
		.size = 11,
		.content = "4294967295\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/big_min_freq",
		.size = 7,
		.content = "902400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/hmp_boost_type",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/hmp_prev_boost_type",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/ltl_cpu_num",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/ltl_divider",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/ltl_max_freq",
		.size = 7,
		.content = "883200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/ltl_min_freq",
		.size = 7,
		.content = "150000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/ltl_min_lock",
		.size = 7,
		.content = "566400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/cpufreq_limit/requests",
		.size = 21,
		.content = "label\t\tmin\tmax\tsince\n",
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
		.size = 1340,
		.content =
			"CPU0\n"
			"\tCPU: 0\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 30\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 6\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU1\n"
			"\tCPU: 1\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 20\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 6\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU2\n"
			"\tCPU: 2\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 22\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 6\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU3\n"
			"\tCPU: 3\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 8\n"
			"\tIs busy: 1\n"
			"\tNot preferred: 0\n"
			"\tNr running: 6\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tNr isolated CPUs: 0\n"
			"\tBoost: 0\n"
			"CPU4\n"
			"\tCPU: 4\n"
			"\tOnline: 1\n"
			"\tIsolated: 1\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNot preferred: 0\n"
			"\tNr running: 1\n"
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
			"\tNr running: 1\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tNr isolated CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU6\n"
			"\tCPU: 6\n"
			"\tOnline: 1\n"
			"\tIsolated: 0\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 11\n"
			"\tIs busy: 0\n"
			"\tNot preferred: 0\n"
			"\tNr running: 1\n"
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
			"\tNr running: 1\n"
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
		.content = "1766400\n",
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
		.size = 136,
		.content = "300000 403200 480000 576000 652800 748800 825600 902400 979200 1056000 1132800 1228800 1324800 1420800 1516800 1612800 1689600 1766400 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 23,
		.content = "performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "748800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 12,
		.content = "osm-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1766400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "748800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 207,
		.content =
			"300000 0\n"
			"403200 0\n"
			"480000 0\n"
			"576000 0\n"
			"652800 0\n"
			"748800 66938\n"
			"825600 1324\n"
			"902400 680\n"
			"979200 569\n"
			"1056000 839\n"
			"1132800 3428\n"
			"1228800 4671\n"
			"1324800 1242\n"
			"1420800 3520\n"
			"1516800 1109\n"
			"1612800 830\n"
			"1689600 477\n"
			"1766400 11777\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 6,
		.content = "11086\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/isolate",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000517f803c\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/sched_load_boost",
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
		.content = "128\n",
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
		.content = "4\n",
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
		.size = 4,
		.content = "512\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/shared_cpu_list",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "01\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/size",
		.size = 5,
		.content = "128K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index3/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index3/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index3/level",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index3/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_map",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index3/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index3/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index3/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index3/write_policy",
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
		.content = "1766400\n",
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
		.size = 136,
		.content = "300000 403200 480000 576000 652800 748800 825600 902400 979200 1056000 1132800 1228800 1324800 1420800 1516800 1612800 1689600 1766400 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 23,
		.content = "performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "748800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
		.size = 12,
		.content = "osm-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1766400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "748800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 207,
		.content =
			"300000 0\n"
			"403200 0\n"
			"480000 0\n"
			"576000 0\n"
			"652800 0\n"
			"748800 67202\n"
			"825600 1324\n"
			"902400 680\n"
			"979200 569\n"
			"1056000 839\n"
			"1132800 3428\n"
			"1228800 4671\n"
			"1324800 1242\n"
			"1420800 3520\n"
			"1516800 1109\n"
			"1612800 830\n"
			"1689600 477\n"
			"1766400 11777\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 6,
		.content = "11086\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/isolate",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000517f803c\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/sched_load_boost",
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
		.content = "128\n",
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
		.content = "4\n",
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
		.size = 4,
		.content = "512\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/shared_cpu_list",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "02\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/size",
		.size = 5,
		.content = "128K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index3/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index3/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index3/level",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index3/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index3/shared_cpu_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index3/shared_cpu_map",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index3/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index3/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index3/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index3/write_policy",
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
		.content = "1766400\n",
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
		.size = 136,
		.content = "300000 403200 480000 576000 652800 748800 825600 902400 979200 1056000 1132800 1228800 1324800 1420800 1516800 1612800 1689600 1766400 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 23,
		.content = "performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "748800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver",
		.size = 12,
		.content = "osm-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1766400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "748800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 207,
		.content =
			"300000 0\n"
			"403200 0\n"
			"480000 0\n"
			"576000 0\n"
			"652800 0\n"
			"748800 67472\n"
			"825600 1326\n"
			"902400 680\n"
			"979200 570\n"
			"1056000 839\n"
			"1132800 3428\n"
			"1228800 4673\n"
			"1324800 1242\n"
			"1420800 3520\n"
			"1516800 1109\n"
			"1612800 830\n"
			"1689600 477\n"
			"1766400 11777\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 6,
		.content = "11090\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/isolate",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000517f803c\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/sched_load_boost",
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
		.content = "128\n",
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
		.content = "4\n",
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
		.size = 4,
		.content = "512\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/shared_cpu_list",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "04\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/size",
		.size = 5,
		.content = "128K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index3/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index3/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index3/level",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index3/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index3/shared_cpu_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index3/shared_cpu_map",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index3/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index3/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index3/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index3/write_policy",
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
		.content = "1766400\n",
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
		.size = 136,
		.content = "300000 403200 480000 576000 652800 748800 825600 902400 979200 1056000 1132800 1228800 1324800 1420800 1516800 1612800 1689600 1766400 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 23,
		.content = "performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "748800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_driver",
		.size = 12,
		.content = "osm-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1766400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "748800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 207,
		.content =
			"300000 0\n"
			"403200 0\n"
			"480000 0\n"
			"576000 0\n"
			"652800 0\n"
			"748800 67730\n"
			"825600 1331\n"
			"902400 683\n"
			"979200 570\n"
			"1056000 840\n"
			"1132800 3428\n"
			"1228800 4677\n"
			"1324800 1242\n"
			"1420800 3520\n"
			"1516800 1111\n"
			"1612800 832\n"
			"1689600 477\n"
			"1766400 11784\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 6,
		.content = "11114\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/isolate",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000517f803c\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/sched_load_boost",
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
		.content = "128\n",
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
		.content = "4\n",
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
		.size = 4,
		.content = "512\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/shared_cpu_list",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "08\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/size",
		.size = 5,
		.content = "128K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index3/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index3/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index3/level",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index3/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index3/shared_cpu_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index3/shared_cpu_map",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index3/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index3/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index3/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index3/write_policy",
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
			"\tBusy%: 4\n"
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
			"\tBusy%: 0\n"
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
		.content = "2803200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "825600\n",
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
		.size = 182,
		.content = "825600 902400 979200 1056000 1209600 1286400 1363200 1459200 1536000 1612800 1689600 1766400 1843200 1920000 1996800 2092800 2169600 2246400 2323200 2400000 2476800 2553600 2649600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors",
		.size = 23,
		.content = "performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "825600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver",
		.size = 12,
		.content = "osm-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2803200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "825600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 280,
		.content =
			"825600 87365\n"
			"902400 90\n"
			"979200 211\n"
			"1056000 540\n"
			"1209600 409\n"
			"1286400 268\n"
			"1363200 443\n"
			"1459200 126\n"
			"1536000 56\n"
			"1612800 499\n"
			"1689600 251\n"
			"1766400 156\n"
			"1843200 87\n"
			"1920000 85\n"
			"1996800 172\n"
			"2092800 115\n"
			"2169600 96\n"
			"2246400 70\n"
			"2323200 687\n"
			"2400000 98\n"
			"2476800 238\n"
			"2553600 75\n"
			"2649600 118\n"
			"2803200 6280\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 5,
		.content = "6193\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/isolate",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000516f802d\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000001\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/sched_load_boost",
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
		.size = 4,
		.content = "512\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/shared_cpu_list",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "10\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/size",
		.size = 5,
		.content = "256K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "8\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index3/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index3/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index3/level",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index3/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index3/shared_cpu_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index3/shared_cpu_map",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index3/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index3/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index3/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cache/index3/write_policy",
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
		.content = "2803200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "825600\n",
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
		.size = 182,
		.content = "825600 902400 979200 1056000 1209600 1286400 1363200 1459200 1536000 1612800 1689600 1766400 1843200 1920000 1996800 2092800 2169600 2246400 2323200 2400000 2476800 2553600 2649600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_governors",
		.size = 23,
		.content = "performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "825600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_driver",
		.size = 12,
		.content = "osm-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2803200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "825600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 280,
		.content =
			"825600 87664\n"
			"902400 90\n"
			"979200 211\n"
			"1056000 540\n"
			"1209600 409\n"
			"1286400 268\n"
			"1363200 443\n"
			"1459200 126\n"
			"1536000 56\n"
			"1612800 499\n"
			"1689600 251\n"
			"1766400 156\n"
			"1843200 87\n"
			"1920000 85\n"
			"1996800 172\n"
			"2092800 115\n"
			"2169600 96\n"
			"2246400 70\n"
			"2323200 687\n"
			"2400000 98\n"
			"2476800 238\n"
			"2553600 75\n"
			"2649600 118\n"
			"2803200 6280\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 5,
		.content = "6193\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/isolate",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000516f802d\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000001\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/sched_load_boost",
		.size = 2,
		.content = "0\n",
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
		.size = 4,
		.content = "512\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/shared_cpu_list",
		.size = 2,
		.content = "5\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "20\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/size",
		.size = 5,
		.content = "256K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "8\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index3/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index3/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index3/level",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index3/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index3/shared_cpu_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index3/shared_cpu_map",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index3/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index3/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index3/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cache/index3/write_policy",
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
		.content = "2803200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "825600\n",
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
		.size = 182,
		.content = "825600 902400 979200 1056000 1209600 1286400 1363200 1459200 1536000 1612800 1689600 1766400 1843200 1920000 1996800 2092800 2169600 2246400 2323200 2400000 2476800 2553600 2649600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_governors",
		.size = 23,
		.content = "performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "825600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_driver",
		.size = 12,
		.content = "osm-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2803200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "825600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 280,
		.content =
			"825600 87975\n"
			"902400 90\n"
			"979200 211\n"
			"1056000 540\n"
			"1209600 409\n"
			"1286400 272\n"
			"1363200 443\n"
			"1459200 126\n"
			"1536000 56\n"
			"1612800 499\n"
			"1689600 251\n"
			"1766400 156\n"
			"1843200 87\n"
			"1920000 85\n"
			"1996800 172\n"
			"2092800 115\n"
			"2169600 96\n"
			"2246400 70\n"
			"2323200 687\n"
			"2400000 98\n"
			"2476800 238\n"
			"2553600 75\n"
			"2649600 118\n"
			"2803200 6280\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 5,
		.content = "6195\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/isolate",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000516f802d\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000001\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/sched_load_boost",
		.size = 2,
		.content = "0\n",
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
		.size = 4,
		.content = "512\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/shared_cpu_list",
		.size = 2,
		.content = "6\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "40\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/size",
		.size = 5,
		.content = "256K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "8\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index3/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index3/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index3/level",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index3/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index3/shared_cpu_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index3/shared_cpu_map",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index3/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index3/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index3/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cache/index3/write_policy",
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
		.content = "2803200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "825600\n",
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
		.size = 182,
		.content = "825600 902400 979200 1056000 1209600 1286400 1363200 1459200 1536000 1612800 1689600 1766400 1843200 1920000 1996800 2092800 2169600 2246400 2323200 2400000 2476800 2553600 2649600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_governors",
		.size = 23,
		.content = "performance schedutil \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "825600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_driver",
		.size = 12,
		.content = "osm-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_governor",
		.size = 10,
		.content = "schedutil\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "2803200\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "825600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 280,
		.content =
			"825600 88241\n"
			"902400 90\n"
			"979200 211\n"
			"1056000 540\n"
			"1209600 409\n"
			"1286400 273\n"
			"1363200 448\n"
			"1459200 127\n"
			"1536000 58\n"
			"1612800 500\n"
			"1689600 253\n"
			"1766400 156\n"
			"1843200 87\n"
			"1920000 85\n"
			"1996800 173\n"
			"2092800 115\n"
			"2169600 96\n"
			"2246400 70\n"
			"2323200 687\n"
			"2400000 98\n"
			"2476800 240\n"
			"2553600 75\n"
			"2649600 118\n"
			"2803200 6293\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 5,
		.content = "6266\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/isolate",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/regs/identification/midr_el1",
		.size = 19,
		.content = "0x00000000516f802d\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/regs/identification/revidr_el1",
		.size = 19,
		.content = "0x0000000000000001\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/sched_load_boost",
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
		.size = 4,
		.content = "512\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/shared_cpu_list",
		.size = 2,
		.content = "7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/shared_cpu_map",
		.size = 3,
		.content = "80\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/size",
		.size = 5,
		.content = "256K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "8\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index2/write_policy",
		.size = 10,
		.content = "WriteBack\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index3/allocation_policy",
		.size = 18,
		.content = "ReadWriteAllocate\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index3/coherency_line_size",
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index3/level",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index3/number_of_sets",
		.size = 5,
		.content = "2048\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index3/shared_cpu_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index3/shared_cpu_map",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index3/size",
		.size = 6,
		.content = "2048K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index3/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index3/ways_of_associativity",
		.size = 3,
		.content = "16\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cache/index3/write_policy",
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
		.value = "1",
	},
	{
		.key = "audio.deep_buffer.media",
		.value = "true",
	},
	{
		.key = "audio.offload.buffer.size.kb",
		.value = "32",
	},
	{
		.key = "audio.offload.gapless.enabled",
		.value = "true",
	},
	{
		.key = "audio.offload.video",
		.value = "true",
	},
	{
		.key = "audioflinger.bootsnd",
		.value = "0",
	},
	{
		.key = "av.offload.enable",
		.value = "true",
	},
	{
		.key = "bt.max.hfpclient.connections",
		.value = "1",
	},
	{
		.key = "config.disable_consumerir",
		.value = "true",
	},
	{
		.key = "config.disable_rtt",
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
		.value = "cortex-a9",
	},
	{
		.key = "dalvik.vm.isa.arm64.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.arm64.variant",
		.value = "kryo300",
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
		.value = "0",
	},
	{
		.key = "debug.force_rtl",
		.value = "0",
	},
	{
		.key = "debug.gralloc.gfx_ubwc_disable",
		.value = "0",
	},
	{
		.key = "debug.mdpcomp.logs",
		.value = "0",
	},
	{
		.key = "debug.sensor.logging.slpi",
		.value = "true",
	},
	{
		.key = "debug.sf.enable_hwc_vds",
		.value = "1",
	},
	{
		.key = "debug.sf.hw",
		.value = "0",
	},
	{
		.key = "debug.sf.latch_unsignaled",
		.value = "1",
	},
	{
		.key = "debug.sf.layerdump",
		.value = "0",
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
		.key = "dev.pm.dyn_samplingrate",
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
		.key = "dev.ssrm.atc_ap_current",
		.value = "0,0,0,0,0,0,0,0",
	},
	{
		.key = "dev.ssrm.atc_ap_power",
		.value = "0,0,0,0,0,0,0,0",
	},
	{
		.key = "dev.ssrm.atc_etc_current",
		.value = "77,16,61,0,0,0,0",
	},
	{
		.key = "dev.ssrm.atc_etc_power",
		.value = "308,64,244,0,0,0,0",
	},
	{
		.key = "dev.ssrm.gamelevel",
		.value = "-4,6,-2,4",
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
		.value = "241",
	},
	{
		.key = "dev.ssrm.smart_switch",
		.value = "true",
	},
	{
		.key = "diag.oriented",
		.value = "APO",
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
		.value = "us",
	},
	{
		.key = "gsm.operator.ispsroaming",
		.value = "false",
	},
	{
		.key = "gsm.operator.isroaming",
		.value = "false",
	},
	{
		.key = "gsm.operator.numeric",
		.value = "310410",
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
		.value = "G960USQU1ARB7",
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
		.key = "init.svc.DR-daemon",
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
		.key = "init.svc.adsprpcd",
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
		.value = "stopped",
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
		.key = "init.svc.cdsprpcd",
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
		.key = "init.svc.cs-early-boot",
		.value = "stopped",
	},
	{
		.key = "init.svc.cs-post-boot",
		.value = "stopped",
	},
	{
		.key = "init.svc.dhkprov1x",
		.value = "stopped",
	},
	{
		.key = "init.svc.dhkprov2x",
		.value = "stopped",
	},
	{
		.key = "init.svc.diag_uart_log",
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
		.key = "init.svc.factory_ssc",
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
		.key = "init.svc.insthk",
		.value = "stopped",
	},
	{
		.key = "init.svc.iod",
		.value = "running",
	},
	{
		.key = "init.svc.iop-hal-2-0",
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
		.key = "init.svc.irisd",
		.value = "running",
	},
	{
		.key = "init.svc.irsc_util",
		.value = "stopped",
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
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.nxpnfc_hal_svc",
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
		.key = "init.svc.per_proxy_helper",
		.value = "stopped",
	},
	{
		.key = "init.svc.perf-hal-1-0",
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
		.key = "init.svc.pvclicense_sample",
		.value = "stopped",
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
		.key = "init.svc.qti_esepowermanager_service",
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
		.key = "init.svc.rmt_storage",
		.value = "running",
	},
	{
		.key = "init.svc.run-mobicore",
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
		.key = "init.svc.secure_storage",
		.value = "running",
	},
	{
		.key = "init.svc.seemp_healthd",
		.value = "running",
	},
	{
		.key = "init.svc.sem_daemon",
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
		.key = "init.svc.smcinvoked",
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
		.key = "init.svc.tbaseLoader",
		.value = "stopped",
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
		.key = "iop.enable_prefetch_ofr",
		.value = "1",
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
		.key = "media.stagefright.enable-aac",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-fma2dp",
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
		.key = "net.tcp.default_init_rwnd",
		.value = "60",
	},
	{
		.key = "nfc.boot.reason",
		.value = "1",
	},
	{
		.key = "nfc.fw.dfl_areacode",
		.value = "ATT",
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
		.value = "MAJ:15, MIN: 2",
	},
	{
		.key = "nfc.fw.ver",
		.value = "NXP 11.1.15",
	},
	{
		.key = "nfc.nxp.fwdnldstatus",
		.value = "0",
	},
	{
		.key = "nfc.product.support.ese",
		.value = "1",
	},
	{
		.key = "nfc.product.support.uicc",
		.value = "0",
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
		.key = "persist.audio.fluence.speaker",
		.value = "true",
	},
	{
		.key = "persist.audio.fluence.voicecall",
		.value = "true",
	},
	{
		.key = "persist.audio.fluence.voicerec",
		.value = "false",
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
		.value = "ATT",
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
		.value = "585600",
	},
	{
		.key = "persist.cne.feature",
		.value = "0",
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
		.key = "persist.data.mode",
		.value = "concurrent",
	},
	{
		.key = "persist.data.netmgrd.qos.enable",
		.value = "false",
	},
	{
		.key = "persist.data.wda.enable",
		.value = "true",
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
		.key = "persist.mm.enable.prefetch",
		.value = "true",
	},
	{
		.key = "persist.nfc.log.index",
		.value = "1",
	},
	{
		.key = "persist.radio.add_power_save",
		.value = "1",
	},
	{
		.key = "persist.radio.apm_sim_not_pwdn",
		.value = "1",
	},
	{
		.key = "persist.radio.atfwd.start",
		.value = "true",
	},
	{
		.key = "persist.radio.latest-modeltype",
		.value = "2",
	},
	{
		.key = "persist.radio.lte_vrte_ltd",
		.value = "1",
	},
	{
		.key = "persist.radio.max_ims_instance",
		.value = "2",
	},
	{
		.key = "persist.radio.multisim.config",
		.value = "ss",
	},
	{
		.key = "persist.radio.new.profid",
		.value = "true",
	},
	{
		.key = "persist.radio.sib16_support",
		.value = "1",
	},
	{
		.key = "persist.radio.silent-reset",
		.value = "56",
	},
	{
		.key = "persist.radio.sim.onoff",
		.value = "1",
	},
	{
		.key = "persist.ril.dfm.srlte",
		.value = "false",
	},
	{
		.key = "persist.ril.ims.eutranParam",
		.value = "3",
	},
	{
		.key = "persist.ril.ims.utranParam",
		.value = "0",
	},
	{
		.key = "persist.ril.modem.board",
		.value = "SDM845",
	},
	{
		.key = "persist.ril.radiocapa.tdscdma",
		.value = "true",
	},
	{
		.key = "persist.rmnet.data.enable",
		.value = "true",
	},
	{
		.key = "persist.service.tspcmd.spay",
		.value = "true",
	},
	{
		.key = "persist.sys.ccm.date",
		.value = "Sun Feb 11 14:49:26 KST 2018",
	},
	{
		.key = "persist.sys.clssprld1",
		.value = "1000",
	},
	{
		.key = "persist.sys.clssprld2",
		.value = "228",
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
		.key = "persist.sys.force_sw_gles",
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
		.value = "en-US",
	},
	{
		.key = "persist.sys.localedefault",
		.value = "",
	},
	{
		.key = "persist.sys.localenosim",
		.value = "en-US",
	},
	{
		.key = "persist.sys.members.cp_support",
		.value = "on",
	},
	{
		.key = "persist.sys.omc.notification",
		.value = "AT&T Chat In",
	},
	{
		.key = "persist.sys.omc.ringtone",
		.value = "AT&T Firefly",
	},
	{
		.key = "persist.sys.omc_etcpath",
		.value = "/odm/omc/ATT/etc",
	},
	{
		.key = "persist.sys.omc_path",
		.value = "/odm/omc/ATT/conf",
	},
	{
		.key = "persist.sys.omc_respath",
		.value = "/odm/omc/ATT/res",
	},
	{
		.key = "persist.sys.omc_support",
		.value = "true",
	},
	{
		.key = "persist.sys.omcnw_path",
		.value = "/odm/omc/ATT/conf",
	},
	{
		.key = "persist.sys.pcovalue",
		.value = "-1",
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
		.value = "ATT",
	},
	{
		.key = "persist.sys.prev_salescode",
		.value = "ATT",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
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
		.value = "America/Los_Angeles",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "mtp,adb",
	},
	{
		.key = "persist.sys.usb.config.extra",
		.value = "none",
	},
	{
		.key = "persist.sys.usb.dualrole",
		.value = "true",
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
		.key = "persist.sys.vold.firstboot",
		.value = "1",
	},
	{
		.key = "persist.sys.vzw_wifi_running",
		.value = "false",
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
		.key = "persist.vendor.audio.ras.enabled",
		.value = "false",
	},
	{
		.key = "persist.vendor.bt.a2dp_offload_cap",
		.value = "sbc-aptx-aptxtws-aptxhd-aac",
	},
	{
		.key = "persist.vendor.radio.custom_ecc",
		.value = "1",
	},
	{
		.key = "persist.vendor.radio.rat_on",
		.value = "combine",
	},
	{
		.key = "persist.vendor.radio.sib16_support",
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
		.key = "qcom.hw.aac.encoder",
		.value = "true",
	},
	{
		.key = "ril.CompleteMsg",
		.value = "OK",
	},
	{
		.key = "ril.ICC_TYPE",
		.value = "0",
	},
	{
		.key = "ril.ICC_TYPE0",
		.value = "0",
	},
	{
		.key = "ril.NwNmId",
		.value = "",
	},
	{
		.key = "ril.RildInit",
		.value = "1",
	},
	{
		.key = "ril.airplane.mode",
		.value = "0",
	},
	{
		.key = "ril.app.pco",
		.value = "-1",
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
		.value = "1_0_0",
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
		.key = "ril.cdma.esn",
		.value = "8034847B",
	},
	{
		.key = "ril.cs_svc",
		.value = "1",
	},
	{
		.key = "ril.data.intfprefix",
		.value = "rmnet_data",
	},
	{
		.key = "ril.debug.modemfactory",
		.value = "CSC Feature State: IMS ON, EPDG ON",
	},
	{
		.key = "ril.debug.ntc",
		.value = "M:ATT, S:ATT, T:GSM, C:USA",
	},
	{
		.key = "ril.ecclist0",
		.value = "911,112,*911,#911,000,08,110,999,118,119",
	},
	{
		.key = "ril.ecclist00",
		.value = "112,911,999,000,110,118,119,911,112,08,000,110,118,119,999",
	},
	{
		.key = "ril.ecclist_net0",
		.value = "",
	},
	{
		.key = "ril.eri_num",
		.value = "1",
	},
	{
		.key = "ril.eri_ver_1",
		.value = "E:None ",
	},
	{
		.key = "ril.hasisim",
		.value = "0",
	},
	{
		.key = "ril.hw_ver",
		.value = "REV1.1",
	},
	{
		.key = "ril.ims.ecsupport",
		.value = "0",
	},
	{
		.key = "ril.initPB",
		.value = "0",
	},
	{
		.key = "ril.lte_ps_only",
		.value = "0",
	},
	{
		.key = "ril.manufacturedate",
		.value = "20180305",
	},
	{
		.key = "ril.modem.board",
		.value = "SDM845",
	},
	{
		.key = "ril.official_cscver",
		.value = "G960UOYN1ARB7",
	},
	{
		.key = "ril.pco.default",
		.value = "-1",
	},
	{
		.key = "ril.pco.hipri",
		.value = "-1",
	},
	{
		.key = "ril.product_code",
		.value = "SM-G960UZKAATT",
	},
	{
		.key = "ril.radiostate",
		.value = "10",
	},
	{
		.key = "ril.region_props",
		.value = "ATT.USA.US.ATT",
	},
	{
		.key = "ril.rfcal_date",
		.value = "2018.03.07",
	},
	{
		.key = "ril.serialnumber",
		.value = "R38K306CCJV",
	},
	{
		.key = "ril.servicestate",
		.value = "2",
	},
	{
		.key = "ril.signal.param",
		.value = "-8,255,255",
	},
	{
		.key = "ril.simoperator",
		.value = "",
	},
	{
		.key = "ril.ss.routing",
		.value = "1",
	},
	{
		.key = "ril.subinfo",
		.value = "0:-2",
	},
	{
		.key = "ril.sw_ver",
		.value = "G960USQU1ARB7",
	},
	{
		.key = "ril.twwan911Timer",
		.value = "0",
	},
	{
		.key = "ro.adb.secure",
		.value = "1",
	},
	{
		.key = "ro.af.client_heap_size_kbyte",
		.value = "7168",
	},
	{
		.key = "ro.allow.mock.location",
		.value = "0",
	},
	{
		.key = "ro.ap_serial",
		.value = "0xC90801DF",
	},
	{
		.key = "ro.baseband",
		.value = "sdm",
	},
	{
		.key = "ro.board.platform",
		.value = "sdm845",
	},
	{
		.key = "ro.boot.ap_serial",
		.value = "0xC90801DF",
	},
	{
		.key = "ro.boot.baseband",
		.value = "sdm",
	},
	{
		.key = "ro.boot.boot_recovery",
		.value = "0",
	},
	{
		.key = "ro.boot.bootdevice",
		.value = "1d84000.ufshc",
	},
	{
		.key = "ro.boot.bootloader",
		.value = "G960USQU1ARB7",
	},
	{
		.key = "ro.boot.carrierid",
		.value = "ATT",
	},
	{
		.key = "ro.boot.carrierid.param.offset",
		.value = "9437644",
	},
	{
		.key = "ro.boot.cp_debug_level",
		.value = "0x55FF",
	},
	{
		.key = "ro.boot.ddr_start_type",
		.value = "1",
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
		.value = "20208BC90801DF11",
	},
	{
		.key = "ro.boot.em.model",
		.value = "SM-G960U",
	},
	{
		.key = "ro.boot.em.status",
		.value = "0x0",
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
		.key = "ro.boot.im.param.offset",
		.value = "9437232",
	},
	{
		.key = "ro.boot.me.param.offset",
		.value = "9437312",
	},
	{
		.key = "ro.boot.other.locked",
		.value = "1",
	},
	{
		.key = "ro.boot.pr.param.offset",
		.value = "9437472",
	},
	{
		.key = "ro.boot.prototype.param.offset",
		.value = "9437660",
	},
	{
		.key = "ro.boot.revision",
		.value = "14",
	},
	{
		.key = "ro.boot.sales.param.offset",
		.value = "9437648",
	},
	{
		.key = "ro.boot.sales_code",
		.value = "ATT",
	},
	{
		.key = "ro.boot.sec_atd.tty",
		.value = "/dev/ttyHS8",
	},
	{
		.key = "ro.boot.security_mode",
		.value = "1526595585",
	},
	{
		.key = "ro.boot.serialno",
		.value = "3551423248573398",
	},
	{
		.key = "ro.boot.sku.param.offset",
		.value = "9437552",
	},
	{
		.key = "ro.boot.sn.param.offset",
		.value = "9437392",
	},
	{
		.key = "ro.boot.swp_config",
		.value = "1",
	},
	{
		.key = "ro.boot.ucs_mode",
		.value = "0",
	},
	{
		.key = "ro.boot.usrf",
		.value = "9438192",
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
		.value = "0",
	},
	{
		.key = "ro.bootimage.build.date",
		.value = "Sun Feb 11 14:49:26 KST 2018",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1518328166",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "samsung/starqltesq/starqltesq:8.0.0/R16NW/G960USQU1ARB7:user/release-keys",
	},
	{
		.key = "ro.bootloader",
		.value = "G960USQU1ARB7",
	},
	{
		.key = "ro.bootmode",
		.value = "unknown",
	},
	{
		.key = "ro.build.PDA",
		.value = "G960USQU1ARB7",
	},
	{
		.key = "ro.build.ab_update",
		.value = "false",
	},
	{
		.key = "ro.build.changelist",
		.value = "13056303",
	},
	{
		.key = "ro.build.characteristics",
		.value = "default",
	},
	{
		.key = "ro.build.date",
		.value = "Sun Feb 11 14:49:26 KST 2018",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1518328166",
	},
	{
		.key = "ro.build.description",
		.value = "starqltesq-user 8.0.0 R16NW G960USQU1ARB7 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "R16NW.G960USQU1ARB7",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "samsung/starqltesq/starqltesq:8.0.0/R16NW/G960USQU1ARB7:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "starqltesq-user",
	},
	{
		.key = "ro.build.host",
		.value = "SWDG4715",
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
		.value = "starqltesq",
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
		.key = "ro.build.shutdown_timeout",
		.value = "0",
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
		.value = "G960USQU1ARB7",
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
		.value = "2018-02-01",
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
		.value = "ATT",
	},
	{
		.key = "ro.carrierid.param.offset",
		.value = "9437644",
	},
	{
		.key = "ro.cfg.dha_cached_max",
		.value = "24",
	},
	{
		.key = "ro.chipname",
		.value = "SDM845",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-samsung-ss",
	},
	{
		.key = "ro.com.google.clientidbase.am",
		.value = "android-att-us",
	},
	{
		.key = "ro.com.google.clientidbase.ms",
		.value = "android-att-us",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "8.0_r4",
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
		.key = "ro.config.tima",
		.value = "1",
	},
	{
		.key = "ro.config.timaversion",
		.value = "3.0",
	},
	{
		.key = "ro.config.vc_call_vol_steps",
		.value = "7",
	},
	{
		.key = "ro.control_privapp_permissions",
		.value = "log",
	},
	{
		.key = "ro.cp_debug_level",
		.value = "0x55FF",
	},
	{
		.key = "ro.crypto.fs_crypto_blkdev",
		.value = "/dev/block/dm-3",
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
		.key = "ro.csc.amazon.partnerid",
		.value = "att",
	},
	{
		.key = "ro.csc.country_code",
		.value = "USA",
	},
	{
		.key = "ro.csc.countryiso_code",
		.value = "US",
	},
	{
		.key = "ro.csc.facebook.partnerid",
		.value = "att:4b2a1409-4fa0-4d4c-a184-95f0f26d4192",
	},
	{
		.key = "ro.csc.omcnw_code",
		.value = "ATT",
	},
	{
		.key = "ro.csc.sales_code",
		.value = "ATT",
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
		.key = "ro.em.did",
		.value = "20208BC90801DF11",
	},
	{
		.key = "ro.em.model",
		.value = "SM-G960U",
	},
	{
		.key = "ro.em.status",
		.value = "0x0",
	},
	{
		.key = "ro.em.version",
		.value = "20",
	},
	{
		.key = "ro.emmc_checksum",
		.value = "unknown",
	},
	{
		.key = "ro.error.receiver.default",
		.value = "com.samsung.receiver.error",
	},
	{
		.key = "ro.expect.recovery_id",
		.value = "0xddcc25f76a8f0f655368af4baaa6d26dc6ce5e57000000000000000000000000",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/persistent",
	},
	{
		.key = "ro.gfx.driver.0",
		.value = "com.samsung.gpudriver.S9Adreno630_80",
	},
	{
		.key = "ro.hardware",
		.value = "qcom",
	},
	{
		.key = "ro.hardware.egl",
		.value = "adreno",
	},
	{
		.key = "ro.hardware.gatekeeper",
		.value = "mdfpp",
	},
	{
		.key = "ro.hardware.keystore",
		.value = "mdfpp",
	},
	{
		.key = "ro.hardware.nfc_nci",
		.value = "nqx.default",
	},
	{
		.key = "ro.hdcp2.rx",
		.value = "tz",
	},
	{
		.key = "ro.hmac_mismatch",
		.value = "unknown",
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
		.value = "9437232",
	},
	{
		.key = "ro.kernel.qemu",
		.value = "0",
	},
	{
		.key = "ro.kernel.qemu.gles",
		.value = "0",
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
		.value = "9437312",
	},
	{
		.key = "ro.multisim.simslotcount",
		.value = "1",
	},
	{
		.key = "ro.nfc.port",
		.value = "I2C",
	},
	{
		.key = "ro.oem.key1",
		.value = "ATT",
	},
	{
		.key = "ro.oem_unlock_supported",
		.value = "1",
	},
	{
		.key = "ro.omc.build.id",
		.value = "16902995",
	},
	{
		.key = "ro.omc.build.version",
		.value = "G960UOYN1ARB7",
	},
	{
		.key = "ro.omc.changetype",
		.value = "DATA_RESET_OFF,TRUE",
	},
	{
		.key = "ro.omc.disabler",
		.value = "TRUE",
	},
	{
		.key = "ro.omc.img_mount",
		.value = "0",
	},
	{
		.key = "ro.omc.region",
		.value = "US",
	},
	{
		.key = "ro.opengles.version",
		.value = "196610",
	},
	{
		.key = "ro.pr.param.offset",
		.value = "9437472",
	},
	{
		.key = "ro.product.board",
		.value = "sdm845",
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
		.value = "starqltesq",
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
		.value = "samsung",
	},
	{
		.key = "ro.product.model",
		.value = "SM-G960U",
	},
	{
		.key = "ro.product.name",
		.value = "starqltesq",
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
		.key = "ro.prototype.param.offset",
		.value = "9437660",
	},
	{
		.key = "ro.qc.sdk.audio.fluencetype",
		.value = "none",
	},
	{
		.key = "ro.qc.sdk.audio.ssr",
		.value = "false",
	},
	{
		.key = "ro.qualcomm.cabl",
		.value = "1",
	},
	{
		.key = "ro.radio.noril",
		.value = "no",
	},
	{
		.key = "ro.revision",
		.value = "14",
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
		.key = "ro.sales.param.offset",
		.value = "9437648",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.securestorage.support",
		.value = "true",
	},
	{
		.key = "ro.security.ese.cosname",
		.value = "JCOP4.0_0050534A",
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
		.value = "2",
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
		.value = "3551423248573398",
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
		.value = "XAA,OMC",
	},
	{
		.key = "ro.sku.param.offset",
		.value = "9437552",
	},
	{
		.key = "ro.sn.param.offset",
		.value = "9437392",
	},
	{
		.key = "ro.swp_config",
		.value = "1",
	},
	{
		.key = "ro.telephony.call_ring.multiple",
		.value = "false",
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
		.key = "ro.use_data_netmgrd",
		.value = "false",
	},
	{
		.key = "ro.usrf",
		.value = "9438192",
	},
	{
		.key = "ro.vendor.audio.sdk.fluencetype",
		.value = "none",
	},
	{
		.key = "ro.vendor.audio.sdk.ssr",
		.value = "false",
	},
	{
		.key = "ro.vendor.build.date",
		.value = "Sun Feb 11 14:49:26 KST 2018",
	},
	{
		.key = "ro.vendor.build.date.utc",
		.value = "1518328166",
	},
	{
		.key = "ro.vendor.build.fingerprint",
		.value = "samsung/starqltesq/starqltesq:8.0.0/R16NW/G960USQU1ARB7:user/release-keys",
	},
	{
		.key = "ro.vendor.camera.sep_cts.verified",
		.value = "false",
	},
	{
		.key = "ro.vendor.extension_library",
		.value = "libqti-perfd-client.so",
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
		.key = "sdm.debug.disable_dest_scalar",
		.value = "1",
	},
	{
		.key = "sdm.debug.disable_display_ubwc_ff_voting",
		.value = "1",
	},
	{
		.key = "sdm.debug.disable_inline_rotator",
		.value = "1",
	},
	{
		.key = "sdm.debug.disable_scalar",
		.value = "0",
	},
	{
		.key = "sdm.debug.prefersplit",
		.value = "1",
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
		.key = "service.media.powersnd",
		.value = "1",
	},
	{
		.key = "service.poa.modem_reset_count",
		.value = "0",
	},
	{
		.key = "service.secureui.screeninfo",
		.value = "1080x2220",
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
		.key = "sys.aa_noti",
		.value = "",
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
		.value = "ttyHS0",
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
		.key = "sys.disable_ext_animation",
		.value = "1",
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
		.key = "sys.is_members",
		.value = "exist",
	},
	{
		.key = "sys.isdumpstaterunning",
		.value = "0",
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
		.key = "sys.mdniecontrolservice.mscon",
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
		.key = "sys.post_boot.parsed",
		.value = "1",
	},
	{
		.key = "sys.pvclicense.loaded",
		.value = "true",
	},
	{
		.key = "sys.qca1530",
		.value = "detect",
	},
	{
		.key = "sys.qseecomd.enable",
		.value = "true",
	},
	{
		.key = "sys.sbf.mnoname0",
		.value = "ATT_US",
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
		.value = "61222",
	},
	{
		.key = "sys.usb.config",
		.value = "mtp,adb",
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
		.value = "mtp,adb",
	},
	{
		.key = "sys.use_fifo_ui",
		.value = "0",
	},
	{
		.key = "sys.vendor.shutdown.waittime",
		.value = "500",
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
		.key = "sys.vzw_sim_state",
		.value = "ABSENT",
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
		.key = "tunnel.audio.encode",
		.value = "true",
	},
	{
		.key = "use.voice.path.for.pcm.voip",
		.value = "true",
	},
	{
		.key = "vendor.audio.adm.buffering.ms",
		.value = "6",
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
		.value = "false",
	},
	{
		.key = "vendor.audio.offload.passthrough",
		.value = "false",
	},
	{
		.key = "vendor.audio.offload.pstimeout.secs",
		.value = "3",
	},
	{
		.key = "vendor.audio.offload.track.enable",
		.value = "true",
	},
	{
		.key = "vendor.audio.parser.ip.buffer.size",
		.value = "262144",
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
		.key = "vendor.audio_hal.in_period_size",
		.value = "144",
	},
	{
		.key = "vendor.audio_hal.period_multiplier",
		.value = "4",
	},
	{
		.key = "vendor.audio_hal.period_size",
		.value = "192",
	},
	{
		.key = "vendor.display.enable_default_color_mode",
		.value = "1",
	},
	{
		.key = "vendor.fm.a2dp.conc.disabled",
		.value = "true",
	},
	{
		.key = "vendor.ril.debug.sales_code",
		.value = "ATT",
	},
	{
		.key = "vendor.sec.rild.libpath",
		.value = "/vendor/lib64/libsec-ril.so",
	},
	{
		.key = "vendor.voice.path.for.pcm.voip",
		.value = "true",
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
