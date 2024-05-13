struct cpuinfo_mock_cpuid cpuid_dump[] = {
	{
		.input_eax = 0x00000000,
		.eax = 0x0000000B,
		.ebx = 0x756E6547,
		.ecx = 0x6C65746E,
		.edx = 0x49656E69,
	},
	{
		.input_eax = 0x00000001,
		.eax = 0x000406C4,
		.ebx = 0x00100800,
		.ecx = 0x43D8E3BF,
		.edx = 0xBFEBFBFF,
	},
	{
		.input_eax = 0x00000002,
		.eax = 0x61B4A001,
		.ebx = 0x0000FFC2,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000003,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000004,
		.input_ecx = 0x00000000,
		.eax = 0x1C000121,
		.ebx = 0x0140003F,
		.ecx = 0x0000003F,
		.edx = 0x00000001,
	},
	{
		.input_eax = 0x00000004,
		.input_ecx = 0x00000001,
		.eax = 0x1C000122,
		.ebx = 0x01C0003F,
		.ecx = 0x0000003F,
		.edx = 0x00000001,
	},
	{
		.input_eax = 0x00000004,
		.input_ecx = 0x00000002,
		.eax = 0x1C00C143,
		.ebx = 0x03C0003F,
		.ecx = 0x000003FF,
		.edx = 0x00000001,
	},
	{
		.input_eax = 0x00000005,
		.eax = 0x00000040,
		.ebx = 0x00000040,
		.ecx = 0x00000003,
		.edx = 0x33000020,
	},
	{
		.input_eax = 0x00000006,
		.eax = 0x00000007,
		.ebx = 0x00000002,
		.ecx = 0x00000009,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000007,
		.input_ecx = 0x00000000,
		.eax = 0x00000000,
		.ebx = 0x00002282,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000008,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000009,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x0000000A,
		.eax = 0x07280203,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000503,
	},
	{
		.input_eax = 0x0000000B,
		.input_ecx = 0x00000000,
		.eax = 0x00000001,
		.ebx = 0x00000001,
		.ecx = 0x00000100,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x0000000B,
		.input_ecx = 0x00000001,
		.eax = 0x00000004,
		.ebx = 0x00000004,
		.ecx = 0x00000201,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x80000000,
		.eax = 0x80000008,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x80000001,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000101,
		.edx = 0x28100000,
	},
	{
		.input_eax = 0x80000002,
		.eax = 0x20202020,
		.ebx = 0x6E492020,
		.ecx = 0x286C6574,
		.edx = 0x41202952,
	},
	{
		.input_eax = 0x80000003,
		.eax = 0x286D6F74,
		.ebx = 0x20294D54,
		.ecx = 0x5A2D3578,
		.edx = 0x30353338,
	},
	{
		.input_eax = 0x80000004,
		.eax = 0x50432020,
		.ebx = 0x20402055,
		.ecx = 0x34342E31,
		.edx = 0x007A4847,
	},
	{
		.input_eax = 0x80000005,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x80000006,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x04008040,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x80000007,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000100,
	},
	{
		.input_eax = 0x80000008,
		.eax = 0x00003024,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
};
struct cpuinfo_mock_file filesystem[] = {
	{
		.path = "/proc/cpuinfo",
		.size = 3752,
		.content =
			"processor\t: 0\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 76\n"
			"model name\t: Intel(R) Atom(TM) x5-Z8350  CPU @ 1.44GHz\n"
			"stepping\t: 4\n"
			"microcode\t: 0x406\n"
			"cpu MHz\t\t: 560.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 0\n"
			"cpu cores\t: 4\n"
			"apicid\t\t: 0\n"
			"initial apicid\t: 0\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 11\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf nonstop_tsc_s3 pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2879.92\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 1\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 76\n"
			"model name\t: Intel(R) Atom(TM) x5-Z8350  CPU @ 1.44GHz\n"
			"stepping\t: 4\n"
			"microcode\t: 0x406\n"
			"cpu MHz\t\t: 560.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 1\n"
			"cpu cores\t: 4\n"
			"apicid\t\t: 2\n"
			"initial apicid\t: 2\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 11\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf nonstop_tsc_s3 pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2879.92\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 2\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 76\n"
			"model name\t: Intel(R) Atom(TM) x5-Z8350  CPU @ 1.44GHz\n"
			"stepping\t: 4\n"
			"microcode\t: 0x406\n"
			"cpu MHz\t\t: 560.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 2\n"
			"cpu cores\t: 4\n"
			"apicid\t\t: 4\n"
			"initial apicid\t: 4\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 11\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf nonstop_tsc_s3 pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2879.92\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 3\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 76\n"
			"model name\t: Intel(R) Atom(TM) x5-Z8350  CPU @ 1.44GHz\n"
			"stepping\t: 4\n"
			"microcode\t: 0x406\n"
			"cpu MHz\t\t: 560.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 3\n"
			"cpu cores\t: 4\n"
			"apicid\t\t: 6\n"
			"initial apicid\t: 6\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 11\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf nonstop_tsc_s3 pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2879.92\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n",
	},
	{
		.path = "/system/build.prop",
		.size = 2513,
		.content =
			"\n"
			"# begin build properties\n"
			"# autogenerated by buildinfo.sh\n"
			"ro.build.id=LMY47I\n"
			"ro.build.display.id=I1-TFD_V1.0_20170503\n"
			"ro.build.version.incremental=eng.dell.20170503.111732\n"
			"ro.build.version.sdk=22\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.all_codenames=REL\n"
			"ro.build.version.release=5.1\n"
			"ro.build.date=Wed May  3 11:20:51 CST 2017\n"
			"ro.build.date.utc=1493781651\n"
			"ro.build.type=user\n"
			"ro.build.user=dell\n"
			"ro.build.host=build\n"
			"ro.build.tags=release-keys\n"
			"ro.build.flavor=cht_cr_mrd-user\n"
			"ro.product.model=I1-TFD\n"
			"ro.product.brand=intel\n"
			"ro.product.name=cht_cr_mrd\n"
			"ro.product.device=CHT_CR_MRD\n"
			"ro.product.board=cht_cr_mrd\n"
			"# ro.product.cpu.abi and ro.product.cpu.abi2 are obsolete,\n"
			"# use ro.product.cpu.abilist instead.\n"
			"ro.product.cpu.abi=x86\n"
			"ro.product.cpu.abilist=x86,armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist32=x86,armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist64=\n"
			"ro.product.manufacturer=cube\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=gmin\n"
			"# ro.build.product is obsolete; use ro.product.device\n"
			"ro.build.product=CHT_CR_MRD\n"
			"persist.sys.timezone=Asia/Shanghai\n"
			"ro.product.locale.language=zh\n"
			"ro.product.locale.region=CN\n"
			"# Do not try to parse description, fingerprint, or thumbprint\n"
			"ro.build.description=cht_cr_mrd-user 5.1 LMY47I eng.dell.20170503.111732 release-keys\n"
			"ro.build.fingerprint=intel/cht_cr_mrd/cht_cr_mrd:5.1/LMY47I/dell05031120:user/release-keys\n"
			"ro.build.characteristics=tablet\n"
			"# end build properties\n"
			"\n"
			"#\n"
			"# ADDITIONAL_BUILD_PROPERTIES\n"
			"#\n"
			"ro.dalvik.vm.isa.arm=x86\n"
			"ro.enable.native.bridge.exec=1\n"
			"sys.powerctl.no.shutdown=1\n"
			"dalvik.vm.heapstartsize=8m\n"
			"dalvik.vm.heapgrowthlimit=100m\n"
			"dalvik.vm.heapsize=174m\n"
			"dalvik.vm.heaptargetutilization=0.75\n"
			"dalvik.vm.heapminfree=512k\n"
			"dalvik.vm.heapmaxfree=8m\n"
			"ro.opengles.version=196609\n"
			"ro.setupwizard.mode=DISABLED\n"
			"ro.com.google.gmsversion=5.1_r2\n"
			"ro.hwui.texture_cache_size=24.0f\n"
			"ro.hwui.text_large_cache_width=2048\n"
			"ro.hwui.text_large_cache_height=512\n"
			"drm.service.enabled=true\n"
			"keyguard.no_require_sim=true\n"
			"ro.com.android.dataroaming=true\n"
			"ro.com.android.dateformat=MM-dd-yyyy\n"
			"ro.config.ringtone=Ring_Synth_04.ogg\n"
			"ro.config.notification_sound=pixiedust.ogg\n"
			"ro.carrier=unknown\n"
			"ro.config.alarm_alert=Alarm_Classic.ogg\n"
			"ro.sf3g.feature=ux\n"
			"persist.sys.dalvik.vm.lib.2=libart.so\n"
			"dalvik.vm.isa.x86.features=sse4_2,aes_in,popcnt,movbe\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"\n"
			"# begin fota properties\n"
			"ro.fota.platform=IntelZ3735F_5.1\n"
			"ro.fota.type=pad\n"
			"ro.fota.oem=emdoor-Z3735F_5.1\n"
			"ro.fota.device=I1-TFD\n"
			"ro.fota.version=I1-TFD_V1.0_2017050320170503-1120\n"
			"# end fota properties\n",
	},
	{
		.path = "/sys/devices/system/cpu/kernel_max",
		.size = 2,
		.content = "3\n",
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
		.path = "/sys/devices/system/cpu/online",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/offline",
		.size = 1,
		.content = "\n",
	},
	{
		.path = "/sys/devices/system/cpu/modalias",
		.size = 446,
		.content = "x86cpu:vendor:0000:family:0006:model:004C:feature:,0000,0001,0002,0003,0004,0005,0006,0007,0008,0009,000B,000C,000D,000E,000F,0010,0011,0013,0015,0016,0017,0018,0019,001A,001B,001C,001D,001F,002B,0034,003B,003D,0068,006B,006C,006D,006F,0070,0072,0074,0075,0076,0078,007C,007E,0080,0081,0082,0083,0084,0085,0087,0088,0089,008D,008E,008F,0093,0094,0096,0097,0098,0099,009E,00C0,00C8,00E0,00E1,00E3,00E7,0100,0101,0102,0103,0104,0121,0127,0129,012D\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/all_time_in_state",
		.size = 538,
		.content =
			"freq\t\tcpu0\t\tcpu1\t\tcpu2\t\tcpu3\t\t\n"
			"480000\t\t13328\t\t13328\t\t13328\t\t13328\t\t\n"
			"560000\t\t179\t\t179\t\t179\t\t179\t\t\n"
			"640000\t\t78\t\t78\t\t78\t\t78\t\t\n"
			"720000\t\t69\t\t69\t\t69\t\t69\t\t\n"
			"800000\t\t63\t\t63\t\t63\t\t63\t\t\n"
			"880000\t\t99\t\t99\t\t99\t\t99\t\t\n"
			"960000\t\t30\t\t30\t\t30\t\t30\t\t\n"
			"1040000\t\t19\t\t19\t\t19\t\t19\t\t\n"
			"1120000\t\t56\t\t56\t\t56\t\t56\t\t\n"
			"1200000\t\t26\t\t26\t\t26\t\t26\t\t\n"
			"1280000\t\t6\t\t6\t\t6\t\t6\t\t\n"
			"1360000\t\t26\t\t26\t\t26\t\t26\t\t\n"
			"1440000\t\t13\t\t13\t\t13\t\t13\t\t\n"
			"1520000\t\t19\t\t19\t\t19\t\t19\t\t\n"
			"1600000\t\t20\t\t20\t\t20\t\t20\t\t\n"
			"1680000\t\t20\t\t20\t\t20\t\t20\t\t\n"
			"1760000\t\t28\t\t28\t\t28\t\t28\t\t\n"
			"1840000\t\t23\t\t23\t\t23\t\t23\t\t\n"
			"1920000\t\t4893\t\t4893\t\t4893\t\t4893\t\t\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_driver",
		.size = 11,
		.content = "intel_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_governor_ro",
		.size = 5,
		.content = "menu\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1920000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 6,
		.content = "10000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 146,
		.content = "1920000 1840000 1760000 1680000 1600000 1520000 1440000 1360000 1280000 1200000 1120000 1040000 960000 880000 800000 720000 640000 560000 480000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 13,
		.content = "acpi-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 207,
		.content =
			"1920000 4893\n"
			"1840000 23\n"
			"1760000 28\n"
			"1680000 20\n"
			"1600000 20\n"
			"1520000 19\n"
			"1440000 13\n"
			"1360000 26\n"
			"1280000 6\n"
			"1200000 26\n"
			"1120000 56\n"
			"1040000 19\n"
			"960000 30\n"
			"880000 99\n"
			"800000 63\n"
			"720000 69\n"
			"640000 78\n"
			"560000 179\n"
			"480000 13422\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "375\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/trans_table",
		.size = 4057,
		.content =
			"   From  :    To\n"
			"         :   1920000   1840000   1760000   1680000   1600000   1520000   1440000   1360000   1280000   1200000   1120000   1040000    960000    880000    800000    720000    640000    560000    480000 \n"
			"  1920000:         0         6         7         3         4         3         4         5         1         4         8         5         6         7         7         3         5         3        35 \n"
			"  1840000:         6         0         0         0         0         1         0         0         0         0         0         0         0         0         0         0         0         0         1 \n"
			"  1760000:         2         2         0         1         0         1         1         0         1         0         0         0         0         0         0         0         0         0         0 \n"
			"  1680000:         2         0         1         0         0         0         0         0         0         0         1         0         0         0         0         0         0         0         1 \n"
			"  1600000:         2         0         0         1         0         1         0         0         0         0         1         0         1         0         0         0         0         0         0 \n"
			"  1520000:         1         0         0         0         2         0         0         0         0         1         0         0         0         0         0         0         0         2         0 \n"
			"  1440000:         4         0         0         0         0         0         0         0         0         0         0         0         0         1         0         0         0         0         1 \n"
			"  1360000:         0         0         0         0         0         0         0         0         0         0         1         0         0         1         0         0         1         1         1 \n"
			"  1280000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1200000:         3         0         0         0         0         0         0         0         0         0         0         0         0         0         0         1         0         0         2 \n"
			"  1120000:         3         0         0         0         0         0         0         0         0         1         0         1         1         4         0         0         1         0         1 \n"
			"  1040000:         0         0         0         0         0         0         0         0         0         0         1         0         0         1         1         1         1         0         1 \n"
			"   960000:         1         0         0         0         0         0         0         0         0         0         0         0         0         1         1         2         2         0         1 \n"
			"   880000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         5         1         3         1         6 \n"
			"   800000:         1         0         0         0         0         0         0         0         0         0         0         0         0         3         0         2         1         2         6 \n"
			"   720000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         1         0         3         5         3 \n"
			"   640000:         4         0         0         0         0         0         0         0         0         0         0         0         0         0         0         4         0         2        10 \n"
			"   560000:         3         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         3         0        20 \n"
			"   480000:        78         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0        10         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings",
		.size = 2,
		.content = "f\n",
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
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list",
		.size = 2,
		.content = "0\n",
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
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/shared_cpu_map",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/size",
		.size = 4,
		.content = "24K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/ways_of_associativity",
		.size = 2,
		.content = "6\n",
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
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/shared_cpu_map",
		.size = 2,
		.content = "1\n",
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
		.content = "8\n",
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
		.content = "0-1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/shared_cpu_map",
		.size = 2,
		.content = "3\n",
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
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1920000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_transition_latency",
		.size = 6,
		.content = "10000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_frequencies",
		.size = 146,
		.content = "1920000 1840000 1760000 1680000 1600000 1520000 1440000 1360000 1280000 1200000 1120000 1040000 960000 880000 800000 720000 640000 560000 480000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
		.size = 13,
		.content = "acpi-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 207,
		.content =
			"1920000 4893\n"
			"1840000 23\n"
			"1760000 28\n"
			"1680000 20\n"
			"1600000 20\n"
			"1520000 19\n"
			"1440000 13\n"
			"1360000 26\n"
			"1280000 6\n"
			"1200000 26\n"
			"1120000 56\n"
			"1040000 19\n"
			"960000 30\n"
			"880000 99\n"
			"800000 63\n"
			"720000 69\n"
			"640000 78\n"
			"560000 179\n"
			"480000 13594\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 4,
		.content = "375\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/trans_table",
		.size = 4057,
		.content =
			"   From  :    To\n"
			"         :   1920000   1840000   1760000   1680000   1600000   1520000   1440000   1360000   1280000   1200000   1120000   1040000    960000    880000    800000    720000    640000    560000    480000 \n"
			"  1920000:         0         6         7         3         4         3         4         5         1         4         8         5         6         7         7         3         5         3        35 \n"
			"  1840000:         6         0         0         0         0         1         0         0         0         0         0         0         0         0         0         0         0         0         1 \n"
			"  1760000:         2         2         0         1         0         1         1         0         1         0         0         0         0         0         0         0         0         0         0 \n"
			"  1680000:         2         0         1         0         0         0         0         0         0         0         1         0         0         0         0         0         0         0         1 \n"
			"  1600000:         2         0         0         1         0         1         0         0         0         0         1         0         1         0         0         0         0         0         0 \n"
			"  1520000:         1         0         0         0         2         0         0         0         0         1         0         0         0         0         0         0         0         2         0 \n"
			"  1440000:         4         0         0         0         0         0         0         0         0         0         0         0         0         1         0         0         0         0         1 \n"
			"  1360000:         0         0         0         0         0         0         0         0         0         0         1         0         0         1         0         0         1         1         1 \n"
			"  1280000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1200000:         3         0         0         0         0         0         0         0         0         0         0         0         0         0         0         1         0         0         2 \n"
			"  1120000:         3         0         0         0         0         0         0         0         0         1         0         1         1         4         0         0         1         0         1 \n"
			"  1040000:         0         0         0         0         0         0         0         0         0         0         1         0         0         1         1         1         1         0         1 \n"
			"   960000:         1         0         0         0         0         0         0         0         0         0         0         0         0         1         1         2         2         0         1 \n"
			"   880000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         5         1         3         1         6 \n"
			"   800000:         1         0         0         0         0         0         0         0         0         0         0         0         0         3         0         2         1         2         6 \n"
			"   720000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         1         0         3         5         3 \n"
			"   640000:         4         0         0         0         0         0         0         0         0         0         0         0         0         0         0         4         0         2        10 \n"
			"   560000:         3         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         3         0        20 \n"
			"   480000:        78         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0        10         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_siblings",
		.size = 2,
		.content = "f\n",
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
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/thread_siblings_list",
		.size = 2,
		.content = "1\n",
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
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/shared_cpu_map",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/size",
		.size = 4,
		.content = "24K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/ways_of_associativity",
		.size = 2,
		.content = "6\n",
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
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/shared_cpu_map",
		.size = 2,
		.content = "2\n",
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
		.content = "8\n",
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
		.content = "0-1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/shared_cpu_map",
		.size = 2,
		.content = "3\n",
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
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1920000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_transition_latency",
		.size = 6,
		.content = "10000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies",
		.size = 146,
		.content = "1920000 1840000 1760000 1680000 1600000 1520000 1440000 1360000 1280000 1200000 1120000 1040000 960000 880000 800000 720000 640000 560000 480000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver",
		.size = 13,
		.content = "acpi-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 207,
		.content =
			"1920000 4893\n"
			"1840000 23\n"
			"1760000 28\n"
			"1680000 20\n"
			"1600000 20\n"
			"1520000 19\n"
			"1440000 13\n"
			"1360000 26\n"
			"1280000 6\n"
			"1200000 26\n"
			"1120000 56\n"
			"1040000 19\n"
			"960000 30\n"
			"880000 99\n"
			"800000 63\n"
			"720000 69\n"
			"640000 78\n"
			"560000 179\n"
			"480000 13767\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 4,
		.content = "375\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/trans_table",
		.size = 4057,
		.content =
			"   From  :    To\n"
			"         :   1920000   1840000   1760000   1680000   1600000   1520000   1440000   1360000   1280000   1200000   1120000   1040000    960000    880000    800000    720000    640000    560000    480000 \n"
			"  1920000:         0         6         7         3         4         3         4         5         1         4         8         5         6         7         7         3         5         3        35 \n"
			"  1840000:         6         0         0         0         0         1         0         0         0         0         0         0         0         0         0         0         0         0         1 \n"
			"  1760000:         2         2         0         1         0         1         1         0         1         0         0         0         0         0         0         0         0         0         0 \n"
			"  1680000:         2         0         1         0         0         0         0         0         0         0         1         0         0         0         0         0         0         0         1 \n"
			"  1600000:         2         0         0         1         0         1         0         0         0         0         1         0         1         0         0         0         0         0         0 \n"
			"  1520000:         1         0         0         0         2         0         0         0         0         1         0         0         0         0         0         0         0         2         0 \n"
			"  1440000:         4         0         0         0         0         0         0         0         0         0         0         0         0         1         0         0         0         0         1 \n"
			"  1360000:         0         0         0         0         0         0         0         0         0         0         1         0         0         1         0         0         1         1         1 \n"
			"  1280000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1200000:         3         0         0         0         0         0         0         0         0         0         0         0         0         0         0         1         0         0         2 \n"
			"  1120000:         3         0         0         0         0         0         0         0         0         1         0         1         1         4         0         0         1         0         1 \n"
			"  1040000:         0         0         0         0         0         0         0         0         0         0         1         0         0         1         1         1         1         0         1 \n"
			"   960000:         1         0         0         0         0         0         0         0         0         0         0         0         0         1         1         2         2         0         1 \n"
			"   880000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         5         1         3         1         6 \n"
			"   800000:         1         0         0         0         0         0         0         0         0         0         0         0         0         3         0         2         1         2         6 \n"
			"   720000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         1         0         3         5         3 \n"
			"   640000:         4         0         0         0         0         0         0         0         0         0         0         0         0         0         0         4         0         2        10 \n"
			"   560000:         3         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         3         0        20 \n"
			"   480000:        78         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0        10         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_id",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_siblings",
		.size = 2,
		.content = "f\n",
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
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/thread_siblings_list",
		.size = 2,
		.content = "2\n",
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
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/shared_cpu_map",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/size",
		.size = 4,
		.content = "24K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/ways_of_associativity",
		.size = 2,
		.content = "6\n",
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
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/shared_cpu_map",
		.size = 2,
		.content = "4\n",
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
		.content = "8\n",
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
		.content = "2-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/shared_cpu_map",
		.size = 2,
		.content = "c\n",
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
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1920000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_transition_latency",
		.size = 6,
		.content = "10000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 146,
		.content = "1920000 1840000 1760000 1680000 1600000 1520000 1440000 1360000 1280000 1200000 1120000 1040000 960000 880000 800000 720000 640000 560000 480000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_driver",
		.size = 13,
		.content = "acpi-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "480000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 207,
		.content =
			"1920000 4912\n"
			"1840000 23\n"
			"1760000 28\n"
			"1680000 20\n"
			"1600000 20\n"
			"1520000 19\n"
			"1440000 13\n"
			"1360000 26\n"
			"1280000 6\n"
			"1200000 26\n"
			"1120000 56\n"
			"1040000 19\n"
			"960000 30\n"
			"880000 99\n"
			"800000 63\n"
			"720000 69\n"
			"640000 78\n"
			"560000 187\n"
			"480000 16528\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "381\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/trans_table",
		.size = 4057,
		.content =
			"   From  :    To\n"
			"         :   1920000   1840000   1760000   1680000   1600000   1520000   1440000   1360000   1280000   1200000   1120000   1040000    960000    880000    800000    720000    640000    560000    480000 \n"
			"  1920000:         0         6         7         3         4         3         4         5         1         4         8         5         6         7         7         3         5         3        37 \n"
			"  1840000:         6         0         0         0         0         1         0         0         0         0         0         0         0         0         0         0         0         0         1 \n"
			"  1760000:         2         2         0         1         0         1         1         0         1         0         0         0         0         0         0         0         0         0         0 \n"
			"  1680000:         2         0         1         0         0         0         0         0         0         0         1         0         0         0         0         0         0         0         1 \n"
			"  1600000:         2         0         0         1         0         1         0         0         0         0         1         0         1         0         0         0         0         0         0 \n"
			"  1520000:         1         0         0         0         2         0         0         0         0         1         0         0         0         0         0         0         0         2         0 \n"
			"  1440000:         4         0         0         0         0         0         0         0         0         0         0         0         0         1         0         0         0         0         1 \n"
			"  1360000:         0         0         0         0         0         0         0         0         0         0         1         0         0         1         0         0         1         1         1 \n"
			"  1280000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 \n"
			"  1200000:         3         0         0         0         0         0         0         0         0         0         0         0         0         0         0         1         0         0         2 \n"
			"  1120000:         3         0         0         0         0         0         0         0         0         1         0         1         1         4         0         0         1         0         1 \n"
			"  1040000:         0         0         0         0         0         0         0         0         0         0         1         0         0         1         1         1         1         0         1 \n"
			"   960000:         1         0         0         0         0         0         0         0         0         0         0         0         0         1         1         2         2         0         1 \n"
			"   880000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         5         1         3         1         6 \n"
			"   800000:         1         0         0         0         0         0         0         0         0         0         0         0         0         3         0         2         1         2         6 \n"
			"   720000:         2         0         0         0         0         0         0         0         0         0         0         0         0         0         1         0         3         5         3 \n"
			"   640000:         4         0         0         0         0         0         0         0         0         0         0         0         0         0         0         4         0         2        10 \n"
			"   560000:         3         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         3         0        21 \n"
			"   480000:        80         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0        11         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_id",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_siblings",
		.size = 2,
		.content = "f\n",
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
		.size = 2,
		.content = "8\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/thread_siblings_list",
		.size = 2,
		.content = "3\n",
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
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/shared_cpu_list",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/shared_cpu_map",
		.size = 2,
		.content = "8\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/size",
		.size = 4,
		.content = "24K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/type",
		.size = 5,
		.content = "Data\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/ways_of_associativity",
		.size = 2,
		.content = "6\n",
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
		.size = 3,
		.content = "64\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/shared_cpu_list",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/shared_cpu_map",
		.size = 2,
		.content = "8\n",
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
		.content = "8\n",
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
		.content = "2-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/shared_cpu_map",
		.size = 2,
		.content = "c\n",
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
	{ NULL },
};
#ifdef __ANDROID__
struct cpuinfo_mock_property properties[] = {
	{
		.key = "AudioComms.vtsv.routed",
		.value = "false",
	},
	{
		.key = "audio.aware.card",
		.value = "cherrytrailaud",
	},
	{
		.key = "audio.device.name",
		.value = "cherrytrailaud",
	},
	{
		.key = "audio.offload.capabilities",
		.value = "1",
	},
	{
		.key = "audio.offload.disable",
		.value = "0",
	},
	{
		.key = "audio.offload.min.duration.secs",
		.value = "20",
	},
	{
		.key = "audio.offload.scalability",
		.value = "1",
	},
	{
		.key = "audio.wov.card",
		.value = "cherrytrailaud",
	},
	{
		.key = "audio.wov.device",
		.value = "5",
	},
	{
		.key = "audio.wov.dsp_log",
		.value = "0",
	},
	{
		.key = "audio.wov.routed",
		.value = "false",
	},
	{
		.key = "camera.disable_zsl_mode",
		.value = "1",
	},
	{
		.key = "dalvik.vm.dex2oat-Xms",
		.value = "64m",
	},
	{
		.key = "dalvik.vm.dex2oat-Xmx",
		.value = "256m",
	},
	{
		.key = "dalvik.vm.heapgrowthlimit",
		.value = "100m",
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
		.value = "174m",
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
		.key = "dalvik.vm.isa.x86.features",
		.value = "sse4_2,aes_in,popcnt,movbe",
	},
	{
		.key = "dalvik.vm.stack-trace-file",
		.value = "/data/anr/traces.txt",
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
		.value = "1",
	},
	{
		.key = "gsm.firmware.upload",
		.value = "ok",
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
		.value = "NOT_READY",
	},
	{
		.key = "hwc.video.extmode.enable",
		.value = "0",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.atomisp-init",
		.value = "stopped",
	},
	{
		.key = "init.svc.bcu_cpufreqrel",
		.value = "stopped",
	},
	{
		.key = "init.svc.bootanim",
		.value = "stopped",
	},
	{
		.key = "init.svc.coreu",
		.value = "running",
	},
	{
		.key = "init.svc.debuggerd",
		.value = "running",
	},
	{
		.key = "init.svc.defaultcrypto",
		.value = "stopped",
	},
	{
		.key = "init.svc.drm",
		.value = "running",
	},
	{
		.key = "init.svc.fg_algo_iface",
		.value = "running",
	},
	{
		.key = "init.svc.flash_recovery",
		.value = "stopped",
	},
	{
		.key = "init.svc.hdcpd",
		.value = "running",
	},
	{
		.key = "init.svc.healthd",
		.value = "running",
	},
	{
		.key = "init.svc.installd",
		.value = "running",
	},
	{
		.key = "init.svc.intel_prop",
		.value = "stopped",
	},
	{
		.key = "init.svc.keymaster_meid",
		.value = "running",
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
		.key = "init.svc.logd",
		.value = "running",
	},
	{
		.key = "init.svc.media",
		.value = "running",
	},
	{
		.key = "init.svc.mkaddr",
		.value = "stopped",
	},
	{
		.key = "init.svc.mkipaddr",
		.value = "stopped",
	},
	{
		.key = "init.svc.msync",
		.value = "running",
	},
	{
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.otpserver",
		.value = "running",
	},
	{
		.key = "init.svc.power_hal_helper",
		.value = "stopped",
	},
	{
		.key = "init.svc.pstore-clean",
		.value = "stopped",
	},
	{
		.key = "init.svc.rfkill-init",
		.value = "stopped",
	},
	{
		.key = "init.svc.rfkill_bt",
		.value = "stopped",
	},
	{
		.key = "init.svc.sdcard",
		.value = "running",
	},
	{
		.key = "init.svc.sensorhubd",
		.value = "running",
	},
	{
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.sl_si_service",
		.value = "stopped",
	},
	{
		.key = "init.svc.surfaceflinger",
		.value = "running",
	},
	{
		.key = "init.svc.thermal-daemon",
		.value = "stopped",
	},
	{
		.key = "init.svc.ueventd",
		.value = "running",
	},
	{
		.key = "init.svc.ufo-init",
		.value = "stopped",
	},
	{
		.key = "init.svc.usb3gmonitor",
		.value = "running",
	},
	{
		.key = "init.svc.vold",
		.value = "running",
	},
	{
		.key = "init.svc.watchdogd",
		.value = "running",
	},
	{
		.key = "init.svc.zygote",
		.value = "running",
	},
	{
		.key = "keyguard.no_require_sim",
		.value = "true",
	},
	{
		.key = "media.settings.xml",
		.value = "/etc/media_profiles_ov2680.xml|/etc/media_profiles_ov2680.xml",
	},
	{
		.key = "net.bt.name",
		.value = "Android",
	},
	{
		.key = "net.change",
		.value = "net.dns2",
	},
	{
		.key = "net.dns1",
		.value = "202.96.134.133",
	},
	{
		.key = "net.dns2",
		.value = "202.96.134.133",
	},
	{
		.key = "net.hostname",
		.value = "android-ce3968db4e196787",
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
		.key = "offload.compress.device",
		.value = "1",
	},
	{
		.key = "offload.mixer.mute.ctl.name",
		.value = "media0_in volume 0 mute",
	},
	{
		.key = "offload.mixer.rp.ctl.name",
		.value = "media0_in volume 0 rampduration",
	},
	{
		.key = "offload.mixer.volume.ctl.name",
		.value = "media0_in volume 0 volume",
	},
	{
		.key = "partition.system.verified",
		.value = "1",
	},
	{
		.key = "persist.intel.ogl.debug",
		.value = "/data/ufo.prop",
	},
	{
		.key = "persist.intel.ogl.dumpdebugvars",
		.value = "1",
	},
	{
		.key = "persist.intel.ogl.username",
		.value = "Developer",
	},
	{
		.key = "persist.media.pfw.verbose",
		.value = "true",
	},
	{
		.key = "persist.nomodem_ui",
		.value = "true",
	},
	{
		.key = "persist.service.bdroid.bdaddr",
		.value = "22:22:d4:4c:2c:93",
	},
	{
		.key = "persist.service.thermal",
		.value = "1",
	},
	{
		.key = "persist.sys.country",
		.value = "US",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.language",
		.value = "en",
	},
	{
		.key = "persist.sys.localevar",
		.value = "",
	},
	{
		.key = "persist.sys.preinstalled",
		.value = "1",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
	},
	{
		.key = "persist.sys.sd.defaultpath",
		.value = "/storage/sdcard0/",
	},
	{
		.key = "persist.sys.timezone",
		.value = "Asia/Shanghai",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "mtp,adb",
	},
	{
		.key = "persist.thermal.display.msg",
		.value = "1",
	},
	{
		.key = "persist.thermal.display.vibra",
		.value = "1",
	},
	{
		.key = "persist.thermal.mode",
		.value = "itux",
	},
	{
		.key = "persist.thermal.shutdown.msg",
		.value = "1",
	},
	{
		.key = "persist.thermal.shutdown.tone",
		.value = "1",
	},
	{
		.key = "persist.thermal.shutdown.vibra",
		.value = "1",
	},
	{
		.key = "ril.coredumpwarning.enable",
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
		.key = "ro.baseband",
		.value = "unknown",
	},
	{
		.key = "ro.board.platform",
		.value = "gmin",
	},
	{
		.key = "ro.boot.bootreason",
		.value = "power_button_pressed",
	},
	{
		.key = "ro.boot.hardware",
		.value = "cht_cr_mrd",
	},
	{
		.key = "ro.boot.serialno",
		.value = "Default0string",
	},
	{
		.key = "ro.bootloader",
		.value = "unknown",
	},
	{
		.key = "ro.bootmode",
		.value = "unknown",
	},
	{
		.key = "ro.bt.bdaddr_path",
		.value = "/config/bt/bd_addr.conf",
	},
	{
		.key = "ro.build.characteristics",
		.value = "tablet",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1493781651",
	},
	{
		.key = "ro.build.date",
		.value = "Wed May  3 11:20:51 CST 2017",
	},
	{
		.key = "ro.build.description",
		.value = "cht_cr_mrd-user 5.1 LMY47I eng.dell.20170503.111732 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "I1-TFD_V1.0_20170503",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "intel/cht_cr_mrd/cht_cr_mrd:5.1/LMY47I/dell05031120:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "cht_cr_mrd-user",
	},
	{
		.key = "ro.build.host",
		.value = "build",
	},
	{
		.key = "ro.build.id",
		.value = "LMY47I",
	},
	{
		.key = "ro.build.product",
		.value = "CHT_CR_MRD",
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
		.value = "dell",
	},
	{
		.key = "ro.build.version.all_codenames",
		.value = "REL",
	},
	{
		.key = "ro.build.version.codename",
		.value = "REL",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "eng.dell.20170503.111732",
	},
	{
		.key = "ro.build.version.release",
		.value = "5.1",
	},
	{
		.key = "ro.build.version.sdk",
		.value = "22",
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
		.key = "ro.com.android.dateformat",
		.value = "MM-dd-yyyy",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "5.1_r2",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Alarm_Classic.ogg",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "pixiedust.ogg",
	},
	{
		.key = "ro.config.ringtone",
		.value = "Ring_Synth_04.ogg",
	},
	{
		.key = "ro.crypto.fs_crypto_blkdev",
		.value = "/dev/block/dm-1",
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
		.key = "ro.dalvik.vm.isa.arm",
		.value = "x86",
	},
	{
		.key = "ro.dalvik.vm.native.bridge",
		.value = "libhoudini.so",
	},
	{
		.key = "ro.debuggable",
		.value = "0",
	},
	{
		.key = "ro.enable.native.bridge.exec",
		.value = "1",
	},
	{
		.key = "ro.factorytest",
		.value = "0",
	},
	{
		.key = "ro.fota.device",
		.value = "I1-TFD",
	},
	{
		.key = "ro.fota.oem",
		.value = "emdoor-Z3735F_5.1",
	},
	{
		.key = "ro.fota.platform",
		.value = "IntelZ3735F_5.1",
	},
	{
		.key = "ro.fota.type",
		.value = "pad",
	},
	{
		.key = "ro.fota.version",
		.value = "I1-TFD_V1.0_2017050320170503-1120",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/by-name/android_persistent",
	},
	{
		.key = "ro.hardware",
		.value = "cht_cr_mrd",
	},
	{
		.key = "ro.hwui.text_large_cache_height",
		.value = "512",
	},
	{
		.key = "ro.hwui.text_large_cache_width",
		.value = "2048",
	},
	{
		.key = "ro.hwui.texture_cache_size",
		.value = "24.0f",
	},
	{
		.key = "ro.iio.accel.x.opt_scale",
		.value = "-1",
	},
	{
		.key = "ro.iio.accel.z.opt_scale",
		.value = "-1",
	},
	{
		.key = "ro.iio.anglvel.x.opt_scale",
		.value = "-1",
	},
	{
		.key = "ro.iio.anglvel.z.opt_scale",
		.value = "-1",
	},
	{
		.key = "ro.iio.magn.x.opt_scale",
		.value = "-1",
	},
	{
		.key = "ro.iio.magn.z.opt_scale",
		.value = "-1",
	},
	{
		.key = "ro.modules.location",
		.value = "/system/lib/modules",
	},
	{
		.key = "ro.opengles.version",
		.value = "196609",
	},
	{
		.key = "ro.product.board",
		.value = "cht_cr_mrd",
	},
	{
		.key = "ro.product.brand",
		.value = "intel",
	},
	{
		.key = "ro.product.cpu.abi",
		.value = "x86",
	},
	{
		.key = "ro.product.cpu.abilist32",
		.value = "x86,armeabi-v7a,armeabi",
	},
	{
		.key = "ro.product.cpu.abilist64",
		.value = "",
	},
	{
		.key = "ro.product.cpu.abilist",
		.value = "x86,armeabi-v7a,armeabi",
	},
	{
		.key = "ro.product.device",
		.value = "CHT_CR_MRD",
	},
	{
		.key = "ro.product.locale.language",
		.value = "zh",
	},
	{
		.key = "ro.product.locale.region",
		.value = "CN",
	},
	{
		.key = "ro.product.manufacturer",
		.value = "cube",
	},
	{
		.key = "ro.product.model",
		.value = "I1-TFD",
	},
	{
		.key = "ro.product.name",
		.value = "cht_cr_mrd",
	},
	{
		.key = "ro.revision",
		.value = "0",
	},
	{
		.key = "ro.runtime.firstboot",
		.value = "1512467818449",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "Default0string",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "DISABLED",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "224",
	},
	{
		.key = "ro.sf3g.feature",
		.value = "ux",
	},
	{
		.key = "ro.ufo.use_coreu",
		.value = "1",
	},
	{
		.key = "ro.ufo.use_msync",
		.value = "1",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
	},
	{
		.key = "ro.zygote",
		.value = "zygote32",
	},
	{
		.key = "selinux.reload_policy",
		.value = "1",
	},
	{
		.key = "service.adb.tcp.port",
		.value = "5555",
	},
	{
		.key = "service.bootanim.exit",
		.value = "1",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
	},
	{
		.key = "sys.charger.connected",
		.value = "1",
	},
	{
		.key = "sys.ifwi.version",
		.value = "5.11",
	},
	{
		.key = "sys.kernel.version",
		.value = "3.14.37-x86_64-L1-R517",
	},
	{
		.key = "sys.power_hal.niproc",
		.value = "2673",
	},
	{
		.key = "sys.powerctl.no.shutdown",
		.value = "1",
	},
	{
		.key = "sys.settings_global_version",
		.value = "4",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "27000",
	},
	{
		.key = "sys.usb.config",
		.value = "mtp,adb",
	},
	{
		.key = "sys.usb.state",
		.value = "mtp,adb",
	},
	{
		.key = "vold.decrypt",
		.value = "trigger_restart_framework",
	},
	{
		.key = "vold.post_fs_data_done",
		.value = "1",
	},
	{
		.key = "widi.abr.enable",
		.value = "true",
	},
	{
		.key = "widi.hdcp.enable",
		.value = "true",
	},
	{
		.key = "widi.setsocketsize.enable",
		.value = "false",
	},
	{
		.key = "widi.socketpriority.enable",
		.value = "false",
	},
	{
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "wlan.driver.status",
		.value = "unloaded",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
