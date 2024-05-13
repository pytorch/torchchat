struct cpuinfo_mock_cpuid cpuid_dump[] = {
	{
		.input_eax = 0x00000000,
		.eax = 0x00000017,
		.ebx = 0x756E6547,
		.ecx = 0x6C65746E,
		.edx = 0x49656E69,
	},
	{
		.input_eax = 0x00000001,
		.eax = 0x0007065A,
		.ebx = 0x02200800,
		.ecx = 0x03D8E20B,
		.edx = 0x1F8BFBFF,
	},
	{
		.input_eax = 0x00000002,
		.eax = 0x61B4A001,
		.ebx = 0x0000FFC0,
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
		.eax = 0x3C000121,
		.ebx = 0x00C0003F,
		.ecx = 0x0000003F,
		.edx = 0x00000001,
	},
	{
		.input_eax = 0x00000004,
		.input_ecx = 0x00000001,
		.eax = 0x3C000122,
		.ebx = 0x01C0003F,
		.ecx = 0x0000003F,
		.edx = 0x00000001,
	},
	{
		.input_eax = 0x00000004,
		.input_ecx = 0x00000002,
		.eax = 0x3C01C143,
		.ebx = 0x03C0003F,
		.ecx = 0x000003FF,
		.edx = 0x00000001,
	},
	{
		.input_eax = 0x00000005,
		.eax = 0x00000040,
		.ebx = 0x00000040,
		.ecx = 0x00000003,
		.edx = 0x03000020,
	},
	{
		.input_eax = 0x00000006,
		.eax = 0x00000004,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
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
		.edx = 0x00000002,
	},
	{
		.input_eax = 0x0000000B,
		.input_ecx = 0x00000001,
		.eax = 0x00000005,
		.ebx = 0x00000008,
		.ecx = 0x00000201,
		.edx = 0x00000002,
	},
	{
		.input_eax = 0x0000000C,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x0000000D,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x0000000E,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x0000000F,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000010,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000011,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000013,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000014,
		.input_ecx = 0x00000000,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000015,
		.eax = 0x00000004,
		.ebx = 0x00000120,
		.ecx = 0x018CBA80,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000016,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000017,
		.input_ecx = 0x00000000,
		.eax = 0x00000003,
		.ebx = 0x00000001,
		.ecx = 0x00000001,
		.edx = 0x00000001,
	},
	{
		.input_eax = 0x00000017,
		.input_ecx = 0x00000001,
		.eax = 0x756E6547,
		.ebx = 0x20656E69,
		.ecx = 0x65727053,
		.edx = 0x72746461,
	},
	{
		.input_eax = 0x00000017,
		.input_ecx = 0x00000002,
		.eax = 0x52286D75,
		.ebx = 0x72502029,
		.ecx = 0x7365636F,
		.edx = 0x20726F73,
	},
	{
		.input_eax = 0x00000017,
		.input_ecx = 0x00000003,
		.eax = 0x20202020,
		.ebx = 0x20202020,
		.ecx = 0x20202020,
		.edx = 0x00202020,
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
		.ebx = 0x20202020,
		.ecx = 0x65727053,
		.edx = 0x72746461,
	},
	{
		.input_eax = 0x80000003,
		.eax = 0x53206D75,
		.ebx = 0x35383943,
		.ecx = 0x492D4933,
		.edx = 0x20202041,
	},
	{
		.input_eax = 0x80000004,
		.eax = 0x00000000,
		.ebx = 0x00000000,
		.ecx = 0x00383237,
		.edx = 0x007A484D,
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
		.size = 6305,
		.content =
			"processor\t: 0\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 117\n"
			"model name\t: Spreadtrum SC9853I-IA\n"
			"stepping\t: 10\n"
			"microcode\t: 0xa0a\n"
			"cpu MHz\t\t: 624.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 8\n"
			"core id\t\t: 0\n"
			"cpu cores\t: 8\n"
			"apicid\t\t: 0\n"
			"initial apicid\t: 0\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 23\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht nx rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology nonstop_tsc pni pclmulqdq monitor ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes lahf_lm 3dnowprefetch arat tsc_adjust smep erms\n"
			"bugs\t\t:\n"
			"bogomips\t: 3744.19\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 1\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 117\n"
			"model name\t: Spreadtrum SC9853I-IA\n"
			"stepping\t: 10\n"
			"microcode\t: 0xa0a\n"
			"cpu MHz\t\t: 624.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 8\n"
			"core id\t\t: 1\n"
			"cpu cores\t: 8\n"
			"apicid\t\t: 2\n"
			"initial apicid\t: 2\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 23\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht nx rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology nonstop_tsc pni pclmulqdq monitor ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes lahf_lm 3dnowprefetch arat tsc_adjust smep erms\n"
			"bugs\t\t:\n"
			"bogomips\t: 3744.19\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 2\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 117\n"
			"model name\t: Spreadtrum SC9853I-IA\n"
			"stepping\t: 10\n"
			"microcode\t: 0xa0a\n"
			"cpu MHz\t\t: 624.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 8\n"
			"core id\t\t: 2\n"
			"cpu cores\t: 8\n"
			"apicid\t\t: 4\n"
			"initial apicid\t: 4\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 23\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht nx rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology nonstop_tsc pni pclmulqdq monitor ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes lahf_lm 3dnowprefetch arat tsc_adjust smep erms\n"
			"bugs\t\t:\n"
			"bogomips\t: 3744.19\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 3\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 117\n"
			"model name\t: Spreadtrum SC9853I-IA\n"
			"stepping\t: 10\n"
			"microcode\t: 0xa0a\n"
			"cpu MHz\t\t: 624.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 8\n"
			"core id\t\t: 3\n"
			"cpu cores\t: 8\n"
			"apicid\t\t: 6\n"
			"initial apicid\t: 6\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 23\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht nx rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology nonstop_tsc pni pclmulqdq monitor ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes lahf_lm 3dnowprefetch arat tsc_adjust smep erms\n"
			"bugs\t\t:\n"
			"bogomips\t: 3744.19\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 4\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 117\n"
			"model name\t: Spreadtrum SC9853I-IA\n"
			"stepping\t: 10\n"
			"microcode\t: 0xa0a\n"
			"cpu MHz\t\t: 624.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 8\n"
			"core id\t\t: 4\n"
			"cpu cores\t: 8\n"
			"apicid\t\t: 8\n"
			"initial apicid\t: 8\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 23\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht nx rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology nonstop_tsc pni pclmulqdq monitor ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes lahf_lm 3dnowprefetch arat tsc_adjust smep erms\n"
			"bugs\t\t:\n"
			"bogomips\t: 3744.19\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 5\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 117\n"
			"model name\t: Spreadtrum SC9853I-IA\n"
			"stepping\t: 10\n"
			"microcode\t: 0xa0a\n"
			"cpu MHz\t\t: 624.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 8\n"
			"core id\t\t: 5\n"
			"cpu cores\t: 8\n"
			"apicid\t\t: 10\n"
			"initial apicid\t: 10\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 23\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht nx rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology nonstop_tsc pni pclmulqdq monitor ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes lahf_lm 3dnowprefetch arat tsc_adjust smep erms\n"
			"bugs\t\t:\n"
			"bogomips\t: 3744.19\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 6\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 117\n"
			"model name\t: Spreadtrum SC9853I-IA\n"
			"stepping\t: 10\n"
			"microcode\t: 0xa0a\n"
			"cpu MHz\t\t: 624.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 8\n"
			"core id\t\t: 6\n"
			"cpu cores\t: 8\n"
			"apicid\t\t: 12\n"
			"initial apicid\t: 12\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 23\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht nx rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology nonstop_tsc pni pclmulqdq monitor ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes lahf_lm 3dnowprefetch arat tsc_adjust smep erms\n"
			"bugs\t\t:\n"
			"bogomips\t: 3744.19\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 7\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 117\n"
			"model name\t: Spreadtrum SC9853I-IA\n"
			"stepping\t: 10\n"
			"microcode\t: 0xa0a\n"
			"cpu MHz\t\t: 624.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 8\n"
			"core id\t\t: 7\n"
			"cpu cores\t: 8\n"
			"apicid\t\t: 14\n"
			"initial apicid\t: 14\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 23\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht nx rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology nonstop_tsc pni pclmulqdq monitor ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes lahf_lm 3dnowprefetch arat tsc_adjust smep erms\n"
			"bugs\t\t:\n"
			"bogomips\t: 3744.19\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"Hardware\t: Spreadtrum SC9853I-IA\n"
			"\n"
			"\n",
	},
	{
		.path = "/system/build.prop",
		.size = 8107,
		.content =
			"\n"
			"# begin build properties\n"
			"# autogenerated by buildinfo.sh\n"
			"ro.build.id=NRD90M\n"
			"###\n"
			"ro.build.version.incremental=eng.root.20180112.175428\n"
			"ro.build.version.sdk=24\n"
			"ro.build.version.preview_sdk=0\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.all_codenames=REL\n"
			"ro.build.version.release=7.0\n"
			"ro.build.version.security_patch=2017-12-05\n"
			"ro.build.version.base_os=\n"
			"ro.build.date=Fri Jan 12 17:54:28 CST 2018\n"
			"ro.build.date.utc=1515750868\n"
			"ro.build.type=user\n"
			"ro.build.user=root\n"
			"ro.build.host=lxh\n"
			"ro.build.tags=release-keys\n"
			"ro.build.flavor=k500_lgt_511-user\n"
			"ro.product.model=T5c\n"
			"ro.product.brand=LEAGOO\n"
			"ro.product.name=T5c\n"
			"ro.product.device=T5c\n"
			"ro.product.board=k500_lgt_511_vmm\n"
			"# ro.product.cpu.abi and ro.product.cpu.abi2 are obsolete,\n"
			"# use ro.product.cpu.abilist instead.\n"
			"ro.product.cpu.abi=x86_64\n"
			"ro.product.cpu.abilist=x86_64,x86,armeabi-v7a,armeabi,arm64-v8a\n"
			"ro.product.cpu.abilist32=x86,armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist64=x86_64,arm64-v8a\n"
			"ro.product.manufacturer=LEAGOO\n"
			"ro.product.locale=en-US\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=sp9853i\n"
			"# ro.build.product is obsolete; use ro.product.device\n"
			"ro.build.product=T5c\n"
			"# Do not try to parse description, fingerprint, or thumbprint\n"
			"ro.build.description=k500_lgt_511-user 7.0 NRD90M eng.root.20180112.175428 release-keys\n"
			"ro.build.fingerprint=LEAGOO/T5c/T5c:7.0/NRD90M/01121754:user/release-keys\n"
			"ro.build.characteristics=nosdcard\n"
			"# end build properties\n"
			"#\n"
			"# from device/sprd/isharkl2/k500_lgt_511/system.prop\n"
			"#\n"
			"# Default density config\n"
			"ro.product.hardware=k500_lgt_511_v1.0.0\n"
			"# Set Opengl ES Version 3.2\n"
			"ro.opengles.version=196610\n"
			"\n"
			"\n"
			"#enable audio nr tuning\n"
			"ro.audio_tunning.nr=1\n"
			"\n"
			"#enable audio dual spk  tuning\n"
			"ro.audio_tunning.dual_spk=0\n"
			"\n"
			"\n"
			"ro.fm.chip.port.UART.androidm=false\n"
			"\n"
			"persist.sys.cam.eois.dc.back=false\n"
			"persist.sys.cam.eois.dc.front=false\n"
			"persist.sys.cam.eois.dv.back=true\n"
			"persist.sys.cam.eois.dv.front=false\n"
			"persist.sys.cam.pipviv=false\n"
			"\n"
			"persist.sys.cam.battery.flash=15\n"
			"\n"
			"#FRP property for pst device\n"
			"ro.frp.pst=/dev/block/platform/sdio_emmc/by-name/persist\n"
			"\n"
			"persist.sys.cam.refocus.enable=false\n"
			"persist.sys.cam.ba.blur.version=0\n"
			"persist.sys.cam.api.version=1\n"
			"\n"
			"persist.sys.blending.enable=true\n"
			"\n"
			"persist.sys.sprd.refocus.bokeh=true\n"
			"qemu.hw.mainkeys=0\n"
			"\n"
			"#Enable sdcardfs feature\n"
			"ro.sys.sdcardfs=true\n"
			"persist.sys.sdcardfs=force_on\n"
			"persist.bindcore.hdr=_4_0_1_2_3\n"
			"\n"
			"#disable partial update\n"
			"debug.hwui.use_partial_updates=false\n"
			"\n"
			"persist.sys.fp.ck.period=3\n"
			"#\n"
			"# from device/sprd/isharkl2/k500_lgt_511/custom_config/custom.prop\n"
			"#\n"
			"ro.macro.custom.leagoo=true\n"
			"ro.com.google.clientidbase=android-leagoo\n"
			"ro.build.display.id=LEAGOO_T5c_OS2.1_E_20180112\n"
			"ro.build.display.spid=SC9853_K500_LGT_511_V2.2_20180112\n"
			"ro.build.display.spid.customer=LEAGOO_T5c_OS2.1_E_20180112\n"
			"ro.leagoo.baseband.version=LEAGOO T5c_OS2.1\n"
			"ro.product.bt.name=LEAGOO T5c\n"
			"ro.leagoo.storage.ui=true\n"
			"ro.modify.settings.icon=true\n"
			"ro.lock.disable.statusbar=true\n"
			"ro.leagoo.power.ui=true\n"
			"ro.message.wake.up.screen=false\n"
			"ro.sim.no.switch.languages=true\n"
			"ro.test.playsound.outside=true\n"
			"ro.hide.smart.control=true\n"
			"persist.sys.fp.ck.period=3\n"
			"ro.single.point.y.index=150\n"
			"ro.email.signatures.same=true\n"
			"ro.modify.message.notify=true\n"
			"ro.temp.ntc=true\n"
			"ro.disable.sound.effects=true\n"
			"ro.not.support.menu.key=true\n"
			"ro.not.support.back.key=true\n"
			"ro.not.support.home.key=true\n"
			"ro.ram.display.config.3gb=true\n"
			"ro.rm.calllog.geocode=true\n"
			"ro.preload.media.internel=true\n"
			"ro.support.video.dream=true\n"
			"#\n"
			"# ADDITIONAL_BUILD_PROPERTIES\n"
			"#\n"
			"persist.radio.modem.config=TL_LF_TD_W_G,W_G\n"
			"persist.radio.modem.workmode=9,255\n"
			"ro.radio.modem.capability=TL_LF_TD_W_G,W_G\n"
			"ro.dalvik.vm.isa.arm=x86\n"
			"ro.enable.native.bridge.exec=1\n"
			"ro.dalvik.vm.isa.arm64=x86_64\n"
			"ro.enable.native.bridge.exec64=1\n"
			"ro.sys.prc_compatibility=1\n"
			"ro.hwui.layer_cache_size=48\n"
			"ro.hwui.r_buffer_cache_size=8\n"
			"ro.hwui.path_cache_size=32\n"
			"ro.hwui.gradient_cache_size=1\n"
			"ro.hwui.drop_shadow_cache_size=6\n"
			"ro.hwui.texture_cache_flushrate=0.4\n"
			"ro.hwui.text_small_cache_width=1024\n"
			"ro.hwui.text_small_cache_height=1024\n"
			"ro.hwui.text_large_cache_width=2048\n"
			"ro.hwui.text_large_cache_height=1024\n"
			"dalvik.vm.sprd_usejitprofiles=false\n"
			"ro.config.notification_sound=LEAGOO-Pops.mp3\n"
			"ro.config.alarm_alert=Alarm_Classic.ogg\n"
			"ro.config.ringtone=LEAGOO-Turkish.mp3\n"
			"ro.config.ringtone0=LEAGOO-Turkish.mp3\n"
			"ro.config.ringtone1=LEAGOO-Turkish.mp3\n"
			"ro.config.message_sound0=Argon.ogg\n"
			"ro.config.message_sound1=Argon.ogg\n"
			"ro.config.message_sound=pixiedust.ogg\n"
			"ro.product.locale.language=en\n"
			"ro.product.locale.region=US\n"
			"persist.sys.timezone=Asia/Calcutta\n"
			"persist.sys.defaulttimezone=Asia/Calcutta\n"
			"ro.product.first_api_level=24\n"
			"ro.carrier=unknown\n"
			"rild.libpath=/system/lib64/libsprd-ril.so\n"
			"ro.radio.modemtype=l\n"
			"ro.telephony.default_network=9\n"
			"keyguard.no_require_sim=true\n"
			"ro.com.android.dataroaming=false\n"
			"ro.simlock.unlock.autoshow=1\n"
			"ro.simlock.unlock.bynv=0\n"
			"ro.simlock.onekey.lock=0\n"
			"persist.sys.engpc.disable=1\n"
			"persist.sys.sprd.modemreset=1\n"
			"ro.product.partitionpath=/dev/block/platform/sdio_emmc/by-name/\n"
			"ro.modem.l.dev=/proc/cptl/\n"
			"ro.modem.l.tty=/dev/stty_lte\n"
			"ro.modem.l.diag=/dev/sdiag_lte\n"
			"ro.modem.l.log=/dev/slog_lte\n"
			"ro.modem.l.assert=/dev/spipe_lte2\n"
			"ro.modem.l.fixnv_size=0xc8000\n"
			"ro.modem.l.runnv_size=0xe8000\n"
			"ro.sp.log=/dev/slog_pm\n"
			"persist.modem.l.nvp=l_\n"
			"persist.modem.l.enable=1\n"
			"ro.storage.flash_type=2\n"
			"sys.internal.emulated=1\n"
			"persist.storage.type=2\n"
			"ro.storage.install2internal=0\n"
			"drm.service.enabled=true\n"
			"persist.sys.sprd.wcnreset=1\n"
			"persist.sys.start_udpdatastall=0\n"
			"persist.sys.apr.enabled=0\n"
			"persist.sys.ag.enable=false\n"
			"ro.adb.secure=1\n"
			"persist.sys.apr.intervaltime=1\n"
			"persist.sys.apr.testgroup=CSSLAB\n"
			"persist.sys.apr.autoupload=1\n"
			"ro.modem.l.eth=seth_lte\n"
			"ro.modem.l.snd=1\n"
			"ro.modem.l.loop=/dev/spipe_lte0\n"
			"ro.modem.l.nv=/dev/spipe_lte1\n"
			"ro.modem.l.vbc=/dev/spipe_lte6\n"
			"ro.modem.l.id=0\n"
			"persist.sys.heartbeat.enable=1\n"
			"persist.sys.powerHint.enable=1\n"
			"persist.ylog.modem.shutdownlog=1\n"
			"persist.sys.cam.photo.gallery=true\n"
			"persist.sys.ucam.puzzle=false\n"
			"persist.sys.ucam.edit=false\n"
			"persist.sys.cam.timestamp=false\n"
			"persist.sys.cam.gif=false\n"
			"persist.sys.cam.scenery=false\n"
			"persist.sys.cam.vgesture=false\n"
			"persist.sys.cam.gps=true\n"
			"persist.sys.cam.normalhdr=true\n"
			"persist.sys.cam.sfv.alter=true\n"
			"persist.sys.cam.arcsoft.filter=true\n"
			"persist.sys.cam.filter.highfps=true\n"
			"persist.sys.cam.new.wideangle=true\n"
			"persist.sys.cam.3dnr=true\n"
			"persist.sys.bsservice.enable=1\n"
			"ro.wcn.hardware.product=marlin2\n"
			"ro.bt.bdaddr_path=/data/misc/bluedroid/btmac.txt\n"
			"persist.sys.volte.enable=true\n"
			"ro.trim.config=true\n"
			"persist.sys.vilte.socket=ap\n"
			"persist.dbg.wfc_avail_ovr=1\n"
			"persist.sys.vowifi.voice=cp\n"
			"ro.modem.wcn.enable=1\n"
			"ro.modem.wcn.diag=/dev/slog_wcn\n"
			"ro.modem.wcn.id=1\n"
			"ro.modem.wcn.count=1\n"
			"ro.modem.l.count=2\n"
			"persist.msms.phone_count=2\n"
			"persist.radio.multisim.config=dsds\n"
			"ro.modem.gnss.diag=/dev/slog_gnss\n"
			"persist.sys.support.vt=true\n"
			"persist.sys.csvt=false\n"
			"persist.radio.ssda.mode=csfb\n"
			"ro.wcn.gpschip=ge2\n"
			"ro.hotspot.enabled=1\n"
			"reset_default_http_response=true\n"
			"ro.void_charge_tip=true\n"
			"ro.softaplte.coexist=true\n"
			"ro.vowifi.softap.ee_warning=false\n"
			"persist.sys.wifi.pocketmode=true\n"
			"ro.wcn=enabled\n"
			"ro.softap.whitelist=true\n"
			"ro.btwifisoftap.coexist=true\n"
			"persist.wifi.func.hidessid=true\n"
			"ro.wifi.softap.maxstanum=10\n"
			"ro.wifi.signal.optimized=true\n"
			"ro.support.auto.roam=disabled\n"
			"ro.wifip2p.coexist=true\n"
			"persist.sys.notify.light.color=1\n"
			"persist.sys.charging.tone=1\n"
			"persist.support.fingerprint=true\n"
			"persist.sprd.fp.lockapp=true\n"
			"persist.sprd.fp.launchapp=true\n"
			"persist.sys.volte.mode=Normal\n"
			"ro.setupwizard.mode=OPTIONAL\n"
			"ro.com.google.gmsversion=7.0_r12\n"
			"ro.launcher.circleslide=true\n"
			"ro.launcher.shakewallpaper=true\n"
			"ro.launcher.defaultfoldername=true\n"
			"ro.launcher.unreadinfo=true\n"
			"ro.launcher.dynamicicon=true\n"
			"ro.launcher.dynamicclock=true\n"
			"ro.launcher.dynamiccalendar=true\n"
			"ril.sim.phone_ex.start=true\n"
			"persist.netmon.linger=10000\n"
			"persist.sys.dalvik.vm.lib.2=libart.so\n"
			"dalvik.vm.isa.x86_64.variant=silvermont\n"
			"dalvik.vm.isa.x86_64.features=default\n"
			"dalvik.vm.isa.x86.variant=silvermont\n"
			"dalvik.vm.isa.x86.features=default\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"\n"
			"# begin redstonefota properties\n"
			"ro.redstone.version=LEAGOO_T5c_OS2.1_E_20180112\n"
			"# end fota properties\n"
			"ro.expect.recovery_id=0x0041a640e4805f06ee66e08584f90e9343c058db000000000000000000000000\n",
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
		.size = 324,
		.content = "cpu:type:x86,ven0000fam0006mod0075:feature:,0000,0001,0002,0003,0004,0005,0006,0007,0008,0009,000B,000C,000D,000E,000F,0010,0011,0013,0017,0018,0019,001A,001B,001C,0034,003B,003D,0068,006B,006F,0070,0072,0074,0075,0076,0078,0080,0081,0083,0089,008D,008E,008F,0093,0094,0096,0097,0098,0099,00C0,00C8,00E1,0121,0127,0129,012D\n",
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
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "624000\n",
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
		.size = 39,
		.content = "624000 936000 1248000 1560000 1872000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 45,
		.content = "userspace powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 15,
		.content = "bgnfld-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 76,
		.content =
			"624000 1030325\n"
			"936000 1271\n"
			"1248000 723\n"
			"1560000 1240\n"
			"1872000 22490\n"
			"2028000 0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 5,
		.content = "2276\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/trans_table",
		.size = 521,
		.content =
			"   From  :    To\n"
			"         :    624000    936000   1248000   1560000   1872000   2028000 \n"
			"   624000:         0       166         0         5       507         0 \n"
			"   936000:       222         0        42         1        76         0 \n"
			"  1248000:        74        39         0        22        88         0 \n"
			"  1560000:        46        19        34         0       132         0 \n"
			"  1872000:       337       117       147       202         0         0 \n"
			"  2028000:         0         0         0         0         0         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings_list",
		.size = 4,
		.content = "0-7\n",
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
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "624000\n",
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
		.size = 39,
		.content = "624000 936000 1248000 1560000 1872000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 45,
		.content = "userspace powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
		.size = 15,
		.content = "bgnfld-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 76,
		.content =
			"624000 1030676\n"
			"936000 1271\n"
			"1248000 723\n"
			"1560000 1240\n"
			"1872000 22490\n"
			"2028000 0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 5,
		.content = "2276\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/trans_table",
		.size = 521,
		.content =
			"   From  :    To\n"
			"         :    624000    936000   1248000   1560000   1872000   2028000 \n"
			"   624000:         0       166         0         5       507         0 \n"
			"   936000:       222         0        42         1        76         0 \n"
			"  1248000:        74        39         0        22        88         0 \n"
			"  1560000:        46        19        34         0       132         0 \n"
			"  1872000:       337       117       147       202         0         0 \n"
			"  2028000:         0         0         0         0         0         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_siblings",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_siblings_list",
		.size = 4,
		.content = "0-7\n",
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
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "624000\n",
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
		.size = 39,
		.content = "624000 936000 1248000 1560000 1872000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 45,
		.content = "userspace powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver",
		.size = 15,
		.content = "bgnfld-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 76,
		.content =
			"624000 1031019\n"
			"936000 1271\n"
			"1248000 723\n"
			"1560000 1240\n"
			"1872000 22495\n"
			"2028000 0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 5,
		.content = "2278\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/trans_table",
		.size = 521,
		.content =
			"   From  :    To\n"
			"         :    624000    936000   1248000   1560000   1872000   2028000 \n"
			"   624000:         0       166         0         5       508         0 \n"
			"   936000:       222         0        42         1        76         0 \n"
			"  1248000:        74        39         0        22        88         0 \n"
			"  1560000:        46        19        34         0       132         0 \n"
			"  1872000:       338       117       147       202         0         0 \n"
			"  2028000:         0         0         0         0         0         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_id",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_siblings",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_siblings_list",
		.size = 4,
		.content = "0-7\n",
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
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/affected_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "624000\n",
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
		.size = 39,
		.content = "624000 936000 1248000 1560000 1872000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 45,
		.content = "userspace powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_driver",
		.size = 15,
		.content = "bgnfld-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 76,
		.content =
			"624000 1031362\n"
			"936000 1271\n"
			"1248000 723\n"
			"1560000 1240\n"
			"1872000 22495\n"
			"2028000 0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 5,
		.content = "2278\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/trans_table",
		.size = 521,
		.content =
			"   From  :    To\n"
			"         :    624000    936000   1248000   1560000   1872000   2028000 \n"
			"   624000:         0       166         0         5       508         0 \n"
			"   936000:       222         0        42         1        76         0 \n"
			"  1248000:        74        39         0        22        88         0 \n"
			"  1560000:        46        19        34         0       132         0 \n"
			"  1872000:       338       117       147       202         0         0 \n"
			"  2028000:         0         0         0         0         0         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_id",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_siblings",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_siblings_list",
		.size = 4,
		.content = "0-7\n",
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
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "624000\n",
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
		.size = 39,
		.content = "624000 936000 1248000 1560000 1872000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors",
		.size = 45,
		.content = "userspace powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver",
		.size = 15,
		.content = "bgnfld-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 65,
		.content =
			"624000 1047415\n"
			"936000 1704\n"
			"1248000 2590\n"
			"1560000 705\n"
			"1872000 5034\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 4,
		.content = "758\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/trans_table",
		.size = 389,
		.content =
			"   From  :    To\n"
			"         :    624000    936000   1248000   1560000   1872000 \n"
			"   624000:         0        72         2        13       122 \n"
			"   936000:        52         0        48         3        29 \n"
			"  1248000:        42        30         0        24        21 \n"
			"  1560000:        23         8        25         0        36 \n"
			"  1872000:        92        22        42        52         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/core_id",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/core_siblings",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/core_siblings_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
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
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "624000\n",
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
		.size = 39,
		.content = "624000 936000 1248000 1560000 1872000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_governors",
		.size = 45,
		.content = "userspace powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_driver",
		.size = 15,
		.content = "bgnfld-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 65,
		.content =
			"624000 1047808\n"
			"936000 1704\n"
			"1248000 2590\n"
			"1560000 705\n"
			"1872000 5034\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 4,
		.content = "758\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/trans_table",
		.size = 389,
		.content =
			"   From  :    To\n"
			"         :    624000    936000   1248000   1560000   1872000 \n"
			"   624000:         0        72         2        13       122 \n"
			"   936000:        52         0        48         3        29 \n"
			"  1248000:        42        30         0        24        21 \n"
			"  1560000:        23         8        25         0        36 \n"
			"  1872000:        92        22        42        52         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/core_id",
		.size = 2,
		.content = "5\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/core_siblings",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/core_siblings_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
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
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "624000\n",
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
		.size = 39,
		.content = "624000 936000 1248000 1560000 1872000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_governors",
		.size = 45,
		.content = "userspace powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_driver",
		.size = 15,
		.content = "bgnfld-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 65,
		.content =
			"624000 1048171\n"
			"936000 1704\n"
			"1248000 2590\n"
			"1560000 705\n"
			"1872000 5034\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 4,
		.content = "758\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/trans_table",
		.size = 389,
		.content =
			"   From  :    To\n"
			"         :    624000    936000   1248000   1560000   1872000 \n"
			"   624000:         0        72         2        13       122 \n"
			"   936000:        52         0        48         3        29 \n"
			"  1248000:        42        30         0        24        21 \n"
			"  1560000:        23         8        25         0        36 \n"
			"  1872000:        92        22        42        52         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/core_id",
		.size = 2,
		.content = "6\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/core_siblings",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/core_siblings_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
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
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/affected_cpus",
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "624000\n",
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
		.size = 39,
		.content = "624000 936000 1248000 1560000 1872000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_governors",
		.size = 45,
		.content = "userspace powersave interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_driver",
		.size = 15,
		.content = "bgnfld-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1872000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "624000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 65,
		.content =
			"624000 1048515\n"
			"936000 1704\n"
			"1248000 2590\n"
			"1560000 705\n"
			"1872000 5034\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 4,
		.content = "758\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/trans_table",
		.size = 389,
		.content =
			"   From  :    To\n"
			"         :    624000    936000   1248000   1560000   1872000 \n"
			"   624000:         0        72         2        13       122 \n"
			"   936000:        52         0        48         3        29 \n"
			"  1248000:        42        30         0        24        21 \n"
			"  1560000:        23         8        25         0        36 \n"
			"  1872000:        92        22        42        52         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/core_id",
		.size = 2,
		.content = "7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/core_siblings",
		.size = 3,
		.content = "ff\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/core_siblings_list",
		.size = 4,
		.content = "0-7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
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
		.key = "camera.disable_zsl_mode",
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
		.key = "dalvik.vm.isa.x86.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.x86.variant",
		.value = "silvermont",
	},
	{
		.key = "dalvik.vm.isa.x86_64.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.x86_64.variant",
		.value = "silvermont",
	},
	{
		.key = "dalvik.vm.sprd_usejitprofiles",
		.value = "false",
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
		.key = "debug.hwui.use_partial_updates",
		.value = "false",
	},
	{
		.key = "debug.sf.protect_mm",
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
		.key = "gsm.current.phone-type",
		.value = "1,1",
	},
	{
		.key = "gsm.network.type",
		.value = "GPRS,Unknown",
	},
	{
		.key = "gsm.operator.alpha",
		.value = "T-Mobile",
	},
	{
		.key = "gsm.operator.iso-country",
		.value = "us",
	},
	{
		.key = "gsm.operator.isroaming",
		.value = "false,false",
	},
	{
		.key = "gsm.operator.numeric",
		.value = "310260",
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
		.key = "gsm.sys.volte.state",
		.value = "0,0",
	},
	{
		.key = "gsm.version.baseband",
		.value = "FM_BASE_17A_Release_415P_P6|sc9853_modem|11-07-2017 04:24:05",
	},
	{
		.key = "gsm.version.baseband1",
		.value = "FM_BASE_17A_Release_415P_P6|sc9853_modem|11-07-2017 04:24:05",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "android reference-ril 1.0",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.apr",
		.value = "stopped",
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
		.key = "init.svc.cameraserver",
		.value = "running",
	},
	{
		.key = "init.svc.cmd_services",
		.value = "stopped",
	},
	{
		.key = "init.svc.cp_diskserver",
		.value = "running",
	},
	{
		.key = "init.svc.dataLogDaemon",
		.value = "stopped",
	},
	{
		.key = "init.svc.debuggerd",
		.value = "running",
	},
	{
		.key = "init.svc.debuggerd64",
		.value = "running",
	},
	{
		.key = "init.svc.drm",
		.value = "running",
	},
	{
		.key = "init.svc.engpcclientag",
		.value = "stopped",
	},
	{
		.key = "init.svc.engpcclientlte",
		.value = "stopped",
	},
	{
		.key = "init.svc.ext_data",
		.value = "running",
	},
	{
		.key = "init.svc.fingerprintd",
		.value = "running",
	},
	{
		.key = "init.svc.flash_recovery",
		.value = "stopped",
	},
	{
		.key = "init.svc.gatekeeperd",
		.value = "running",
	},
	{
		.key = "init.svc.gnss_download",
		.value = "running",
	},
	{
		.key = "init.svc.healthd",
		.value = "running",
	},
	{
		.key = "init.svc.ims_bridged",
		.value = "running",
	},
	{
		.key = "init.svc.installd",
		.value = "running",
	},
	{
		.key = "init.svc.ip_monitor",
		.value = "running",
	},
	{
		.key = "init.svc.ju_ipsec_server",
		.value = "running",
	},
	{
		.key = "init.svc.keystore",
		.value = "running",
	},
	{
		.key = "init.svc.lmfs",
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
		.key = "init.svc.modem_control",
		.value = "running",
	},
	{
		.key = "init.svc.modemd",
		.value = "running",
	},
	{
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.path_wrapper",
		.value = "stopped",
	},
	{
		.key = "init.svc.phasecheckserver",
		.value = "running",
	},
	{
		.key = "init.svc.pinnerfile",
		.value = "running",
	},
	{
		.key = "init.svc.refnotify",
		.value = "running",
	},
	{
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.slogmodem",
		.value = "running",
	},
	{
		.key = "init.svc.spril-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.surfaceflinger",
		.value = "running",
	},
	{
		.key = "init.svc.systemDebuggerd",
		.value = "running",
	},
	{
		.key = "init.svc.thermald",
		.value = "running",
	},
	{
		.key = "init.svc.ueventd",
		.value = "running",
	},
	{
		.key = "init.svc.vold",
		.value = "running",
	},
	{
		.key = "init.svc.watchdogd",
		.value = "stopped",
	},
	{
		.key = "init.svc.wcnd",
		.value = "running",
	},
	{
		.key = "init.svc.wdtee_deamon",
		.value = "running",
	},
	{
		.key = "init.svc.wdtee_listener",
		.value = "running",
	},
	{
		.key = "init.svc.ylog",
		.value = "stopped",
	},
	{
		.key = "init.svc.zram",
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
		.key = "lmk.autocalc",
		.value = "true",
	},
	{
		.key = "logd.kernel",
		.value = "false",
	},
	{
		.key = "logd.klogd",
		.value = "false",
	},
	{
		.key = "net.bt.name",
		.value = "Android",
	},
	{
		.key = "net.change",
		.value = "net.qtaguid_enabled",
	},
	{
		.key = "net.hostname",
		.value = "android-96d5a2b66d789c21",
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
		.key = "persist.bindcore.hdr",
		.value = "_4_0_1_2_3",
	},
	{
		.key = "persist.dbg.wfc_avail_ovr",
		.value = "1",
	},
	{
		.key = "persist.modem.l.enable",
		.value = "1",
	},
	{
		.key = "persist.modem.l.nvp",
		.value = "l_",
	},
	{
		.key = "persist.msms.phone_count",
		.value = "2",
	},
	{
		.key = "persist.netmon.linger",
		.value = "10000",
	},
	{
		.key = "persist.radio.fd.disable",
		.value = "0",
	},
	{
		.key = "persist.radio.modem.config",
		.value = "TL_LF_TD_W_G,W_G",
	},
	{
		.key = "persist.radio.modem.workmode",
		.value = "9,255",
	},
	{
		.key = "persist.radio.multisim.config",
		.value = "dsds",
	},
	{
		.key = "persist.radio.ssda.mode",
		.value = "csfb",
	},
	{
		.key = "persist.sprd.fp.launchapp",
		.value = "true",
	},
	{
		.key = "persist.sprd.fp.lockapp",
		.value = "true",
	},
	{
		.key = "persist.support.fingerprint",
		.value = "true",
	},
	{
		.key = "persist.sys.ag.enable",
		.value = "false",
	},
	{
		.key = "persist.sys.apr.autoupload",
		.value = "1",
	},
	{
		.key = "persist.sys.apr.enabled",
		.value = "0",
	},
	{
		.key = "persist.sys.apr.intervaltime",
		.value = "1",
	},
	{
		.key = "persist.sys.apr.testgroup",
		.value = "CSSLAB",
	},
	{
		.key = "persist.sys.blending.enable",
		.value = "true",
	},
	{
		.key = "persist.sys.bsservice.enable",
		.value = "1",
	},
	{
		.key = "persist.sys.cam.3dnr",
		.value = "true",
	},
	{
		.key = "persist.sys.cam.api.version",
		.value = "1",
	},
	{
		.key = "persist.sys.cam.arcsoft.filter",
		.value = "true",
	},
	{
		.key = "persist.sys.cam.ba.blur.version",
		.value = "0",
	},
	{
		.key = "persist.sys.cam.battery.flash",
		.value = "15",
	},
	{
		.key = "persist.sys.cam.eois.dc.back",
		.value = "false",
	},
	{
		.key = "persist.sys.cam.eois.dc.front",
		.value = "false",
	},
	{
		.key = "persist.sys.cam.eois.dv.back",
		.value = "true",
	},
	{
		.key = "persist.sys.cam.eois.dv.front",
		.value = "false",
	},
	{
		.key = "persist.sys.cam.filter.highfps",
		.value = "true",
	},
	{
		.key = "persist.sys.cam.gif",
		.value = "false",
	},
	{
		.key = "persist.sys.cam.gps",
		.value = "true",
	},
	{
		.key = "persist.sys.cam.new.wideangle",
		.value = "true",
	},
	{
		.key = "persist.sys.cam.normalhdr",
		.value = "true",
	},
	{
		.key = "persist.sys.cam.photo.gallery",
		.value = "true",
	},
	{
		.key = "persist.sys.cam.pipviv",
		.value = "false",
	},
	{
		.key = "persist.sys.cam.refocus.enable",
		.value = "false",
	},
	{
		.key = "persist.sys.cam.scenery",
		.value = "false",
	},
	{
		.key = "persist.sys.cam.sfv.alter",
		.value = "true",
	},
	{
		.key = "persist.sys.cam.timestamp",
		.value = "false",
	},
	{
		.key = "persist.sys.cam.unfreeze_time",
		.value = "180",
	},
	{
		.key = "persist.sys.cam.vgesture",
		.value = "false",
	},
	{
		.key = "persist.sys.charging.tone",
		.value = "1",
	},
	{
		.key = "persist.sys.cmccpolicy.disable",
		.value = "false",
	},
	{
		.key = "persist.sys.cmdservice.enable",
		.value = "disable",
	},
	{
		.key = "persist.sys.cp2log",
		.value = "0",
	},
	{
		.key = "persist.sys.csvt",
		.value = "false",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.defaulttimezone",
		.value = "Asia/Calcutta",
	},
	{
		.key = "persist.sys.engpc.disable",
		.value = "1",
	},
	{
		.key = "persist.sys.fp.ck.period",
		.value = "3",
	},
	{
		.key = "persist.sys.gps",
		.value = "enabled",
	},
	{
		.key = "persist.sys.heartbeat.enable",
		.value = "1",
	},
	{
		.key = "persist.sys.isfirstboot",
		.value = "false",
	},
	{
		.key = "persist.sys.locale",
		.value = "en-US",
	},
	{
		.key = "persist.sys.modemassertdump",
		.value = "true",
	},
	{
		.key = "persist.sys.nosleep.enabled",
		.value = "0",
	},
	{
		.key = "persist.sys.notify.light.color",
		.value = "1",
	},
	{
		.key = "persist.sys.parts.main2sensor",
		.value = "sp2509r_mipi_raw",
	},
	{
		.key = "persist.sys.parts.mainsensor",
		.value = "s5k3l8xxm3r_mipi_raw",
	},
	{
		.key = "persist.sys.parts.subsensor",
		.value = "gc5024_mipi_raw",
	},
	{
		.key = "persist.sys.power.hispeed0",
		.value = "1872000",
	},
	{
		.key = "persist.sys.power.hispeed1",
		.value = "1872000",
	},
	{
		.key = "persist.sys.power.target_loads0",
		.value = "90\n",
	},
	{
		.key = "persist.sys.power.target_loads1",
		.value = "90\n",
	},
	{
		.key = "persist.sys.powerHint.enable",
		.value = "1",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
	},
	{
		.key = "persist.sys.qosstate",
		.value = "0",
	},
	{
		.key = "persist.sys.sdcardfs",
		.value = "force_on",
	},
	{
		.key = "persist.sys.sprd.modemreset",
		.value = "1",
	},
	{
		.key = "persist.sys.sprd.refocus.bokeh",
		.value = "true",
	},
	{
		.key = "persist.sys.sprd.wcnlog.result",
		.value = "0",
	},
	{
		.key = "persist.sys.sprd.wcnreset",
		.value = "1",
	},
	{
		.key = "persist.sys.start_udpdatastall",
		.value = "0",
	},
	{
		.key = "persist.sys.support.vt",
		.value = "true",
	},
	{
		.key = "persist.sys.sysdump",
		.value = "off",
	},
	{
		.key = "persist.sys.thermal.pagoven",
		.value = "0",
	},
	{
		.key = "persist.sys.timezone",
		.value = "America/New_York",
	},
	{
		.key = "persist.sys.ucam.edit",
		.value = "false",
	},
	{
		.key = "persist.sys.ucam.puzzle",
		.value = "false",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "ptp,adb",
	},
	{
		.key = "persist.sys.vilte.socket",
		.value = "ap",
	},
	{
		.key = "persist.sys.volte.enable",
		.value = "true",
	},
	{
		.key = "persist.sys.volte.mode",
		.value = "Normal",
	},
	{
		.key = "persist.sys.vowifi.voice",
		.value = "cp",
	},
	{
		.key = "persist.sys.wcn.status",
		.value = "assert",
	},
	{
		.key = "persist.sys.webview.vmsize",
		.value = "164942040",
	},
	{
		.key = "persist.sys.wifi.pocketmode",
		.value = "true",
	},
	{
		.key = "persist.wifi.func.hidessid",
		.value = "true",
	},
	{
		.key = "persist.ylog.modem.shutdownlog",
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
		.value = "verify-profile",
	},
	{
		.key = "pm.dexopt.core-app",
		.value = "speed",
	},
	{
		.key = "pm.dexopt.first-boot",
		.value = "interpret-only",
	},
	{
		.key = "pm.dexopt.forced-dexopt",
		.value = "speed",
	},
	{
		.key = "pm.dexopt.install",
		.value = "interpret-only",
	},
	{
		.key = "pm.dexopt.nsys-library",
		.value = "speed",
	},
	{
		.key = "pm.dexopt.shared-apk",
		.value = "speed",
	},
	{
		.key = "qemu.hw.mainkeys",
		.value = "0",
	},
	{
		.key = "reset_default_http_response",
		.value = "true",
	},
	{
		.key = "ril.autotest",
		.value = "0",
	},
	{
		.key = "ril.ecclist",
		.value = "999,112,995",
	},
	{
		.key = "ril.ecclist1",
		.value = "999,112,995",
	},
	{
		.key = "ril.modem.has_alived",
		.value = "1",
	},
	{
		.key = "ril.sim.phone_ex.start",
		.value = "true",
	},
	{
		.key = "rild.libpath",
		.value = "/system/lib64/libsprd-ril.so",
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
		.key = "ro.audio_tunning.dual_spk",
		.value = "0",
	},
	{
		.key = "ro.audio_tunning.nr",
		.value = "1",
	},
	{
		.key = "ro.baseband",
		.value = "unknown",
	},
	{
		.key = "ro.board.platform",
		.value = "sp9853i",
	},
	{
		.key = "ro.boot.flash.locked",
		.value = "1",
	},
	{
		.key = "ro.boot.hardware",
		.value = "k500_lgt_511",
	},
	{
		.key = "ro.boot.serialno",
		.value = "HUMPHC051NG00375",
	},
	{
		.key = "ro.boot.verifiedbootstate",
		.value = "green",
	},
	{
		.key = "ro.bootimage.build.date",
		.value = "Fri Jan 12 17:54:28 CST 2018",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1515750868",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "LEAGOO/T5c/T5c:7.0/NRD90M/01121754:user/release-keys",
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
		.value = "/data/misc/bluedroid/btmac.txt",
	},
	{
		.key = "ro.btwifisoftap.coexist",
		.value = "true",
	},
	{
		.key = "ro.build.characteristics",
		.value = "nosdcard",
	},
	{
		.key = "ro.build.date",
		.value = "Fri Jan 12 17:54:28 CST 2018",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1515750868",
	},
	{
		.key = "ro.build.description",
		.value = "k500_lgt_511-user 7.0 NRD90M eng.root.20180112.175428 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "LEAGOO_T5c_OS2.1_E_20180112",
	},
	{
		.key = "ro.build.display.spid",
		.value = "SC9853_K500_LGT_511_V2.2_20180112",
	},
	{
		.key = "ro.build.display.spid.customer",
		.value = "LEAGOO_T5c_OS2.1_E_20180112",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "LEAGOO/T5c/T5c:7.0/NRD90M/01121754:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "k500_lgt_511-user",
	},
	{
		.key = "ro.build.host",
		.value = "lxh",
	},
	{
		.key = "ro.build.id",
		.value = "NRD90M",
	},
	{
		.key = "ro.build.product",
		.value = "T5c",
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
		.value = "",
	},
	{
		.key = "ro.build.version.codename",
		.value = "REL",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "eng.root.20180112.175428",
	},
	{
		.key = "ro.build.version.preview_sdk",
		.value = "0",
	},
	{
		.key = "ro.build.version.release",
		.value = "7.0",
	},
	{
		.key = "ro.build.version.sdk",
		.value = "24",
	},
	{
		.key = "ro.build.version.security_patch",
		.value = "2017-12-05",
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
		.key = "ro.com.google.clientidbase",
		.value = "android-leagoo",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "7.0_r12",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Alarm_Classic.ogg",
	},
	{
		.key = "ro.config.low_ram",
		.value = "false",
	},
	{
		.key = "ro.config.message_sound",
		.value = "pixiedust.ogg",
	},
	{
		.key = "ro.config.message_sound0",
		.value = "Argon.ogg",
	},
	{
		.key = "ro.config.message_sound1",
		.value = "Argon.ogg",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "LEAGOO-Pops.mp3",
	},
	{
		.key = "ro.config.ringtone",
		.value = "LEAGOO-Turkish.mp3",
	},
	{
		.key = "ro.config.ringtone0",
		.value = "LEAGOO-Turkish.mp3",
	},
	{
		.key = "ro.config.ringtone1",
		.value = "LEAGOO-Turkish.mp3",
	},
	{
		.key = "ro.config.zram.support",
		.value = "true",
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
		.key = "ro.crypto.type",
		.value = "block",
	},
	{
		.key = "ro.dalvik.vm.isa.arm",
		.value = "x86",
	},
	{
		.key = "ro.dalvik.vm.isa.arm64",
		.value = "x86_64",
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
		.key = "ro.disable.sound.effects",
		.value = "true",
	},
	{
		.key = "ro.email.signatures.same",
		.value = "true",
	},
	{
		.key = "ro.enable.native.bridge.exec",
		.value = "1",
	},
	{
		.key = "ro.enable.native.bridge.exec64",
		.value = "1",
	},
	{
		.key = "ro.expect.recovery_id",
		.value = "0x0041a640e4805f06ee66e08584f90e9343c058db000000000000000000000000",
	},
	{
		.key = "ro.fm.chip.port.UART.androidm",
		.value = "false",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/platform/sdio_emmc/by-name/persist",
	},
	{
		.key = "ro.gpu.fbc",
		.value = "1",
	},
	{
		.key = "ro.hardware",
		.value = "k500_lgt_511",
	},
	{
		.key = "ro.hide.smart.control",
		.value = "true",
	},
	{
		.key = "ro.hotspot.enabled",
		.value = "1",
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
		.key = "ro.kernel.qemu.gles",
		.value = "0",
	},
	{
		.key = "ro.launcher.circleslide",
		.value = "true",
	},
	{
		.key = "ro.launcher.defaultfoldername",
		.value = "true",
	},
	{
		.key = "ro.launcher.dynamiccalendar",
		.value = "true",
	},
	{
		.key = "ro.launcher.dynamicclock",
		.value = "true",
	},
	{
		.key = "ro.launcher.dynamicicon",
		.value = "true",
	},
	{
		.key = "ro.launcher.shakewallpaper",
		.value = "true",
	},
	{
		.key = "ro.launcher.unreadinfo",
		.value = "true",
	},
	{
		.key = "ro.leagoo.baseband.version",
		.value = "LEAGOO T5c_OS2.1",
	},
	{
		.key = "ro.leagoo.power.ui",
		.value = "true",
	},
	{
		.key = "ro.leagoo.storage.ui",
		.value = "true",
	},
	{
		.key = "ro.lock.disable.statusbar",
		.value = "true",
	},
	{
		.key = "ro.macro.custom.leagoo",
		.value = "true",
	},
	{
		.key = "ro.message.wake.up.screen",
		.value = "false",
	},
	{
		.key = "ro.modem.gnss.diag",
		.value = "/dev/slog_gnss",
	},
	{
		.key = "ro.modem.l.assert",
		.value = "/dev/spipe_lte2",
	},
	{
		.key = "ro.modem.l.count",
		.value = "2",
	},
	{
		.key = "ro.modem.l.dev",
		.value = "/proc/cptl/",
	},
	{
		.key = "ro.modem.l.diag",
		.value = "/dev/sdiag_lte",
	},
	{
		.key = "ro.modem.l.eth",
		.value = "seth_lte",
	},
	{
		.key = "ro.modem.l.fixnv_size",
		.value = "0xc8000",
	},
	{
		.key = "ro.modem.l.id",
		.value = "0",
	},
	{
		.key = "ro.modem.l.log",
		.value = "/dev/slog_lte",
	},
	{
		.key = "ro.modem.l.loop",
		.value = "/dev/spipe_lte0",
	},
	{
		.key = "ro.modem.l.nv",
		.value = "/dev/spipe_lte1",
	},
	{
		.key = "ro.modem.l.runnv_size",
		.value = "0xe8000",
	},
	{
		.key = "ro.modem.l.snd",
		.value = "1",
	},
	{
		.key = "ro.modem.l.tty",
		.value = "/dev/stty_lte",
	},
	{
		.key = "ro.modem.l.vbc",
		.value = "/dev/spipe_lte6",
	},
	{
		.key = "ro.modem.wcn.count",
		.value = "1",
	},
	{
		.key = "ro.modem.wcn.diag",
		.value = "/dev/slog_wcn",
	},
	{
		.key = "ro.modem.wcn.enable",
		.value = "1",
	},
	{
		.key = "ro.modem.wcn.id",
		.value = "1",
	},
	{
		.key = "ro.modify.message.notify",
		.value = "true",
	},
	{
		.key = "ro.modify.settings.icon",
		.value = "true",
	},
	{
		.key = "ro.not.support.back.key",
		.value = "true",
	},
	{
		.key = "ro.not.support.home.key",
		.value = "true",
	},
	{
		.key = "ro.not.support.menu.key",
		.value = "true",
	},
	{
		.key = "ro.opengles.version",
		.value = "196610",
	},
	{
		.key = "ro.preload.media.internel",
		.value = "true",
	},
	{
		.key = "ro.product.board",
		.value = "k500_lgt_511_vmm",
	},
	{
		.key = "ro.product.brand",
		.value = "LEAGOO",
	},
	{
		.key = "ro.product.bt.name",
		.value = "LEAGOO T5c",
	},
	{
		.key = "ro.product.cpu.abi",
		.value = "x86_64",
	},
	{
		.key = "ro.product.cpu.abilist",
		.value = "x86_64,x86,armeabi-v7a,armeabi,arm64-v8a",
	},
	{
		.key = "ro.product.cpu.abilist32",
		.value = "x86,armeabi-v7a,armeabi",
	},
	{
		.key = "ro.product.cpu.abilist64",
		.value = "x86_64,arm64-v8a",
	},
	{
		.key = "ro.product.device",
		.value = "T5c",
	},
	{
		.key = "ro.product.first_api_level",
		.value = "24",
	},
	{
		.key = "ro.product.hardware",
		.value = "k500_lgt_511_v1.0.0",
	},
	{
		.key = "ro.product.locale",
		.value = "en-US",
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
		.value = "LEAGOO",
	},
	{
		.key = "ro.product.model",
		.value = "T5c",
	},
	{
		.key = "ro.product.name",
		.value = "T5c",
	},
	{
		.key = "ro.product.partitionpath",
		.value = "/dev/block/platform/sdio_emmc/by-name/",
	},
	{
		.key = "ro.radio.modem.capability",
		.value = "TL_LF_TD_W_G,W_G",
	},
	{
		.key = "ro.radio.modemtype",
		.value = "l",
	},
	{
		.key = "ro.ram.display.config.3gb",
		.value = "true",
	},
	{
		.key = "ro.ramsize",
		.value = "3072M",
	},
	{
		.key = "ro.redstone.version",
		.value = "LEAGOO_T5c_OS2.1_E_20180112",
	},
	{
		.key = "ro.revision",
		.value = "0",
	},
	{
		.key = "ro.rm.calllog.geocode",
		.value = "true",
	},
	{
		.key = "ro.run.ramconfig",
		.value = "8",
	},
	{
		.key = "ro.runtime.firstboot",
		.value = "1521117824875",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "HUMPHC051NG00375",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "OPTIONAL",
	},
	{
		.key = "ro.sf.fbc",
		.value = "1",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "480",
	},
	{
		.key = "ro.sim.no.switch.languages",
		.value = "true",
	},
	{
		.key = "ro.simlock.onekey.lock",
		.value = "0",
	},
	{
		.key = "ro.simlock.unlock.autoshow",
		.value = "1",
	},
	{
		.key = "ro.simlock.unlock.bynv",
		.value = "0",
	},
	{
		.key = "ro.single.point.y.index",
		.value = "150",
	},
	{
		.key = "ro.softap.whitelist",
		.value = "true",
	},
	{
		.key = "ro.softaplte.coexist",
		.value = "true",
	},
	{
		.key = "ro.sp.log",
		.value = "/dev/slog_pm",
	},
	{
		.key = "ro.storage.flash_type",
		.value = "2",
	},
	{
		.key = "ro.storage.install2internal",
		.value = "0",
	},
	{
		.key = "ro.support.auto.roam",
		.value = "disabled",
	},
	{
		.key = "ro.support.video.dream",
		.value = "true",
	},
	{
		.key = "ro.sys.prc_compatibility",
		.value = "1",
	},
	{
		.key = "ro.sys.sdcardfs",
		.value = "true",
	},
	{
		.key = "ro.telephony.default_network",
		.value = "9",
	},
	{
		.key = "ro.temp.ntc",
		.value = "true",
	},
	{
		.key = "ro.test.playsound.outside",
		.value = "true",
	},
	{
		.key = "ro.trim.config",
		.value = "true",
	},
	{
		.key = "ro.void_charge_tip",
		.value = "true",
	},
	{
		.key = "ro.vowifi.softap.ee_warning",
		.value = "false",
	},
	{
		.key = "ro.wcn",
		.value = "enabled",
	},
	{
		.key = "ro.wcn.gpschip",
		.value = "ge2",
	},
	{
		.key = "ro.wcn.hardware.product",
		.value = "marlin2",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
	},
	{
		.key = "ro.wifi.signal.optimized",
		.value = "true",
	},
	{
		.key = "ro.wifi.softap.maxstanum",
		.value = "10",
	},
	{
		.key = "ro.wifip2p.coexist",
		.value = "true",
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
		.key = "selinux.reload_policy",
		.value = "1",
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
		.key = "sys.init_log_level",
		.value = "1",
	},
	{
		.key = "sys.oem_unlock_allowed",
		.value = "0",
	},
	{
		.key = "sys.sprd.power.ispowered",
		.value = "1",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "24300",
	},
	{
		.key = "sys.usb.config",
		.value = "ptp,adb",
	},
	{
		.key = "sys.usb.config.ready",
		.value = "1",
	},
	{
		.key = "sys.usb.configfs",
		.value = "1",
	},
	{
		.key = "sys.usb.controller",
		.value = "e2500000.usb2",
	},
	{
		.key = "sys.usb.ffs.ready",
		.value = "1",
	},
	{
		.key = "sys.usb.mode",
		.value = "normal",
	},
	{
		.key = "sys.usb.state",
		.value = "ptp,adb",
	},
	{
		.key = "sys.vm.swappiness",
		.value = "100",
	},
	{
		.key = "vold.decrypt",
		.value = "trigger_restart_framework",
	},
	{
		.key = "vold.has_adoptable",
		.value = "1",
	},
	{
		.key = "vold.post_fs_data_done",
		.value = "1",
	},
	{
		.key = "vold.realdata.mount",
		.value = "ok",
	},
	{
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "zram.disksize",
		.value = "1024",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
