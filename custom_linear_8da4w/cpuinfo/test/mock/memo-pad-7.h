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
		.eax = 0x00030678,
		.ebx = 0x00100800,
		.ecx = 0x43D8E3BF,
		.edx = 0xBFEBFBFF,
	},
	{
		.input_eax = 0x00000002,
		.eax = 0x61B3A001,
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
		.edx = 0x00004503,
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
		.ebx = 0x20202020,
		.ecx = 0x65746E49,
		.edx = 0x2952286C,
	},
	{
		.input_eax = 0x80000003,
		.eax = 0x6F744120,
		.ebx = 0x4D54286D,
		.ecx = 0x50432029,
		.edx = 0x5A202055,
	},
	{
		.input_eax = 0x80000004,
		.eax = 0x35343733,
		.ebx = 0x20402020,
		.ecx = 0x33332E31,
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
		.size = 3848,
		.content =
			"processor\t: 0\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 55\n"
			"model name\t: Intel(R) Atom(TM) CPU  Z3745  @ 1.33GHz\n"
			"stepping\t: 8\n"
			"microcode\t: 0x882e0100\n"
			"cpu MHz\t\t: 1862.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 0\n"
			"cpu cores\t: 4\n"
			"apicid\t\t: 0\n"
			"initial apicid\t: 0\n"
			"fdiv_bug\t: no\n"
			"f00f_bug\t: no\n"
			"coma_bug\t: no\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 11\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe nx rdtscp lm constant_tsc arch_perfmon pebs bts xtopology nonstop_tsc aperfmperf nonstop_tsc_s3 pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2666.77\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 1\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 55\n"
			"model name\t: Intel(R) Atom(TM) CPU  Z3745  @ 1.33GHz\n"
			"stepping\t: 8\n"
			"microcode\t: 0x882e0100\n"
			"cpu MHz\t\t: 1862.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 1\n"
			"cpu cores\t: 4\n"
			"apicid\t\t: 2\n"
			"initial apicid\t: 2\n"
			"fdiv_bug\t: no\n"
			"f00f_bug\t: no\n"
			"coma_bug\t: no\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 11\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe nx rdtscp lm constant_tsc arch_perfmon pebs bts xtopology nonstop_tsc aperfmperf nonstop_tsc_s3 pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2666.77\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 2\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 55\n"
			"model name\t: Intel(R) Atom(TM) CPU  Z3745  @ 1.33GHz\n"
			"stepping\t: 8\n"
			"microcode\t: 0x882e0100\n"
			"cpu MHz\t\t: 1862.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 2\n"
			"cpu cores\t: 4\n"
			"apicid\t\t: 4\n"
			"initial apicid\t: 4\n"
			"fdiv_bug\t: no\n"
			"f00f_bug\t: no\n"
			"coma_bug\t: no\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 11\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe nx rdtscp lm constant_tsc arch_perfmon pebs bts xtopology nonstop_tsc aperfmperf nonstop_tsc_s3 pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2666.77\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 3\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 55\n"
			"model name\t: Intel(R) Atom(TM) CPU  Z3745  @ 1.33GHz\n"
			"stepping\t: 8\n"
			"microcode\t: 0x882e0100\n"
			"cpu MHz\t\t: 1862.000\n"
			"cache size\t: 1024 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 3\n"
			"cpu cores\t: 4\n"
			"apicid\t\t: 6\n"
			"initial apicid\t: 6\n"
			"fdiv_bug\t: no\n"
			"f00f_bug\t: no\n"
			"coma_bug\t: no\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 11\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe nx rdtscp lm constant_tsc arch_perfmon pebs bts xtopology nonstop_tsc aperfmperf nonstop_tsc_s3 pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2666.77\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n",
	},
	{
		.path = "/system/build.prop",
		.size = 3011,
		.content =
			"# begin build properties\n"
			"# autogenerated by buildinfo.sh\n"
			"ro.build.id=KOT49H\n"
			"ro.build.display.id=WW-3.2.23.191\n"
			"ro.build.csc.version=WW-ME176C-3.2.23.191-release-user-20141030-signed\n"
			"ro.build.version.incremental=WW_K013-WW_user_3.2.23.191_20141030-user-20141030\n"
			"ro.build.version.sdk=19\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.release=4.4.2\n"
			"ro.build.date=2014xE5xB9xB4 10xE6x9Cx88 30xE6x97xA5 xE6x98x9FxE6x9Cx9FxE5x9Bx9B 18:42:37 CST\n"
			"ro.build.date.utc=1414665757\n"
			"ro.build.type=user\n"
			"ro.build.user=jenkins\n"
			"ro.build.host=TDC-Build\n"
			"ro.build.tags=release-keys\n"
			"ro.product.model=K013\n"
			"ro.product.brand=asus\n"
			"ro.product.name=WW_K013\n"
			"ro.product.device=K013\n"
			"ro.product.board=baylake\n"
			"ro.product.cpu.abi=x86\n"
			"ro.product.manufacturer=asus\n"
			"ro.build.asus.sku=WW\n"
			"ro.build.asus.version=3.2.23.191\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=baytrail\n"
			"# ro.build.product is obsolete; use ro.product.device\n"
			"ro.build.product=K013\n"
			"# Do not try to parse ro.build.description or .fingerprint\n"
			"ro.build.description=WW_K013-user 4.4.2 KOT49H WW_user_3.2.23.191_20141030 release-keys\n"
			"ro.build.fingerprint=asus/WW_K013/K013:4.4.2/KOT49H/WW_user_3.2.23.191_20141030:user/release-keys\n"
			"ro.build.characteristics=tablet\n"
			"# end build properties\n"
			"\n"
			"#\n"
			"# ADDITIONAL_BUILD_PROPERTIES\n"
			"#\n"
			"ro.streaming.video.drs=true\n"
			"ro.build.app.version=044000222_201403250253\n"
			"ro.asus.ui=1.0\n"
			"ro.contact.simtype=0\n"
			"ro.config.ringtone=Festival.ogg\n"
			"ro.config.notification_sound=NewMessage.ogg\n"
			"ro.config.newmail_sound=NewMail.ogg\n"
			"ro.config.sentmail_sound=SentMail.ogg\n"
			"ro.config.calendaralert_sound=CalendarEvent.ogg\n"
			"ro.config.alarm_alert=BusyBugs.ogg\n"
			"ro.additionalbutton.operation=0\n"
			"ro.build.asus.version.pensdk=1\n"
			"ro.setupwizard.mode=OPTIONAL\n"
			"ro.com.google.gmsversion=4.4_r5\n"
			"ro.com.google.clientidbase=android-asus-rev\n"
			"ro.com.google.clientidbase.ms=android-asus-rev\n"
			"ro.com.google.clientidbase.am=android-asus-rev\n"
			"ro.com.google.clientidbase.gmm=android-asus-rev\n"
			"ro.com.google.clientidbase.yt=android-asus-rev\n"
			"keyguard.no_require_sim=true\n"
			"ro.com.android.dataroaming=true\n"
			"ro.com.android.dateformat=MM-dd-yyyy\n"
			"ro.carrier=wifi-only\n"
			"dalvik.vm.heapstartsize=4m\n"
			"dalvik.vm.heapgrowthlimit=96m\n"
			"dalvik.vm.heapsize=256m\n"
			"dalvik.vm.heaptargetutilization=0.75\n"
			"dalvik.vm.heapminfree=2m\n"
			"dalvik.vm.heapmaxfree=6m\n"
			"dalvik.jit.code_cache_size=1048576\n"
			"ro.product.locale.language=en\n"
			"ro.product.locale.region=US\n"
			"drm.service.enabled=true\n"
			"ro.opengles.version=196608\n"
			"ro.sf.lcd_density=213\n"
			"bt.version.driver=B1_002.002.004.0132.0141_reduced_2dB\n"
			"ro.blankphone_id=1\n"
			"gps.version.driver=V6.19.6.192204\n"
			"ro.spid.gps.tty=ttyMFD1\n"
			"ro.config.max_starting_bg=9\n"
			"persist.sys.dalvik.vm.lib=libdvm.so\n"
			"ro.ril.status.polling.enable=0\n"
			"ro.product.cpu.abi2=armeabi-v7a\n"
			"ro.config.personality=compat_layout\n"
			"rs.gpu.renderscript=1\n"
			"rs.gpu.filterscript=1\n"
			"rs.gpu.rsIntrinsic=1\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"wifi.version.driver=V1.88.47\n"
			"widi.media.extmode.enable=false\n"
			"widi.hdcp.enable=true\n"
			"persist.asus.cb.gcf.mode=0\n"
			"ro.bsp.app2sd=true\n"
			"ro.config.hwrlib=T9_x86\n"
			"ro.config.xt9ime.max_subtype=7\n"
			"ro.ime.lowmemory=false\n"
			"ro.intel.corp.email=1\n"
			"\n",
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
		.size = 426,
		.content = "x86cpu:vendor:0000:family:0006:model:0037:feature:,0000,0001,0002,0003,0004,0005,0006,0007,0008,0009,000B,000C,000D,000E,000F,0010,0011,0013,0015,0016,0017,0018,0019,001A,001B,001C,001D,001F,0034,003B,003D,0066,0068,006B,006C,006D,0072,0076,0078,007C,007E,0080,0081,0082,0083,0084,0085,0087,0088,0089,008D,008E,008F,0093,0094,0096,0097,0098,0099,009E,00C0,00C8,00E0,00E1,00E3,00E7,0100,0101,0102,0103,0104,0121,0127,0129,012D\n",
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
		.content = "1862000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "532000\n",
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
		.size = 85,
		.content = "1862000 1729000 1596000 1463000 1330000 1197000 1064000 931000 798000 665000 532000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "532000\n",
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
		.content = "532000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 125,
		.content =
			"1862000 12139\n"
			"1729000 47\n"
			"1596000 50\n"
			"1463000 331\n"
			"1330000 29\n"
			"1197000 71\n"
			"1064000 50\n"
			"931000 74\n"
			"798000 101\n"
			"665000 386\n"
			"532000 6409\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "420\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/trans_table",
		.size = 1481,
		.content =
			"   From  :    To\n"
			"         :   1862000   1729000   1596000   1463000   1330000   1197000   1064000    931000    798000    665000    532000 \n"
			"  1862000:         0        10        11        14         5         8         5         6         7         8        52 \n"
			"  1729000:         9         0         2         0         0         0         0         1         0         0         2 \n"
			"  1596000:         8         3         0         0         0         0         0         0         0         1         4 \n"
			"  1463000:        16         1         3         0         1         2         0         1         1         1         2 \n"
			"  1330000:         1         0         0         1         0         2         0         0         0         0         2 \n"
			"  1197000:         5         0         0         0         0         0         4         1         1         0         2 \n"
			"  1064000:         2         0         0         0         0         1         0         1         1         2         4 \n"
			"   931000:         3         0         0         0         0         0         2         0         3         2         5 \n"
			"   798000:         4         0         0         0         0         0         0         5         0         2         9 \n"
			"   665000:        15         0         0         2         0         0         0         0         7         0        33 \n"
			"   532000:        62         0         0        11         0         0         0         0         0        41         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings",
		.size = 2,
		.content = "f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/thread_siblings",
		.size = 2,
		.content = "1\n",
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
		.content = "1862000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "532000\n",
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
		.size = 85,
		.content = "1862000 1729000 1596000 1463000 1330000 1197000 1064000 931000 798000 665000 532000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "532000\n",
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
		.content = "532000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 125,
		.content =
			"1862000 12167\n"
			"1729000 47\n"
			"1596000 50\n"
			"1463000 331\n"
			"1330000 29\n"
			"1197000 71\n"
			"1064000 50\n"
			"931000 74\n"
			"798000 101\n"
			"665000 388\n"
			"532000 6701\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 4,
		.content = "427\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/trans_table",
		.size = 1481,
		.content =
			"   From  :    To\n"
			"         :   1862000   1729000   1596000   1463000   1330000   1197000   1064000    931000    798000    665000    532000 \n"
			"  1862000:         0        10        11        14         5         8         5         6         7         9        54 \n"
			"  1729000:         9         0         2         0         0         0         0         1         0         0         2 \n"
			"  1596000:         8         3         0         0         0         0         0         0         0         1         4 \n"
			"  1463000:        16         1         3         0         1         2         0         1         1         1         2 \n"
			"  1330000:         1         0         0         1         0         2         0         0         0         0         2 \n"
			"  1197000:         5         0         0         0         0         0         4         1         1         0         2 \n"
			"  1064000:         2         0         0         0         0         1         0         1         1         2         4 \n"
			"   931000:         3         0         0         0         0         0         2         0         3         2         5 \n"
			"   798000:         4         0         0         0         0         0         0         5         0         2         9 \n"
			"   665000:        15         0         0         2         0         0         0         0         7         0        34 \n"
			"   532000:        65         0         0        11         0         0         0         0         0        41         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_siblings_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_siblings",
		.size = 2,
		.content = "f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/core_id",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/thread_siblings_list",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/thread_siblings",
		.size = 2,
		.content = "2\n",
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
		.content = "1862000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "532000\n",
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
		.size = 85,
		.content = "1862000 1729000 1596000 1463000 1330000 1197000 1064000 931000 798000 665000 532000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "532000\n",
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
		.content = "532000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 125,
		.content =
			"1862000 12194\n"
			"1729000 47\n"
			"1596000 50\n"
			"1463000 331\n"
			"1330000 29\n"
			"1197000 71\n"
			"1064000 50\n"
			"931000 75\n"
			"798000 101\n"
			"665000 400\n"
			"532000 6969\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 4,
		.content = "437\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/trans_table",
		.size = 1481,
		.content =
			"   From  :    To\n"
			"         :   1862000   1729000   1596000   1463000   1330000   1197000   1064000    931000    798000    665000    532000 \n"
			"  1862000:         0        10        11        14         5         8         5         7         7         9        56 \n"
			"  1729000:         9         0         2         0         0         0         0         1         0         0         2 \n"
			"  1596000:         8         3         0         0         0         0         0         0         0         1         4 \n"
			"  1463000:        16         1         3         0         1         2         0         1         1         1         2 \n"
			"  1330000:         1         0         0         1         0         2         0         0         0         0         2 \n"
			"  1197000:         5         0         0         0         0         0         4         1         1         0         2 \n"
			"  1064000:         2         0         0         0         0         1         0         1         1         2         4 \n"
			"   931000:         3         0         0         0         0         0         2         0         3         2         6 \n"
			"   798000:         4         0         0         0         0         0         0         5         0         2         9 \n"
			"   665000:        16         0         0         2         0         0         0         0         7         0        35 \n"
			"   532000:        67         0         0        11         0         0         0         0         0        43         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_siblings_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_siblings",
		.size = 2,
		.content = "f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/core_id",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/thread_siblings_list",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/thread_siblings",
		.size = 2,
		.content = "4\n",
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
		.content = "1862000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "532000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_transition_latency",
		.size = 6,
		.content = "10000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/related_cpus",
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 85,
		.content = "1862000 1729000 1596000 1463000 1330000 1197000 1064000 931000 798000 665000 532000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "532000\n",
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
		.content = "532000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 125,
		.content =
			"1862000 12209\n"
			"1729000 47\n"
			"1596000 50\n"
			"1463000 331\n"
			"1330000 29\n"
			"1197000 71\n"
			"1064000 50\n"
			"931000 75\n"
			"798000 101\n"
			"665000 414\n"
			"532000 7250\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "443\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/trans_table",
		.size = 1481,
		.content =
			"   From  :    To\n"
			"         :   1862000   1729000   1596000   1463000   1330000   1197000   1064000    931000    798000    665000    532000 \n"
			"  1862000:         0        10        11        14         5         8         5         7         7         9        57 \n"
			"  1729000:         9         0         2         0         0         0         0         1         0         0         2 \n"
			"  1596000:         8         3         0         0         0         0         0         0         0         1         4 \n"
			"  1463000:        16         1         3         0         1         2         0         1         1         1         2 \n"
			"  1330000:         1         0         0         1         0         2         0         0         0         0         2 \n"
			"  1197000:         5         0         0         0         0         0         4         1         1         0         2 \n"
			"  1064000:         2         0         0         0         0         1         0         1         1         2         4 \n"
			"   931000:         3         0         0         0         0         0         2         0         3         2         6 \n"
			"   798000:         4         0         0         0         0         0         0         5         0         2         9 \n"
			"   665000:        17         0         0         2         0         0         0         0         7         0        36 \n"
			"   532000:        68         0         0        11         0         0         0         0         0        45         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_siblings_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_siblings",
		.size = 2,
		.content = "f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/core_id",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/thread_siblings_list",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/thread_siblings",
		.size = 2,
		.content = "8\n",
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
		.key = "Audiocomms.Audience.IsPresent",
		.value = "false",
	},
	{
		.key = "Audiocomms.BT.HFP.Supported",
		.value = "true",
	},
	{
		.key = "Audiocomms.Vibrator.IsPresent",
		.value = "true",
	},
	{
		.key = "alsa.mixer.builtinMic",
		.value = "Mic1",
	},
	{
		.key = "alsa.mixer.defaultCard",
		.value = "baytrailaudio",
	},
	{
		.key = "alsa.mixer.defaultGain",
		.value = "1.0",
	},
	{
		.key = "alsa.mixer.earpiece",
		.value = "Headphone",
	},
	{
		.key = "alsa.mixer.headphone",
		.value = "Headphone",
	},
	{
		.key = "alsa.mixer.headsetMic",
		.value = "Mic1",
	},
	{
		.key = "alsa.mixer.headset",
		.value = "Headphone",
	},
	{
		.key = "alsa.mixer.speaker",
		.value = "Speaker",
	},
	{
		.key = "ap.interface",
		.value = "wlan0",
	},
	{
		.key = "audio.device.name",
		.value = "baytrailaudio",
	},
	{
		.key = "audio.offload.capabilities",
		.value = "0",
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
		.key = "audiocomms.XMM.isDualSimModem",
		.value = "false",
	},
	{
		.key = "audiocomms.XMM.primaryChannel",
		.value = "/dev/gsmtty13",
	},
	{
		.key = "audiocomms.XMM.secondaryChannel",
		.value = "",
	},
	{
		.key = "audiocomms.modemLib",
		.value = "",
	},
	{
		.key = "boot.factoryreset.type",
		.value = "0",
	},
	{
		.key = "bt.version.driver",
		.value = "B1_002.002.004.0132.0141_reduced_2dB",
	},
	{
		.key = "camera.hal.control",
		.value = "24",
	},
	{
		.key = "coreu.dpst.aggressiveness",
		.value = "2",
	},
	{
		.key = "dalvik.jit.code_cache_size",
		.value = "1048576",
	},
	{
		.key = "dalvik.vm.heapgrowthlimit",
		.value = "96m",
	},
	{
		.key = "dalvik.vm.heapmaxfree",
		.value = "6m",
	},
	{
		.key = "dalvik.vm.heapminfree",
		.value = "2m",
	},
	{
		.key = "dalvik.vm.heapsize",
		.value = "256m",
	},
	{
		.key = "dalvik.vm.heapstartsize",
		.value = "4m",
	},
	{
		.key = "dalvik.vm.heaptargetutilization",
		.value = "0.75",
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
		.key = "debug.rs.gpu.filterscript",
		.value = "1",
	},
	{
		.key = "debug.rs.gpu.renderscript",
		.value = "1",
	},
	{
		.key = "debug.rs.gpu.rsIntrinsic",
		.value = "1",
	},
	{
		.key = "dev.bootcomplete",
		.value = "1",
	},
	{
		.key = "dhcp.wlan0.dns1",
		.value = "208.67.222.222",
	},
	{
		.key = "dhcp.wlan0.dns2",
		.value = "208.67.220.220",
	},
	{
		.key = "dhcp.wlan0.dns3",
		.value = "",
	},
	{
		.key = "dhcp.wlan0.dns4",
		.value = "",
	},
	{
		.key = "dhcp.wlan0.domain",
		.value = "tfbnw.net",
	},
	{
		.key = "dhcp.wlan0.gateway",
		.value = "172.22.160.1",
	},
	{
		.key = "dhcp.wlan0.ipaddress",
		.value = "172.22.182.182",
	},
	{
		.key = "dhcp.wlan0.leasetime",
		.value = "1800",
	},
	{
		.key = "dhcp.wlan0.mask",
		.value = "255.255.224.0",
	},
	{
		.key = "dhcp.wlan0.mtu",
		.value = "",
	},
	{
		.key = "dhcp.wlan0.pid",
		.value = "1046",
	},
	{
		.key = "dhcp.wlan0.reason",
		.value = "ROUTERADVERT",
	},
	{
		.key = "dhcp.wlan0.result",
		.value = "ok",
	},
	{
		.key = "dhcp.wlan0.server",
		.value = "192.168.47.185",
	},
	{
		.key = "dhcp.wlan0.vendorInfo",
		.value = "",
	},
	{
		.key = "drm.service.enabled",
		.value = "true",
	},
	{
		.key = "gps.version.driver",
		.value = "V6.19.6.192204",
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
		.key = "gsm.sim.state",
		.value = "NOT_READY",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.akmd",
		.value = "running",
	},
	{
		.key = "init.svc.asus_audbg",
		.value = "stopped",
	},
	{
		.key = "init.svc.baytrail-setup",
		.value = "stopped",
	},
	{
		.key = "init.svc.bd_prov",
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
		.key = "init.svc.dhcpcd_wlan0",
		.value = "running",
	},
	{
		.key = "init.svc.drm",
		.value = "running",
	},
	{
		.key = "init.svc.enable_houdini",
		.value = "stopped",
	},
	{
		.key = "init.svc.fg_conf",
		.value = "stopped",
	},
	{
		.key = "init.svc.flash_recovery",
		.value = "stopped",
	},
	{
		.key = "init.svc.gpsd",
		.value = "running",
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
		.key = "init.svc.keystore",
		.value = "running",
	},
	{
		.key = "init.svc.media",
		.value = "running",
	},
	{
		.key = "init.svc.net_eth0-start",
		.value = "stopped",
	},
	{
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.p2p_supplicant",
		.value = "running",
	},
	{
		.key = "init.svc.partlink",
		.value = "running",
	},
	{
		.key = "init.svc.rfid_monzaxd",
		.value = "running",
	},
	{
		.key = "init.svc.rfkill_bt",
		.value = "stopped",
	},
	{
		.key = "init.svc.rmasusdir",
		.value = "stopped",
	},
	{
		.key = "init.svc.sdcard",
		.value = "running",
	},
	{
		.key = "init.svc.securityfile",
		.value = "running",
	},
	{
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.set_property",
		.value = "stopped",
	},
	{
		.key = "init.svc.surfaceflinger",
		.value = "running",
	},
	{
		.key = "init.svc.ueventd",
		.value = "running",
	},
	{
		.key = "init.svc.upi_ug31xx",
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
		.key = "init.svc.wdogcounter",
		.value = "stopped",
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
		.key = "lpa.audiosetup.time",
		.value = "85",
	},
	{
		.key = "lpa.deepbuffer.enable",
		.value = "1",
	},
	{
		.key = "media.settings.xml",
		.value = "/etc/media_profiles_LQ.xml",
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
		.value = "208.67.222.222",
	},
	{
		.key = "net.dns2",
		.value = "208.67.220.220",
	},
	{
		.key = "net.hostname",
		.value = "android-9a1fb0ff95290378",
	},
	{
		.key = "net.qtaguid_enabled",
		.value = "1",
	},
	{
		.key = "net.tcp.buffersize.default",
		.value = "4096,87380,110208,4096,16384,110208",
	},
	{
		.key = "net.tcp.buffersize.edge",
		.value = "4093,26280,110208,4096,16384,110208",
	},
	{
		.key = "net.tcp.buffersize.evdo",
		.value = "4094,87380,262144,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.gprs",
		.value = "4092,8760,65536,4096,8760,65536",
	},
	{
		.key = "net.tcp.buffersize.hsdpa",
		.value = "4094,87380,262144,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.hspa",
		.value = "4094,87380,262144,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.hspap",
		.value = "4094,87380,1220608,4096,16384,1220608",
	},
	{
		.key = "net.tcp.buffersize.hsupa",
		.value = "4094,87380,262144,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.lte",
		.value = "524288,1048576,2097152,262144,524288,1048576",
	},
	{
		.key = "net.tcp.buffersize.umts",
		.value = "4094,87380,110208,4096,16384,110208",
	},
	{
		.key = "net.tcp.buffersize.wifi",
		.value = "524288,1048576,2097152,131072,262144,393216",
	},
	{
		.key = "offload.compress.device",
		.value = "2",
	},
	{
		.key = "persist.asus.audbg",
		.value = "0",
	},
	{
		.key = "persist.asus.cb.gcf.mode",
		.value = "0",
	},
	{
		.key = "persist.dual_sim",
		.value = "none",
	},
	{
		.key = "persist.ril-daemon.disable",
		.value = "0",
	},
	{
		.key = "persist.service.cwsmgr.coex",
		.value = "1",
	},
	{
		.key = "persist.service.cwsmgr.nortcoex",
		.value = "1",
	},
	{
		.key = "persist.service.thermal",
		.value = "1",
	},
	{
		.key = "persist.sys.dalvik.vm.lib",
		.value = "libdvm.so",
	},
	{
		.key = "persist.sys.power_saving.IM",
		.value = "0",
	},
	{
		.key = "persist.sys.power_saving",
		.value = "1",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
	},
	{
		.key = "persist.sys.setupwizard.active",
		.value = "false",
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
		.key = "persist.thermal.display.msg",
		.value = "1",
	},
	{
		.key = "persist.thermal.display.vibra",
		.value = "1",
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
		.key = "persist.thermal.turbo.dynamic",
		.value = "1",
	},
	{
		.key = "ro.adb.secure",
		.value = "1",
	},
	{
		.key = "ro.additionalbutton.operation",
		.value = "0",
	},
	{
		.key = "ro.allow.mock.location",
		.value = "0",
	},
	{
		.key = "ro.arch",
		.value = "x86",
	},
	{
		.key = "ro.asus.ui",
		.value = "1.0",
	},
	{
		.key = "ro.baseband",
		.value = "unknown",
	},
	{
		.key = "ro.blankphone_id",
		.value = "1",
	},
	{
		.key = "ro.board.platform",
		.value = "baytrail",
	},
	{
		.key = "ro.boot.boardid",
		.value = "03.02",
	},
	{
		.key = "ro.boot.bootmedia",
		.value = "sdcard",
	},
	{
		.key = "ro.boot.hardware",
		.value = "K013",
	},
	{
		.key = "ro.boot.mode",
		.value = "main",
	},
	{
		.key = "ro.boot.serialno",
		.value = "000F8AD429",
	},
	{
		.key = "ro.boot.spid",
		.value = "0000:0000:0000:0007:0000:0005",
	},
	{
		.key = "ro.boot.wakesrc",
		.value = "05",
	},
	{
		.key = "ro.bootloader",
		.value = "unknown",
	},
	{
		.key = "ro.bootmode",
		.value = "main",
	},
	{
		.key = "ro.bsp.app2sd",
		.value = "true",
	},
	{
		.key = "ro.bt.bdaddr_path",
		.value = "/config/bt/bd_addr.conf",
	},
	{
		.key = "ro.bt.conf_file",
		.value = "/system/etc/bluetooth/bt_K013.conf",
	},
	{
		.key = "ro.build.app.version",
		.value = "044000222_201403250253",
	},
	{
		.key = "ro.build.asus.sku",
		.value = "WW",
	},
	{
		.key = "ro.build.asus.version.pensdk",
		.value = "1",
	},
	{
		.key = "ro.build.asus.version",
		.value = "3.2.23.191",
	},
	{
		.key = "ro.build.characteristics",
		.value = "tablet",
	},
	{
		.key = "ro.build.csc.version",
		.value = "WW-ME176C-3.2.23.191-release-user-20141030-signed",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1414665757",
	},
	{
		.key = "ro.build.date",
		.value = "2014xE5xB9xB4 10xE6x9Cx88 30xE6x97xA5 xE6x98x9FxE6x9Cx9FxE5x9Bx9B 18:42:37 CST",
	},
	{
		.key = "ro.build.description",
		.value = "WW_K013-user 4.4.2 KOT49H WW_user_3.2.23.191_20141030 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "WW-3.2.23.191",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "asus/WW_K013/K013_1:4.4.2/KOT49H/WW_user_3.2.23.191_20141030:user/release-keys",
	},
	{
		.key = "ro.build.host",
		.value = "TDC-Build",
	},
	{
		.key = "ro.build.id",
		.value = "KOT49H",
	},
	{
		.key = "ro.build.product",
		.value = "K013",
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
		.value = "jenkins",
	},
	{
		.key = "ro.build.version.codename",
		.value = "REL",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "WW_K013-WW_user_3.2.23.191_20141030-user-20141030",
	},
	{
		.key = "ro.build.version.release",
		.value = "4.4.2",
	},
	{
		.key = "ro.build.version.sdk",
		.value = "19",
	},
	{
		.key = "ro.camera.sound.forced",
		.value = "0",
	},
	{
		.key = "ro.carrier",
		.value = "wifi-only",
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
		.key = "ro.com.google.clientidbase.am",
		.value = "android-asus-rev",
	},
	{
		.key = "ro.com.google.clientidbase.gmm",
		.value = "android-asus-rev",
	},
	{
		.key = "ro.com.google.clientidbase.ms",
		.value = "android-asus-rev",
	},
	{
		.key = "ro.com.google.clientidbase.yt",
		.value = "android-asus-rev",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-asus-rev",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "4.4_r5",
	},
	{
		.key = "ro.config.CID",
		.value = "ASUS",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "BusyBugs.ogg",
	},
	{
		.key = "ro.config.calendaralert_sound",
		.value = "CalendarEvent.ogg",
	},
	{
		.key = "ro.config.hwrlib",
		.value = "T9_x86",
	},
	{
		.key = "ro.config.idcode",
		.value = "1A",
	},
	{
		.key = "ro.config.max_starting_bg",
		.value = "9",
	},
	{
		.key = "ro.config.newmail_sound",
		.value = "NewMail.ogg",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "NewMessage.ogg",
	},
	{
		.key = "ro.config.personality",
		.value = "compat_layout",
	},
	{
		.key = "ro.config.ringtone",
		.value = "Festival.ogg",
	},
	{
		.key = "ro.config.sentmail_sound",
		.value = "SentMail.ogg",
	},
	{
		.key = "ro.config.versatility",
		.value = "US",
	},
	{
		.key = "ro.config.xt9ime.max_subtype",
		.value = "7",
	},
	{
		.key = "ro.contact.simtype",
		.value = "0",
	},
	{
		.key = "ro.crypto.fuse_sdcard",
		.value = "true",
	},
	{
		.key = "ro.crypto.state",
		.value = "unencrypted",
	},
	{
		.key = "ro.debuggable",
		.value = "0",
	},
	{
		.key = "ro.epad.mount_point.microsd",
		.value = "/Removable/MicroSD",
	},
	{
		.key = "ro.epad.mount_point.sdreader",
		.value = "/Removable/SD",
	},
	{
		.key = "ro.epad.mount_point.usbdisk1",
		.value = "/Removable/USBdisk1",
	},
	{
		.key = "ro.epad.mount_point.usbdisk2",
		.value = "/Removable/USBdisk2",
	},
	{
		.key = "ro.factorytest",
		.value = "0",
	},
	{
		.key = "ro.fastboot_openadb",
		.value = "0",
	},
	{
		.key = "ro.fmrx.sound.forced",
		.value = "1",
	},
	{
		.key = "ro.hardware",
		.value = "K013",
	},
	{
		.key = "ro.ime.lowmemory",
		.value = "false",
	},
	{
		.key = "ro.intel.corp.email",
		.value = "1",
	},
	{
		.key = "ro.isn",
		.value = "N0CY1421MB0012069",
	},
	{
		.key = "ro.opengles.version",
		.value = "196608",
	},
	{
		.key = "ro.product.board",
		.value = "baylake",
	},
	{
		.key = "ro.product.brand",
		.value = "asus",
	},
	{
		.key = "ro.product.cpu.abi2",
		.value = "armeabi-v7a",
	},
	{
		.key = "ro.product.cpu.abi",
		.value = "x86",
	},
	{
		.key = "ro.product.device",
		.value = "K013_1",
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
		.value = "asus",
	},
	{
		.key = "ro.product.model",
		.value = "K013",
	},
	{
		.key = "ro.product.name",
		.value = "WW_K013",
	},
	{
		.key = "ro.rebootchargermode",
		.value = "true",
	},
	{
		.key = "ro.revision",
		.value = "0",
	},
	{
		.key = "ro.ril.status.polling.enable",
		.value = "0",
	},
	{
		.key = "ro.runtime.firstboot",
		.value = "1506382280231",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "E5OKCY436782",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "OPTIONAL",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "213",
	},
	{
		.key = "ro.spid.gps.pmm",
		.value = "disabled",
	},
	{
		.key = "ro.spid.gps.tty",
		.value = "ttyMFD1",
	},
	{
		.key = "ro.streaming.video.drs",
		.value = "true",
	},
	{
		.key = "ro.thermal.ituxversion",
		.value = "2.0",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
	},
	{
		.key = "rs.gpu.filterscript",
		.value = "1",
	},
	{
		.key = "rs.gpu.renderscript",
		.value = "1",
	},
	{
		.key = "rs.gpu.rsIntrinsic",
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
		.key = "sys.chaabi.version",
		.value = "0.7.51.1115",
	},
	{
		.key = "sys.config.maxxaudio",
		.value = "1",
	},
	{
		.key = "sys.ia32.version",
		.value = "0065.002F",
	},
	{
		.key = "sys.ifwi.version",
		.value = "0065.002F",
	},
	{
		.key = "sys.kernel.version",
		.value = "3.10.20-g7b4e4b8",
	},
	{
		.key = "sys.pdr.version",
		.value = "0000.0000",
	},
	{
		.key = "sys.settings_secure_version",
		.value = "4",
	},
	{
		.key = "sys.settings_system_version",
		.value = "1",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "12000",
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
		.key = "sys.usb.vbus",
		.value = "critical",
	},
	{
		.key = "system_init.startsurfaceflinger",
		.value = "0",
	},
	{
		.key = "vold.post_fs_data_done",
		.value = "1",
	},
	{
		.key = "widi.audio.module",
		.value = "submix",
	},
	{
		.key = "widi.hdcp.enable",
		.value = "true",
	},
	{
		.key = "widi.media.extmode.enable",
		.value = "false",
	},
	{
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "wifi.version.driver",
		.value = "V1.88.47",
	},
	{
		.key = "wlan.driver.status",
		.value = "ok",
	},
	{
		.key = "wlan.driver.vendor",
		.value = "bcm",
	},
	{
		.key = "wpa_supplicant.pid",
		.value = "593",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
