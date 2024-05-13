struct cpuinfo_mock_cpuid cpuid_dump[] = {
	{
		.input_eax = 0x00000000,
		.eax = 0x0000000A,
		.ebx = 0x756E6547,
		.ecx = 0x6C65746E,
		.edx = 0x49656E69,
	},
	{
		.input_eax = 0x00000001,
		.eax = 0x00030651,
		.ebx = 0x03040800,
		.ecx = 0x0040C3BD,
		.edx = 0xBFE9FBFF,
	},
	{
		.input_eax = 0x00000002,
		.eax = 0x4FBA5901,
		.ebx = 0x0E3080C0,
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
		.eax = 0x04004121,
		.ebx = 0x0140003F,
		.ecx = 0x0000003F,
		.edx = 0x00000001,
	},
	{
		.input_eax = 0x00000004,
		.input_ecx = 0x00000001,
		.eax = 0x04004122,
		.ebx = 0x01C0003F,
		.ecx = 0x0000003F,
		.edx = 0x00000001,
	},
	{
		.input_eax = 0x00000004,
		.input_ecx = 0x00000002,
		.eax = 0x04004143,
		.ebx = 0x01C0003F,
		.ecx = 0x000003FF,
		.edx = 0x00000001,
	},
	{
		.input_eax = 0x00000005,
		.eax = 0x00000040,
		.ebx = 0x00000040,
		.ecx = 0x00000003,
		.edx = 0x03020220,
	},
	{
		.input_eax = 0x00000006,
		.eax = 0x00000005,
		.ebx = 0x00000002,
		.ecx = 0x00000001,
		.edx = 0x00000000,
	},
	{
		.input_eax = 0x00000007,
		.input_ecx = 0x00000000,
		.eax = 0x00000000,
		.ebx = 0x00000000,
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
		.ecx = 0x00000001,
		.edx = 0x00100000,
	},
	{
		.input_eax = 0x80000002,
		.eax = 0x20202020,
		.ebx = 0x20202020,
		.ecx = 0x746E4920,
		.edx = 0x52286C65,
	},
	{
		.input_eax = 0x80000003,
		.eax = 0x74412029,
		.ebx = 0x54286D6F,
		.ecx = 0x4320294D,
		.edx = 0x5A205550,
	},
	{
		.input_eax = 0x80000004,
		.eax = 0x30323532,
		.ebx = 0x20402020,
		.ecx = 0x30322E31,
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
		.ecx = 0x02006040,
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
		.eax = 0x00002020,
		.ebx = 0x00000000,
		.ecx = 0x00000000,
		.edx = 0x00000000,
	},
};
struct cpuinfo_mock_file filesystem[] = {
	{
		.path = "/proc/cpuinfo",
		.size = 3241,
		.content =
			"processor\t: 0\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 53\n"
			"model name\t: Intel(R) Atom(TM) CPU Z2520  @ 1.20GHz\n"
			"stepping\t: 1\n"
			"microcode\t: 0x110\n"
			"cpu MHz\t\t: 800.000\n"
			"cache size\t: 512 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 0\n"
			"cpu cores\t: 2\n"
			"apicid\t\t: 0\n"
			"initial apicid\t: 0\n"
			"fdiv_bug\t: no\n"
			"f00f_bug\t: no\n"
			"coma_bug\t: no\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 10\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe nx constant_tsc arch_perfmon pebs bts nonstop_tsc aperfmperf nonstop_tsc_s3 pni dtes64 monitor ds_cpl vmx est tm2 ssse3 xtpr pdcm movbe lahf_lm arat dtherm tpr_shadow vnmi flexpriority\n"
			"bogomips\t: 2396.16\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 32 bits physical, 32 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 1\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 53\n"
			"model name\t: Intel(R) Atom(TM) CPU Z2520  @ 1.20GHz\n"
			"stepping\t: 1\n"
			"microcode\t: 0x110\n"
			"cpu MHz\t\t: 1200.000\n"
			"cache size\t: 512 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 0\n"
			"cpu cores\t: 2\n"
			"apicid\t\t: 1\n"
			"initial apicid\t: 1\n"
			"fdiv_bug\t: no\n"
			"f00f_bug\t: no\n"
			"coma_bug\t: no\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 10\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe nx constant_tsc arch_perfmon pebs bts nonstop_tsc aperfmperf nonstop_tsc_s3 pni dtes64 monitor ds_cpl vmx est tm2 ssse3 xtpr pdcm movbe lahf_lm arat dtherm tpr_shadow vnmi flexpriority\n"
			"bogomips\t: 2396.16\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 32 bits physical, 32 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 2\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 53\n"
			"model name\t: Intel(R) Atom(TM) CPU Z2520  @ 1.20GHz\n"
			"stepping\t: 1\n"
			"microcode\t: 0x110\n"
			"cpu MHz\t\t: 800.000\n"
			"cache size\t: 512 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 1\n"
			"cpu cores\t: 2\n"
			"apicid\t\t: 2\n"
			"initial apicid\t: 2\n"
			"fdiv_bug\t: no\n"
			"f00f_bug\t: no\n"
			"coma_bug\t: no\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 10\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe nx constant_tsc arch_perfmon pebs bts nonstop_tsc aperfmperf nonstop_tsc_s3 pni dtes64 monitor ds_cpl vmx est tm2 ssse3 xtpr pdcm movbe lahf_lm arat dtherm tpr_shadow vnmi flexpriority\n"
			"bogomips\t: 2396.16\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 32 bits physical, 32 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 3\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 53\n"
			"model name\t: Intel(R) Atom(TM) CPU Z2520  @ 1.20GHz\n"
			"stepping\t: 1\n"
			"microcode\t: 0x110\n"
			"cpu MHz\t\t: 800.000\n"
			"cache size\t: 512 KB\n"
			"physical id\t: 0\n"
			"siblings\t: 4\n"
			"core id\t\t: 1\n"
			"cpu cores\t: 2\n"
			"apicid\t\t: 3\n"
			"initial apicid\t: 3\n"
			"fdiv_bug\t: no\n"
			"f00f_bug\t: no\n"
			"coma_bug\t: no\n"
			"fpu\t\t: yes\n"
			"fpu_exception\t: yes\n"
			"cpuid level\t: 10\n"
			"wp\t\t: yes\n"
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe nx constant_tsc arch_perfmon pebs bts nonstop_tsc aperfmperf nonstop_tsc_s3 pni dtes64 monitor ds_cpl vmx est tm2 ssse3 xtpr pdcm movbe lahf_lm arat dtherm tpr_shadow vnmi flexpriority\n"
			"bogomips\t: 2396.16\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 32 bits physical, 32 bits virtual\n"
			"power management:\n"
			"\n",
	},
	{
		.path = "/system/build.prop",
		.size = 3317,
		.content =
			"# begin build properties\n"
			"# autogenerated by buildinfo.sh\n"
			"ro.build.id=KVT49L\n"
			"ro.build.display.id=ASUS_Z007_WW_user_4.11.40.79_20150305 release-keys\n"
			"ro.build.version.incremental=WW_zc451cg-WW_user_4.11.40.79_20150305-user-20150305\n"
			"ro.build.version.sdk=19\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.release=4.4.2\n"
			"ro.build.date=2015xE5xB9xB4 03xE6x9Cx88 05xE6x97xA5 xE6x98x9FxE6x9Cx9FxE5x9Bx9B 17:56:08 CST\n"
			"ro.build.date.utc=1425549368\n"
			"ro.build.type=user\n"
			"ro.build.user=builder3\n"
			"ro.build.host=BUILDER3\n"
			"ro.build.tags=release-keys\n"
			"ro.epad.model=ASUS_Z007\n"
			"ro.product.model=ASUS_Z007\n"
			"ro.product.brand=asus\n"
			"ro.product.name=WW_zc451cg\n"
			"ro.product.device=ASUS_Z007\n"
			"ro.product.board=clovertrail\n"
			"ro.product.cpu.abi=x86\n"
			"ro.product.manufacturer=asus\n"
			"ro.product.locale.language=en\n"
			"ro.product.locale.region=US\n"
			"ro.build.asus.sku=WW\n"
			"ro.build.asus.version=4.11.40.79\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=clovertrail\n"
			"# ro.build.product is obsolete; use ro.product.device\n"
			"ro.build.product=ASUS_Z007\n"
			"# Do not try to parse ro.build.description or .fingerprint\n"
			"ro.build.description=zc451cg-user 4.4.2 KVT49L WW_user_4.11.40.79_20150305 release-keys\n"
			"ro.build.fingerprint=asus/WW_zc451cg/ASUS_Z007:4.4.2/KVT49L/WW_user_4.11.40.79_20150305:user/release-keys\n"
			"ro.build.characteristics=default\n"
			"ro.build.csc.version=WW-ZC451CG-4.11.40.79-rel-user-20150305-175507-signed\n"
			"# end build properties\n"
			"\n"
			"#\n"
			"# ADDITIONAL_BUILD_PROPERTIES\n"
			"#\n"
			"ro.streaming.video.drs=true\n"
			"ro.build.app.version=044000307_201407220024\n"
			"ro.asus.ui=1.0\n"
			"ro.spid.gps.tty=ttyMFD3\n"
			"ro.spid.gps.FrqPlan=FRQ_PLAN_26MHZ_2PPM_26MHZ_100PPB\n"
			"ro.spid.gps.RfType=GL_RF_4752_BRCM_EXT_LNA\n"
			"ro.spid.gps.pmm=disabled\n"
			"persist.gps.chip=bcm4752\n"
			"widi.media.extmode.enable=false\n"
			"widi.uibc.enable=false\n"
			"ro.contact.simtype=1\n"
			"ro.config.ringtone=Festival.ogg\n"
			"ro.config.notification_sound=NewMessage.ogg\n"
			"ro.config.newmail_sound=NewMail.ogg\n"
			"ro.config.sentmail_sound=SentMail.ogg\n"
			"ro.config.calendaralert_sound=CalendarEvent.ogg\n"
			"ro.config.alarm_alert=BusyBugs.ogg\n"
			"ro.additionalbutton.operation=0\n"
			"ro.build.asus.version.pensdk=1\n"
			"ro.asus.amax.lite=1\n"
			"ro.setupwizard.mode=OPTIONAL\n"
			"ro.com.google.gmsversion=4.4_r5\n"
			"ro.com.google.clientidbase=android-asus\n"
			"ro.com.google.clientidbase.ms=android-asus\n"
			"ro.com.google.clientidbase.am=android-asus\n"
			"ro.com.google.clientidbase.gmm=android-asus\n"
			"ro.com.google.clientidbase.yt=android-asus\n"
			"keyguard.no_require_sim=true\n"
			"ro.com.android.dataroaming=false\n"
			"ro.com.android.dateformat=MM-dd-yyyy\n"
			"ro.carrier=unknown\n"
			"dalvik.vm.heapstartsize=8m\n"
			"dalvik.vm.heapgrowthlimit=96m\n"
			"dalvik.vm.heapsize=256m\n"
			"dalvik.vm.heaptargetutilization=0.75\n"
			"dalvik.vm.heapminfree=2m\n"
			"dalvik.vm.heapmaxfree=6m\n"
			"dalvik.jit.code_cache_size=1048576\n"
			"ro.hwui.texture_cache_size=24.0f\n"
			"ro.hwui.text_large_cache_width=2048\n"
			"ro.hwui.text_large_cache_height=512\n"
			"persist.tel.hot_swap.support=true\n"
			"drm.service.enabled=true\n"
			"ro.blankphone_id=1\n"
			"ro.system.simtype=2\n"
			"panel.physicalWidthmm=55\n"
			"panel.physicalHeightmm=98\n"
			"persist.sys.dalvik.vm.lib=libdvm.so\n"
			"ro.ril.status.polling.enable=0\n"
			"ro.product.cpu.abi2=armeabi-v7a\n"
			"ro.config.personality=compat_layout\n"
			"rs.gpu.renderscript=0\n"
			"rs.gpu.filterscript=0\n"
			"rs.gpu.rsIntrinsic=0\n"
			"ro.sf.lcd_density=240\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"persist.telproviders.debug=0\n"
			"persist.asus.message.gcf.mode=0\n"
			"persist.asus.message.debug=0\n"
			"ro.config.hwrlib=T9_x86\n"
			"ro.config.xt9ime.max_subtype=7\n"
			"ro.ime.lowmemory=false\n",
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
		.size = 321,
		.content = "x86cpu:vendor:0000:family:0006:model:0035:feature:,0000,0001,0002,0003,0004,0005,0006,0007,0008,0009,000B,000C,000D,000E,000F,0010,0013,0015,0016,0017,0018,0019,001A,001B,001C,001D,001F,0034,0066,0068,006B,006C,006D,0072,0078,007C,007E,0080,0082,0083,0084,0085,0087,0088,0089,008E,008F,0096,00C0,00E1,00E7,0100,0101,0102\n",
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
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1200000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 23,
		.content = "1200000 933000 800000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 12,
		.content = "sfi-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 36,
		.content =
			"1200000 4635\n"
			"933000 258\n"
			"800000 8780\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "213\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/trans_table",
		.size = 185,
		.content =
			"   From  :    To\n"
			"         :   1200000    933000    800000 \n"
			"  1200000:         0         5        76 \n"
			"   933000:        38         0         9 \n"
			"   800000:        43        42         0 \n",
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
		.size = 4,
		.content = "0-1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/thread_siblings",
		.size = 2,
		.content = "3\n",
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
		.size = 4,
		.content = "0-1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index0/shared_cpu_map",
		.size = 2,
		.content = "3\n",
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
		.size = 4,
		.content = "0-1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index1/shared_cpu_map",
		.size = 2,
		.content = "3\n",
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
		.size = 5,
		.content = "512K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "8\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/affected_cpus",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1200000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/related_cpus",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_frequencies",
		.size = 23,
		.content = "1200000 933000 800000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
		.size = 12,
		.content = "sfi-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1200000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 36,
		.content =
			"1200000 4873\n"
			"933000 208\n"
			"800000 8864\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 4,
		.content = "223\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/trans_table",
		.size = 185,
		.content =
			"   From  :    To\n"
			"         :   1200000    933000    800000 \n"
			"  1200000:         0        12        79 \n"
			"   933000:        35         0         9 \n"
			"   800000:        56        32         0 \n",
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
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/thread_siblings_list",
		.size = 4,
		.content = "0-1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/topology/thread_siblings",
		.size = 2,
		.content = "3\n",
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
		.size = 4,
		.content = "0-1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index0/shared_cpu_map",
		.size = 2,
		.content = "3\n",
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
		.size = 4,
		.content = "0-1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index1/shared_cpu_map",
		.size = 2,
		.content = "3\n",
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
		.size = 5,
		.content = "512K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "8\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/affected_cpus",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1200000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/related_cpus",
		.size = 2,
		.content = "2\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies",
		.size = 23,
		.content = "1200000 933000 800000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver",
		.size = 12,
		.content = "sfi-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1200000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 36,
		.content =
			"1200000 5134\n"
			"933000 340\n"
			"800000 8738\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 4,
		.content = "257\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/trans_table",
		.size = 185,
		.content =
			"   From  :    To\n"
			"         :   1200000    933000    800000 \n"
			"  1200000:         0        12        84 \n"
			"   933000:        39         0        19 \n"
			"   800000:        57        46         0 \n",
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
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/thread_siblings_list",
		.size = 4,
		.content = "2-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/topology/thread_siblings",
		.size = 2,
		.content = "c\n",
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
		.size = 4,
		.content = "2-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index0/shared_cpu_map",
		.size = 2,
		.content = "c\n",
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
		.size = 4,
		.content = "2-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index1/shared_cpu_map",
		.size = 2,
		.content = "c\n",
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
		.size = 5,
		.content = "512K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "8\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/affected_cpus",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1200000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/related_cpus",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 23,
		.content = "1200000 933000 800000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_driver",
		.size = 12,
		.content = "sfi-cpufreq\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1200000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 36,
		.content =
			"1200000 5255\n"
			"933000 336\n"
			"800000 8901\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "246\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/trans_table",
		.size = 185,
		.content =
			"   From  :    To\n"
			"         :   1200000    933000    800000 \n"
			"  1200000:         0        16        77 \n"
			"   933000:        42         0        17 \n"
			"   800000:        51        43         0 \n",
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
		.content = "1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/thread_siblings_list",
		.size = 4,
		.content = "2-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/topology/thread_siblings",
		.size = 2,
		.content = "c\n",
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
		.size = 4,
		.content = "2-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index0/shared_cpu_map",
		.size = 2,
		.content = "c\n",
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
		.size = 4,
		.content = "2-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index1/shared_cpu_map",
		.size = 2,
		.content = "c\n",
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
		.size = 5,
		.content = "512K\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/type",
		.size = 8,
		.content = "Unified\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cache/index2/ways_of_associativity",
		.size = 2,
		.content = "8\n",
	},
	{ NULL },
};

#ifdef __ANDROID__
struct cpuinfo_mock_property properties[] = {
	{
		.key = "Audio.Media.CodecDelayMs",
		.value = "20",
	},
	{
		.key = "Audiocomms.Audience.IsPresent",
		.value = "true",
	},
	{
		.key = "Audiocomms.BT.HFP.Supported",
		.value = "true",
	},
	{
		.key = "alsa.mixer.builtinMic",
		.value = "Mic1",
	},
	{
		.key = "alsa.mixer.defaultCard",
		.value = "cloverviewaudio",
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
		.key = "atd.keybox.ready",
		.value = "1",
	},
	{
		.key = "audio.device.name",
		.value = "cloverviewaudio",
	},
	{
		.key = "audio.media_pb.decoder.dump",
		.value = "disable",
	},
	{
		.key = "audio.media_pb.flinger.dump",
		.value = "disable",
	},
	{
		.key = "audio.media_pb.parser.dump",
		.value = "disable",
	},
	{
		.key = "audio.media_rc.flinger.dump",
		.value = "disable",
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
		.value = "true",
	},
	{
		.key = "audiocomms.XMM.primaryChannel",
		.value = "/dev/gsmtty20",
	},
	{
		.key = "audiocomms.XMM.secondaryChannel",
		.value = "/dev/gsmtty36",
	},
	{
		.key = "audiocomms.modemLib",
		.value = "libmamgr-xmm.so",
	},
	{
		.key = "boot.factoryreset.type",
		.value = "0",
	},
	{
		.key = "bt.version.driver",
		.value = "V18.23.38.1.0",
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
		.value = "8m",
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
		.value = "172.22.192.1",
	},
	{
		.key = "dhcp.wlan0.ipaddress",
		.value = "172.22.201.148",
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
		.value = "1083",
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
		.value = "192.168.137.185",
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
		.value = "6.19.6.198372",
	},
	{
		.key = "gsm.current.phone-type2",
		.value = "1",
	},
	{
		.key = "gsm.current.phone-type",
		.value = "1",
	},
	{
		.key = "gsm.dsds.simactivity",
		.value = "0",
	},
	{
		.key = "gsm.firmware.upload",
		.value = "ok",
	},
	{
		.key = "gsm.internal.oem.concurrency",
		.value = "notallowed",
	},
	{
		.key = "gsm.net.interface",
		.value = "rmnet0",
	},
	{
		.key = "gsm.network.type",
		.value = "Unknown",
	},
	{
		.key = "gsm.sim.state",
		.value = "ABSENT",
	},
	{
		.key = "gsm.sim2.state",
		.value = "ABSENT",
	},
	{
		.key = "gsm.simmanager.set_off_sim1",
		.value = "false",
	},
	{
		.key = "gsm.simmanager.set_off_sim2",
		.value = "false",
	},
	{
		.key = "gsm.version.baseband",
		.value = "1508_1.19.60.1_0226",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "Intrinsyc Rapid-RIL M6.59 for Android 4.2 (Build September 17/2013)",
	},
	{
		.key = "gsm.version.tlv",
		.value = "\"TLV_ZC451CG_V0.06\"",
	},
	{
		.key = "init.svc.RILD1_DSDS",
		.value = "running",
	},
	{
		.key = "init.svc.RILD2_DSDS",
		.value = "running",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.akmd",
		.value = "stopped",
	},
	{
		.key = "init.svc.asus_audbg",
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
		.key = "init.svc.enable_adb",
		.value = "stopped",
	},
	{
		.key = "init.svc.enable_houdini",
		.value = "stopped",
	},
	{
		.key = "init.svc.gpscerd",
		.value = "stopped",
	},
	{
		.key = "init.svc.gpsd_bcm4752",
		.value = "running",
	},
	{
		.key = "init.svc.gpsd_bcm47531",
		.value = "stopped",
	},
	{
		.key = "init.svc.healthd",
		.value = "running",
	},
	{
		.key = "init.svc.info_setting",
		.value = "stopped",
	},
	{
		.key = "init.svc.init_logdate",
		.value = "stopped",
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
		.key = "init.svc.mmgr",
		.value = "running",
	},
	{
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.nvmmanager",
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
		.key = "init.svc.pitservice",
		.value = "stopped",
	},
	{
		.key = "init.svc.proxy-tunneling",
		.value = "stopped",
	},
	{
		.key = "init.svc.proxy",
		.value = "stopped",
	},
	{
		.key = "init.svc.pvrsrvctl",
		.value = "stopped",
	},
	{
		.key = "init.svc.rfkill_bt",
		.value = "stopped",
	},
	{
		.key = "init.svc.ril-daemon",
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
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.simid",
		.value = "stopped",
	},
	{
		.key = "init.svc.surfaceflinger",
		.value = "running",
	},
	{
		.key = "init.svc.tkbd",
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
		.key = "init.svc.wlan_prov",
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
		.key = "lpa.deepbuffer.enable",
		.value = "1",
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
		.value = "android-421bda0b2126bf26",
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
		.key = "panel.physicalHeightmm",
		.value = "98",
	},
	{
		.key = "panel.physicalWidthmm",
		.value = "55",
	},
	{
		.key = "persist.asus.audbg",
		.value = "0",
	},
	{
		.key = "persist.asus.dclick",
		.value = "0",
	},
	{
		.key = "persist.asus.glove",
		.value = "0",
	},
	{
		.key = "persist.asus.instant_camera",
		.value = "0",
	},
	{
		.key = "persist.asus.message.debug",
		.value = "0",
	},
	{
		.key = "persist.asus.message.gcf.mode",
		.value = "0",
	},
	{
		.key = "persist.asuslog.dump.date",
		.value = "2017_1019_205935",
	},
	{
		.key = "persist.dual_sim",
		.value = "dsds",
	},
	{
		.key = "persist.dynamic-data-sim",
		.value = "reboot",
	},
	{
		.key = "persist.gps.chip",
		.value = "bcm4752",
	},
	{
		.key = "persist.modem.tlv.cpaction",
		.value = "done",
	},
	{
		.key = "persist.radio.device.imei2",
		.value = "357067065748715",
	},
	{
		.key = "persist.radio.device.imei",
		.value = "357067065748707",
	},
	{
		.key = "persist.radio.ril_modem_state",
		.value = "1",
	},
	{
		.key = "persist.ril-daemon.disable",
		.value = "dsds",
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
		.value = "America/New_York",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "mtp,adb",
	},
	{
		.key = "persist.system.at-proxy.mode",
		.value = "0",
	},
	{
		.key = "persist.tcs.config_name",
		.value = "XMM6360_CONF_FLASHLESS",
	},
	{
		.key = "persist.tel.hot_swap.support",
		.value = "true",
	},
	{
		.key = "persist.telproviders.debug",
		.value = "0",
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
		.key = "ril.coredumpwarning.enable",
		.value = "1",
	},
	{
		.key = "ril.ecclist1",
		.value = "112,911,000,08,110,118,119,999",
	},
	{
		.key = "ril.ecclist",
		.value = "112,911,000,08,110,118,119,999",
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
		.key = "ro.asus.amax.lite",
		.value = "1",
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
		.value = "clovertrail",
	},
	{
		.key = "ro.boot.bootmedia",
		.value = "sdcard",
	},
	{
		.key = "ro.boot.hardware",
		.value = "redhookbay",
	},
	{
		.key = "ro.boot.mode",
		.value = "main",
	},
	{
		.key = "ro.boot.serialno",
		.value = "MedfieldE35E5BF7",
	},
	{
		.key = "ro.boot.spid",
		.value = "0000:0000:0003:0002:0000:0021",
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
		.key = "ro.build.app.version",
		.value = "044000307_201407220024",
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
		.value = "4.11.40.79",
	},
	{
		.key = "ro.build.characteristics",
		.value = "default",
	},
	{
		.key = "ro.build.csc.version",
		.value = "WW-ZC451CG-4.11.40.79-rel-user-20150305-175507-signed",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1425549368",
	},
	{
		.key = "ro.build.date",
		.value = "2015xE5xB9xB4 03xE6x9Cx88 05xE6x97xA5 xE6x98x9FxE6x9Cx9FxE5x9Bx9B 17:56:08 CST",
	},
	{
		.key = "ro.build.description",
		.value = "zc451cg-user 4.4.2 KVT49L WW_user_4.11.40.79_20150305 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "ASUS_Z007_WW_user_4.11.40.79_20150305 release-keys",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "asus/WW_zc451cg/ASUS_Z007:4.4.2/KVT49L/WW_user_4.11.40.79_20150305:user/release-keys",
	},
	{
		.key = "ro.build.host",
		.value = "BUILDER3",
	},
	{
		.key = "ro.build.id",
		.value = "KVT49L",
	},
	{
		.key = "ro.build.product",
		.value = "ASUS_Z007",
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
		.value = "builder3",
	},
	{
		.key = "ro.build.version.codename",
		.value = "REL",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "WW_zc451cg-WW_user_4.11.40.79_20150305-user-20150305",
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
		.value = "unknown",
	},
	{
		.key = "ro.com.android.dataroaming",
		.value = "false",
	},
	{
		.key = "ro.com.android.dateformat",
		.value = "MM-dd-yyyy",
	},
	{
		.key = "ro.com.google.clientidbase.am",
		.value = "android-asus",
	},
	{
		.key = "ro.com.google.clientidbase.gmm",
		.value = "android-asus",
	},
	{
		.key = "ro.com.google.clientidbase.ms",
		.value = "android-asus",
	},
	{
		.key = "ro.com.google.clientidbase.yt",
		.value = "android-asus",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-asus",
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
		.value = "WW",
	},
	{
		.key = "ro.config.xt9ime.max_subtype",
		.value = "7",
	},
	{
		.key = "ro.contact.simtype",
		.value = "1",
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
		.key = "ro.epad.model",
		.value = "ASUS_Z007",
	},
	{
		.key = "ro.epad.mount_point.microsd",
		.value = "/storage/MicroSD",
	},
	{
		.key = "ro.epad.mount_point.usbdisk1",
		.value = "/storage/USBdisk1",
	},
	{
		.key = "ro.factorytest",
		.value = "0",
	},
	{
		.key = "ro.fmrx.sound.forced",
		.value = "1",
	},
	{
		.key = "ro.gsm.fac.mode",
		.value = "0",
	},
	{
		.key = "ro.hardware",
		.value = "redhookbay",
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
		.key = "ro.ime.lowmemory",
		.value = "false",
	},
	{
		.key = "ro.isn",
		.value = "HD5390HQ80E3RS",
	},
	{
		.key = "ro.opengles.version",
		.value = "131072",
	},
	{
		.key = "ro.product.board",
		.value = "clovertrail",
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
		.value = "ASUS_Z007",
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
		.value = "ASUS_Z007",
	},
	{
		.key = "ro.product.name",
		.value = "WW_zc451cg",
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
		.value = "1508461210442",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "F3AZB701E114",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "OPTIONAL",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "240",
	},
	{
		.key = "ro.spid.gps.FrqPlan",
		.value = "FRQ_PLAN_26MHZ_2PPM_26MHZ_100PPB",
	},
	{
		.key = "ro.spid.gps.RfType",
		.value = "GL_RF_4752_BRCM_EXT_LNA",
	},
	{
		.key = "ro.spid.gps.pmm",
		.value = "disabled",
	},
	{
		.key = "ro.spid.gps.tty",
		.value = "ttyMFD3",
	},
	{
		.key = "ro.streaming.video.drs",
		.value = "true",
	},
	{
		.key = "ro.system.simtype",
		.value = "2",
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
		.value = "0",
	},
	{
		.key = "rs.gpu.renderscript",
		.value = "0",
	},
	{
		.key = "rs.gpu.rsIntrinsic",
		.value = "0",
	},
	{
		.key = "service.bootanim.exit",
		.value = "1",
	},
	{
		.key = "sys.adbon.oneshot",
		.value = "0",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
	},
	{
		.key = "sys.config.maxxaudio",
		.value = "1",
	},
	{
		.key = "sys.ia32.version",
		.value = "00.57",
	},
	{
		.key = "sys.ifwi.version",
		.value = "84.11",
	},
	{
		.key = "sys.kernel.version",
		.value = "3.10.20-g4f9772c",
	},
	{
		.key = "sys.punit.version",
		.value = "A1.40",
	},
	{
		.key = "sys.scu.version",
		.value = "20.3E",
	},
	{
		.key = "sys.settings_global_version",
		.value = "5",
	},
	{
		.key = "sys.settings_secure_version",
		.value = "4",
	},
	{
		.key = "sys.suppia32.version",
		.value = "00.57",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "4803",
	},
	{
		.key = "sys.tkb.enable",
		.value = "false",
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
		.key = "sys.valhooks.version",
		.value = "84.11",
	},
	{
		.key = "system.at-proxy.mode",
		.value = "0",
	},
	{
		.key = "vold.post_fs_data_done",
		.value = "1",
	},
	{
		.key = "widi.media.extmode.enable",
		.value = "false",
	},
	{
		.key = "widi.uibc.enable",
		.value = "false",
	},
	{
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "wifi.version.driver",
		.value = "7.10.323.28",
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
		.value = "694",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
