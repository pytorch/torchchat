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
		.eax = 0x000506A0,
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
		.eax = 0x30383533,
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
		.size = 3682,
		.content =
			"processor\t: 0\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 90\n"
			"model name\t: Intel(R) Atom(TM) CPU  Z3580  @ 1.33GHz\n"
			"stepping\t: 0\n"
			"microcode\t: 0x38\n"
			"cpu MHz\t\t: 1333.000\n"
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
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2662.40\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 1\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 90\n"
			"model name\t: Intel(R) Atom(TM) CPU  Z3580  @ 1.33GHz\n"
			"stepping\t: 0\n"
			"microcode\t: 0x38\n"
			"cpu MHz\t\t: 1333.000\n"
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
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2662.40\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 2\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 90\n"
			"model name\t: Intel(R) Atom(TM) CPU  Z3580  @ 1.33GHz\n"
			"stepping\t: 0\n"
			"microcode\t: 0x38\n"
			"cpu MHz\t\t: 500.000\n"
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
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2662.40\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 3\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 90\n"
			"model name\t: Intel(R) Atom(TM) CPU  Z3580  @ 1.33GHz\n"
			"stepping\t: 0\n"
			"microcode\t: 0x38\n"
			"cpu MHz\t\t: 500.000\n"
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
			"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes rdrand lahf_lm 3dnowprefetch ida arat epb dtherm tpr_shadow vnmi flexpriority ept vpid tsc_adjust smep erms\n"
			"bogomips\t: 2662.40\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 36 bits physical, 48 bits virtual\n"
			"power management:\n"
			"\n",
	},
	{
		.path = "/system/build.prop",
		.size = 3983,
		.content =
			"\n"
			"# begin build properties\n"
			"# autogenerated by buildinfo.sh\n"
			"ro.build.id=MMB29P\n"
			"ro.build.display.id=MMB29P.WW-ASUS_Z00A-4.21.40.352_20170623_7598_user\n"
			"ro.build.version.incremental=WW_Z00A-WW_4.21.40.352_20170623_7598_user_rel-user-20170623\n"
			"ro.build.version.sdk=23\n"
			"ro.build.version.preview_sdk=0\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.all_codenames=REL\n"
			"ro.build.version.release=6.0.1\n"
			"ro.build.version.houdini=6.1.1a\n"
			"ro.build.version.security_patch=2017-05-01\n"
			"ro.build.version.base_os=\n"
			"ro.build.date=xE4xBAx94  6xE6x9Cx88 23 00:08:45 CST 2017\n"
			"ro.build.date.utc=1498147725\n"
			"ro.build.type=user\n"
			"ro.build.user=jenkins\n"
			"ro.build.host=fdc-01-jenkins\n"
			"ro.build.tags=release-keys\n"
			"ro.build.flavor=asusmofd_fhd-user\n"
			"ro.product.model=ASUS_Z00A\n"
			"ro.product.brand=asus\n"
			"ro.product.name=WW_Z00A\n"
			"ro.product.device=Z00A\n"
			"ro.product.board=moorefield\n"
			"# ro.product.cpu.abi and ro.product.cpu.abi2 are obsolete,\n"
			"# use ro.product.cpu.abilist instead.\n"
			"ro.product.cpu.abi=x86\n"
			"ro.product.cpu.abilist=x86,armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist32=x86,armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist64=\n"
			"ro.product.first_api_level=21\n"
			"ro.product.manufacturer=asus\n"
			"ro.product.locale=en-US\n"
			"ro.build.asus.sku=WW\n"
			"ro.build.asus.version=4.21.40.352\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=moorefield\n"
			"# ro.build.product is obsolete; use ro.product.device\n"
			"ro.build.product=mofd_v1\n"
			"# Do not try to parse description, fingerprint, or thumbprint\n"
			"ro.build.description=asusmofd_fhd-user 6.0.1 MMB29P 4.21.40.352_20170623_7598_user release-keys\n"
			"ro.build.fingerprint=asus/WW_Z00A/Z00A:6.0.1/MMB29P/4.21.40.352_20170623_7598_user:user/release-keys\n"
			"ro.build.characteristics=nosdcard\n"
			"ro.build.csc.version=WW_ZE551ML_4.21.40.352_20170623\n"
			"ro.camera.sound.forced=0\n"
			"# end build properties\n"
			"\n"
			"#\n"
			"# ADDITIONAL_BUILD_PROPERTIES\n"
			"#\n"
			"ro.build.app.version=060020736_201603210001\n"
			"ro.asus.ui=1.0\n"
			"ro.ril.ecclist=112,911\n"
			"ro.com.google.clientidbase=android-asus\n"
			"ro.com.google.clientidbase.ms=android-asus\n"
			"ro.com.google.clientidbase.am=android-asus\n"
			"ro.com.google.clientidbase.gmm=android-asus\n"
			"ro.com.google.clientidbase.yt=android-asus\n"
			"ro.spid.gps.tty=ttyMFD2\n"
			"ro.spid.gps.FrqPlan=FRQ_PLAN_26MHZ_2PPM\n"
			"ro.spid.gps.RfType=GL_RF_47531_BRCM\n"
			"hwc.video.extmode.enable=0\n"
			"ro.nfc.conf=mofd-ffd2-a\n"
			"ro.nfc.clk=pll\n"
			"keyguard.no_require_sim=true\n"
			"ro.com.android.dateformat=MM-dd-yyyy\n"
			"ro.carrier=unknown\n"
			"ro.telephony.default_network=9\n"
			"ro.asus.network.types=2\n"
			"persist.tel.hot_swap.support=true\n"
			"ro.asus.phone.ipcall=0\n"
			"ro.asus.phone.sipcall=1\n"
			"drm.service.enabled=true\n"
			"ro.blankphone_id=1\n"
			"ro.dalvik.vm.isa.arm=x86\n"
			"ro.enable.native.bridge.exec=1\n"
			"dalvik.vm.heapstartsize=16m\n"
			"dalvik.vm.heapgrowthlimit=256m\n"
			"dalvik.vm.heapsize=512m\n"
			"dalvik.vm.heaptargetutilization=0.75\n"
			"dalvik.vm.heapminfree=512k\n"
			"dalvik.vm.heapmaxfree=8m\n"
			"ro.hwui.texture_cache_size=72\n"
			"ro.hwui.layer_cache_size=48\n"
			"ro.hwui.r_buffer_cache_size=8\n"
			"ro.hwui.gradient_cache_size=1\n"
			"ro.hwui.path_cache_size=32\n"
			"ro.hwui.drop_shadow_cache_size=6\n"
			"ro.hwui.texture_cache_flushrate=0.4\n"
			"ro.hwui.text_small_cache_width=1024\n"
			"ro.hwui.text_small_cache_height=1024\n"
			"ro.hwui.text_large_cache_width=2048\n"
			"ro.hwui.text_large_cache_height=1024\n"
			"ro.camera.sound.forced=0\n"
			"ro.config.ringtone=Festival.ogg\n"
			"ro.config.notification_sound=NewMessage.ogg\n"
			"ro.config.newmail_sound=NewMail.ogg\n"
			"ro.config.sentmail_sound=SentMail.ogg\n"
			"ro.config.calendaralert_sound=CalendarEvent.ogg\n"
			"ro.config.alarm_alert=BusyBugs.ogg\n"
			"ro.additionalbutton.operation=0\n"
			"ro.setupwizard.mode=OPTIONAL\n"
			"ro.com.google.gmsversion=6.0_r11\n"
			"ro.ril.status.polling.enable=0\n"
			"rild.libpath=/system/lib/librapid-ril-core.so\n"
			"bt.hfp.WideBandSpeechEnabled=true\n"
			"gps.version.driver=66.19.20.275658\n"
			"wifi.version.driver=6.37.45.11\n"
			"bt.version.driver=V10.00.02\n"
			"persist.sys.dalvik.vm.lib.2=libart\n"
			"dalvik.vm.isa.x86.variant=x86\n"
			"dalvik.vm.isa.x86.features=default\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"ro.config.hwrlib=T9_x86\n"
			"ro.config.xt9ime.max_subtype=7\n"
			"ro.ime.lowmemory=false\n"
			"ro.intel.corp.email=1\n"
			"ro.expect.recovery_id=0x9c0e1ee4a82056edf9114ab36dc033fd65faac41000000000000000000000000\n"
			"\n"
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
		.size = 436,
		.content = "x86cpu:vendor:0000:family:0006:model:005A:feature:,0000,0001,0002,0003,0004,0005,0006,0007,0008,0009,000B,000C,000D,000E,000F,0010,0011,0013,0015,0016,0017,0018,0019,001A,001B,001C,001D,001F,002B,0034,003B,003D,0068,006B,006C,006D,006F,0070,0072,0074,0076,0078,007C,0080,0081,0082,0083,0084,0085,0087,0088,0089,008D,008E,008F,0093,0094,0096,0097,0098,0099,009E,00C0,00C8,00E0,00E1,00E3,00E7,0100,0101,0102,0103,0104,0121,0127,0129,012D\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/all_time_in_state",
		.size = 769,
		.content =
			"freq\t\tcpu0\t\tcpu1\t\tcpu2\t\tcpu3\t\t\n"
			"500000\t\t4283894\t\t4283894\t\t4404405\t\t4404405\t\t\n"
			"583000\t\t14933\t\t14933\t\t3589\t\t3589\t\t\n"
			"666000\t\t3580\t\t3580\t\t1684\t\t1684\t\t\n"
			"750000\t\t3550\t\t3550\t\t1120\t\t1120\t\t\n"
			"833000\t\t1801\t\t1801\t\t912\t\t912\t\t\n"
			"916000\t\t1745\t\t1745\t\t849\t\t849\t\t\n"
			"1000000\t\t1321\t\t1321\t\t630\t\t630\t\t\n"
			"1083000\t\t994\t\t994\t\t730\t\t730\t\t\n"
			"1166000\t\t875\t\t875\t\t593\t\t593\t\t\n"
			"1250000\t\t872\t\t872\t\t724\t\t724\t\t\n"
			"1333000\t\t936\t\t936\t\t637\t\t637\t\t\n"
			"1416000\t\t753\t\t753\t\t716\t\t716\t\t\n"
			"1500000\t\t800\t\t800\t\t758\t\t758\t\t\n"
			"1583000\t\t708\t\t708\t\t703\t\t703\t\t\n"
			"1666000\t\t703\t\t703\t\t781\t\t781\t\t\n"
			"1750000\t\t867\t\t867\t\t610\t\t610\t\t\n"
			"1833000\t\t36252\t\t36252\t\t8354\t\t8354\t\t\n"
			"1916000\t\t642\t\t642\t\t515\t\t515\t\t\n"
			"2000000\t\t790\t\t790\t\t600\t\t600\t\t\n"
			"2083000\t\t657\t\t657\t\t682\t\t682\t\t\n"
			"2166000\t\t785\t\t785\t\t509\t\t509\t\t\n"
			"2250000\t\t776\t\t776\t\t656\t\t656\t\t\n"
			"2333000\t\t243140\t\t243140\t\t170617\t\t170617\t\t\n",
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
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2333000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "500000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 179,
		.content = "2333000 2250000 2166000 2083000 2000000 1916000 1833000 1750000 1666000 1583000 1500000 1416000 1333000 1250000 1166000 1083000 1000000 916000 833000 750000 666000 583000 500000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "500000\n",
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
		.content = "500000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 286,
		.content =
			"2333000 243140\n"
			"2250000 776\n"
			"2166000 785\n"
			"2083000 657\n"
			"2000000 790\n"
			"1916000 642\n"
			"1833000 36252\n"
			"1750000 867\n"
			"1666000 703\n"
			"1583000 708\n"
			"1500000 800\n"
			"1416000 753\n"
			"1333000 936\n"
			"1250000 872\n"
			"1166000 875\n"
			"1083000 994\n"
			"1000000 1321\n"
			"916000 1745\n"
			"833000 1801\n"
			"750000 3550\n"
			"666000 3580\n"
			"583000 14933\n"
			"500000 4283973\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 6,
		.content = "18659\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/trans_table",
		.size = 4095,
		.content =
			"   From  :    To\n"
			"         :   2333000   2250000   2166000   2083000   2000000   1916000   1833000   1750000   1666000   1583000   1500000   1416000   1333000   1250000   1166000   1083000   1000000    916000    833000    750000    666000    583000    500000 \n"
			"  2333000:         0       147       147       107       128        92       102       101        85        80        92        71        78        66        57        53        55        57        62        60        63        91      1654 \n"
			"  2250000:       156         0         6         2         3         3         1         4         3         4         2         2         1         2         1         1         4         2         1         0         0         1        11 \n"
			"  2166000:       117        26         0         7        13         1         2         6         3         3         2         1         2         2         1         0         1         2         1         0         0         0         8 \n"
			"  2083000:        57        25        22         0         2         5         2         4         4         4         4         4         0         1         1         1         2         1         3         5         0         0         5 \n"
			"  2000000:        66        12        18        20         0         6         4         2         7         2         5         5         5         1         4         2         4         1         1         1         0         2        13 \n"
			"  1916000:        52         0         5        12        20         0         5         5         5         2         3         2         4         0         4         1         1         2         0         2         2         2        15 \n"
			"  1833000:        48         0         0         4        13        22         0        26        19        23        21        20        24        31        21        29        49        67        60       187        79       321      1716 \n"
			"  1750000:        67         0         0         0         2        15        27         0         5         5         9         8         4         4         5         5         2         4         1         2         1         1        19 \n"
			"  1666000:        50         0         0         0         0         0        27        18         0         7         4         4         6         6         3         5         6         2         1         2         1         4        10 \n"
			"  1583000:        49         0         0         0         0         0         4        20        13         0         6         7         8         6         5         6         5         6         2         4         0         1        18 \n"
			"  1500000:        57         0         0         0         0         0         6         0        12        16         0        12         7         4         6        11         9         4         9         1         3         2        17 \n"
			"  1416000:        51         0         0         0         0         0         9         0         0        14        21         0         6         9         6         4         8         3         7         5         3         4        18 \n"
			"  1333000:        43         0         0         0         0         0         8         0         0         0         7        20         0         3        12         7        13         9         8         9         2         4        35 \n"
			"  1250000:        41         0         0         0         0         0        14         0         0         0         0        12        21         0         6         6        12        12         5         7         6         4        25 \n"
			"  1166000:        40         0         0         0         0         0         8         0         0         0         0         0        13        29         0        12        11        11         4         7         6         3        22 \n"
			"  1083000:        28         0         0         0         0         0         6         0         0         0         0         0         0         7        26         0        21        14        11      ",
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
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2333000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "500000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/related_cpus",
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_frequencies",
		.size = 179,
		.content = "2333000 2250000 2166000 2083000 2000000 1916000 1833000 1750000 1666000 1583000 1500000 1416000 1333000 1250000 1166000 1083000 1000000 916000 833000 750000 666000 583000 500000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "500000\n",
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
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "500000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 286,
		.content =
			"2333000 243140\n"
			"2250000 776\n"
			"2166000 785\n"
			"2083000 657\n"
			"2000000 790\n"
			"1916000 642\n"
			"1833000 36252\n"
			"1750000 867\n"
			"1666000 703\n"
			"1583000 708\n"
			"1500000 800\n"
			"1416000 753\n"
			"1333000 936\n"
			"1250000 872\n"
			"1166000 875\n"
			"1083000 994\n"
			"1000000 1321\n"
			"916000 1745\n"
			"833000 1801\n"
			"750000 3550\n"
			"666000 3580\n"
			"583000 14933\n"
			"500000 4284229\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 6,
		.content = "18659\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/trans_table",
		.size = 4095,
		.content =
			"   From  :    To\n"
			"         :   2333000   2250000   2166000   2083000   2000000   1916000   1833000   1750000   1666000   1583000   1500000   1416000   1333000   1250000   1166000   1083000   1000000    916000    833000    750000    666000    583000    500000 \n"
			"  2333000:         0       147       147       107       128        92       102       101        85        80        92        71        78        66        57        53        55        57        62        60        63        91      1654 \n"
			"  2250000:       156         0         6         2         3         3         1         4         3         4         2         2         1         2         1         1         4         2         1         0         0         1        11 \n"
			"  2166000:       117        26         0         7        13         1         2         6         3         3         2         1         2         2         1         0         1         2         1         0         0         0         8 \n"
			"  2083000:        57        25        22         0         2         5         2         4         4         4         4         4         0         1         1         1         2         1         3         5         0         0         5 \n"
			"  2000000:        66        12        18        20         0         6         4         2         7         2         5         5         5         1         4         2         4         1         1         1         0         2        13 \n"
			"  1916000:        52         0         5        12        20         0         5         5         5         2         3         2         4         0         4         1         1         2         0         2         2         2        15 \n"
			"  1833000:        48         0         0         4        13        22         0        26        19        23        21        20        24        31        21        29        49        67        60       187        79       321      1716 \n"
			"  1750000:        67         0         0         0         2        15        27         0         5         5         9         8         4         4         5         5         2         4         1         2         1         1        19 \n"
			"  1666000:        50         0         0         0         0         0        27        18         0         7         4         4         6         6         3         5         6         2         1         2         1         4        10 \n"
			"  1583000:        49         0         0         0         0         0         4        20        13         0         6         7         8         6         5         6         5         6         2         4         0         1        18 \n"
			"  1500000:        57         0         0         0         0         0         6         0        12        16         0        12         7         4         6        11         9         4         9         1         3         2        17 \n"
			"  1416000:        51         0         0         0         0         0         9         0         0        14        21         0         6         9         6         4         8         3         7         5         3         4        18 \n"
			"  1333000:        43         0         0         0         0         0         8         0         0         0         7        20         0         3        12         7        13         9         8         9         2         4        35 \n"
			"  1250000:        41         0         0         0         0         0        14         0         0         0         0        12        21         0         6         6        12        12         5         7         6         4        25 \n"
			"  1166000:        40         0         0         0         0         0         8         0         0         0         0         0        13        29         0        12        11        11         4         7         6         3        22 \n"
			"  1083000:        28         0         0         0         0         0         6         0         0         0         0         0         0         7        26         0        21        14        11      ",
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
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2333000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "500000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/related_cpus",
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies",
		.size = 179,
		.content = "2333000 2250000 2166000 2083000 2000000 1916000 1833000 1750000 1666000 1583000 1500000 1416000 1333000 1250000 1166000 1083000 1000000 916000 833000 750000 666000 583000 500000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "500000\n",
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
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "500000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 281,
		.content =
			"2333000 170617\n"
			"2250000 656\n"
			"2166000 509\n"
			"2083000 682\n"
			"2000000 600\n"
			"1916000 515\n"
			"1833000 8354\n"
			"1750000 610\n"
			"1666000 781\n"
			"1583000 703\n"
			"1500000 758\n"
			"1416000 716\n"
			"1333000 637\n"
			"1250000 724\n"
			"1166000 593\n"
			"1083000 730\n"
			"1000000 630\n"
			"916000 849\n"
			"833000 912\n"
			"750000 1120\n"
			"666000 1684\n"
			"583000 3589\n"
			"500000 4405005\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 5,
		.content = "7130\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/trans_table",
		.size = 4095,
		.content =
			"   From  :    To\n"
			"         :   2333000   2250000   2166000   2083000   2000000   1916000   1833000   1750000   1666000   1583000   1500000   1416000   1333000   1250000   1166000   1083000   1000000    916000    833000    750000    666000    583000    500000 \n"
			"  2333000:         0       104        85        92        74        68        78        67        57        52        61        54        47        51        31        45        31        37        36        37        36        51       530 \n"
			"  2250000:       114         0         6         7         3         6         2         2         4         2         0         0         5         0         1         1         0         1         1         1         0         0         5 \n"
			"  2166000:        59        22         0         4         3         5         5         1         4         3         2         2         1         1         1         1         1         1         2         0         1         0         5 \n"
			"  2083000:        62        21        18         0         5         4         6         4         4         3         3         1         1         3         2         2         1         4         0         1         1         1         4 \n"
			"  2000000:        21        14        13        25         0         7         8         3         5         1         6         2         3         2         1         1         3         2         3         0         0         0         6 \n"
			"  1916000:        33         0         2        20        20         0         6         2         3         4         2         3         2         3         2         1         2         0         0         1         2         0        15 \n"
			"  1833000:        46         0         0         3        18        23         0        18        24        22        18        13        17        12         9        13        12        12         4         7         3         9       166 \n"
			"  1750000:        35         0         0         0         3         9        37         0         6         4         2         4         4         4         4         3         1         1         1         3         0         1        15 \n"
			"  1666000:        29         0         0         0         0         0        29        20         0        11         8         3         5        12         2         2         2         5         2         0         2         0        16 \n"
			"  1583000:        31         0         0         0         0         0         5        19        18         0         7         9         5         3         6         5         3         5         4         1         2         0        15 \n"
			"  1500000:        30         0         0         0         0         0         0         0        22        20         0        14         6         6         7         6         5         6         1         1         2         6        17 \n"
			"  1416000:        42         0         0         0         0         0         3         0         0        16        23         0         5        14         2         5         1         6         7         2         3         1        19 \n"
			"  1333000:        34         0         0         0         0         0         6         1         0         0        17        30         0         4         1         6         8         4         2         7         4         2        15 \n"
			"  1250000:        27         0         0         0         0         0         4         0         0         0         0        14        27         0        11         7         6         5         4         4         5         4        20 \n"
			"  1166000:        19         0         0         0         0         0         4         0         0         0         0         0        13        20         0         5         4         5         9         4         5         1        28 \n"
			"  1083000:        20         0         0         0         0         0         4         0         0         0         0         0         0         3        29         0        11         7         6      ",
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
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2333000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "500000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "100000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/related_cpus",
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 179,
		.content = "2333000 2250000 2166000 2083000 2000000 1916000 1833000 1750000 1666000 1583000 1500000 1416000 1333000 1250000 1166000 1083000 1000000 916000 833000 750000 666000 583000 500000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 44,
		.content = "ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "500000\n",
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
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "500000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 281,
		.content =
			"2333000 170617\n"
			"2250000 656\n"
			"2166000 509\n"
			"2083000 682\n"
			"2000000 600\n"
			"1916000 515\n"
			"1833000 8354\n"
			"1750000 610\n"
			"1666000 781\n"
			"1583000 703\n"
			"1500000 758\n"
			"1416000 716\n"
			"1333000 637\n"
			"1250000 724\n"
			"1166000 593\n"
			"1083000 730\n"
			"1000000 630\n"
			"916000 849\n"
			"833000 912\n"
			"750000 1120\n"
			"666000 1684\n"
			"583000 3589\n"
			"500000 4405273\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 5,
		.content = "7130\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/trans_table",
		.size = 4095,
		.content =
			"   From  :    To\n"
			"         :   2333000   2250000   2166000   2083000   2000000   1916000   1833000   1750000   1666000   1583000   1500000   1416000   1333000   1250000   1166000   1083000   1000000    916000    833000    750000    666000    583000    500000 \n"
			"  2333000:         0       104        85        92        74        68        78        67        57        52        61        54        47        51        31        45        31        37        36        37        36        51       530 \n"
			"  2250000:       114         0         6         7         3         6         2         2         4         2         0         0         5         0         1         1         0         1         1         1         0         0         5 \n"
			"  2166000:        59        22         0         4         3         5         5         1         4         3         2         2         1         1         1         1         1         1         2         0         1         0         5 \n"
			"  2083000:        62        21        18         0         5         4         6         4         4         3         3         1         1         3         2         2         1         4         0         1         1         1         4 \n"
			"  2000000:        21        14        13        25         0         7         8         3         5         1         6         2         3         2         1         1         3         2         3         0         0         0         6 \n"
			"  1916000:        33         0         2        20        20         0         6         2         3         4         2         3         2         3         2         1         2         0         0         1         2         0        15 \n"
			"  1833000:        46         0         0         3        18        23         0        18        24        22        18        13        17        12         9        13        12        12         4         7         3         9       166 \n"
			"  1750000:        35         0         0         0         3         9        37         0         6         4         2         4         4         4         4         3         1         1         1         3         0         1        15 \n"
			"  1666000:        29         0         0         0         0         0        29        20         0        11         8         3         5        12         2         2         2         5         2         0         2         0        16 \n"
			"  1583000:        31         0         0         0         0         0         5        19        18         0         7         9         5         3         6         5         3         5         4         1         2         0        15 \n"
			"  1500000:        30         0         0         0         0         0         0         0        22        20         0        14         6         6         7         6         5         6         1         1         2         6        17 \n"
			"  1416000:        42         0         0         0         0         0         3         0         0        16        23         0         5        14         2         5         1         6         7         2         3         1        19 \n"
			"  1333000:        34         0         0         0         0         0         6         1         0         0        17        30         0         4         1         6         8         4         2         7         4         2        15 \n"
			"  1250000:        27         0         0         0         0         0         4         0         0         0         0        14        27         0        11         7         6         5         4         4         5         4        20 \n"
			"  1166000:        19         0         0         0         0         0         4         0         0         0         0         0        13        20         0         5         4         5         9         4         5         1        28 \n"
			"  1083000:        20         0         0         0         0         0         4         0         0         0         0         0         0         3        29         0        11         7         6      ",
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
		.key = "AudioComms.PFW.ConfName",
		.value = "ParameterFrameworkConfiguration.xml",
	},
	{
		.key = "AudioComms.PFW.ConfPath",
		.value = "/system/etc/catalog/V1_DSDA/audiocomms_config/parameter-framework/",
	},
	{
		.key = "AudioComms.RoutePFW.ConfName",
		.value = "ParameterFrameworkConfigurationRoute.xml",
	},
	{
		.key = "AudioComms.RoutePFW.ConfPath",
		.value = "/system/etc/catalog/V1_DSDA/audiocomms_config/parameter-framework/",
	},
	{
		.key = "AudioComms.Vibrator.ConfName",
		.value = "ParameterFrameworkConfigurationVibrator.xml",
	},
	{
		.key = "AudioComms.Vibrator.ConfPath",
		.value = "/system/etc/catalog/V1_DSDA/audiocomms_config/parameter-framework/",
	},
	{
		.key = "AudioComms.vtsv.routed",
		.value = "false",
	},
	{
		.key = "ap.interface",
		.value = "wlan0",
	},
	{
		.key = "atd.voucher.exist",
		.value = "1",
	},
	{
		.key = "atd.voucher.intact",
		.value = "1",
	},
	{
		.key = "audio.aware.card",
		.value = "rt5647audio",
	},
	{
		.key = "audio.device.name",
		.value = "rt5647audio",
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
		.key = "audio.vtsv.card",
		.value = "rt5647audio",
	},
	{
		.key = "audio.vtsv.device",
		.value = "5",
	},
	{
		.key = "audio.vtsv.dsp_log",
		.value = "0",
	},
	{
		.key = "bt.hfp.WideBandSpeechEnabled",
		.value = "true",
	},
	{
		.key = "bt.version.driver",
		.value = "V10.00.02",
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
		.key = "dalvik.vm.isa.x86.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.x86.variant",
		.value = "x86",
	},
	{
		.key = "dalvik.vm.stack-trace-file",
		.value = "/data/anr/traces.txt",
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
		.key = "debug.rs.default-CPU-driver",
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
		.key = "gps.version.driver",
		.value = "66.19.20.275658",
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
		.key = "gsm.sim.gid1",
		.value = ",",
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
		.key = "gsm.sim.pin1_count",
		.value = "",
	},
	{
		.key = "gsm.sim.pin1_count.1",
		.value = "",
	},
	{
		.key = "gsm.sim.pin2_count",
		.value = "",
	},
	{
		.key = "gsm.sim.pin2_count.1",
		.value = "",
	},
	{
		.key = "gsm.sim.puk1_count",
		.value = "",
	},
	{
		.key = "gsm.sim.puk1_count.1",
		.value = "",
	},
	{
		.key = "gsm.sim.puk2_count",
		.value = "",
	},
	{
		.key = "gsm.sim.puk2_count.1",
		.value = "",
	},
	{
		.key = "gsm.sim.spn",
		.value = ",",
	},
	{
		.key = "gsm.sim.state",
		.value = "ABSENT,ABSENT",
	},
	{
		.key = "gsm.sim1.present",
		.value = "0",
	},
	{
		.key = "gsm.sim2.present",
		.value = "0",
	},
	{
		.key = "gsm.version.baseband",
		.value = "1603_5.0.68.10_0224,1546_7.0.30.0_0427",
	},
	{
		.key = "gsm.version.baseband1",
		.value = "1603_5.0.68.10_0224",
	},
	{
		.key = "gsm.version.baseband2",
		.value = "1546_7.0.30.0_0427",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "Intrinsyc Rapid-RIL M6.59 for Android 4.2 (Build September 17/2013)",
	},
	{
		.key = "hwc.video.extmode.enable",
		.value = "0",
	},
	{
		.key = "init.svc.VerifyVouchers",
		.value = "stopped",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.bcu_cpufreqrel",
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
		.key = "init.svc.brcm_config_init",
		.value = "stopped",
	},
	{
		.key = "init.svc.cdrom",
		.value = "stopped",
	},
	{
		.key = "init.svc.config-zram",
		.value = "stopped",
	},
	{
		.key = "init.svc.config_init",
		.value = "stopped",
	},
	{
		.key = "init.svc.crashlog_config",
		.value = "stopped",
	},
	{
		.key = "init.svc.csts",
		.value = "stopped",
	},
	{
		.key = "init.svc.debuggerd",
		.value = "running",
	},
	{
		.key = "init.svc.dpst",
		.value = "running",
	},
	{
		.key = "init.svc.drm",
		.value = "running",
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
		.key = "init.svc.gatekeeperd",
		.value = "running",
	},
	{
		.key = "init.svc.gpsd",
		.value = "running",
	},
	{
		.key = "init.svc.healthd",
		.value = "running",
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
		.key = "init.svc.intel_prop",
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
		.key = "init.svc.mmgr",
		.value = "running",
	},
	{
		.key = "init.svc.mmgr2",
		.value = "running",
	},
	{
		.key = "init.svc.modem-c_main-sh",
		.value = "stopped",
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
		.key = "init.svc.nvmmanager2",
		.value = "running",
	},
	{
		.key = "init.svc.p2p_supplicant",
		.value = "running",
	},
	{
		.key = "init.svc.pclinkd",
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
		.value = "running",
	},
	{
		.key = "init.svc.ril-daemon1",
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
		.key = "init.svc.surfaceflinger",
		.value = "running",
	},
	{
		.key = "init.svc.ueventd",
		.value = "running",
	},
	{
		.key = "init.svc.umount_apd",
		.value = "stopped",
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
		.key = "intel.dpst.alg",
		.value = "3",
	},
	{
		.key = "intel.dpst.enable",
		.value = "1",
	},
	{
		.key = "intel.dpst.userlevel",
		.value = "3",
	},
	{
		.key = "keyguard.no_require_sim",
		.value = "true",
	},
	{
		.key = "logtool.class",
		.value = "com.asus.logtool.LogService",
	},
	{
		.key = "logtool.package",
		.value = "com.asus.internal.fdctoolstate",
	},
	{
		.key = "lpa.deepbuffer.enable",
		.value = "1",
	},
	{
		.key = "media.camera.facing",
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
		.key = "net.dns3",
		.value = "",
	},
	{
		.key = "net.dns4",
		.value = "",
	},
	{
		.key = "net.hostname",
		.value = "android-60227a9441d8bd8f",
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
		.key = "persist.asus.cover.dbwake",
		.value = "23",
	},
	{
		.key = "persist.asus.coverenabled",
		.value = "1",
	},
	{
		.key = "persist.asus.dclick",
		.value = "1",
	},
	{
		.key = "persist.asus.enduser.dialog",
		.value = "0",
	},
	{
		.key = "persist.asus.flipcovermode",
		.value = "0",
	},
	{
		.key = "persist.asus.gesture.type",
		.value = "1111111",
	},
	{
		.key = "persist.asus.glove",
		.value = "0",
	},
	{
		.key = "persist.asus.inoutdoor",
		.value = "0",
	},
	{
		.key = "persist.asus.power.mode",
		.value = "normal",
	},
	{
		.key = "persist.asuslog.dump.date",
		.value = "2017_0921_070843",
	},
	{
		.key = "persist.audio.fmroute.speaker",
		.value = "0",
	},
	{
		.key = "persist.dual_sim",
		.value = "none",
	},
	{
		.key = "persist.fwlog.enable",
		.value = "1",
	},
	{
		.key = "persist.ims_support",
		.value = "0",
	},
	{
		.key = "persist.radio.device.imei",
		.value = "357798077141487",
	},
	{
		.key = "persist.radio.device.imei2",
		.value = "357798077141495",
	},
	{
		.key = "persist.radio.multisim.config",
		.value = "dsda",
	},
	{
		.key = "persist.radio.operator_code",
		.value = "",
	},
	{
		.key = "persist.radio.operator_code1",
		.value = "",
	},
	{
		.key = "persist.service.cdrom.enable",
		.value = "1",
	},
	{
		.key = "persist.service.cwsmgr.coex",
		.value = "1",
	},
	{
		.key = "persist.service.cwsmgr.nortcoex",
		.value = "0",
	},
	{
		.key = "persist.service.thermal",
		.value = "1",
	},
	{
		.key = "persist.stm.dvc.mid_disabled",
		.value = "punit",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.enableAPD",
		.value = "0",
	},
	{
		.key = "persist.sys.highercost",
		.value = "0",
	},
	{
		.key = "persist.sys.locale",
		.value = "en-US",
	},
	{
		.key = "persist.sys.media.avsync",
		.value = "1",
	},
	{
		.key = "persist.sys.mmgr1.blob_hash",
		.value = "131b9ce6869fadee3c6462839a96080f",
	},
	{
		.key = "persist.sys.mmgr1.config_hash",
		.value = "022cef5498b7acd3efe3cb5303d28bf7",
	},
	{
		.key = "persist.sys.mmgr1.reboot",
		.value = "0",
	},
	{
		.key = "persist.sys.mmgr2.blob_hash",
		.value = "131b9ce6869fadee3c6462839a96080f",
	},
	{
		.key = "persist.sys.mmgr2.config_hash",
		.value = "348173b40e6b47091456524b19a52fe6",
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
		.value = "mtp,mass_storage,adb",
	},
	{
		.key = "persist.sys.webview.vmsize",
		.value = "141853048",
	},
	{
		.key = "persist.tel.hot_swap.support",
		.value = "true",
	},
	{
		.key = "persist.thermal.debug.xml",
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
		.key = "persist.thermal.turbo.dynamic",
		.value = "1",
	},
	{
		.key = "qemu.hw.mainkeys",
		.value = "1",
	},
	{
		.key = "ril.ecclist",
		.value = "112,911,000,08,110,118,119,999",
	},
	{
		.key = "ril.ecclist1",
		.value = "112,911,000,08,110,118,119,999",
	},
	{
		.key = "ril.lteca.mode",
		.value = "0",
	},
	{
		.key = "rild.libpath",
		.value = "/system/lib/librapid-ril-core.so",
	},
	{
		.key = "ro.adb.secure",
		.value = "1",
	},
	{
		.key = "ro.adbon.oneshot",
		.value = "0",
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
		.key = "ro.asus.network.types",
		.value = "2",
	},
	{
		.key = "ro.asus.persistent",
		.value = "/factory/AntiTheft.file",
	},
	{
		.key = "ro.asus.phone.ipcall",
		.value = "0",
	},
	{
		.key = "ro.asus.phone.sipcall",
		.value = "1",
	},
	{
		.key = "ro.asus.ui",
		.value = "1.0",
	},
	{
		.key = "ro.audio.vibra.ring.vol.idx",
		.value = "5",
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
		.value = "moorefield",
	},
	{
		.key = "ro.boot.bootreason",
		.value = "reboot",
	},
	{
		.key = "ro.boot.hardware",
		.value = "mofd_v1",
	},
	{
		.key = "ro.boot.min.cap",
		.value = "0",
	},
	{
		.key = "ro.boot.mode",
		.value = "main",
	},
	{
		.key = "ro.boot.serialno",
		.value = "MedfieldCEAC8E0B",
	},
	{
		.key = "ro.boot.spid",
		.value = "0000:0000:0000:0008:0000:0000",
	},
	{
		.key = "ro.boot.verifiedbootstate",
		.value = "green",
	},
	{
		.key = "ro.boot.wakesrc",
		.value = "05",
	},
	{
		.key = "ro.bootimage.build.date",
		.value = "xE4xBAx94 6xE6x9Cx88 23 00:15:28 CST 2017",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1498148128",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "asus/WW_Z00A/Z00A:6.0.1/MMB29P/4.21.40.352_20170623_7598_user:user/release-keys",
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
		.key = "ro.bt.bdaddr_path",
		.value = "/config/bt/bd_addr.conf",
	},
	{
		.key = "ro.build.app.version",
		.value = "060020736_201603210001",
	},
	{
		.key = "ro.build.asus.sku",
		.value = "WW",
	},
	{
		.key = "ro.build.asus.version",
		.value = "4.21.40.352",
	},
	{
		.key = "ro.build.characteristics",
		.value = "nosdcard",
	},
	{
		.key = "ro.build.csc.version",
		.value = "WW_ZE551ML_4.21.40.352_20170623",
	},
	{
		.key = "ro.build.date",
		.value = "xE4xBAx94  6xE6x9Cx88 23 00:08:45 CST 2017",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1498147725",
	},
	{
		.key = "ro.build.description",
		.value = "asusmofd_fhd-user 6.0.1 MMB29P 4.21.40.352_20170623_7598_user release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "MMB29P.WW-ASUS_Z00A-4.21.40.352_20170623_7598_user",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "asus/WW_Z00A/Z00A:6.0.1/MMB29P/4.21.40.352_20170623_7598_user:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "asusmofd_fhd-user",
	},
	{
		.key = "ro.build.host",
		.value = "fdc-01-jenkins",
	},
	{
		.key = "ro.build.id",
		.value = "MMB29P",
	},
	{
		.key = "ro.build.product",
		.value = "mofd_v1",
	},
	{
		.key = "ro.build.servaddr",
		.value = "1",
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
		.key = "ro.build.version.houdini",
		.value = "6.1.1a",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "WW_Z00A-WW_4.21.40.352_20170623_7598_user_rel-user-20170623",
	},
	{
		.key = "ro.build.version.preview_sdk",
		.value = "0",
	},
	{
		.key = "ro.build.version.release",
		.value = "6.0.1",
	},
	{
		.key = "ro.build.version.sdk",
		.value = "23",
	},
	{
		.key = "ro.build.version.security_patch",
		.value = "2017-05-01",
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
		.key = "ro.com.google.clientidbase",
		.value = "android-asus",
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
		.key = "ro.com.google.gmsversion",
		.value = "6.0_r11",
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
		.value = "6G",
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
		.key = "ro.config.packingcode",
		.value = "",
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
		.key = "ro.cpufreq",
		.value = "2.3GHz",
	},
	{
		.key = "ro.cpufreq.limit",
		.value = "1",
	},
	{
		.key = "ro.crypto.state",
		.value = "unencrypted",
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
		.key = "ro.expect.recovery_id",
		.value = "0x9c0e1ee4a82056edf9114ab36dc033fd65faac41000000000000000000000000",
	},
	{
		.key = "ro.fmrx.sound.forced",
		.value = "1",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/by-name/persistent",
	},
	{
		.key = "ro.gnss.sv.status",
		.value = "true",
	},
	{
		.key = "ro.hardware",
		.value = "mofd_v1",
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
		.key = "ro.ime.lowmemory",
		.value = "false",
	},
	{
		.key = "ro.intel.corp.email",
		.value = "1",
	},
	{
		.key = "ro.isn",
		.value = "QTCYZ11BT65501288",
	},
	{
		.key = "ro.memsize",
		.value = "4G",
	},
	{
		.key = "ro.nfc.clk",
		.value = "pll",
	},
	{
		.key = "ro.nfc.conf",
		.value = "mofd-ffd2-a",
	},
	{
		.key = "ro.nfc.nfcc",
		.value = "bcm_2079x",
	},
	{
		.key = "ro.nfc.se.ese",
		.value = "false",
	},
	{
		.key = "ro.nfc.se.uicc",
		.value = "false",
	},
	{
		.key = "ro.nfc.use_csm",
		.value = "false",
	},
	{
		.key = "ro.opengles.version",
		.value = "196609",
	},
	{
		.key = "ro.product.board",
		.value = "moorefield",
	},
	{
		.key = "ro.product.brand",
		.value = "asus",
	},
	{
		.key = "ro.product.carrier",
		.value = "US-ASUS_Z00AD-WW_Z00A",
	},
	{
		.key = "ro.product.cpu.abi",
		.value = "x86",
	},
	{
		.key = "ro.product.cpu.abilist",
		.value = "x86,armeabi-v7a,armeabi",
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
		.key = "ro.product.device",
		.value = "Z00A",
	},
	{
		.key = "ro.product.first_api_level",
		.value = "21",
	},
	{
		.key = "ro.product.locale",
		.value = "en-US",
	},
	{
		.key = "ro.product.manufacturer",
		.value = "asus",
	},
	{
		.key = "ro.product.model",
		.value = "ASUS_Z00AD",
	},
	{
		.key = "ro.product.name",
		.value = "WW_Z00A",
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
		.key = "ro.ril.ecclist",
		.value = "112,911",
	},
	{
		.key = "ro.ril.status.polling.enable",
		.value = "0",
	},
	{
		.key = "ro.runtime.firstboot",
		.value = "1505978091191",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "G5AZFG01L789BJH",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "OPTIONAL",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "480",
	},
	{
		.key = "ro.spid.gps.FrqPlan",
		.value = "FRQ_PLAN_26MHZ_2PPM",
	},
	{
		.key = "ro.spid.gps.RfType",
		.value = "GL_RF_47531_BRCM",
	},
	{
		.key = "ro.spid.gps.tty",
		.value = "ttyMFD2",
	},
	{
		.key = "ro.swconf.info",
		.value = "V1_DSDA_ZE550ML_US",
	},
	{
		.key = "ro.telephony.default_network",
		.value = "9",
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
		.key = "security.perf_harden",
		.value = "1",
	},
	{
		.key = "selinux.reload_policy",
		.value = "1",
	},
	{
		.key = "service.amtl1.cfg",
		.value = "moorefield_XMM_7260",
	},
	{
		.key = "service.amtl2.cfg",
		.value = "moorefield_XMM_2230",
	},
	{
		.key = "service.bootanim.exit",
		.value = "1",
	},
	{
		.key = "sys.asuslog.date.temp",
		.value = "2017_0921_070843",
	},
	{
		.key = "sys.asuslog.fdate.temp",
		.value = "2017_0921_070843",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
	},
	{
		.key = "sys.chaabi.version",
		.value = "006E.0801",
	},
	{
		.key = "sys.charger.connected",
		.value = "1",
	},
	{
		.key = "sys.config.maxxaudio",
		.value = "2",
	},
	{
		.key = "sys.foregroundapp",
		.value = "com.asus.launcher",
	},
	{
		.key = "sys.ia32.version",
		.value = "0002.0016",
	},
	{
		.key = "sys.ifwi.version",
		.value = "0094.0183",
	},
	{
		.key = "sys.kernel.version",
		.value = "3.10.72-x86_64_moor-gb6d574d",
	},
	{
		.key = "sys.mia.version",
		.value = "00B0.3230",
	},
	{
		.key = "sys.nfc.brcm.cfg",
		.value = "/etc/ze551ml_gold_libnfc-brcm.conf",
	},
	{
		.key = "sys.nfc.brcm.chip_cfg",
		.value = "/etc/ze551ml_gold_libnfc-brcm-20795a20.conf",
	},
	{
		.key = "sys.nfc.loc",
		.value = "gold",
	},
	{
		.key = "sys.nfc.project_id_str",
		.value = "ze551ml",
	},
	{
		.key = "sys.oem_unlock_allowed",
		.value = "0",
	},
	{
		.key = "sys.pmic.nvm.version",
		.value = "1C",
	},
	{
		.key = "sys.punit.version",
		.value = "0000.0032",
	},
	{
		.key = "sys.scu.version",
		.value = "00B0.0035",
	},
	{
		.key = "sys.scubs.version",
		.value = "00B0.0001",
	},
	{
		.key = "sys.settings_global_version",
		.value = "1",
	},
	{
		.key = "sys.settings_secure_version",
		.value = "1",
	},
	{
		.key = "sys.settings_system_version",
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
		.key = "sys.ucode.version",
		.value = "0000.0038",
	},
	{
		.key = "sys.usb.config",
		.value = "mtp,adb,mass_storage",
	},
	{
		.key = "sys.usb.configfs",
		.value = "0",
	},
	{
		.key = "sys.usb.ffs.ready",
		.value = "1",
	},
	{
		.key = "sys.usb.modemevt",
		.value = "0",
	},
	{
		.key = "sys.usb.state",
		.value = "mtp,adb,mass_storage",
	},
	{
		.key = "sys.usb.vbus",
		.value = "normal",
	},
	{
		.key = "sys.valhooks.version",
		.value = "005E.0031",
	},
	{
		.key = "sys.watchdog.previous.counter",
		.value = "0",
	},
	{
		.key = "system_init.startsurfaceflinger",
		.value = "0",
	},
	{
		.key = "video.playback.slow-motion",
		.value = "1",
	},
	{
		.key = "vold.has_adoptable",
		.value = "0",
	},
	{
		.key = "vold.microsd.fstype",
		.value = "none",
	},
	{
		.key = "vold.post_fs_data_done",
		.value = "1",
	},
	{
		.key = "vppsettings.frc",
		.value = "1",
	},
	{
		.key = "widi.abr.enable",
		.value = "true",
	},
	{
		.key = "widi.audio.module",
		.value = "submix",
	},
	{
		.key = "widi.hdcp.enable",
		.value = "auto",
	},
	{
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "wifi.version.driver",
		.value = "6.37.45.11",
	},
	{
		.key = "wlan.driver.status",
		.value = "ok",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
