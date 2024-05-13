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
		.ebx = 0x02040800,
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
		.eax = 0x30363532,
		.ebx = 0x20402020,
		.ecx = 0x30362E31,
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
		.size = 3242,
		.content =
			"processor\t: 0\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 53\n"
			"model name\t: Intel(R) Atom(TM) CPU Z2560  @ 1.60GHz\n"
			"stepping\t: 1\n"
			"microcode\t: 0x110\n"
			"cpu MHz\t\t: 1600.000\n"
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
			"bogomips\t: 3194.88\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 32 bits physical, 32 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 1\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 53\n"
			"model name\t: Intel(R) Atom(TM) CPU Z2560  @ 1.60GHz\n"
			"stepping\t: 1\n"
			"microcode\t: 0x110\n"
			"cpu MHz\t\t: 1600.000\n"
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
			"bogomips\t: 3194.88\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 32 bits physical, 32 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 2\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 53\n"
			"model name\t: Intel(R) Atom(TM) CPU Z2560  @ 1.60GHz\n"
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
			"bogomips\t: 3194.88\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 32 bits physical, 32 bits virtual\n"
			"power management:\n"
			"\n"
			"processor\t: 3\n"
			"vendor_id\t: GenuineIntel\n"
			"cpu family\t: 6\n"
			"model\t\t: 53\n"
			"model name\t: Intel(R) Atom(TM) CPU Z2560  @ 1.60GHz\n"
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
			"bogomips\t: 3194.88\n"
			"clflush size\t: 64\n"
			"cache_alignment\t: 64\n"
			"address sizes\t: 32 bits physical, 32 bits virtual\n"
			"power management:\n"
			"\n",
	},
	{
		.path = "/system/build.prop",
		.size = 5245,
		.content =
			"\n"
			"# begin build properties\n"
			"# autogenerated by buildinfo.sh\n"
			"ro.build.id=LRX22C\n"
			"ro.build.display.id=ATT_12.16.10.95\n"
			"ro.build.version.incremental=ATT_Phone-12.16.10.95-20160223\n"
			"ro.build.version.sdk=21\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.all_codenames=REL\n"
			"ro.build.version.release=5.0.1\n"
			"ro.build.date=Tue Feb 23 17:42:49 CST 2016\n"
			"ro.build.date.utc=1456220569\n"
			"ro.build.type=user\n"
			"ro.build.user=android\n"
			"ro.build.host=mcrd1-40\n"
			"ro.build.tags=release-keys\n"
			"ro.product.model=ASUS ZenFone 2E\n"
			"ro.product.brand=asus\n"
			"ro.product.name=ATT_Phone\n"
			"ro.product.device=ASUS_Z00D\n"
			"ro.product.board=clovertrail\n"
			"# ro.product.cpu.abi and ro.product.cpu.abi2 are obsolete,\n"
			"# use ro.product.cpu.abilist instead.\n"
			"ro.product.cpu.abi=x86\n"
			"ro.product.cpu.abilist=x86,armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist32=x86,armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist64=\n"
			"ro.product.manufacturer=asus\n"
			"ro.product.locale.language=en\n"
			"ro.product.locale.region=US\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=clovertrail\n"
			"# ro.build.product is obsolete; use ro.product.device\n"
			"ro.build.product=ze500cl\n"
			"# Do not try to parse description, fingerprint, or thumbprint\n"
			"ro.build.description=ATT_Phone-user 5.0.1 LRX22C ATT_Phone-12.16.10.95-20160223 release-keys\n"
			"ro.build.fingerprint=asus/ATT_Phone/ASUS_Z00D:5.0.1/LRX22C/ATT_Phone-12.16.10.95-20160223:user/release-keys\n"
			"ro.build.version.base_os=asus/ATT_Phone/ASUS_Z00D:5.0.1/LRX22C/ATT_Phone-12.16.10.95-20160223:user/release-keys\n"
			"ro.build.version.security_patch=2015-09-11\n"
			"ro.build.characteristics=default\n"
			"ro.build.csc.version=ATT_ZE500CL-12.16.10.95-20160223\n"
			"# end build properties\n"
			"#\n"
			"# from device/intel/clovertrail/redhookbay/system.prop\n"
			"#\n"
			"#\n"
			"# System.prop for Clovertrail\n"
			"#\n"
			"\n"
			"# [ASUS BSP] Gavin_Chang - Add MicroSD path\n"
			"ro.epad.mount_point.microsd = /storage/MicroSD\n"
			"ro.epad.mount_point.sdcard1 = /storage/MicroSD\n"
			"# [ASUS BSP] Gavin_Chang - Add MicroSD path\n"
			"\n"
			"# [ASUS BSP] Gavin_Chang - Add APP2SD\n"
			"ro.bsp.app2sd=true\n"
			"# [ASUS BSP] Gavin_Chang - Add APP2SD\n"
			"\n"
			"# [ASUS BSP] Cheryl Chen - PF450CL006S - Disable Power Off Dialog for Power On/Off Stress Test\n"
			"persist.sys.test.poweronoff = 0\n"
			"# [ASUS BSP] Cheryl Chen - PF450CL006E\n"
			"persist.asus.mupload.enable = 0\n"
			"\n"
			"# [ASUS BSP] Jacob Kung - ZE500CL support golve mode property\n"
			"persist.asus.glove = 0\n"
			"\n"
			"# [ASUS BSP] Jacob Kung - ZE500CL support double tap mode property\n"
			"persist.asus.dclick = 0\n"
			"\n"
			"# [ASUS BSP] Jacob Kung - ZE500CL support gesture mode property\n"
			"persist.asus.gesture.type = 0000000\n"
			"\n"
			"# [ASUS BSP] Gary3_Chen - need miracast video clone mode to be default\n"
			"widi.media.extmode.enable = false\n"
			"# [ASUS BSP] Gary3_Chen - need miracast video clone mode to be default\n"
			"\n"
			"# [ASUS BSP] Joy_Lin For ASUS Power task+++\n"
			"asus.powertask = 0\n"
			"# [ASUS BSP] Joy_Lin For ASUS Power task---\n"
			"\n"
			"#\n"
			"# ASUS ze500cl project specific properties\n"
			"#\n"
			"\n"
			"\n"
			"#\n"
			"# ADDITIONAL_BUILD_PROPERTIES\n"
			"#\n"
			"ro.build.asus.sku=ATT\n"
			"ro.streaming.video.drs=true\n"
			"ro.build.app.version=044030426_201501050106\n"
			"ro.asus.ui=1.0\n"
			"ro.dalvik.vm.isa.arm=x86\n"
			"ro.enable.native.bridge.exec=1\n"
			"ro.config.specific=att\n"
			"ro.spid.gps.pmm=disabled\n"
			"ro.spid.gps.tty=ttyMFD3\n"
			"ro.spid.gps.FrqPlan=FRQ_PLAN_26MHZ_2PPM\n"
			"ro.spid.gps.RfType=GL_RF_4752_BRCM_EXT_LNA\n"
			"persist.tel.lteOnGsmDevice=true\n"
			"ro.telephony.default_network=9\n"
			"telephony.lteOnCdmaDevice=0\n"
			"audiocomms.vp.fw_name=vpimg_es325.bin\n"
			"ro.com.android.dataroaming=false\n"
			"keyguard.no_require_sim=true\n"
			"ro.com.android.dateformat=MM-dd-yyyy\n"
			"ro.carrier=unknown\n"
			"dalvik.vm.heapstartsize=8m\n"
			"dalvik.vm.heapgrowthlimit=128m\n"
			"dalvik.vm.heapsize=174m\n"
			"dalvik.vm.heaptargetutilization=0.75\n"
			"dalvik.vm.heapminfree=512k\n"
			"dalvik.vm.heapmaxfree=8m\n"
			"dalvik.jit.code_cache_size=1048576\n"
			"ro.hwui.texture_cache_size=24.0f\n"
			"ro.hwui.text_large_cache_width=2048\n"
			"ro.hwui.text_large_cache_height=512\n"
			"persist.tel.hot_swap.support=true\n"
			"drm.service.enabled=true\n"
			"ro.frp.pst=/dev/block/mmcblk0p2\n"
			"ro.blankphone_id=1\n"
			"ro.com.google.clientidbase=android-asus\n"
			"ro.contact.simtype=1\n"
			"ro.camera.sound.forced=0\n"
			"ro.config.ringtone=Festival.ogg\n"
			"ro.config.notification_sound=NewMessage.ogg\n"
			"ro.config.newmail_sound=NewMail.ogg\n"
			"ro.config.sentmail_sound=SentMail.ogg\n"
			"ro.config.calendaralert_sound=CalendarEvent.ogg\n"
			"ro.config.alarm_alert=BusyBugs.ogg\n"
			"ro.additionalbutton.operation=0\n"
			"ro.asus.browser.uap=ASUS-ZE500CL\n"
			"ro.setupwizard.mode=OPTIONAL\n"
			"ro.com.google.gmsversion=5.0_r2\n"
			"ro.com.google.clientidbase.ms=android-att-us\n"
			"ro.com.google.clientidbase.am=android-att-us\n"
			"ro.com.google.clientidbase.gmm=android-asus\n"
			"ro.com.google.clientidbase.yt=android-asus\n"
			"persist.sys.dalvik.vm.lib.2=libart.so\n"
			"ro.ril.status.polling.enable=0\n"
			"persist.tcs.hw_filename=/etc/telephony/XMM7160_CONF_5.xml\n"
			"rs.gpu.renderscript=0\n"
			"rs.gpu.filterscript=0\n"
			"rs.gpu.rsIntrinsic=0\n"
			"panel.physicalWidthmm=62\n"
			"panel.physicalHeightmm=111\n"
			"ro.opengles.version=131072\n"
			"gsm.net.interface=rmnet0\n"
			"rild.libpath=/system/lib/librapid-ril-core.so\n"
			"dalvik.vm.isa.x86.features=ssse3,movbe\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"sys.tcpdump.file=/data/logs/capture.pcap\n"
			"ro.internal.tcpdump.file=/data/logs/capture.pcap\n"
			"persist.data_netmgrd_mtu=1410\n"
			"persist.asus.bri_ratio=80\n"
			"persist.telproviders.debug=0\n"
			"persist.asus.cb.debug=0\n"
			"persist.asus.message.gcf.mode=0\n"
			"persist.asus.message.debug=0\n"
			"ro.config.hwrlib=T9_x86\n"
			"ro.config.xt9ime.max_subtype=7\n"
			"ro.ime.lowmemory=false\n"
			"ro.intel.corp.email=1\n"
			"persist.sys.modem.restart=12\n",
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
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1600000\n",
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
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 31,
		.content = "1600000 1333000 933000 800000 \n",
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
		.size = 49,
		.content =
			"1600000 6382\n"
			"1333000 580\n"
			"933000 362\n"
			"800000 19032\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "535\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/trans_table",
		.size = 277,
		.content =
			"   From  :    To\n"
			"         :   1600000   1333000    933000    800000 \n"
			"  1600000:         0        46        18       130 \n"
			"  1333000:        36         0         4        34 \n"
			"   933000:        21        12         0        35 \n"
			"   800000:       137        16        46         0 \n",
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
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1600000\n",
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
		.size = 4,
		.content = "0 1\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_frequencies",
		.size = 31,
		.content = "1600000 1333000 933000 800000 \n",
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
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 49,
		.content =
			"1600000 6415\n"
			"1333000 580\n"
			"933000 362\n"
			"800000 19247\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 4,
		.content = "539\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/trans_table",
		.size = 277,
		.content =
			"   From  :    To\n"
			"         :   1600000   1333000    933000    800000 \n"
			"  1600000:         0        46        18       132 \n"
			"  1333000:        36         0         4        34 \n"
			"   933000:        21        12         0        35 \n"
			"   800000:       139        16        46         0 \n",
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
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1600000\n",
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
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies",
		.size = 31,
		.content = "1600000 1333000 933000 800000 \n",
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
		.content = "1600000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 49,
		.content =
			"1600000 6073\n"
			"1333000 374\n"
			"933000 465\n"
			"800000 19948\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 4,
		.content = "527\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/trans_table",
		.size = 277,
		.content =
			"   From  :    To\n"
			"         :   1600000   1333000    933000    800000 \n"
			"  1600000:         0        41        15       129 \n"
			"  1333000:        27         0         5        31 \n"
			"   933000:        17        12         0        45 \n"
			"   800000:       141        10        54         0 \n",
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
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1600000\n",
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
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 31,
		.content = "1600000 1333000 933000 800000 \n",
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
		.content = "1600000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "800000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 49,
		.content =
			"1600000 6085\n"
			"1333000 374\n"
			"933000 465\n"
			"800000 20190\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "529\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/trans_table",
		.size = 277,
		.content =
			"   From  :    To\n"
			"         :   1600000   1333000    933000    800000 \n"
			"  1600000:         0        41        15       130 \n"
			"  1333000:        27         0         5        31 \n"
			"   933000:        17        12         0        45 \n"
			"   800000:       142        10        54         0 \n",
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
		.key = "asus.powertask",
		.value = "0",
	},
	{
		.key = "atd.keybox.ready",
		.value = "TRUE",
	},
	{
		.key = "atd.start.key.install",
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
		.value = "1",
	},
	{
		.key = "audio.offload.min.duration.secs",
		.value = "20",
	},
	{
		.key = "audiocomms.vp.fw_name",
		.value = "vpimg_es325.bin",
	},
	{
		.key = "crashlogd.processing.ongoing",
		.value = "1",
	},
	{
		.key = "dalvik.jit.code_cache_size",
		.value = "1048576",
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
		.value = "128m",
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
		.value = "ssse3,movbe",
	},
	{
		.key = "dalvik.vm.stack-trace-file",
		.value = "/data/anr/traces.txt",
	},
	{
		.key = "debug.asus.android_reboot",
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
		.key = "gps.version.driver",
		.value = "5_0_17.19.13.230734_sardine",
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
		.key = "gsm.first.imei",
		.value = "true",
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
		.key = "gsm.sim.ccid",
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
		.value = "M3.20.1_ZE500CL_1507.03",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "Intrinsyc Rapid-RIL M6.59 for Android 4.2 (Build September 17/2013)",
	},
	{
		.key = "init.svc.CheckProp",
		.value = "stopped",
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
		.key = "init.svc.apk_logfs",
		.value = "stopped",
	},
	{
		.key = "init.svc.asus-usb-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.asus_audbg",
		.value = "stopped",
	},
	{
		.key = "init.svc.asus_checkaudbg",
		.value = "stopped",
	},
	{
		.key = "init.svc.asus_chk_asdf",
		.value = "stopped",
	},
	{
		.key = "init.svc.asus_kernelmsg",
		.value = "stopped",
	},
	{
		.key = "init.svc.asusgesture",
		.value = "stopped",
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
		.key = "init.svc.btwifimac",
		.value = "stopped",
	},
	{
		.key = "init.svc.check-datalog",
		.value = "stopped",
	},
	{
		.key = "init.svc.console",
		.value = "stopped",
	},
	{
		.key = "init.svc.crashlogd",
		.value = "stopped",
	},
	{
		.key = "init.svc.debuggerd",
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
		.key = "init.svc.gpsd",
		.value = "running",
	},
	{
		.key = "init.svc.healthd",
		.value = "running",
	},
	{
		.key = "init.svc.ins_key",
		.value = "restarting",
	},
	{
		.key = "init.svc.ins_moudle",
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
		.key = "init.svc.media",
		.value = "running",
	},
	{
		.key = "init.svc.mmgr",
		.value = "running",
	},
	{
		.key = "init.svc.mount-cache2",
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
		.key = "init.svc.p2p_supplicant",
		.value = "running",
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
		.value = "running",
	},
	{
		.key = "init.svc.runcheckJB",
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
		.key = "init.svc.widevine",
		.value = "stopped",
	},
	{
		.key = "init.svc.wifi_info",
		.value = "stopped",
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
		.key = "lpa.audiosetup.time",
		.value = "70",
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
		.value = "net.qtaguid_enabled",
	},
	{
		.key = "net.hostname",
		.value = "android-6860077ab02fcd35",
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
		.value = "4093,26280,35040,4096,16384,35040",
	},
	{
		.key = "net.tcp.buffersize.evdo",
		.value = "4094,87380,262144,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.gprs",
		.value = "4092,8760,11680,4096,8760,11680",
	},
	{
		.key = "net.tcp.buffersize.hsdpa",
		.value = "32768,262144,1220608,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.hspa",
		.value = "32768,262144,1220608,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.hspap",
		.value = "32768,262144,1220608,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.hsupa",
		.value = "32768,262144,1220608,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.lte",
		.value = "655360,1310720,2621440,327680,655360,1310720",
	},
	{
		.key = "net.tcp.buffersize.umts",
		.value = "4094,87380,220416,4096,16384,110208",
	},
	{
		.key = "net.tcp.buffersize.wifi",
		.value = "524288,1048576,2097152,262144,524288,1048576",
	},
	{
		.key = "net.tcp.default_init_rwnd",
		.value = "60",
	},
	{
		.key = "offload.compress.device",
		.value = "2",
	},
	{
		.key = "panel.physicalHeightmm",
		.value = "111",
	},
	{
		.key = "panel.physicalWidthmm",
		.value = "62",
	},
	{
		.key = "persist.asus.audbg",
		.value = "0",
	},
	{
		.key = "persist.asus.bri_ratio",
		.value = "80",
	},
	{
		.key = "persist.asus.cb.debug",
		.value = "0",
	},
	{
		.key = "persist.asus.dclick",
		.value = "0",
	},
	{
		.key = "persist.asus.fuse_MicroSD",
		.value = "done",
	},
	{
		.key = "persist.asus.gesture.type",
		.value = "0000000",
	},
	{
		.key = "persist.asus.glove",
		.value = "0",
	},
	{
		.key = "persist.asus.kernelmessage",
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
		.key = "persist.asus.mupload.enable",
		.value = "0",
	},
	{
		.key = "persist.asus.qxdmlog.filesize",
		.value = "100",
	},
	{
		.key = "persist.asus.qxdmlog.maxfiles",
		.value = "6",
	},
	{
		.key = "persist.asus.qxdmlog.message",
		.value = "0",
	},
	{
		.key = "persist.asus.qxdmlog.sd1mmc0",
		.value = "0",
	},
	{
		.key = "persist.asus.startlog",
		.value = "0",
	},
	{
		.key = "persist.data_netmgrd_mtu",
		.value = "1410",
	},
	{
		.key = "persist.radio.device.imei",
		.value = "014327004196110",
	},
	{
		.key = "persist.radio.device.imeisv",
		.value = "14",
	},
	{
		.key = "persist.radio.ril_modem_state",
		.value = "1",
	},
	{
		.key = "persist.selective.ota_budget",
		.value = "1800000",
	},
	{
		.key = "persist.service.thermal",
		.value = "1",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.modem.restart",
		.value = "12",
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
		.key = "persist.sys.test.poweronoff",
		.value = "0",
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
		.value = "XMM7160_CONF_5",
	},
	{
		.key = "persist.tcs.hw_filename",
		.value = "/etc/telephony/XMM7160_CONF_5.xml",
	},
	{
		.key = "persist.tel.hot_swap.support",
		.value = "true",
	},
	{
		.key = "persist.tel.lteOnGsmDevice",
		.value = "true",
	},
	{
		.key = "persist.telproviders.debug",
		.value = "0",
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
		.key = "ril.coredumpwarning.enable",
		.value = "1",
	},
	{
		.key = "ril.ecclist",
		.value = "112,911,000,08,110,118,119,999",
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
		.key = "ro.additionalbutton.operation",
		.value = "0",
	},
	{
		.key = "ro.allow.mock.location",
		.value = "0",
	},
	{
		.key = "ro.asus.browser.uap",
		.value = "ASUS-ZE500CL",
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
		.key = "ro.boot.min.cap",
		.value = "3",
	},
	{
		.key = "ro.boot.mode",
		.value = "main",
	},
	{
		.key = "ro.boot.serialno",
		.value = "01234567890123456789",
	},
	{
		.key = "ro.boot.spid",
		.value = "0000:0000:0003:0002:0000:0021",
	},
	{
		.key = "ro.boot.wakesrc",
		.value = "01",
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
		.value = "/factory/bd_addr.conf",
	},
	{
		.key = "ro.bt.conf_file",
		.value = "/system/etc/bluetooth/bt_ATT_Phone.conf",
	},
	{
		.key = "ro.btmac",
		.value = "9C:5C:8E:1C:61:0D",
	},
	{
		.key = "ro.build.app.version",
		.value = "044030426_201501050106",
	},
	{
		.key = "ro.build.asus.sku",
		.value = "ATT",
	},
	{
		.key = "ro.build.characteristics",
		.value = "default",
	},
	{
		.key = "ro.build.csc.version",
		.value = "ATT_ZE500CL-12.16.10.95-20160223",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1456220569",
	},
	{
		.key = "ro.build.date",
		.value = "Tue Feb 23 17:42:49 CST 2016",
	},
	{
		.key = "ro.build.description",
		.value = "ATT_Phone-user 5.0.1 LRX22C ATT_Phone-12.16.10.95-20160223 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "ATT_12.16.10.95",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "asus/ATT_Phone/ASUS_Z00D:5.0.1/LRX22C/ATT_Phone-12.16.10.95-20160223:user/release-keys",
	},
	{
		.key = "ro.build.host",
		.value = "mcrd1-40",
	},
	{
		.key = "ro.build.id",
		.value = "LRX22C",
	},
	{
		.key = "ro.build.product",
		.value = "ze500cl",
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
		.value = "android",
	},
	{
		.key = "ro.build.version.all_codenames",
		.value = "REL",
	},
	{
		.key = "ro.build.version.base_os",
		.value = "asus/ATT_Phone/ASUS_Z00D:5.0.1/LRX22C/ATT_Phone-12.16.10.95-20160223:user/release-keys",
	},
	{
		.key = "ro.build.version.codename",
		.value = "REL",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "ATT_Phone-12.16.10.95-20160223",
	},
	{
		.key = "ro.build.version.release",
		.value = "5.0.1",
	},
	{
		.key = "ro.build.version.sdk",
		.value = "21",
	},
	{
		.key = "ro.build.version.security_patch",
		.value = "2015-09-11",
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
		.value = "android-att-us",
	},
	{
		.key = "ro.com.google.clientidbase.gmm",
		.value = "android-asus",
	},
	{
		.key = "ro.com.google.clientidbase.ms",
		.value = "android-att-us",
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
		.value = "5.0_r2",
	},
	{
		.key = "ro.config.CID",
		.value = "ATT",
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
		.value = "1B",
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
		.key = "ro.config.ringtone",
		.value = "Festival.ogg",
	},
	{
		.key = "ro.config.sentmail_sound",
		.value = "SentMail.ogg",
	},
	{
		.key = "ro.config.specific",
		.value = "att",
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
		.key = "ro.epad.mount_point.MicroSD",
		.value = "auto",
	},
	{
		.key = "ro.epad.mount_point.microsd",
		.value = "/storage/MicroSD",
	},
	{
		.key = "ro.epad.mount_point.sdcard1",
		.value = "/storage/MicroSD",
	},
	{
		.key = "ro.epad.mount_point.usbdisk1",
		.value = "/Removable/USBdisk1",
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
		.key = "ro.frp.pst",
		.value = "/dev/block/mmcblk0p2",
	},
	{
		.key = "ro.gnss.sv.status",
		.value = "true",
	},
	{
		.key = "ro.hardware.id",
		.value = "1.2",
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
		.key = "ro.intel.corp.email",
		.value = "1",
	},
	{
		.key = "ro.internal.tcpdump.file",
		.value = "/data/logs/capture.pcap",
	},
	{
		.key = "ro.isn",
		.value = "N0CY1601MB0054039",
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
		.value = "ASUS_Z00D",
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
		.value = "ASUS ZenFone 2E",
	},
	{
		.key = "ro.product.name",
		.value = "ATT_Phone",
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
		.value = "1508168516847",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "G1AZCY02T543",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "OPTIONAL",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "320",
	},
	{
		.key = "ro.spid.gps.FrqPlan",
		.value = "FRQ_PLAN_26MHZ_2PPM",
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
		.key = "ro.telephony.default_network",
		.value = "9",
	},
	{
		.key = "ro.thermal.ituxversion",
		.value = "3.0",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
	},
	{
		.key = "ro.wifimac",
		.value = "9C5C8E1C610E",
	},
	{
		.key = "ro.zygote",
		.value = "zygote32",
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
		.key = "selinux.reload_policy",
		.value = "1",
	},
	{
		.key = "service.amtl1.cfg",
		.value = "clovertrail_XMM_7160",
	},
	{
		.key = "service.bootanim.exit",
		.value = "1",
	},
	{
		.key = "sys.atcmd.ready",
		.value = "0",
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
		.key = "sys.config.asussound",
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
		.value = "91.37",
	},
	{
		.key = "sys.kernel.version",
		.value = "3.10.20-i386_ctp-00001-g2bc4601",
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
		.value = "4",
	},
	{
		.key = "sys.supp_ia32.version",
		.value = "00.57",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "10800",
	},
	{
		.key = "sys.tcpdump.file",
		.value = "/data/logs/capture.pcap",
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
		.value = "91.38",
	},
	{
		.key = "sys.watchdog.previous.counter",
		.value = "0",
	},
	{
		.key = "system.at-proxy.mode",
		.value = "0",
	},
	{
		.key = "telephony.lteOnCdmaDevice",
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
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "wifi.version.driver",
		.value = "7.10.324.7",
	},
	{
		.key = "wlan.att.attwifi.enable",
		.value = "1",
	},
	{
		.key = "wlan.att.attwifi.netid",
		.value = "0",
	},
	{
		.key = "wlan.driver.status",
		.value = "ok",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
