struct cpuinfo_mock_file filesystem[] = {
	{
		.path = "/proc/cpuinfo",
		.size = 263,
		.content =
			"Processor\t: ARMv7 Processor rev 1 (v7l)\n"
			"processor\t: 0\n"
			"BogoMIPS\t: 1022.18\n"
			"\n"
			"Features\t: swp half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 \n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 7\n"
			"CPU variant\t: 0x0\n"
			"\n"
			"Hardware\t: gs702c\n"
			"Revision\t: 0000\n"
			"Serial\t\t: 0000000000000000\n",
	},
	{
		.path = "/system/build.prop",
		.size = 4697,
		.content =
			"ro.build.id=KTU84P\n"
			"ro.build.display.id=actions_7029c-userdebug 4.4.4 KTU84P eng.root.20161122.162559 release-keys\n"
			"ro.build.version.incremental=eng.root.20161122.162559\n"
			"ro.build.version.sdk=19\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.release=4.4.4\n"
			"ro.build.date=Tue Nov 22 16:28:10 CST 2016\n"
			"ro.build.date.utc=1479803290\n"
			"ro.build.type=userdebug\n"
			"ro.build.user=root\n"
			"ro.build.host=tom-desktop\n"
			"ro.build.tags=release-keys\n"
			"ro.product.model=7029c_J0918\n"
			"ro.product.brand=Android\n"
			"ro.product.name= actions_7029c\n"
			"ro.product.device=J0918\n"
			"ro.product.board=atm7029c_twd_sd_J0918_8723bsvq0\n"
			"ro.product.cpu.abi=armeabi-v7a\n"
			"ro.product.cpu.abi2=armeabi\n"
			"ro.product.manufacturer=Actions\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=ATM702X\n"
			"ro.build.product= actions_7029c_J0918\n"
			"ro.build.description=actions_7029c-userdebug 4.4.4 KTU84P eng.root.20161122.162559 release-keys\n"
			"ro.build.fingerprint=Android/actions_7029c/J0918:4.4.4/KTU84P/eng.root.20161122.162559:userdebug/release-keys\n"
			"ro.build.characteristics=tablet\n"
			"ro.sf.lcd_density=160\n"
			"ro.opengles.version = 131072\n"
			"ro.config.softopengles = 0\n"
			"ro.config.used_hw_vsync = 1\n"
			"ro.settings.config.hdmi=off\n"
			"ro.soundrecorder.format=amr\n"
			"ro.systemui.volumekey=enable\n"
			"ro.systemui.capture=enable\n"
			"ro.systemui.hidebutton=disable\n"
			"ro.systemui.morebutton=disable\n"
			"ro.launcher.swipe=enable\n"
			"ro.launcher.config.cling=enable\n"
			"ro.launcher.hideactivity=disable\n"
			"ro.launcher.allapp.landX=7\n"
			"ro.launcher.allapp.landY=4\n"
			"ro.launcher.allapp.portX=4\n"
			"ro.launcher.allapp.portY=6\n"
			"ro.launcher.workspace.landX=6\n"
			"ro.launcher.workspace.landY=3\n"
			"ro.launcher.workspace.portX=6\n"
			"ro.launcher.workspace.portY=3\n"
			"ro.launcher.hotseatcellcount=7\n"
			"ro.launcher.hotseatallappsindex=3\n"
			"ro.launcher.hotseat.landY=0\n"
			"ro.launcher.hotseat.portY=0\n"
			"ro.calendar.localaccount=disable\n"
			"ro.product.usbdevice.VID=10d6\n"
			"ro.product.usbdevice.PID=fffe\n"
			"ro.product.mtpdevice.PID=4e41\n"
			"ro.product.ptpdevice.PID=4e43\n"
			"ro.shutmenu.recovery=disable\n"
			"ro.shutmenu.planemode=disable\n"
			"ro.shutmenu.restart=enable\n"
			"ro.usb.descriptor=actions,leopard,3.00\n"
			"ro.usbdevice.volumelabel=GS702C\n"
			"ro.serialno=4512482adf0feeee\n"
			"ro.config.quickboot = 0\n"
			"ro.im.keysounddefenable=true\n"
			"ro.support.gpswithwifi=1\n"
			"ro.wifi.signal.level.1=-70\n"
			"ro.wifi.signal.level.2=-65\n"
			"ro.wifi.signal.level.3=-60\n"
			"ro.wifi.signal.level.4=-55\n"
			"ro.product.pribrand=actions   \n"
			"ro.product.primodel=owlx1\n"
			"ro.ota.autorecovery=enable\n"
			"ro.device.model=actions_7029c_J0918\n"
			"ro.recovery.wipe=false\n"
			"ro.browser.maxtabs=8\n"
			"ro.config.shutdown.screenoff=0\n"
			"ro.ota.server= http://ota.actions-semi.net/GS702C/\n"
			"system.ctl.recoverywhencrash=4\n"
			"ro.com.android.dateformat=MM-dd-yyyy\n"
			"ro.config.ringtone=Orion.ogg\n"
			"ro.config.notification_sound=Adara.ogg\n"
			"ro.config.alarm_alert=Cesium.ogg\n"
			"ro.carrier=unknown\n"
			"hwui.render_dirty_regions=false\n"
			"ro.product.locale.language=en\n"
			"ro.product.locale.region=US\n"
			"dalvik.vm.dexopt-flags=v=n,o=v\n"
			"dalvik.vm.checkjni=false\n"
			"dalvik.vm.heapgrowthlimit=80m\n"
			"dalvik.vm.heapminfree=2m\n"
			"persist.radio.mediatek.chipid=0x6620\n"
			"dalvik.vm.heapstartsize=8m\n"
			"dalvik.vm.heapsize=384m\n"
			"dalvik.vm.heaptargetutilization=0.75\n"
			"dalvik.vm.heapmaxfree=8m\n"
			"ro.setupwizard.mode=OPTIONAL\n"
			"ro.com.google.gmsversion=4.4_r4\n"
			"persist.sys.dalvik.vm.lib=libdvm.so\n"
			"dalvik.vm.lockprof.threshold=500\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"ro.sf.hwrotation=90\n"
			"ro.sf.hdmi_rotation=0\n"
			"ro.sf.default_rotation=1\n"
			"persist.demo.hdmirotationlock=true\n"
			"persist.sys.shutok=init\n"
			"ro.config.max_starting_bg=3\n"
			"ro.config.wipe_corrupt_fs=true\n"
			"ro.camerahal.configorientation=90\n"
			"ro.camerahal.prevres0=SVGA,HD\n"
			"ro.camerahal.imageres0=SVGA,2M\n"
			"ro.camerahal.prevresdft0=SVGA\n"
			"ro.camerahal.imageresdft0=2M\n"
			"ro.camerahal.fpsdft0=30\n"
			"ro.camerahal.prevres1=VGA\n"
			"ro.camerahal.imageres1=QVGA,VGA\n"
			"ro.camerahal.prevresdft1=VGA\n"
			"ro.camerahal.imageresdft1=VGA\n"
			"ro.camerahal.fpsdft1=30\n"
			"ro.camerahal.flash0=1   \n"
			"ro.camerahal.flash1=0   \n"
			"camcorder.settings.xml=/data/camera/camcorder_profiles.xml\n"
			"ro.smartbacklight.minpwm=150\n"
			"ro.smartbacklight.strength=100\n"
			"ro.hdmi.onoffmode=alwayson\n"
			"ro.config.forcetabletui =1\n"
			"ro.build_mode=NULL\n"
			"persist.sys.timezone=America/New_York\n"
			"persist.sys.usb.config=mass_storage,adb\n"
			"persist.service.adb.enable=1\n"
			"ro.allow.mock.location=0\n"
			"ro.debuggable=1\n"
			"ro.adb.secure=0\n"
			"ro.config.hdmi_secure_check=0\n"
			"persist.sys.extra_features=1\n"
			"ro.change_property.enable=true\n"
			"ro.browser.ua=3\n"
			"ro.settings.support.bluetooth=true\n"
			"ro.settings.datausage=true\n"
			"ro.settings.hotspot=true\n"
			"ro.settings.mobilenetworks=true\n"
			"ro.settings.phonestatus=true\n"
			"ro.g3.display=true\n"
			"ro.airplanemode.display=true\n"
			"ro.settings.support.ethernet=true\n"
			"ro.settings.quickboot=1\n"
			"ro.camerahal.hangle0=56.0\n"
			"ro.camerahal.hangle1=46.0\n"
			"ro.net.config=0\n"
			"ro.settings.backuptransport=com.google.android.backup/.BackupTransportService\n"
			"ro.camerahal.hdr0=0\n"
			"ro.camerahal.hdr1=0\n"
			"ro.camerahal.hdrghost=0\n",
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
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/offline",
		.size = 4,
		.content = "1-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpuidle/current_driver",
		.size = 13,
		.content = "leopard_idle\n",
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
		.content = "1320000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "288000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 4,
		.content = "2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 54,
		.content = "288000 732000 948000 1020000 1104000 1260000 1320000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 57,
		.content = "conservative ondemand userspace interactive performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "288000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 15,
		.content = "leopard_cpufreq",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
		.size = 12,
		.content = "interactive\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "288000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 77,
		.content =
			"288000 147856\n"
			"732000 843\n"
			"948000 0\n"
			"1020000 0\n"
			"1104000 7945\n"
			"1260000 0\n"
			"1320000 0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "553\n",
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
	{ NULL },
};
#ifdef __ANDROID__
struct cpuinfo_mock_property properties[] = {
	{
		.key = "camcorder.settings.xml",
		.value = "/data/camera/camcorder_profiles.xml",
	},
	{
		.key = "dalvik.vm.checkjni",
		.value = "false",
	},
	{
		.key = "dalvik.vm.dexopt-flags",
		.value = "v=n,o=v",
	},
	{
		.key = "dalvik.vm.heapgrowthlimit",
		.value = "64m",
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
		.value = "128m",
	},
	{
		.key = "dalvik.vm.heapstartsize",
		.value = "5m",
	},
	{
		.key = "dalvik.vm.heaptargetutilization",
		.value = "0.75",
	},
	{
		.key = "dalvik.vm.lockprof.threshold",
		.value = "500",
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
		.key = "gsm.current.phone-type",
		.value = "1",
	},
	{
		.key = "gsm.network.type",
		.value = "none",
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
		.key = "hw.tvout.virtual_tv_xscale",
		.value = "100",
	},
	{
		.key = "hw.tvout.virtual_tv_yscale",
		.value = "100",
	},
	{
		.key = "hw.tvout_hdmi_select_vid",
		.value = "-1",
	},
	{
		.key = "hwui.render_dirty_regions",
		.value = "false",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.areadaheadd",
		.value = "stopped",
	},
	{
		.key = "init.svc.bootanim",
		.value = "stopped",
	},
	{
		.key = "init.svc.console",
		.value = "running",
	},
	{
		.key = "init.svc.cp_vendor",
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
		.key = "init.svc.engsetbtmacaddr",
		.value = "stopped",
	},
	{
		.key = "init.svc.engsetmacaddr",
		.value = "stopped",
	},
	{
		.key = "init.svc.flash_recovery",
		.value = "stopped",
	},
	{
		.key = "init.svc.fuse_sdcard",
		.value = "running",
	},
	{
		.key = "init.svc.healthd",
		.value = "running",
	},
	{
		.key = "init.svc.hwclock_update",
		.value = "stopped",
	},
	{
		.key = "init.svc.insmod_camera",
		.value = "stopped",
	},
	{
		.key = "init.svc.insmod_gsensor",
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
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.pfmnceserver",
		.value = "running",
	},
	{
		.key = "init.svc.rtw_suppl_con",
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
		.key = "init.svc.vold",
		.value = "running",
	},
	{
		.key = "init.svc.zygote",
		.value = "running",
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
		.value = "android-5f4820463c6ae816",
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
		.value = "524288,1048576,2097152,262144,524288,1048576",
	},
	{
		.key = "net.tcp.default_init_rwnd",
		.value = "60",
	},
	{
		.key = "persist.demo.hdmirotationlock",
		.value = "true",
	},
	{
		.key = "persist.radio.mediatek.chipid",
		.value = "0x6620",
	},
	{
		.key = "persist.service.adb.enable",
		.value = "",
	},
	{
		.key = "persist.sys.dalvik.vm.lib",
		.value = "libdvm.so",
	},
	{
		.key = "persist.sys.extra_features",
		.value = "1",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
	},
	{
		.key = "persist.sys.shutok",
		.value = "init",
	},
	{
		.key = "persist.sys.strictmode.disable",
		.value = "true",
	},
	{
		.key = "persist.sys.timezone",
		.value = "America/New_York",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "mass_storage,adb",
	},
	{
		.key = "persist.vold.set_label_done",
		.value = "1",
	},
	{
		.key = "ro.adb.secure",
		.value = "0",
	},
	{
		.key = "ro.airplanemode.display",
		.value = "true",
	},
	{
		.key = "ro.allow.mock.location",
		.value = "0",
	},
	{
		.key = "ro.baseband",
		.value = "",
	},
	{
		.key = "ro.board.platform",
		.value = "ATM702X",
	},
	{
		.key = "ro.boot.bootdev",
		.value = "nand",
	},
	{
		.key = "ro.boot.dvfslevel",
		.value = "0x7031",
	},
	{
		.key = "ro.boot.serialno",
		.value = "A07B4ED162A4EB39",
	},
	{
		.key = "ro.boot.supportduallogo",
		.value = "0",
	},
	{
		.key = "ro.bootdev",
		.value = "nand",
	},
	{
		.key = "ro.bootloader",
		.value = "gs702c-2014",
	},
	{
		.key = "ro.bootmode",
		.value = "normal",
	},
	{
		.key = "ro.browser.maxtabs",
		.value = "8",
	},
	{
		.key = "ro.browser.ua",
		.value = "3",
	},
	{
		.key = "ro.bt.bdaddr_path",
		.value = "/data/misc/bluedroid/bdaddr",
	},
	{
		.key = "ro.build.characteristics",
		.value = "tablet",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1479803290",
	},
	{
		.key = "ro.build.date",
		.value = "Tue Nov 22 16:28:10 CST 2016",
	},
	{
		.key = "ro.build.description",
		.value = "actions_7029c-userdebug 4.4.4 KTU84P eng.root.20161122.162559 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "actions_7029c-userdebug 4.4.4 KTU84P eng.root.20161122.162559 release-keys",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "Android/actions_7029c/J0918:4.4.4/KTU84P/eng.root.20161122.162559:userdebug/release-keys",
	},
	{
		.key = "ro.build.host",
		.value = "tom-desktop",
	},
	{
		.key = "ro.build.id",
		.value = "KTU84P",
	},
	{
		.key = "ro.build.product",
		.value = "actions_7029c_J0918",
	},
	{
		.key = "ro.build.tags",
		.value = "release-keys",
	},
	{
		.key = "ro.build.type",
		.value = "userdebug",
	},
	{
		.key = "ro.build.user",
		.value = "root",
	},
	{
		.key = "ro.build.version.codename",
		.value = "REL",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "eng.root.20161122.162559",
	},
	{
		.key = "ro.build.version.release",
		.value = "4.4.4",
	},
	{
		.key = "ro.build.version.sdk",
		.value = "19",
	},
	{
		.key = "ro.build_mode",
		.value = "NULL",
	},
	{
		.key = "ro.calendar.localaccount",
		.value = "disable",
	},
	{
		.key = "ro.camerahal.configorientation",
		.value = "90",
	},
	{
		.key = "ro.camerahal.flash0",
		.value = "1",
	},
	{
		.key = "ro.camerahal.flash1",
		.value = "0",
	},
	{
		.key = "ro.camerahal.fpsdft0",
		.value = "30",
	},
	{
		.key = "ro.camerahal.fpsdft1",
		.value = "30",
	},
	{
		.key = "ro.camerahal.hangle0",
		.value = "56.0",
	},
	{
		.key = "ro.camerahal.hangle1",
		.value = "46.0",
	},
	{
		.key = "ro.camerahal.hdr0",
		.value = "0",
	},
	{
		.key = "ro.camerahal.hdr1",
		.value = "0",
	},
	{
		.key = "ro.camerahal.hdrghost",
		.value = "0",
	},
	{
		.key = "ro.camerahal.imageres0",
		.value = "SVGA,2M",
	},
	{
		.key = "ro.camerahal.imageres1",
		.value = "QVGA,VGA",
	},
	{
		.key = "ro.camerahal.imageresdft0",
		.value = "2M",
	},
	{
		.key = "ro.camerahal.imageresdft1",
		.value = "VGA",
	},
	{
		.key = "ro.camerahal.prevres0",
		.value = "SVGA,HD",
	},
	{
		.key = "ro.camerahal.prevres1",
		.value = "VGA",
	},
	{
		.key = "ro.camerahal.prevresdft0",
		.value = "SVGA",
	},
	{
		.key = "ro.camerahal.prevresdft1",
		.value = "VGA",
	},
	{
		.key = "ro.carrier",
		.value = "unknown",
	},
	{
		.key = "ro.change_property.enable",
		.value = "true",
	},
	{
		.key = "ro.com.android.dateformat",
		.value = "MM-dd-yyyy",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "4.4_r4",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Cesium.ogg",
	},
	{
		.key = "ro.config.forcetabletui",
		.value = "1",
	},
	{
		.key = "ro.config.hdmi_secure_check",
		.value = "0",
	},
	{
		.key = "ro.config.low_ram",
		.value = "true",
	},
	{
		.key = "ro.config.max_starting_bg",
		.value = "3",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "Adara.ogg",
	},
	{
		.key = "ro.config.quickboot",
		.value = "0",
	},
	{
		.key = "ro.config.ringtone",
		.value = "Orion.ogg",
	},
	{
		.key = "ro.config.shutdown.screenoff",
		.value = "0",
	},
	{
		.key = "ro.config.softopengles",
		.value = "0",
	},
	{
		.key = "ro.config.used_hw_vsync",
		.value = "1",
	},
	{
		.key = "ro.config.wipe_corrupt_fs",
		.value = "true",
	},
	{
		.key = "ro.debuggable",
		.value = "1",
	},
	{
		.key = "ro.device.model",
		.value = "actions_7029c_J0918",
	},
	{
		.key = "ro.dvfslevel",
		.value = "0x7031",
	},
	{
		.key = "ro.factorytest",
		.value = "0",
	},
	{
		.key = "ro.g3.display",
		.value = "true",
	},
	{
		.key = "ro.hardware",
		.value = "gs702c",
	},
	{
		.key = "ro.hdmi.onoffmode",
		.value = "alwayson",
	},
	{
		.key = "ro.im.keysounddefenable",
		.value = "true",
	},
	{
		.key = "ro.launcher.allapp.landX",
		.value = "7",
	},
	{
		.key = "ro.launcher.allapp.landY",
		.value = "4",
	},
	{
		.key = "ro.launcher.allapp.portX",
		.value = "4",
	},
	{
		.key = "ro.launcher.allapp.portY",
		.value = "6",
	},
	{
		.key = "ro.launcher.config.cling",
		.value = "enable",
	},
	{
		.key = "ro.launcher.hideactivity",
		.value = "disable",
	},
	{
		.key = "ro.launcher.hotseat.landY",
		.value = "0",
	},
	{
		.key = "ro.launcher.hotseat.portY",
		.value = "0",
	},
	{
		.key = "ro.launcher.hotseatallappsindex",
		.value = "3",
	},
	{
		.key = "ro.launcher.hotseatcellcount",
		.value = "7",
	},
	{
		.key = "ro.launcher.swipe",
		.value = "enable",
	},
	{
		.key = "ro.launcher.workspace.landX",
		.value = "6",
	},
	{
		.key = "ro.launcher.workspace.landY",
		.value = "3",
	},
	{
		.key = "ro.launcher.workspace.portX",
		.value = "6",
	},
	{
		.key = "ro.launcher.workspace.portY",
		.value = "3",
	},
	{
		.key = "ro.net.config",
		.value = "0",
	},
	{
		.key = "ro.opengles.version",
		.value = "131072",
	},
	{
		.key = "ro.ota.autorecovery",
		.value = "enable",
	},
	{
		.key = "ro.ota.server",
		.value = "http://ota.actions-semi.net/GS702C/",
	},
	{
		.key = "ro.phone.dynamicoomadj",
		.value = "1",
	},
	{
		.key = "ro.product.board",
		.value = "atm7029c_twd_sd_J0918_8723bsvq0",
	},
	{
		.key = "ro.product.brand",
		.value = "Android",
	},
	{
		.key = "ro.product.cpu.abi2",
		.value = "armeabi",
	},
	{
		.key = "ro.product.cpu.abi",
		.value = "armeabi-v7a",
	},
	{
		.key = "ro.product.device",
		.value = "J0918",
	},
	{
		.key = "ro.product.devicenumber",
		.value = "",
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
		.value = "Actions",
	},
	{
		.key = "ro.product.model",
		.value = "7029c_J0918",
	},
	{
		.key = "ro.product.mtpdevice.PID",
		.value = "4e41",
	},
	{
		.key = "ro.product.name",
		.value = "actions_7029c",
	},
	{
		.key = "ro.product.pribrand",
		.value = "actions",
	},
	{
		.key = "ro.product.primodel",
		.value = "owlx1",
	},
	{
		.key = "ro.product.ptpdevice.PID",
		.value = "4e43",
	},
	{
		.key = "ro.product.usbdevice.PID",
		.value = "fffe",
	},
	{
		.key = "ro.product.usbdevice.VID",
		.value = "10d6",
	},
	{
		.key = "ro.recovery.wipe",
		.value = "false",
	},
	{
		.key = "ro.revision",
		.value = "0",
	},
	{
		.key = "ro.runtime.firstboot",
		.value = "1313055355096",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "A07B4ED162A4EB39",
	},
	{
		.key = "ro.settings.backuptransport",
		.value = "com.google.android.backup/.BackupTransportService",
	},
	{
		.key = "ro.settings.config.hdmi",
		.value = "off",
	},
	{
		.key = "ro.settings.datausage",
		.value = "true",
	},
	{
		.key = "ro.settings.hotspot",
		.value = "true",
	},
	{
		.key = "ro.settings.mobilenetworks",
		.value = "true",
	},
	{
		.key = "ro.settings.phonestatus",
		.value = "true",
	},
	{
		.key = "ro.settings.quickboot",
		.value = "1",
	},
	{
		.key = "ro.settings.support.bluetooth",
		.value = "true",
	},
	{
		.key = "ro.settings.support.ethernet",
		.value = "true",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "DISABLE",
	},
	{
		.key = "ro.sf.default_rotation",
		.value = "1",
	},
	{
		.key = "ro.sf.hdmi_rotation",
		.value = "0",
	},
	{
		.key = "ro.sf.hwrotation",
		.value = "90",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "160",
	},
	{
		.key = "ro.shutmenu.planemode",
		.value = "disable",
	},
	{
		.key = "ro.shutmenu.recovery",
		.value = "disable",
	},
	{
		.key = "ro.shutmenu.restart",
		.value = "enable",
	},
	{
		.key = "ro.skia.font.cache",
		.value = "2097152",
	},
	{
		.key = "ro.skia.min.font.cache",
		.value = "262144",
	},
	{
		.key = "ro.smartbacklight.minpwm",
		.value = "150",
	},
	{
		.key = "ro.smartbacklight.strength",
		.value = "100",
	},
	{
		.key = "ro.soundrecorder.format",
		.value = "amr",
	},
	{
		.key = "ro.support.gpswithwifi",
		.value = "1",
	},
	{
		.key = "ro.systemui.capture",
		.value = "enable",
	},
	{
		.key = "ro.systemui.hidebutton",
		.value = "disable",
	},
	{
		.key = "ro.systemui.morebutton",
		.value = "disable",
	},
	{
		.key = "ro.systemui.volumekey",
		.value = "enable",
	},
	{
		.key = "ro.usb.descriptor",
		.value = "actions,leopard,3.00",
	},
	{
		.key = "ro.usbdevice.volumelabel",
		.value = "GS702C",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
	},
	{
		.key = "ro.wifi.signal.level.1",
		.value = "-70",
	},
	{
		.key = "ro.wifi.signal.level.2",
		.value = "-65",
	},
	{
		.key = "ro.wifi.signal.level.3",
		.value = "-60",
	},
	{
		.key = "ro.wifi.signal.level.4",
		.value = "-55",
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
		.key = "sys.image_enhanced_system",
		.value = "0",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "4500",
	},
	{
		.key = "sys.usb.config",
		.value = "mass_storage,adb",
	},
	{
		.key = "sys.usb.state",
		.value = "mass_storage,adb",
	},
	{
		.key = "system.ctl.recoverywhencrash",
		.value = "4",
	},
	{
		.key = "system.ram.total",
		.value = "512",
	},
	{
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "wifi.supplicant_scan_interval",
		.value = "120",
	},
	{
		.key = "wlan.driver.status",
		.value = "ok",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
