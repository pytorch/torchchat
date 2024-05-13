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
		.path = "/system/build.prop",
		.size = 8740,
		.content =
			"\n"
			"# begin build properties\n"
			"# autogenerated by buildinfo.sh\n"
			"ro.build.id=NRD90M\n"
			"ro.build.display.id=NRD90M.G950USQU1AQC9\n"
			"ro.build.version.incremental=G950USQU1AQC9\n"
			"ro.build.version.sdk=24\n"
			"ro.build.version.preview_sdk=0\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.all_codenames=REL\n"
			"ro.build.version.release=7.0\n"
			"ro.build.version.security_patch=2017-03-01\n"
			"ro.build.version.base_os=\n"
			"ro.build.date=Sat Mar 11 22:44:17 KST 2017\n"
			"ro.build.date.utc=1489239857\n"
			"ro.build.type=user\n"
			"ro.build.user=dpi\n"
			"ro.build.host=SWHE7721\n"
			"ro.build.tags=release-keys\n"
			"ro.build.flavor=dreamqltesq-user\n"
			"ro.product.model=SM-G950U\n"
			"ro.product.brand=samsung\n"
			"ro.product.name=dreamqltesq\n"
			"ro.product.device=dreamqltesq\n"
			"ro.product.board=msm8998\n"
			"# ro.product.cpu.abi and ro.product.cpu.abi2 are obsolete,\n"
			"# use ro.product.cpu.abilist instead.\n"
			"ro.product.cpu.abi=arm64-v8a\n"
			"ro.product.cpu.abilist=arm64-v8a,armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist32=armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist64=arm64-v8a\n"
			"ro.product.manufacturer=samsung\n"
			"ro.product.locale=en-US\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=msm8998\n"
			"# ro.build.product is obsolete; use ro.product.device\n"
			"ro.build.product=dreamqltesq\n"
			"# Do not try to parse description, fingerprint, or thumbprint\n"
			"ro.build.description=dreamqltesq-user 7.0 NRD90M G950USQU1AQC9 release-keys\n"
			"ro.build.fingerprint=samsung/dreamqltesq/dreamqltesq:7.0/NRD90M/G950USQU1AQC9:user/release-keys\n"
			"ro.build.characteristics=default\n"
			"# Samsung Specific Properties\n"
			"ro.build.PDA=G950USQU1AQC9\n"
			"ro.build.official.release=true\n"
			"ro.config.rm_preload_enabled=1\n"
			"ro.build.changelist=10895874\n"
			"ro.product_ship=true\n"
			"ro.chipname=MSM8998\n"
			"# end build properties\n"
			"\n"
			"#\n"
			"# HWUI_BUILD_PROPERTIES\n"
			"#\n"
			"ro.hwui.texture_cache_size=88\n"
			"ro.hwui.layer_cache_size=58\n"
			"ro.hwui.path_cache_size=16\n"
			"ro.hwui.texture_cache_flushrate=0.4\n"
			"ro.hwui.shape_cache_size=4\n"
			"ro.hwui.gradient_cache_size=2\n"
			"ro.hwui.drop_shadow_cache_size=6\n"
			"ro.hwui.r_buffer_cache_size=8\n"
			"ro.hwui.text_small_cache_width=1024\n"
			"ro.hwui.text_small_cache_height=1024\n"
			"ro.hwui.text_large_cache_width=4096\n"
			"ro.hwui.text_large_cache_height=2048\n"
			"#\n"
			"# from device/samsung/dreamqltesq/system.prop\n"
			"#\n"
			"#\n"
			"# system.prop for cobalt\n"
			"#\n"
			"ro.sf.lcd_density=480\n"
			"ro.sf.init.lcd_density=640\n"
			"\n"
			"DEVICE_PROVISIONED=1\n"
			"\n"
			"debug.sf.hw=1\n"
			"debug.gralloc.enable_fb_ubwc=1\n"
			"dalvik.vm.heapsize=36m\n"
			"dev.pm.dyn_samplingrate=1\n"
			"persist.demo.hdmirotationlock=false\n"
			"\n"
			"#ro.hdmi.enable=true\n"
			"#\n"
			"# system props for the cne module\n"
			"#\n"
			"persist.cne.feature=0\n"
			"persist.cne.dpm=0\n"
			"persist.dpm.feature=0\n"
			"\n"
			"#system props for the MM modules\n"
			"media.stagefright.enable-player=true\n"
			"media.stagefright.enable-http=true\n"
			"media.stagefright.enable-aac=true\n"
			"media.stagefright.enable-qcp=true\n"
			"media.stagefright.enable-scan=true\n"
			"mmp.enable.3g2=true\n"
			"media.aac_51_output_enabled=true\n"
			"mm.enable.smoothstreaming=true\n"
			"#3183219 is decimal sum of supported codecs in AAL\n"
			"#codecs:(PARSER_)AAC AC3 AMR_NB AMR_WB ASF AVI DTS FLV 3GP 3G2 MKV MP2PS MP2TS MP3 OGG QCP WAV FLAC AIFF APE\n"
			"mm.enable.qcom_parser=1048575\n"
			"persist.mm.enable.prefetch=true\n"
			"\n"
			"#\n"
			"# system props for the data modules\n"
			"#\n"
			"ro.use_data_netmgrd=true\n"
			"persist.data.netmgrd.qos.enable=true\n"
			"persist.data.mode=concurrent\n"
			"#system props for time-services\n"
			"persist.timed.enable=true\n"
			"\n"
			"#\n"
			"# system prop for opengles version\n"
			"#\n"
			"# 196608 is decimal for 0x30000 to report version 3\n"
			"ro.opengles.version=196610\n"
			"\n"
			"# system property for maximum number of HFP client connections\n"
			"bt.max.hfpclient.connections=1\n"
			"\n"
			"# System property for cabl\n"
			"ro.qualcomm.cabl=0\n"
			"\n"
			"#Simulate sdcard on /data/media\n"
			"#\n"
			"persist.fuse_sdcard=true\n"
			"\n"
			"#system prop for Bluetooth SOC type\n"
			"qcom.bluetooth.soc=cherokee\n"
			"\n"
			"#system prop for A4WP profile support\n"
			"ro.bluetooth.a4wp=false\n"
			"\n"
			"#system prop for wipower support\n"
			"ro.bluetooth.wipower=true\n"
			"\n"
			"#\n"
			"#snapdragon value add features\n"
			"#\n"
			"\n"
			"#system prop for RmNet Data\n"
			"persist.rmnet.data.enable=true\n"
			"persist.data.wda.enable=true\n"
			"persist.data.df.dl_mode=5\n"
			"persist.data.df.ul_mode=5\n"
			"persist.data.df.agg.dl_pkt=10\n"
			"persist.data.df.agg.dl_size=4096\n"
			"persist.data.df.mux_count=8\n"
			"persist.data.df.iwlan_mux=9\n"
			"persist.data.df.dev_name=rmnet_usb0\n"
			"\n"
			"#property to enable user to access Google WFD settings\n"
			"persist.debug.wfd.enable=1\n"
			"##property to choose between virtual/external wfd display\n"
			"persist.sys.wfd.virtual=0\n"
			"\n"
			"# system prop for NFC DT\n"
			"ro.nfc.port=I2C\n"
			"\n"
			"#hwui properties\n"
			"ro.hwui.texture_cache_size=72\n"
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
			"\n"
			"#config for bringup\n"
			"config.disable_atlas=true\n"
			"debug.batt.no_battery=true\n"
			"\n"
			"# enable navigation bar\n"
			"qemu.hw.mainkeys=0\n"
			"\n"
			"#property to enable VDS WFD solution\n"
			"persist.hwc.enable_vds=1\n"
			"\n"
			"#Set SSC Debug Level on AP Side\n"
			"persist.debug.sensors.hal=I\n"
			"debug.qualcomm.sns.daemon=I\n"
			"debug.qualcomm.sns.libsensor1=I\n"
			"\n"
			"#Disable Sensor Feature\n"
			"ro.qti.sensors.georv=false\n"
			"ro.qti.sensors.cmc=false\n"
			"ro.qti.sensors.dpc=false\n"
			"ro.qti.sensors.facing=false\n"
			"ro.qti.sensors.fast_amd=false\n"
			"ro.qti.sensors.scrn_ortn=false\n"
			"\n"
			"#\n"
			"# ADDITIONAL_BUILD_PROPERTIES\n"
			"#\n"
			"ro.astcenc.astcsupport=1\n"
			"ro.mct.compressiontype=ETC1\n"
			"ro.config.dmverity=true\n"
			"ro.config.kap_default_on=true\n"
			"ro.config.kap=true\n"
			"ro.knox.enhance.zygote.aslr=0\n"
			"ro.tether.denied=false\n"
			"rild.libpath=/system/lib64/libsec-ril.so\n"
			"ro.radio.noril=no\n"
			"ro.use_data_netmgrd=true\n"
			"persist.radio.sib16_support=0\n"
			"ro.product.first_api_level=24\n"
			"dalvik.vm.heapstartsize=8m\n"
			"dalvik.vm.heapgrowthlimit=256m\n"
			"dalvik.vm.heapsize=512m\n"
			"dalvik.vm.heaptargetutilization=0.75\n"
			"dalvik.vm.heapminfree=2m\n"
			"dalvik.vm.heapmaxfree=8m\n"
			"keyguard.no_require_sim=true\n"
			"ro.carrier=unknown\n"
			"ro.security.icd.flagmode=multi\n"
			"security.ASKS.policy_version=000000\n"
			"ro.com.google.clientidbase=android-samsung\n"
			"ro.vendor.extension_library=libqti-perfd-client.so\n"
			"persist.radio.apm_sim_not_pwdn=1\n"
			"persist.radio.custom_ecc=1\n"
			"af.fast_track_multiplier=1\n"
			"audio_hal.period_size=192\n"
			"audio.adm.buffering.ms=3\n"
			"ro.qc.sdk.audio.fluencetype=none\n"
			"persist.audio.fluence.voicecall=true\n"
			"persist.audio.fluence.voicerec=false\n"
			"persist.audio.fluence.speaker=true\n"
			"tunnel.audio.encode=false\n"
			"audio.offload.buffer.size.kb=32\n"
			"audio.offload.video=true\n"
			"audio.offload.pcm.16bit.enable=true\n"
			"audio.offload.pcm.24bit.enable=true\n"
			"audio.offload.track.enable=true\n"
			"audio.deep_buffer.media=true\n"
			"audio.heap.size.multiplier=7\n"
			"use.voice.path.for.pcm.voip=true\n"
			"audio.offload.multiaac.enable=true\n"
			"audio.dolby.ds2.enabled=true\n"
			"audio.dolby.ds2.hardbypass=true\n"
			"audio.offload.multiple.enabled=false\n"
			"audio.offload.passthrough=true\n"
			"ro.qc.sdk.audio.ssr=false\n"
			"audio.offload.gapless.enabled=false\n"
			"audio.safx.pbe.enabled=true\n"
			"audio.parser.ip.buffer.size=262144\n"
			"flac.sw.decoder.24bit.support=true\n"
			"persist.bt.a2dp_offload_cap=sbc-aptx\n"
			"use.qti.sw.alac.decoder=true\n"
			"use.qti.sw.ape.decoder=true\n"
			"qcom.hw.aac.encoder=true\n"
			"fm.a2dp.conc.disabled=false\n"
			"audio.noisy.broadcast.delay=600\n"
			"ro.build.scafe.version=2017A\n"
			"ro.error.receiver.default=com.samsung.receiver.error\n"
			"ro.frp.pst=/dev/block/persistent\n"
			"ro.hdcp2.rx=tz\n"
			"ro.securestorage.support=true\n"
			"ro.wsmd.enable=true\n"
			"ro.mst.support=1\n"
			"security.mdpp.mass=skmm\n"
			"security.mdpp=None\n"
			"ro.security.mdpp.ver=3.0\n"
			"ro.security.mdpp.release=1\n"
			"ro.security.wlan.ver=1.0\n"
			"ro.security.wlan.release=1\n"
			"security.mdpp.result=None\n"
			"ro.hardware.keystore=mdfpp\n"
			"ro.hardware.gatekeeper=mdfpp\n"
			"ro.security.vpnpp.ver=1.4\n"
			"ro.security.vpnpp.release=8.1\n"
			"ro.security.mdpp.ux=Enabled\n"
			"sys.config.amp_perf_enable=true\n"
			"ro.config.dha_cached_min=6\n"
			"ro.config.dha_cached_max=16\n"
			"ro.config.dha_empty_min=8\n"
			"ro.config.dha_empty_init=32\n"
			"ro.config.dha_empty_max=32\n"
			"ro.config.dha_th_rate=2.0\n"
			"ro.config.dha_pwhitelist_enable=1\n"
			"ro.config.dha_pwhl_key=7938\n"
			"ro.config.fall_prevent_enable=true\n"
			"ro.config.infinite_bg_enable=true\n"
			"ro.sec.ice.key_update=true\n"
			"ro.config.ringtone=Over_the_Horizon.ogg\n"
			"ro.config.notification_sound=Skyline.ogg\n"
			"ro.config.alarm_alert=Morning_Glory.ogg\n"
			"ro.config.media_sound=Media_preview_Touch_the_light.ogg\n"
			"ro.config.ringtone_2=Basic_Bell.ogg\n"
			"ro.config.notification_sound_2=S_Charming_Bell.ogg\n"
			"ro.gfx.driver.0=com.samsung.gpudriver.S8Adreno540_70\n"
			"ro.hardware.egl=adreno\n"
			"ro.setupwizard.mode=OPTIONAL\n"
			"ro.com.google.gmsversion=7.0_r4\n"
			"ro.opa.eligible_device=true\n"
			"ro.build.selinux=1\n"
			"persist.sys.dalvik.vm.lib.2=libart.so\n"
			"dalvik.vm.isa.arm64.variant=generic\n"
			"dalvik.vm.isa.arm64.features=default\n"
			"dalvik.vm.isa.arm.variant=cortex-a9\n"
			"dalvik.vm.isa.arm.features=default\n"
			"ro.config.knox=v30\n"
			"ro.config.tima=1\n"
			"ro.config.timaversion=3.0\n"
			"ro.config.iccc_version=1.0\n"
			"ro.kernel.qemu=0\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"ro.mdtp.package_name2=com.qualcomm.qti.securemsm.mdtp.MdtpDemo\n"
			"ro.build.version.sem=2403\n"
			"ro.build.version.sep=80100\n"
			"ro.expect.recovery_id=0x8552e40588718421e1b203e068e9106d55fdab7d000000000000000000000000\n"
			"\n",
	},
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
		.content = "5\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/dev",
		.size = 6,
		.content = "236:0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/available_frequencies",
		.size = 80,
		.content = "180000000 257000000 342000000 414000000 515000000 596000000 670000000 710000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/available_governors",
		.size = 163,
		.content = "spdm_bw_hyp mem_latency bw_hwmon msm-vidc-vmem+ msm-vidc-vmem msm-vidc-ddr bw_vbif gpubw_mon cpufreq msm-adreno-tz userspace powersave performance simple_ondemand\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
		.size = 10,
		.content = "257000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/governor",
		.size = 14,
		.content = "msm-adreno-tz\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/gpu_load",
		.size = 2,
		.content = "6\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/max_freq",
		.size = 10,
		.content = "670000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/min_freq",
		.size = 10,
		.content = "257000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/polling_interval",
		.size = 3,
		.content = "10\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/suspend_time",
		.size = 7,
		.content = "566611\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/target_freq",
		.size = 10,
		.content = "257000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/trans_stat",
		.size = 533,
		.content =
			"   From  :   To\n"
			"         :670000000596000000515000000414000000342000000257000000   time(ms)\n"
			" 670000000:       0       2       0       0       0       0       750\n"
			" 596000000:       0       0       2       0       0       0       180\n"
			" 515000000:       0       0       0       0       0       2       980\n"
			" 414000000:       0       0       0       0       0       0         0\n"
			" 342000000:       0       0       0       0       0       0         0\n"
			"*257000000:       2       0       0       0       0       0   8001780\n"
			"Total transition : 8\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/freq_table_mhz",
		.size = 25,
		.content = "670 596 515 414 342 257 \n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/ft_fast_hang_detect",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/ft_hang_intr_status",
		.size = 2,
		.content = "1\n",
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
		.size = 61,
		.content = "670000000 596000000 515000000 414000000 342000000 257000000 \n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
		.size = 5,
		.content = "10 %\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_clock_stats",
		.size = 31,
		.content = "153582 25967 9621 0 0 6205976 \n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_model",
		.size = 12,
		.content = "Adreno540v2\n",
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
		.content = "1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/max_gpuclk",
		.size = 10,
		.content = "670000000\n",
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
		.content = "5\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/num_pwrlevels",
		.size = 2,
		.content = "6\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/pmqos_active_latency",
		.size = 4,
		.content = "501\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/popp",
		.size = 2,
		.content = "0\n",
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
		.size = 5,
		.content = "2953\n",
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
		.size = 18,
		.content =
			"dreamqltesq-user\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/image_version",
		.size = 25,
		.content =
			"10:NRD90M:G950USQU1AQC9\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/images",
		.size = 613,
		.content =
			"0:\n"
			"\tCRM:\t\t00:BOOT.XF.1.2-00322\n"
			"\tVariant:\tMsm8998LA\n"
			"\tVersion:\t:SWHE7721\n"
			"1:\n"
			"\tCRM:\t\t01:TZ.BF.4.0.6.C1-00028\n"
			"\tVariant:\t \n"
			"\tVersion:\t:CRM\n"
			"3:\n"
			"\tCRM:\t\t03:RPM.BF.1.7.C2-00007\n"
			"\tVariant:\tAAAAANAZR\n"
			"\tVersion:\t:SWHE7721\n"
			"10:\n"
			"\tCRM:\t\t10:NRD90M:G950USQU1AQC9\n"
			"\n"
			"\tVariant:\tdreamqltesq-user\n"
			"\n"
			"\tVersion:\tREL\n"
			"\n"
			"11:\n"
			"\tCRM:\t\t11:MPSS.AT.2.0.C2.2-87307\n"
			"\tVariant:\t8998.gen.prodQ\n"
			"\tVersion:\t:SWDG4503-VM02\n"
			"12:\n"
			"\tCRM:\t\t12:ADSP.HT.3.0.c2-00032-CB8998-1\n"
			"\tVariant:\tAAAAAAAAQ\n"
			"\tVersion:\t:SWDG4503-VM02\n"
			"14:\n"
			"\tCRM:\t\t14:VIDEO.VE.4.2-00046\n"
			"\tVariant:\tPROD\n"
			"\tVersion:\t\n"
			"15:\n"
			"\tCRM:\t\t15:SLPI.HB.2.0.c2-00009-M8998AZL-1.88149.3\n"
			"\tVariant:\tAAAAAAAAQ\n"
			"\tVersion:\t:SWHE7721\n",
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
		.content = "3582555872\n",
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
		.path = "/sys/devices/system/cpu/cpu0/core_ctl/global_state",
		.size = 1008,
		.content =
			"CPU0\n"
			"\tCPU: 0\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 1\n"
			"\tIs busy: 1\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tBoost: 0\n"
			"CPU1\n"
			"\tCPU: 1\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 0\n"
			"\tIs busy: 1\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tBoost: 0\n"
			"CPU2\n"
			"\tCPU: 2\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 0\n"
			"\tIs busy: 1\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tBoost: 0\n"
			"CPU3\n"
			"\tCPU: 3\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 0\n"
			"\tIs busy: 1\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tBoost: 0\n"
			"CPU4\n"
			"\tCPU: 4\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU5\n"
			"\tCPU: 5\n"
			"\tOnline: 1\n"
			"\tActive: 0\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU6\n"
			"\tCPU: 6\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU7\n"
			"\tCPU: 7\n"
			"\tOnline: 1\n"
			"\tActive: 0\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
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
			"\tCPU:0 0\n"
			"\tCPU:1 0\n"
			"\tCPU:2 0\n"
			"\tCPU:3 0\n",
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
		.content = "595200\n",
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
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 282,
		.content =
			"300000 592725\n"
			"364800 11362\n"
			"441600 3149\n"
			"518400 2763\n"
			"595200 2209\n"
			"672000 1661\n"
			"748800 1772\n"
			"825600 1916\n"
			"883200 1551\n"
			"960000 2374\n"
			"1036800 2238\n"
			"1094400 1659\n"
			"1171200 41468\n"
			"1248000 13041\n"
			"1324800 1904\n"
			"1401600 4057\n"
			"1478400 2750\n"
			"1555200 1331\n"
			"1670400 2015\n"
			"1747200 3649\n"
			"1824000 6966\n"
			"1900800 98490\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 6,
		.content = "45130\n",
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
		.content = "300000\n",
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
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 282,
		.content =
			"300000 593150\n"
			"364800 11370\n"
			"441600 3153\n"
			"518400 2765\n"
			"595200 2209\n"
			"672000 1661\n"
			"748800 1776\n"
			"825600 1918\n"
			"883200 1551\n"
			"960000 2374\n"
			"1036800 2238\n"
			"1094400 1659\n"
			"1171200 41492\n"
			"1248000 13041\n"
			"1324800 1906\n"
			"1401600 4059\n"
			"1478400 2750\n"
			"1555200 1331\n"
			"1670400 2015\n"
			"1747200 3651\n"
			"1824000 6970\n"
			"1900800 98497\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 6,
		.content = "45157\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/isolate",
		.size = 2,
		.content = "0\n",
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
		.size = 8,
		.content = "1171200\n",
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
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 282,
		.content =
			"300000 593536\n"
			"364800 11384\n"
			"441600 3159\n"
			"518400 2767\n"
			"595200 2211\n"
			"672000 1661\n"
			"748800 1780\n"
			"825600 1918\n"
			"883200 1551\n"
			"960000 2374\n"
			"1036800 2238\n"
			"1094400 1659\n"
			"1171200 41506\n"
			"1248000 13041\n"
			"1324800 1906\n"
			"1401600 4059\n"
			"1478400 2750\n"
			"1555200 1331\n"
			"1670400 2015\n"
			"1747200 3651\n"
			"1824000 6970\n"
			"1900800 98497\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 6,
		.content = "45182\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/isolate",
		.size = 2,
		.content = "0\n",
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
		.content = "300000\n",
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
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_max_freq",
		.size = 8,
		.content = "1900800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "300000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 282,
		.content =
			"300000 593868\n"
			"364800 11386\n"
			"441600 3165\n"
			"518400 2777\n"
			"595200 2213\n"
			"672000 1665\n"
			"748800 1788\n"
			"825600 1922\n"
			"883200 1555\n"
			"960000 2376\n"
			"1036800 2242\n"
			"1094400 1661\n"
			"1171200 41542\n"
			"1248000 13045\n"
			"1324800 1906\n"
			"1401600 4065\n"
			"1478400 2752\n"
			"1555200 1331\n"
			"1670400 2015\n"
			"1747200 3653\n"
			"1824000 6972\n"
			"1900800 98497\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 6,
		.content = "45235\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/isolate",
		.size = 2,
		.content = "0\n",
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
		.path = "/sys/devices/system/cpu/cpu4/core_ctl/global_state",
		.size = 1008,
		.content =
			"CPU0\n"
			"\tCPU: 0\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 2\n"
			"\tIs busy: 1\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tBoost: 0\n"
			"CPU1\n"
			"\tCPU: 1\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 0\n"
			"\tIs busy: 1\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tBoost: 0\n"
			"CPU2\n"
			"\tCPU: 2\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 0\n"
			"\tIs busy: 1\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tBoost: 0\n"
			"CPU3\n"
			"\tCPU: 3\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 0\n"
			"\tBusy%: 0\n"
			"\tIs busy: 1\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 4\n"
			"\tNeed CPUs: 4\n"
			"\tBoost: 0\n"
			"CPU4\n"
			"\tCPU: 4\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU5\n"
			"\tCPU: 5\n"
			"\tOnline: 1\n"
			"\tActive: 0\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU6\n"
			"\tCPU: 6\n"
			"\tOnline: 1\n"
			"\tActive: 1\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
			"\tBoost: 0\n"
			"CPU7\n"
			"\tCPU: 7\n"
			"\tOnline: 1\n"
			"\tActive: 0\n"
			"\tFirst CPU: 4\n"
			"\tBusy%: 0\n"
			"\tIs busy: 0\n"
			"\tNr running: 0\n"
			"\tActive CPUs: 2\n"
			"\tNeed CPUs: 2\n"
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
			"\tCPU:4 0\n"
			"\tCPU:6 0\n"
			"\tCPU:5 0\n"
			"\tCPU:7 0\n",
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
		.content = "2361600\n",
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
		.size = 231,
		.content = "300000 345600 422400 499200 576000 652800 729600 806400 902400 979200 1056000 1132800 1190400 1267200 1344000 1420800 1497600 1574400 1651200 1728000 1804800 1881600 1958400 2035200 2112000 2208000 2265600 2323200 2342400 2361600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "902400\n",
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
		.content = "902400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 351,
		.content =
			"300000 38889\n"
			"345600 23\n"
			"422400 41\n"
			"499200 26\n"
			"576000 67\n"
			"652800 73\n"
			"729600 59\n"
			"806400 46\n"
			"902400 742666\n"
			"979200 310\n"
			"1056000 593\n"
			"1132800 249\n"
			"1190400 181\n"
			"1267200 202\n"
			"1344000 242\n"
			"1420800 276\n"
			"1497600 1249\n"
			"1574400 4755\n"
			"1651200 572\n"
			"1728000 458\n"
			"1804800 320\n"
			"1881600 369\n"
			"1958400 1012\n"
			"2035200 689\n"
			"2112000 775\n"
			"2208000 365\n"
			"2265600 204\n"
			"2323200 106\n"
			"2342400 45\n"
			"2361600 7956\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 5,
		.content = "7626\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/isolate",
		.size = 2,
		.content = "1\n",
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
		.content = "2361600\n",
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
		.size = 231,
		.content = "300000 345600 422400 499200 576000 652800 729600 806400 902400 979200 1056000 1132800 1190400 1267200 1344000 1420800 1497600 1574400 1651200 1728000 1804800 1881600 1958400 2035200 2112000 2208000 2265600 2323200 2342400 2361600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "902400\n",
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
		.content = "902400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 351,
		.content =
			"300000 38889\n"
			"345600 23\n"
			"422400 41\n"
			"499200 26\n"
			"576000 67\n"
			"652800 73\n"
			"729600 59\n"
			"806400 46\n"
			"902400 743103\n"
			"979200 310\n"
			"1056000 593\n"
			"1132800 249\n"
			"1190400 181\n"
			"1267200 202\n"
			"1344000 242\n"
			"1420800 276\n"
			"1497600 1249\n"
			"1574400 4755\n"
			"1651200 572\n"
			"1728000 458\n"
			"1804800 320\n"
			"1881600 369\n"
			"1958400 1012\n"
			"2035200 689\n"
			"2112000 775\n"
			"2208000 365\n"
			"2265600 204\n"
			"2323200 106\n"
			"2342400 45\n"
			"2361600 7956\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 5,
		.content = "7626\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/isolate",
		.size = 2,
		.content = "0\n",
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
		.content = "2361600\n",
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
		.size = 231,
		.content = "300000 345600 422400 499200 576000 652800 729600 806400 902400 979200 1056000 1132800 1190400 1267200 1344000 1420800 1497600 1574400 1651200 1728000 1804800 1881600 1958400 2035200 2112000 2208000 2265600 2323200 2342400 2361600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "902400\n",
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
		.content = "902400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 351,
		.content =
			"300000 38889\n"
			"345600 23\n"
			"422400 41\n"
			"499200 26\n"
			"576000 67\n"
			"652800 73\n"
			"729600 59\n"
			"806400 46\n"
			"902400 743514\n"
			"979200 310\n"
			"1056000 593\n"
			"1132800 249\n"
			"1190400 181\n"
			"1267200 204\n"
			"1344000 244\n"
			"1420800 276\n"
			"1497600 1249\n"
			"1574400 4757\n"
			"1651200 572\n"
			"1728000 458\n"
			"1804800 320\n"
			"1881600 369\n"
			"1958400 1012\n"
			"2035200 689\n"
			"2112000 775\n"
			"2208000 365\n"
			"2265600 204\n"
			"2323200 106\n"
			"2342400 45\n"
			"2361600 7956\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 5,
		.content = "7631\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/isolate",
		.size = 2,
		.content = "0\n",
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
		.content = "2361600\n",
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
		.size = 231,
		.content = "300000 345600 422400 499200 576000 652800 729600 806400 902400 979200 1056000 1132800 1190400 1267200 1344000 1420800 1497600 1574400 1651200 1728000 1804800 1881600 1958400 2035200 2112000 2208000 2265600 2323200 2342400 2361600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "902400\n",
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
		.content = "902400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 351,
		.content =
			"300000 38889\n"
			"345600 23\n"
			"422400 41\n"
			"499200 26\n"
			"576000 67\n"
			"652800 73\n"
			"729600 59\n"
			"806400 46\n"
			"902400 743938\n"
			"979200 310\n"
			"1056000 593\n"
			"1132800 249\n"
			"1190400 181\n"
			"1267200 204\n"
			"1344000 244\n"
			"1420800 276\n"
			"1497600 1249\n"
			"1574400 4757\n"
			"1651200 572\n"
			"1728000 458\n"
			"1804800 320\n"
			"1881600 369\n"
			"1958400 1012\n"
			"2035200 689\n"
			"2112000 775\n"
			"2208000 365\n"
			"2265600 204\n"
			"2323200 106\n"
			"2342400 45\n"
			"2361600 7956\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 5,
		.content = "7631\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/isolate",
		.size = 2,
		.content = "1\n",
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
		.key = "audio.adm.buffering.ms",
		.value = "3",
	},
	{
		.key = "audio.deep_buffer.media",
		.value = "true",
	},
	{
		.key = "audio.dolby.ds2.enabled",
		.value = "true",
	},
	{
		.key = "audio.dolby.ds2.hardbypass",
		.value = "true",
	},
	{
		.key = "audio.heap.size.multiplier",
		.value = "7",
	},
	{
		.key = "audio.noisy.broadcast.delay",
		.value = "600",
	},
	{
		.key = "audio.offload.buffer.size.kb",
		.value = "32",
	},
	{
		.key = "audio.offload.gapless.enabled",
		.value = "false",
	},
	{
		.key = "audio.offload.multiaac.enable",
		.value = "true",
	},
	{
		.key = "audio.offload.multiple.enabled",
		.value = "false",
	},
	{
		.key = "audio.offload.passthrough",
		.value = "true",
	},
	{
		.key = "audio.offload.pcm.16bit.enable",
		.value = "true",
	},
	{
		.key = "audio.offload.pcm.24bit.enable",
		.value = "true",
	},
	{
		.key = "audio.offload.track.enable",
		.value = "true",
	},
	{
		.key = "audio.offload.video",
		.value = "true",
	},
	{
		.key = "audio.parser.ip.buffer.size",
		.value = "262144",
	},
	{
		.key = "audio.safx.pbe.enabled",
		.value = "true",
	},
	{
		.key = "audio_hal.period_size",
		.value = "192",
	},
	{
		.key = "audioflinger.bootsnd",
		.value = "0",
	},
	{
		.key = "boot.sfbootcomplete",
		.value = "0",
	},
	{
		.key = "bt.max.hfpclient.connections",
		.value = "1",
	},
	{
		.key = "config.disable_atlas",
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
		.key = "debug.batt.no_battery",
		.value = "true",
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
		.key = "debug.qualcomm.sns.daemon",
		.value = "I",
	},
	{
		.key = "debug.qualcomm.sns.libsensor1",
		.value = "I",
	},
	{
		.key = "debug.sensor.logging.slpi",
		.value = "true",
	},
	{
		.key = "debug.sf.hw",
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
		.key = "dev.kies.deviceowner",
		.value = "0",
	},
	{
		.key = "dev.kies.drivedisplay",
		.value = "0",
	},
	{
		.key = "dev.kies.drivedisplay.trust",
		.value = "1",
	},
	{
		.key = "dev.kies.sommode",
		.value = "TRUE",
	},
	{
		.key = "dev.kiessupport",
		.value = "TRUE",
	},
	{
		.key = "dev.knoxapp.running",
		.value = "false",
	},
	{
		.key = "dev.mtp.opensession",
		.value = "0",
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
		.key = "dev.ssrm.gamelevel",
		.value = "-4,6,-3,-3",
	},
	{
		.key = "dev.ssrm.init",
		.value = "1",
	},
	{
		.key = "dev.ssrm.live_thumbnail",
		.value = "1",
	},
	{
		.key = "dev.ssrm.mode",
		.value = "",
	},
	{
		.key = "dev.ssrm.pst",
		.value = "310",
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
		.key = "flac.sw.decoder.24bit.support",
		.value = "true",
	},
	{
		.key = "fm.a2dp.conc.disabled",
		.value = "false",
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
		.key = "gsm.operator.ispsroaming",
		.value = "false",
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
		.value = "G950USQU1AQC9",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "Samsung RIL v3.0",
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
		.key = "init.svc.SIDESYNC_service",
		.value = "running",
	},
	{
		.key = "init.svc.SMD-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.TvoutService_C",
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
		.key = "init.svc.at_distributor",
		.value = "stopped",
	},
	{
		.key = "init.svc.atfwd",
		.value = "running",
	},
	{
		.key = "init.svc.audiod",
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
		.key = "init.svc.bootanim",
		.value = "stopped",
	},
	{
		.key = "init.svc.bootchecker",
		.value = "stopped",
	},
	{
		.key = "init.svc.cameraserver",
		.value = "running",
	},
	{
		.key = "init.svc.ccm",
		.value = "running",
	},
	{
		.key = "init.svc.compact_memory",
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
		.key = "init.svc.debuggerd",
		.value = "running",
	},
	{
		.key = "init.svc.debuggerd64",
		.value = "running",
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
		.key = "init.svc.energy-awareness",
		.value = "stopped",
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
		.key = "init.svc.factory_adsp",
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
		.key = "init.svc.healthd",
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
		.key = "init.svc.iop",
		.value = "running",
	},
	{
		.key = "init.svc.ipacm",
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
		.key = "init.svc.keystore",
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
		.key = "init.svc.otp",
		.value = "running",
	},
	{
		.key = "init.svc.p2p_supplicant",
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
		.key = "init.svc.powersnd",
		.value = "stopped",
	},
	{
		.key = "init.svc.prepare_param",
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
		.key = "init.svc.qseecomd",
		.value = "running",
	},
	{
		.key = "init.svc.qti-testscripts",
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
		.key = "init.svc.sec-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.secure_storage",
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
		.key = "init.svc.sensorhubservice",
		.value = "running",
	},
	{
		.key = "init.svc.sensors",
		.value = "running",
	},
	{
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.ss_ramdump",
		.value = "stopped",
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
		.key = "init.svc.time_daemon",
		.value = "running",
	},
	{
		.key = "init.svc.ueventd",
		.value = "running",
	},
	{
		.key = "init.svc.visiond",
		.value = "running",
	},
	{
		.key = "init.svc.vold",
		.value = "running",
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
		.key = "keyguard.no_require_sim",
		.value = "true",
	},
	{
		.key = "logd.auditd",
		.value = "false",
	},
	{
		.key = "logd.kernel",
		.value = "false",
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
		.value = "1048575",
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
		.key = "net.change",
		.value = "net.iptype",
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
		.value = "android-ad0b498ffaa8dd47",
	},
	{
		.key = "net.iptype",
		.value = "506:v4v6v6",
	},
	{
		.key = "net.knox.shareddevice.version",
		.value = "2.8.0",
	},
	{
		.key = "net.knoxscep.version",
		.value = "2.2.0",
	},
	{
		.key = "net.knoxvpn.version",
		.value = "2.4.0",
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
		.key = "net.tcp.buffersize.wifi",
		.value = "524288,2097152,4194304,262144,524288,1048576",
	},
	{
		.key = "net.tcp.default_init_rwnd",
		.value = "60",
	},
	{
		.key = "nfc.delay.boot",
		.value = "0",
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
		.value = "3",
	},
	{
		.key = "nfc.fw.rfreg_ver",
		.value = "MAJ: D, MIN: 3",
	},
	{
		.key = "nfc.fw.ver",
		.value = "NXP 11.1.d",
	},
	{
		.key = "nfc.nxp.fwdnldstatus",
		.value = "0",
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
		.key = "persist.audio.omc.ringtone",
		.value = "AT&amp;amp;T Firefly.ogg",
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
		.value = "0",
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
		.key = "persist.bluetooth_fw_ver",
		.value = "bcm4361B0_V0194.0195_murata.hcd",
	},
	{
		.key = "persist.bt.a2dp_offload_cap",
		.value = "sbc-aptx",
	},
	{
		.key = "persist.camera.debug.logfile",
		.value = "0",
	},
	{
		.key = "persist.camera.gyro.disable",
		.value = "0",
	},
	{
		.key = "persist.cne.dpm",
		.value = "0",
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
		.key = "persist.data.dpm.enable",
		.value = "true",
	},
	{
		.key = "persist.data.dropssdp",
		.value = "false",
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
		.key = "persist.debug.sensors.hal",
		.value = "I",
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
		.key = "persist.dpm.feature",
		.value = "0",
	},
	{
		.key = "persist.eons.enabled",
		.value = "true",
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
		.key = "persist.radio.add_power_save",
		.value = "1",
	},
	{
		.key = "persist.radio.apm_sim_not_pwdn",
		.value = "1",
	},
	{
		.key = "persist.radio.custom_ecc",
		.value = "1",
	},
	{
		.key = "persist.radio.initphone-type",
		.value = "1",
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
		.key = "persist.radio.new.profid",
		.value = "true",
	},
	{
		.key = "persist.radio.plmnname",
		.value = "",
	},
	{
		.key = "persist.radio.sib16_support",
		.value = "0",
	},
	{
		.key = "persist.radio.silent-reset",
		.value = "41",
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
		.value = "MSM8998",
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
		.key = "persist.service.bdroid.version",
		.value = "5.0",
	},
	{
		.key = "persist.service.bgkeycount",
		.value = "",
	},
	{
		.key = "persist.service.tspcmd.spay",
		.value = "true",
	},
	{
		.key = "persist.sys.ccm.date",
		.value = "Sat Mar 11 22:44:17 KST 2017",
	},
	{
		.key = "persist.sys.clipboardedge.intro",
		.value = "false",
	},
	{
		.key = "persist.sys.clssprld2",
		.value = "428",
	},
	{
		.key = "persist.sys.clssprld3",
		.value = "1149",
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
		.key = "persist.sys.debug_omc",
		.value = "/system/omc/ATT",
	},
	{
		.key = "persist.sys.debug_omcnw",
		.value = "/system/omc/ATT",
	},
	{
		.key = "persist.sys.display_density",
		.value = "480",
	},
	{
		.key = "persist.sys.force_sw_gles",
		.value = "0",
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
		.key = "persist.sys.omc.enable",
		.value = "false",
	},
	{
		.key = "persist.sys.omc_etcpath",
		.value = "/system/omc/ATT/etc",
	},
	{
		.key = "persist.sys.omc_path",
		.value = "/system/omc/ATT",
	},
	{
		.key = "persist.sys.omc_respath",
		.value = "/system/omc/ATT/res",
	},
	{
		.key = "persist.sys.omc_support",
		.value = "true",
	},
	{
		.key = "persist.sys.omcnw_path",
		.value = "/system/omc/ATT",
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
		.key = "persist.sys.sm_mode",
		.value = "1",
	},
	{
		.key = "persist.sys.ssr.enable_ramdumps",
		.value = "0",
	},
	{
		.key = "persist.sys.storage_preload",
		.value = "2",
	},
	{
		.key = "persist.sys.timezone",
		.value = "Asia/Seoul",
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
		.key = "qcom.bluetooth.soc",
		.value = "cherokee",
	},
	{
		.key = "qcom.hw.aac.encoder",
		.value = "true",
	},
	{
		.key = "qemu.hw.mainkeys",
		.value = "0",
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
		.key = "ril.RildInit",
		.value = "1",
	},
	{
		.key = "ril.airplane.mode",
		.value = "1",
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
		.key = "ril.cs_svc",
		.value = "1",
	},
	{
		.key = "ril.data.intfprefix",
		.value = "rmnet_data",
	},
	{
		.key = "ril.debug_modemfactory",
		.value = "CSC Feature State: IMS ON, EPDG ON",
	},
	{
		.key = "ril.ecclist0",
		.value = "911,112,*911,#911,000,08,110,999,118,119",
	},
	{
		.key = "ril.ecclist00",
		.value = "112,911,999,000,08,110,118,119",
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
		.value = "REV1.0",
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
		.key = "ril.isctc",
		.value = "0",
	},
	{
		.key = "ril.manufacturedate",
		.value = "20170420",
	},
	{
		.key = "ril.modem.board",
		.value = "MSM8998",
	},
	{
		.key = "ril.official_cscver",
		.value = "G950UOYN1AQC9",
	},
	{
		.key = "ril.product_code",
		.value = "SM-G950UZKAATT",
	},
	{
		.key = "ril.radiostate",
		.value = "0",
	},
	{
		.key = "ril.rfcal_date",
		.value = "2017.04.22",
	},
	{
		.key = "ril.serialnumber",
		.value = "R38J40MVEZR",
	},
	{
		.key = "ril.servicestate",
		.value = "3",
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
		.value = "G950USQU1AQC9",
	},
	{
		.key = "ril.twwan911Timer",
		.value = "40",
	},
	{
		.key = "ril.voicecapable",
		.value = "true",
	},
	{
		.key = "rild.libpath",
		.value = "/system/lib64/libsec-ril.so",
	},
	{
		.key = "ro.adb.secure",
		.value = "1",
	},
	{
		.key = "ro.alarm_boot",
		.value = "false",
	},
	{
		.key = "ro.allow.mock.location",
		.value = "0",
	},
	{
		.key = "ro.astcenc.astcsupport",
		.value = "1",
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
		.key = "ro.bluetooth.tty",
		.value = "ttyHS0",
	},
	{
		.key = "ro.bluetooth.wipower",
		.value = "true",
	},
	{
		.key = "ro.board.platform",
		.value = "msm8998",
	},
	{
		.key = "ro.boot.baseband",
		.value = "msm",
	},
	{
		.key = "ro.boot.boot_recovery",
		.value = "0",
	},
	{
		.key = "ro.boot.bootdevice",
		.value = "1da4000.ufshc",
	},
	{
		.key = "ro.boot.bootloader",
		.value = "G950USQU1AQC9",
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
		.key = "ro.boot.cp_reserved_mem",
		.value = "off",
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
		.key = "ro.boot.em.did",
		.value = "205ED58976E0",
	},
	{
		.key = "ro.boot.em.model",
		.value = "SM-G950U",
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
		.key = "ro.boot.revision",
		.value = "12",
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
		.value = "988837435645343543",
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
		.key = "ro.boot.ucs_mode",
		.value = "0",
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
		.value = "Sat Mar 11 22:44:17 KST 2017",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1489239857",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "samsung/dreamqltesq/dreamqltesq:7.0/NRD90M/G950USQU1AQC9:user/test-keys",
	},
	{
		.key = "ro.bootloader",
		.value = "G950USQU1AQC9",
	},
	{
		.key = "ro.bootmode",
		.value = "unknown",
	},
	{
		.key = "ro.bt.bdaddr_path",
		.value = "/efs/bluetooth/bt_addr",
	},
	{
		.key = "ro.build.PDA",
		.value = "G950USQU1AQC9",
	},
	{
		.key = "ro.build.changelist",
		.value = "10895874",
	},
	{
		.key = "ro.build.characteristics",
		.value = "default",
	},
	{
		.key = "ro.build.date",
		.value = "Sat Mar 11 22:44:17 KST 2017",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1489239857",
	},
	{
		.key = "ro.build.description",
		.value = "dreamqltesq-user 7.0 NRD90M G950USQU1AQC9 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "NRD90M.G950USQU1AQC9",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "samsung/dreamqltesq/dreamqltesq:7.0/NRD90M/G950USQU1AQC9:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "dreamqltesq-user",
	},
	{
		.key = "ro.build.host",
		.value = "SWHE7721",
	},
	{
		.key = "ro.build.id",
		.value = "NRD90M",
	},
	{
		.key = "ro.build.official.release",
		.value = "true",
	},
	{
		.key = "ro.build.product",
		.value = "dreamqltesq",
	},
	{
		.key = "ro.build.scafe.version",
		.value = "2017A",
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
		.value = "G950USQU1AQC9",
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
		.value = "2017-03-01",
	},
	{
		.key = "ro.build.version.sem",
		.value = "2403",
	},
	{
		.key = "ro.build.version.sep",
		.value = "80100",
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
		.key = "ro.chipname",
		.value = "MSM8998",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-samsung",
	},
	{
		.key = "ro.com.google.clientidbase.am",
		.value = "android-att-us",
	},
	{
		.key = "ro.com.google.clientidbase.gmm",
		.value = "android-samsung",
	},
	{
		.key = "ro.com.google.clientidbase.ms",
		.value = "android-att-us",
	},
	{
		.key = "ro.com.google.clientidbase.yt",
		.value = "android-samsung",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "7.0_r4",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Morning_Glory.ogg",
	},
	{
		.key = "ro.config.dha_cached_max",
		.value = "16",
	},
	{
		.key = "ro.config.dha_cached_min",
		.value = "6",
	},
	{
		.key = "ro.config.dha_empty_init",
		.value = "32",
	},
	{
		.key = "ro.config.dha_empty_max",
		.value = "32",
	},
	{
		.key = "ro.config.dha_empty_min",
		.value = "8",
	},
	{
		.key = "ro.config.dha_pwhitelist_enable",
		.value = "1",
	},
	{
		.key = "ro.config.dha_pwhl_key",
		.value = "7938",
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
		.value = "1.0",
	},
	{
		.key = "ro.config.infinite_bg_enable",
		.value = "true",
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
		.key = "ro.cp_debug_level",
		.value = "0x55FF",
	},
	{
		.key = "ro.crypto.fs_crypto_blkdev",
		.value = "/dev/block/dm-1",
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
		.key = "ro.ddr_start_type",
		.value = "1",
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
		.key = "ro.em.did",
		.value = "205ED58976E0",
	},
	{
		.key = "ro.em.model",
		.value = "SM-G950U",
	},
	{
		.key = "ro.em.status",
		.value = "0x0",
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
		.value = "0x8552e40588718421e1b203e068e9106d55fdab7d000000000000000000000000",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/persistent",
	},
	{
		.key = "ro.gfx.driver.0",
		.value = "com.samsung.gpudriver.S8Adreno540_70",
	},
	{
		.key = "ro.gpu.available_frequencies",
		.value = "670000000 596000000 515000000 414000000 342000000 257000000 ",
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
		.key = "ro.input.resamplelatency",
		.value = "1",
	},
	{
		.key = "ro.kernel.qemu",
		.value = "0",
	},
	{
		.key = "ro.knox.enhance.zygote.aslr",
		.value = "0",
	},
	{
		.key = "ro.mct.compressiontype",
		.value = "ETC1",
	},
	{
		.key = "ro.mdtp.package_name2",
		.value = "com.qualcomm.qti.securemsm.mdtp.MdtpDemo",
	},
	{
		.key = "ro.me.param.offset",
		.value = "9437312",
	},
	{
		.key = "ro.mst.support",
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
		.key = "ro.opa.eligible_device",
		.value = "true",
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
		.value = "msm8998",
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
		.value = "dreamqltesq",
	},
	{
		.key = "ro.product.first_api_level",
		.value = "24",
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
		.value = "SM-G950U",
	},
	{
		.key = "ro.product.name",
		.value = "dreamqltesq",
	},
	{
		.key = "ro.product_ship",
		.value = "true",
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
		.key = "ro.qti.sensors.cmc",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.dpc",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.facing",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.fast_amd",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.georv",
		.value = "false",
	},
	{
		.key = "ro.qti.sensors.scrn_ortn",
		.value = "false",
	},
	{
		.key = "ro.qualcomm.cabl",
		.value = "0",
	},
	{
		.key = "ro.radio.noril",
		.value = "no",
	},
	{
		.key = "ro.revision",
		.value = "12",
	},
	{
		.key = "ro.ril.network_code",
		.value = "ATT",
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
		.key = "ro.rtn_config",
		.value = "unknown",
	},
	{
		.key = "ro.runtime.firstboot",
		.value = "1524159263637",
	},
	{
		.key = "ro.sales.param.offset",
		.value = "9437648",
	},
	{
		.key = "ro.sec.ice.key_update",
		.value = "true",
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
		.key = "ro.security.icd.flagmode",
		.value = "multi",
	},
	{
		.key = "ro.security.mdpp.release",
		.value = "1",
	},
	{
		.key = "ro.security.mdpp.ux",
		.value = "Enabled",
	},
	{
		.key = "ro.security.mdpp.ver",
		.value = "3.0",
	},
	{
		.key = "ro.security.reactive.version",
		.value = "2.0.11",
	},
	{
		.key = "ro.security.vpnpp.release",
		.value = "8.1",
	},
	{
		.key = "ro.security.vpnpp.ver",
		.value = "1.4",
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
		.value = "988837435645343543",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "OPTIONAL",
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
		.key = "ro.sku.param.offset",
		.value = "9437552",
	},
	{
		.key = "ro.sn.param.offset",
		.value = "9437392",
	},
	{
		.key = "ro.telephony.call_ring.multiple",
		.value = "false",
	},
	{
		.key = "ro.telephony.default_network",
		.value = "9",
	},
	{
		.key = "ro.tether.denied",
		.value = "false",
	},
	{
		.key = "ro.use_data_netmgrd",
		.value = "true",
	},
	{
		.key = "ro.vendor.extension_library",
		.value = "libqti-perfd-client.so",
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
		.key = "secmm.codecsolution.ready",
		.value = "1",
	},
	{
		.key = "secmm.player.uhqamode",
		.value = "True",
	},
	{
		.key = "security.ASKS.policy_version",
		.value = "161228",
	},
	{
		.key = "security.mdpp",
		.value = "Ready",
	},
	{
		.key = "security.mdpp.mass",
		.value = "skmm",
	},
	{
		.key = "security.mdpp.result",
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
		.key = "selinux.policy_version",
		.value = "SEPF_SECMOBILE_7.0_0005",
	},
	{
		.key = "selinux.reload_policy",
		.value = "1",
	},
	{
		.key = "service.bootanim.exit",
		.value = "0",
	},
	{
		.key = "service.camera.client",
		.value = "",
	},
	{
		.key = "service.camera.hdmi_preview",
		.value = "0",
	},
	{
		.key = "service.camera.match.id",
		.value = "0",
	},
	{
		.key = "service.camera.running",
		.value = "0",
	},
	{
		.key = "service.camera.running_0",
		.value = "0",
	},
	{
		.key = "service.camera.samsung.enabled",
		.value = "0",
	},
	{
		.key = "service.media.powersnd",
		.value = "1",
	},
	{
		.key = "service.secureui.screeninfo",
		.value = "1080x2076",
	},
	{
		.key = "storage.mmc.size",
		.value = "63916998656",
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
		.key = "sys.aasservice.aason",
		.value = "true",
	},
	{
		.key = "sys.bartender.batterystats.ver",
		.value = "16",
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
		.key = "sys.config.amp_perf_enable",
		.value = "true",
	},
	{
		.key = "sys.config.mars_version",
		.value = "2.00",
	},
	{
		.key = "sys.config.slginfo_debug",
		.value = "false",
	},
	{
		.key = "sys.config.slginfo_dha",
		.value = "true",
	},
	{
		.key = "sys.config.slginfo_enable",
		.value = "false",
	},
	{
		.key = "sys.config.slginfo_max_count",
		.value = "1000",
	},
	{
		.key = "sys.config.slginfo_meminfo",
		.value = "true",
	},
	{
		.key = "sys.config.slginfo_vmstat",
		.value = "true",
	},
	{
		.key = "sys.dockstate",
		.value = "0",
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
		.key = "sys.enterprise.otp.version",
		.value = "2.6.0",
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
		.key = "sys.knox.exists",
		.value = "0",
	},
	{
		.key = "sys.knox.store",
		.value = "0",
	},
	{
		.key = "sys.listeners.registered",
		.value = "true",
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
		.key = "sys.qseecomd.enable",
		.value = "true",
	},
	{
		.key = "sys.reset_reason",
		.value = "N|RP",
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
		.key = "sys.ssrm.game_running",
		.value = "false",
	},
	{
		.key = "sys.ssrm.mdnie",
		.value = "-1",
	},
	{
		.key = "sys.sysctl.compact_memory",
		.value = "0",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "49950",
	},
	{
		.key = "sys.sysctl.tcp_def_init_rwnd",
		.value = "60",
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
		.key = "sys.usb.rndis.func.name",
		.value = "gsi",
	},
	{
		.key = "sys.usb.rps_mask",
		.value = "2",
	},
	{
		.key = "sys.usb.state",
		.value = "mtp,adb",
	},
	{
		.key = "sys.vs.mode",
		.value = "false",
	},
	{
		.key = "system.camera.CC.disable",
		.value = "0",
	},
	{
		.key = "telephony.lteOnCdmaDevice",
		.value = "0",
	},
	{
		.key = "tunnel.audio.encode",
		.value = "false",
	},
	{
		.key = "use.qti.sw.alac.decoder",
		.value = "true",
	},
	{
		.key = "use.qti.sw.ape.decoder",
		.value = "true",
	},
	{
		.key = "use.voice.path.for.pcm.voip",
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
	{
		.key = "wlan.driver.status",
		.value = "ok",
	},
	{
		.key = "wlan.p2p.chkintent",
		.value = "8",
	},
	{
		.key = "wlan.wfd.status",
		.value = "disconnected",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
