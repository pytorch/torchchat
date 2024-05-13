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
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 4\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 5\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 6\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 7\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc MSM8953\n",
	},
#elif CPUINFO_ARCH_ARM
	{
		.path = "/proc/cpuinfo",
		.size = 2004,
		.content =
			"Processor\t: AArch64 Processor rev 4 (aarch64)\n"
			"processor\t: 0\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 4\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 5\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 6\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 7\n"
			"BogoMIPS\t: 38.40\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt lpae evtstrm aes pmull sha1 sha2 crc32\n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 8\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc MSM8953\n",
	},
#endif
	{
		.path = "/system/build.prop",
		.size = 10355,
		.content =
			"\n"
			"# begin build properties\n"
			"# autogenerated by buildinfo.sh\n"
			"ro.build.id=NRD90M\n"
			"ro.build.display.id=AL1512-mido-build-20170803193727\n"
			"ro.build.version.incremental=V8.5.4.0.NCFMIED\n"
			"ro.build.version.sdk=24\n"
			"ro.build.version.preview_sdk=0\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.all_codenames=REL\n"
			"ro.build.version.release=7.0\n"
			"ro.build.version.security_patch=2017-07-01\n"
			"ro.build.version.base_os=\n"
			"ro.build.date=Thu Aug  3 19:37:25 WIB 2017\n"
			"ro.build.date.utc=1501763845\n"
			"ro.build.type=user\n"
			"ro.build.user=builder\n"
			"ro.build.host=mi-server\n"
			"ro.build.tags=release-keys\n"
			"ro.build.flavor=mido-user\n"
			"ro.product.model=Redmi Note 4\n"
			"ro.product.brand=xiaomi\n"
			"ro.product.name=mido\n"
			"ro.product.device=mido\n"
			"ro.product.mod_device=mido_global\n"
			"ro.product.board=msm8953\n"
			"# ro.product.cpu.abi and ro.product.cpu.abi2 are obsolete,\n"
			"# use ro.product.cpu.abilist instead.\n"
			"ro.product.cpu.abi=arm64-v8a\n"
			"ro.product.cpu.abilist=arm64-v8a,armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist32=armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist64=arm64-v8a\n"
			"ro.product.locale=en-GB\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=msm8953\n"
			"# ro.build.product is obsolete; use ro.product.device\n"
			"ro.build.product=mido\n"
			"# Do not try to parse description, fingerprint, or thumbprint\n"
			"ro.build.description=mido-user 7.0 NRD90M V8.5.4.0.NCFMIED release-keys\n"
			"ro.build.fingerprint=xiaomi/mido/mido:7.0/NRD90M/V8.5.4.0.NCFMIED:user/release-keys\n"
			"ro.build.characteristics=nosdcard\n"
			"# end build properties\n"
			"#\n"
			"# from device/xiaomi/mido/system.prop\n"
			"#\n"
			"#\n"
			"# system.prop for msm8953\n"
			"#\n"
			"\n"
			"rild.libpath=/vendor/lib64/libril-qc-qmi-1.so\n"
			"rild.libargs=-d /dev/smd0\n"
			"persist.rild.nitz_plmn=\n"
			"persist.rild.nitz_long_ons_0=\n"
			"persist.rild.nitz_long_ons_1=\n"
			"persist.rild.nitz_long_ons_2=\n"
			"persist.rild.nitz_long_ons_3=\n"
			"persist.rild.nitz_short_ons_0=\n"
			"persist.rild.nitz_short_ons_1=\n"
			"persist.rild.nitz_short_ons_2=\n"
			"persist.rild.nitz_short_ons_3=\n"
			"ril.subscription.types=NV,RUIM\n"
			"DEVICE_PROVISIONED=1\n"
			"#\n"
			"# Set network mode to (T/L/G/W/1X/EVDO, T/G/W/L) for 7+5 mode device on DSDS mode\n"
			"#\n"
			"\n"
			"debug.sf.hw=0\n"
			"debug.egl.hw=0\n"
			"persist.hwc.mdpcomp.enable=true\n"
			"debug.mdpcomp.logs=0\n"
			"dalvik.vm.heapsize=36m\n"
			"dev.pm.dyn_samplingrate=1\n"
			"persist.demo.hdmirotationlock=false\n"
			"debug.enable.sglscale=1\n"
			"debug.gralloc.enable_fb_ubwc=1\n"
			"#ro.hdmi.enable=true\n"
			"#\n"
			"# system props for the cne module\n"
			"#\n"
			"persist.cne.feature=1\n"
			"\n"
			"#\n"
			"# system props for the dpm module\n"
			"#\n"
			"persist.dpm.feature=1\n"
			"\n"
			"#system props for the MM modules\n"
			"media.msm8956hw=0\n"
			"mm.enable.smoothstreaming=true\n"
			"mmp.enable.3g2=true\n"
			"media.aac_51_output_enabled=true\n"
			"av.debug.disable.pers.cache=1\n"
			"\n"
			"#codecs:(PARSER_)AAC AC3 AMR_NB AMR_WB ASF AVI DTS FLV 3GP 3G2 MKV MP2PS MP2TS MP3 OGG QCP WAV FLAC AIFF APE\n"
			"#mm.enable.qcom_parser=1048575\n"
			"\n"
			"#\n"
			"# system props for the data modules\n"
			"#\n"
			"ro.use_data_netmgrd=true\n"
			"persist.data.netmgrd.qos.enable=true\n"
			"persist.data.mode=concurrent\n"
			"\n"
			"#system props for time-services\n"
			"persist.timed.enable=true\n"
			"\n"
			"#\n"
			"# system prop for opengles version\n"
			"#\n"
			"# 196608 is decimal for 0x30000 to report major/minor versions as 3/0\n"
			"# 196609 is decimal for 0x30001 to report major/minor versions as 3/1\n"
			"ro.opengles.version=196610\n"
			"\n"
			"# System property for cabl\n"
			"ro.qualcomm.cabl=0\n"
			"\n"
			"#\n"
			"# System props for telephony\n"
			"# System prop to turn on CdmaLTEPhone always\n"
			"telephony.lteOnCdmaDevice=1\n"
			"\n"
			"#\n"
			"# System props for bluetooh\n"
			"# System prop to turn on hfp client\n"
			"bluetooth.hfp.client=1\n"
			"\n"
			"#Simulate sdcard on /data/media\n"
			"#\n"
			"persist.fuse_sdcard=true\n"
			"\n"
			"#System property for FM transmitter\n"
			"ro.fm.transmitter=false\n"
			"\n"
			"#property to enable user to access Google WFD settings\n"
			"persist.debug.wfd.enable=1\n"
			"#property to enable VDS WFD solution\n"
			"persist.hwc.enable_vds=1\n"
			"\n"
			"#selects CoreSight configuration to enable\n"
			"persist.debug.coresight.config=stm-events\n"
			"\n"
			"#selects Console configuration to enable\n"
			"persist.console.silent.config=1\n"
			"\n"
			"#property for vendor specific library\n"
			"ro.vendor.gt_library=libqti-gt.so\n"
			"ro.vendor.at_library=libqti-at.so\n"
			"\n"
			"#property for game detection feature\n"
			"debug.enable.gamed=0\n"
			"\n"
			"#property to enable narrow search range for video encoding\n"
			"vidc.enc.disable_bframes=1\n"
			"vidc.disable.split.mode=1\n"
			"vidc.dec.downscalar_width=1920\n"
			"vidc.dec.downscalar_height=1088\n"
			"\n"
			"# disable PQ feature by default\n"
			"vidc.enc.disable.pq=true\n"
			"\n"
			"# Additional buffers shared between Camera and Video\n"
			"vidc.enc.dcvs.extra-buff-count=2\n"
			"\n"
			"# system property to accelerate Progressive Download using STA\n"
			"persist.mm.sta.enable=0\n"
			"\n"
			"#property to enable fingerprint\n"
			"persist.qfp=false\n"
			"\n"
			"#min/max cpu in core control\n"
			"ro.core_ctl_min_cpu=2\n"
			"ro.core_ctl_max_cpu=4\n"
			"\n"
			"#HWUI properties\n"
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
			"#Enable B service adj transition by default\n"
			"ro.sys.fw.bservice_enable=true\n"
			"ro.sys.fw.bservice_limit=5\n"
			"ro.sys.fw.bservice_age=5000\n"
			"\n"
			"#Memperf properties\n"
			"ro.memperf.lib=libmemperf.so\n"
			"ro.memperf.enable=false\n"
			"\n"
			"#Trim properties\n"
			"ro.sys.fw.use_trim_settings=true\n"
			"ro.sys.fw.empty_app_percent=50\n"
			"ro.sys.fw.trim_empty_percent=100\n"
			"ro.sys.fw.trim_cache_percent=100\n"
			"ro.sys.fw.trim_enable_memory=2147483648\n"
			"\n"
			"# Enable Delay Service Restart\n"
			"ro.am.reschedule_service=true\n"
			"\n"
			"#Optimal dex2oat threads for faster app installation\n"
			"ro.sys.fw.dex2oat_thread_count=4\n"
			"\n"
			"# Create zram disk\n"
			"#ro.config.zram=true\n"
			"\n"
			"# set cutoff voltage to 3400mV\n"
			"ro.cutoff_voltage_mv=3400\n"
			"\n"
			"#set device emmc size\n"
			"ro.emmc_size=16GB\n"
			"\n"
			"#force HAL1 for below packages\n"
			"camera.hal1.packagelist=com.skype.raider,com.google.android.talk\n"
			"\n"
			"#Enable FR27607-RIL to send ONLINE cmd in bootup\n"
			"#persist.radio.poweron_opt=1\n"
			"\n"
			"#low power mode for camera\n"
			"camera.lowpower.record.enable=1\n"
			"\n"
			"#In video expect camera time source as monotonic\n"
			"media.camera.ts.monotonic=1\n"
			"\n"
			"#properties for limiting preview size in camera\n"
			"camera.display.umax=1920x1080\n"
			"camera.display.lmax=1280x720\n"
			"\n"
			"persist.camera.stats.test=5\n"
			"#contacts.autosync\n"
			"persist.env.contacts.autosync=true\n"
			"\n"
			"#set cutoff voltage to 3400mV\n"
			"ro.cutoff_voltage_mv=3400\n"
			"\n"
			"# set default multisim config to dsds\n"
			"persist.radio.multisim.config=dsds\n"
			"\n"
			"\n"
			"#enable rnr for camera\n"
			"persist.camera.feature.cac=1\n"
			"persist.camera.imglib.cac3=2\n"
			"\n"
			"\n"
			"#Add for hardware version\n"
			"ro.build.hardware.version=V1\n"
			"\n"
			"#properties for Xiaomi LCM display\n"
			"persist.sys.display_prefer=2\n"
			"persist.sys.display_eyecare=0\n"
			"persist.sys.ltm_enable=true\n"
			"persist.sys.display_ce=11\n"
			"persist.sys.display_cabc=1\n"
			"persist.sys.gamut_mode=0\n"
			"\n"
			"#disable isp clock optimization for camera\n"
			"persist.camera.isp.clock.optmz=0\n"
			"\n"
			"#\n"
			"# system props for the cne module\n"
			"#\n"
			"persist.cne.feature=1\n"
			"#\n"
			"# system props for the dpm module\n"
			"#\n"
			"persist.dpm.feature=1\n"
			"#\n"
			"# from device/xiaomi/mido/additional_global.prop\n"
			"#\n"
			"#codecs:(PARSER_)AAC AC3 AMR_NB AMR_WB ASF AVI DTS FLV 3GP 3G2 MKV MP2PS MP2TS MP3 OGG QCP WAV FLAC AIFF APE\n"
			"mm.enable.qcom_parser=261773\n"
			"\n"
			"#\n"
			"# ADDITIONAL_BUILD_PROPERTIES\n"
			"#\n"
			"ro.miui.version.code_time=1492621200\n"
			"ro.miui.ui.version.code=6\n"
			"ro.miui.ui.version.name=V8\n"
			"ro.product.first_api_level=23\n"
			"persist.sys.mcd_config_file=/system/etc/mcd_default.conf\n"
			"persist.sys.perf.debug=true\n"
			"persist.sys.whetstone.level=2\n"
			"ro.setupwizard.require_network=any\n"
			"ro.setupwizard.mode=OPTIONAL\n"
			"ro.com.google.gmsversion=7.0_r3\n"
			"drm.service.enabled=true\n"
			"ro.ss.version=5.1.111-004\n"
			"ro.ss.nohidden=true\n"
			"dalvik.vm.heapminfree=4m\n"
			"dalvik.vm.heapstartsize=16m\n"
			"persist.delta_time.enable=true\n"
			"dalvik.vm.heapgrowthlimit=192m\n"
			"dalvik.vm.heapsize=512m\n"
			"dalvik.vm.heaptargetutilization=0.75\n"
			"dalvik.vm.heapmaxfree=8m\n"
			"ro.carrier=unknown\n"
			"ro.vendor.extension_library=libqti-perfd-client.so\n"
			"persist.radio.sib16_support=1\n"
			"persist.radio.custom_ecc=1\n"
			"ro.frp.pst=/dev/block/bootdevice/by-name/config\n"
			"af.fast_track_multiplier=2\n"
			"audio_hal.period_size=192\n"
			"ro.qc.sdk.audio.fluencetype=fluence\n"
			"persist.audio.fluence.voicecall=true\n"
			"persist.audio.fluence.voicerec=true\n"
			"persist.audio.fluence.speaker=true\n"
			"audio.offload.disable=true\n"
			"tunnel.audio.encode=false\n"
			"audio.offload.buffer.size.kb=64\n"
			"audio.offload.min.duration.secs=30\n"
			"audio.offload.video=true\n"
			"audio.offload.pcm.16bit.enable=true\n"
			"audio.offload.pcm.24bit.enable=true\n"
			"audio.offload.track.enable=true\n"
			"audio.deep_buffer.media=true\n"
			"media.stagefright.audio.sink=280\n"
			"use.voice.path.for.pcm.voip=true\n"
			"audio.offload.multiaac.enable=true\n"
			"audio.dolby.ds2.enabled=true\n"
			"audio.dolby.ds2.hardbypass=true\n"
			"audio.offload.multiple.enabled=false\n"
			"audio.offload.passthrough=false\n"
			"ro.qc.sdk.audio.ssr=false\n"
			"audio.offload.gapless.enabled=true\n"
			"audio.safx.pbe.enabled=true\n"
			"audio.parser.ip.buffer.size=0\n"
			"audio.playback.mch.downsample=true\n"
			"use.qti.sw.alac.decoder=true\n"
			"use.qti.sw.ape.decoder=true\n"
			"audio.pp.asphere.enabled=false\n"
			"voice.playback.conc.disabled=true\n"
			"voice.record.conc.disabled=false\n"
			"voice.voip.conc.disabled=true\n"
			"voice.conc.fallbackpath=deep-buffer\n"
			"persist.speaker.prot.enable=false\n"
			"qcom.hw.aac.encoder=true\n"
			"flac.sw.decoder.24bit.support=true\n"
			"persist.sys.dalvik.vm.lib.2=libart.so\n"
			"dalvik.vm.isa.arm64.variant=generic\n"
			"dalvik.vm.isa.arm64.features=default\n"
			"dalvik.vm.isa.arm.variant=cortex-a53\n"
			"dalvik.vm.isa.arm.features=default\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"ro.miui.has_real_blur=1\n"
			"ro.miui.has_handy_mode_sf=1\n"
			"fw.max_users=5\n"
			"persist.radio.calls.on.ims=0\n"
			"persist.radio.jbims=0\n"
			"persist.radio.csvt.enabled=false\n"
			"persist.radio.rat_on=combine\n"
			"persist.radio.mt_sms_ack=20\n"
			"ro.mdtp.package_name2=com.qualcomm.qti.securemsm.mdtp.MdtpDemo\n"
			"ro.product.mod_device=mido_global\n"
			"ro.config.sms_received_sound=FadeIn.ogg\n"
			"ro.config.sms_delivered_sound=MessageComplete.ogg\n"
			"ro.com.android.mobiledata=false\n"
			"ro.product.manufacturer=Xiaomi\n"
			"ro.config.elder-ringtone=Angel.mp3\n"
			"keyguard.no_require_sim=true\n"
			"ro.com.android.dataroaming=false\n"
			"persist.sys.mitalk.enable=true\n"
			"ro.config.ringtone=Ring_Synth_04.ogg\n"
			"ro.config.notification_sound=pixiedust.ogg\n"
			"ro.config.alarm_alert=Alarm_Classic.ogg\n"
			"ro.product.cuptsm=XIAOMI|ESE|02|01\n"
			"persist.power.useautobrightadj=true\n"
			"persist.radio.apm_sim_not_pwdn=1\n"
			"qemu.hw.mainkeys=1\n"
			"ro.sf.lcd_density=480\n"
			"ro.com.google.clientidbase=android-xiaomi\n"
			"persist.dbg.volte_avail_ovr=1\n"
			"persist.dbg.vt_avail_ovr=1\n"
			"sys.haptic.down.weak=0,12,24,32\n"
			"sys.haptic.down.normal=0,24,20,46\n"
			"sys.haptic.down.strong=0,36,20,64\n"
			"ro.telephony.default_network=20,20\n"
			"ro.expect.recovery_id=0x1e60c0a1d98a20844b25d79bebd86d2774c5dbb3000000000000000000000000\n"
			"\n"
			"\n"
			"import /system/vendor/vendor.prop\n"
			"\n"
			"import /system/vendor/default.prop\n"
			"\n"
			"import /system/vendor/power.prop\n"
			"\n"
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
		.content = "320\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/deep_nap_timer",
		.size = 5,
		.content = "1000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/default_pwrlevel",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/dev",
		.size = 6,
		.content = "240:0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/available_frequencies",
		.size = 70,
		.content = "650000000 560000000 510000000 400000000 320000000 216000000 133330000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/available_governors",
		.size = 165,
		.content = "spdm_bw_hyp bw_hwmon venus-ddr-gov msm-vidc-vmem+ msm-vidc-vmem msm-vidc-ddr bw_vbif gpubw_mon msm-adreno-tz cpufreq userspace powersave performance simple_ondemand\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
		.size = 10,
		.content = "320000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/governor",
		.size = 14,
		.content = "msm-adreno-tz\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/gpu_load",
		.size = 2,
		.content = "4\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/max_freq",
		.size = 10,
		.content = "650000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/min_freq",
		.size = 10,
		.content = "133330000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/polling_interval",
		.size = 3,
		.content = "10\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/suspend_time",
		.size = 7,
		.content = "152826\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/target_freq",
		.size = 10,
		.content = "510000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/devfreq/trans_stat",
		.size = 670,
		.content =
			"   From  :   To\n"
			"         :650000000560000000510000000400000000320000000216000000133330000   time(ms)\n"
			" 650000000:       0       0       0       0       0       0       0         0\n"
			" 560000000:       0       0       1       0       0       0       0       210\n"
			"*510000000:       0       1       0       2       0       0       0     97470\n"
			" 400000000:       0       0       2       0       5       0       0    119280\n"
			" 320000000:       0       0       1       4       0      54       0     96060\n"
			" 216000000:       0       0       0       0      38       0     100     59980\n"
			" 133330000:       0       0       0       1      15      84       0     78320\n"
			"Total transition : 308\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/freq_table_mhz",
		.size = 29,
		.content = "650 560 510 400 320 216 133 \n",
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
		.size = 71,
		.content = "650000000 560000000 510000000 400000000 320000000 216000000 133330000 \n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
		.size = 5,
		.content = "25 %\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_clock_stats",
		.size = 48,
		.content = "0 115594 161290 437693 5723869 9516728 4951633 \n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpu_model",
		.size = 12,
		.content = "Adreno506v1\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpubusy",
		.size = 16,
		.content = "      0       0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/gpuclk",
		.size = 10,
		.content = "320000000\n",
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
		.content = "650000000\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/max_pwrlevel",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/class/kgsl/kgsl-3d0/min_clock_mhz",
		.size = 4,
		.content = "133\n",
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
		.content = "213\n",
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
		.size = 4,
		.content = "264\n",
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
		.size = 25,
		.content = "8953A-JAASANAZA-40000000\n",
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
		.content = "QRD\n",
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
		.size = 11,
		.content =
			"mido-user\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/image_version",
		.size = 28,
		.content =
			"10:NRD90M:V8.5.4.0.NCFMIED\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/images",
		.size = 562,
		.content =
			"0:\n"
			"\tCRM:\t\t00:BOOT.BF.3.3-00199\n"
			"\tVariant:\tJAASANAZA\n"
			"\tVersion:\tmodem-ci\n"
			"\n"
			"1:\n"
			"\tCRM:\t\t01:TZ.BF.4.0.5-00030\n"
			"\tVariant:\t\n"
			"\tVersion:\tCRM\n"
			"\n"
			"3:\n"
			"\tCRM:\t\t03:RPM.BF.2.4-00054\n"
			"\tVariant:\tAAAAANAAR\n"
			"\tVersion:\tmodem-ci\n"
			"\n"
			"10:\n"
			"\tCRM:\t\t10:NRD90M:V8.5.4.0.NCFMIED\n"
			"\n"
			"\tVariant:\tmido-user\n"
			"\n"
			"\tVersion:\tREL\n"
			"\n"
			"\n"
			"11:\n"
			"\tCRM:\t\t11:MPSS.TA.2.2.C1-109373\n"
			"\tVariant:\t8953.gen.prodQ\n"
			"\tVersion:\tmodem-ci\n"
			"\n"
			"12:\n"
			"\tCRM:\t\t12:ADSP.8953.2.8.2-00046\n"
			"\tVariant:\tAAAAAAAAQ\n"
			"\tVersion:\tmodem-ci\n"
			"\n"
			"13:\n"
			"\tCRM:\t\t13:CNSS.PR.4.0-107273\n"
			"\tVariant:\tSCAQNAZM\n"
			"\tVersion:\tCRM\n"
			"\n"
			"14:\n"
			"\tCRM:\t\t14:VIDEO.VE.4.2-00031\n"
			"\tVariant:\tPROD\n"
			"\tVersion:\t:CRM\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/machine",
		.size = 8,
		.content = "MSM8953\n",
	},
	{
		.path = "/sys/devices/soc0/platform_subtype",
		.size = 4,
		.content = "QRD\n",
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
		.size = 6,
		.content = "65536\n",
	},
	{
		.path = "/sys/devices/soc0/pmic_model",
		.size = 6,
		.content = "65558\n",
	},
	{
		.path = "/sys/devices/soc0/raw_id",
		.size = 3,
		.content = "70\n",
	},
	{
		.path = "/sys/devices/soc0/raw_version",
		.size = 2,
		.content = "1\n",
	},
	{
		.path = "/sys/devices/soc0/revision",
		.size = 4,
		.content = "1.1\n",
	},
	{
		.path = "/sys/devices/soc0/select_image",
		.size = 3,
		.content = "10\n",
	},
	{
		.path = "/sys/devices/soc0/serial_number",
		.size = 11,
		.content = "1249859759\n",
	},
	{
		.path = "/sys/devices/soc0/soc_id",
		.size = 4,
		.content = "293\n",
	},
	{
		.path = "/sys/devices/soc0/vendor",
		.size = 9,
		.content = "Qualcomm\n",
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
		.path = "/sys/devices/system/cpu/cpufreq/all_time_in_state",
		.size = 55,
		.content = "freq\t\tcpu0\t\tcpu1\t\tcpu2\t\tcpu3\t\tcpu4\t\tcpu5\t\tcpu6\t\tcpu7\t\t\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/current_in_state",
		.size = 300,
		.content =
			"CPU4:652800=0 1036800=0 1401600=0 1689600=0 1843200=0 1958400=0 2016000=0 \n"
			"CPU5:652800=0 1036800=0 1401600=0 1689600=0 1843200=0 1958400=0 2016000=0 \n"
			"CPU6:652800=0 1036800=0 1401600=0 1689600=0 1843200=0 1958400=0 2016000=0 \n"
			"CPU7:652800=0 1036800=0 1401600=0 1689600=0 1843200=0 1958400=0 2016000=0 \n",
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
		.path = "/sys/devices/system/cpu/cpu0/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/affected_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2016000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "652800 1036800 1401600 1689600 1843200 1958400 2016000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "652800\n",
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
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 86,
		.content =
			"652800 35835\n"
			"1036800 525\n"
			"1401600 1882\n"
			"1689600 426\n"
			"1843200 291\n"
			"1958400 84\n"
			"2016000 6157\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "998\n",
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
		.path = "/sys/devices/system/cpu/cpu1/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/affected_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2016000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/related_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "652800 1036800 1401600 1689600 1843200 1958400 2016000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "652800\n",
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
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 86,
		.content =
			"652800 36081\n"
			"1036800 525\n"
			"1401600 1882\n"
			"1689600 426\n"
			"1843200 291\n"
			"1958400 84\n"
			"2016000 6157\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 4,
		.content = "998\n",
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
		.path = "/sys/devices/system/cpu/cpu2/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/affected_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2016000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/related_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "652800 1036800 1401600 1689600 1843200 1958400 2016000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "652800\n",
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
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 86,
		.content =
			"652800 36327\n"
			"1036800 525\n"
			"1401600 1882\n"
			"1689600 426\n"
			"1843200 291\n"
			"1958400 84\n"
			"2016000 6157\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 4,
		.content = "998\n",
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
		.path = "/sys/devices/system/cpu/cpu3/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/affected_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2016000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/related_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "652800 1036800 1401600 1689600 1843200 1958400 2016000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "652800\n",
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
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 86,
		.content =
			"652800 36575\n"
			"1036800 525\n"
			"1401600 1882\n"
			"1689600 426\n"
			"1843200 291\n"
			"1958400 84\n"
			"2016000 6157\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "998\n",
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
		.path = "/sys/devices/system/cpu/cpu4/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/affected_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2016000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/related_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "652800 1036800 1401600 1689600 1843200 1958400 2016000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "652800\n",
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
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 86,
		.content =
			"652800 36826\n"
			"1036800 525\n"
			"1401600 1882\n"
			"1689600 426\n"
			"1843200 291\n"
			"1958400 84\n"
			"2016000 6157\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 4,
		.content = "998\n",
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
		.path = "/sys/devices/system/cpu/cpu5/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/affected_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2016000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/related_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "652800 1036800 1401600 1689600 1843200 1958400 2016000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "652800\n",
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
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 86,
		.content =
			"652800 37076\n"
			"1036800 525\n"
			"1401600 1882\n"
			"1689600 426\n"
			"1843200 291\n"
			"1958400 84\n"
			"2016000 6157\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 4,
		.content = "998\n",
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
		.path = "/sys/devices/system/cpu/cpu6/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/affected_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2016000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/related_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "652800 1036800 1401600 1689600 1843200 1958400 2016000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "652800\n",
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
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 86,
		.content =
			"652800 37327\n"
			"1036800 525\n"
			"1401600 1882\n"
			"1689600 426\n"
			"1843200 291\n"
			"1958400 84\n"
			"2016000 6157\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 4,
		.content = "998\n",
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
		.path = "/sys/devices/system/cpu/cpu7/cpuidle/driver/name",
		.size = 9,
		.content = "msm_idle\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/affected_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "2016000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_transition_latency",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/related_cpus",
		.size = 16,
		.content = "0 1 2 3 4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_frequencies",
		.size = 56,
		.content = "652800 1036800 1401600 1689600 1843200 1958400 2016000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "652800\n",
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
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "652800\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 86,
		.content =
			"652800 37576\n"
			"1036800 525\n"
			"1401600 1882\n"
			"1689600 426\n"
			"1843200 291\n"
			"1958400 84\n"
			"2016000 6157\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 4,
		.content = "998\n",
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
		.key = "audio.offload.buffer.size.kb",
		.value = "64",
	},
	{
		.key = "audio.offload.disable",
		.value = "true",
	},
	{
		.key = "audio.offload.gapless.enabled",
		.value = "true",
	},
	{
		.key = "audio.offload.min.duration.secs",
		.value = "30",
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
		.value = "false",
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
		.value = "0",
	},
	{
		.key = "audio.playback.mch.downsample",
		.value = "true",
	},
	{
		.key = "audio.pp.asphere.enabled",
		.value = "false",
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
		.key = "av.debug.disable.pers.cache",
		.value = "1",
	},
	{
		.key = "bluetooth.hfp.client",
		.value = "1",
	},
	{
		.key = "camera.display.lmax",
		.value = "1280x720",
	},
	{
		.key = "camera.display.umax",
		.value = "1920x1080",
	},
	{
		.key = "camera.hal1.packagelist",
		.value = "com.skype.raider,com.google.android.talk",
	},
	{
		.key = "camera.lowpower.record.enable",
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
		.value = "4m",
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
		.key = "dalvik.vm.isa.arm.features",
		.value = "default",
	},
	{
		.key = "dalvik.vm.isa.arm.variant",
		.value = "cortex-a53",
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
		.key = "debug.egl.hw",
		.value = "0",
	},
	{
		.key = "debug.enable.gamed",
		.value = "0",
	},
	{
		.key = "debug.enable.sglscale",
		.value = "1",
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
		.key = "debug.mdpcomp.logs",
		.value = "0",
	},
	{
		.key = "debug.sf.hw",
		.value = "0",
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
		.key = "events.cpu",
		.value = "true",
	},
	{
		.key = "flac.sw.decoder.24bit.support",
		.value = "true",
	},
	{
		.key = "fpc.fp.miui.token",
		.value = "0",
	},
	{
		.key = "fw.max_users",
		.value = "5",
	},
	{
		.key = "gsm.apn.sim.operator.numeric",
		.value = ",",
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
		.key = "gsm.operator.orig.alpha",
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
		.value = "953_GEN_PACK-1.102026.1.109373.1",
	},
	{
		.key = "gsm.version.baseband1",
		.value = "953_GEN_PACK-1.102026.1.109373.1",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "Qualcomm RIL 1.0",
	},
	{
		.key = "init.svc.adbd",
		.value = "stopping",
	},
	{
		.key = "init.svc.adsprpcd",
		.value = "running",
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
		.key = "init.svc.bootanim",
		.value = "stopped",
	},
	{
		.key = "init.svc.cameraserver",
		.value = "running",
	},
	{
		.key = "init.svc.carrier_switcher",
		.value = "stopped",
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
		.key = "init.svc.cnss_diag",
		.value = "stopped",
	},
	{
		.key = "init.svc.config_bluetooth",
		.value = "stopped",
	},
	{
		.key = "init.svc.config_bt_addr",
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
		.key = "init.svc.displayfeature",
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
		.key = "init.svc.energy-awareness",
		.value = "stopped",
	},
	{
		.key = "init.svc.fdpp",
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
		.key = "init.svc.getcolorid",
		.value = "stopped",
	},
	{
		.key = "init.svc.gx_fpd",
		.value = "stopped",
	},
	{
		.key = "init.svc.healthd",
		.value = "running",
	},
	{
		.key = "init.svc.hvdcp_opti",
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
		.key = "init.svc.mcd_init",
		.value = "stopped",
	},
	{
		.key = "init.svc.mcd_service",
		.value = "running",
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
		.key = "init.svc.miui-post-boot",
		.value = "stopped",
	},
	{
		.key = "init.svc.mlipayservice",
		.value = "running",
	},
	{
		.key = "init.svc.mqsasd",
		.value = "running",
	},
	{
		.key = "init.svc.msm_irqbalance",
		.value = "running",
	},
	{
		.key = "init.svc.mtservice",
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
		.key = "init.svc.port-bridge",
		.value = "stopped",
	},
	{
		.key = "init.svc.ppd",
		.value = "running",
	},
	{
		.key = "init.svc.ptt_socket_app",
		.value = "stopped",
	},
	{
		.key = "init.svc.qcamerasvr",
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
		.key = "init.svc.qseecomd",
		.value = "running",
	},
	{
		.key = "init.svc.qseeproxydaemon",
		.value = "running",
	},
	{
		.key = "init.svc.qti",
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
		.key = "init.svc.ril-daemon2",
		.value = "running",
	},
	{
		.key = "init.svc.rmt_storage",
		.value = "running",
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
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.setlockstate",
		.value = "stopped",
	},
	{
		.key = "init.svc.shelld",
		.value = "running",
	},
	{
		.key = "init.svc.ssService",
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
		.key = "init.svc.time_daemon",
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
		.key = "init.svc.vsimservice",
		.value = "running",
	},
	{
		.key = "init.svc.wcnss-service",
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
		.key = "mcd.extra.params",
		.value = "",
	},
	{
		.key = "media.aac_51_output_enabled",
		.value = "true",
	},
	{
		.key = "media.camera.ts.monotonic",
		.value = "1",
	},
	{
		.key = "media.msm8956hw",
		.value = "0",
	},
	{
		.key = "media.stagefright.audio.sink",
		.value = "280",
	},
	{
		.key = "mm.enable.qcom_parser",
		.value = "261773",
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
		.value = "net.qtaguid_enabled",
	},
	{
		.key = "net.hostname",
		.value = "RedmiNote4-Redmi",
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
		.key = "persist.activate_mbn.enabled",
		.value = "false",
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
		.value = "true",
	},
	{
		.key = "persist.camera.feature.cac",
		.value = "1",
	},
	{
		.key = "persist.camera.gyro.disable",
		.value = "0",
	},
	{
		.key = "persist.camera.imglib.cac3",
		.value = "2",
	},
	{
		.key = "persist.camera.isp.clock.optmz",
		.value = "0",
	},
	{
		.key = "persist.camera.stats.test",
		.value = "5",
	},
	{
		.key = "persist.cne.feature",
		.value = "1",
	},
	{
		.key = "persist.console.silent.config",
		.value = "1",
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
		.key = "persist.dbg.ims_volte_enable",
		.value = "1",
	},
	{
		.key = "persist.dbg.volte_avail_ovr",
		.value = "1",
	},
	{
		.key = "persist.dbg.vt_avail_ovr",
		.value = "1",
	},
	{
		.key = "persist.dbg.wfc_avail_ovr",
		.value = "0",
	},
	{
		.key = "persist.debug.coresight.config",
		.value = "stm-events",
	},
	{
		.key = "persist.debug.wfd.enable",
		.value = "1",
	},
	{
		.key = "persist.delta_time.enable",
		.value = "true",
	},
	{
		.key = "persist.demo.hdmirotationlock",
		.value = "false",
	},
	{
		.key = "persist.device.type",
		.value = "omt",
	},
	{
		.key = "persist.dpm.feature",
		.value = "1",
	},
	{
		.key = "persist.env.contacts.autosync",
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
		.key = "persist.hwc.mdpcomp.enable",
		.value = "true",
	},
	{
		.key = "persist.mm.sta.enable",
		.value = "0",
	},
	{
		.key = "persist.net.doxlat",
		.value = "true",
	},
	{
		.key = "persist.power.useautobrightadj",
		.value = "true",
	},
	{
		.key = "persist.qfp",
		.value = "false",
	},
	{
		.key = "persist.radio.adb_log_on",
		.value = "0",
	},
	{
		.key = "persist.radio.apm_sim_not_pwdn",
		.value = "1",
	},
	{
		.key = "persist.radio.calls.on.ims",
		.value = "0",
	},
	{
		.key = "persist.radio.csvt.enabled",
		.value = "false",
	},
	{
		.key = "persist.radio.custom_ecc",
		.value = "1",
	},
	{
		.key = "persist.radio.eons.enabled",
		.value = "false",
	},
	{
		.key = "persist.radio.hw_mbn_update",
		.value = "0",
	},
	{
		.key = "persist.radio.jbims",
		.value = "0",
	},
	{
		.key = "persist.radio.msim.stackid_0",
		.value = "0",
	},
	{
		.key = "persist.radio.msim.stackid_1",
		.value = "1",
	},
	{
		.key = "persist.radio.mt_sms_ack",
		.value = "20",
	},
	{
		.key = "persist.radio.multisim.config",
		.value = "dsds",
	},
	{
		.key = "persist.radio.rat_on",
		.value = "combine",
	},
	{
		.key = "persist.radio.ril_payload_on",
		.value = "0",
	},
	{
		.key = "persist.radio.sglte_target",
		.value = "0",
	},
	{
		.key = "persist.radio.sib16_support",
		.value = "1",
	},
	{
		.key = "persist.radio.sn",
		.value = "2B739F051576",
	},
	{
		.key = "persist.radio.stack_id_0",
		.value = "0",
	},
	{
		.key = "persist.radio.stack_id_1",
		.value = "1",
	},
	{
		.key = "persist.radio.sw_mbn_loaded",
		.value = "1",
	},
	{
		.key = "persist.radio.sw_mbn_update",
		.value = "0",
	},
	{
		.key = "persist.radio.trigger.silence",
		.value = "true",
	},
	{
		.key = "persist.radio.videopause.mode",
		.value = "1",
	},
	{
		.key = "persist.regional.wipedata.level",
		.value = "all",
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
		.key = "persist.speaker.prot.enable",
		.value = "false",
	},
	{
		.key = "persist.sys.cnd.iwlan",
		.value = "1",
	},
	{
		.key = "persist.sys.dalvik.vm.lib.2",
		.value = "libart.so",
	},
	{
		.key = "persist.sys.display_cabc",
		.value = "1",
	},
	{
		.key = "persist.sys.display_ce",
		.value = "11",
	},
	{
		.key = "persist.sys.display_eyecare",
		.value = "0",
	},
	{
		.key = "persist.sys.display_prefer",
		.value = "2",
	},
	{
		.key = "persist.sys.enable_pinfile",
		.value = "true",
	},
	{
		.key = "persist.sys.gamut_mode",
		.value = "0",
	},
	{
		.key = "persist.sys.ifaa",
		.value = "0",
	},
	{
		.key = "persist.sys.klo",
		.value = "on",
	},
	{
		.key = "persist.sys.klo.rec_start",
		.value = "31249",
	},
	{
		.key = "persist.sys.locale",
		.value = "en-US",
	},
	{
		.key = "persist.sys.ltm_enable",
		.value = "true",
	},
	{
		.key = "persist.sys.mcd_config_file",
		.value = "/system/etc/mcd_default.conf",
	},
	{
		.key = "persist.sys.memctrl",
		.value = "on",
	},
	{
		.key = "persist.sys.mitalk.enable",
		.value = "true",
	},
	{
		.key = "persist.sys.notification_device",
		.value = "503",
	},
	{
		.key = "persist.sys.notification_num",
		.value = "3",
	},
	{
		.key = "persist.sys.notification_rank",
		.value = "2",
	},
	{
		.key = "persist.sys.perf.debug",
		.value = "true",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
	},
	{
		.key = "persist.sys.screenshot_mode",
		.value = "1",
	},
	{
		.key = "persist.sys.smartcover_mode",
		.value = "0",
	},
	{
		.key = "persist.sys.task_isolation",
		.value = "true",
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
		.key = "persist.sys.usb.config.extra",
		.value = "none",
	},
	{
		.key = "persist.sys.webview.vmsize",
		.value = "116905264",
	},
	{
		.key = "persist.sys.whetstone.level",
		.value = "2",
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
		.key = "qcom.hw.aac.encoder",
		.value = "true",
	},
	{
		.key = "qemu.hw.mainkeys",
		.value = "1",
	},
	{
		.key = "ril.ecclist",
		.value = "*911,#911,000,08,110,999,118,119,120,122,911,112",
	},
	{
		.key = "ril.ecclist1",
		.value = "*911,#911,000,08,110,999,118,119,120,122,911,112",
	},
	{
		.key = "ril.limit_service_mnc",
		.value = "GSM_310",
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
		.key = "rild.libargs",
		.value = "-d /dev/smd0",
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
		.key = "ro.alarm_boot",
		.value = "false",
	},
	{
		.key = "ro.allow.mock.location",
		.value = "0",
	},
	{
		.key = "ro.am.reschedule_service",
		.value = "true",
	},
	{
		.key = "ro.baseband",
		.value = "msm",
	},
	{
		.key = "ro.bluetooth.dun",
		.value = "true",
	},
	{
		.key = "ro.bluetooth.hfp.ver",
		.value = "1.7",
	},
	{
		.key = "ro.bluetooth.sap",
		.value = "true",
	},
	{
		.key = "ro.board.platform",
		.value = "msm8953",
	},
	{
		.key = "ro.boot.authorized_kernel",
		.value = "true",
	},
	{
		.key = "ro.boot.baseband",
		.value = "msm",
	},
	{
		.key = "ro.boot.boot_reason",
		.value = "",
	},
	{
		.key = "ro.boot.bootdevice",
		.value = "7824900.sdhci",
	},
	{
		.key = "ro.boot.console",
		.value = "ttyHSL0",
	},
	{
		.key = "ro.boot.emmc",
		.value = "true",
	},
	{
		.key = "ro.boot.fpsensor",
		.value = "fpc",
	},
	{
		.key = "ro.boot.hardware",
		.value = "qcom",
	},
	{
		.key = "ro.boot.keymaster",
		.value = "1",
	},
	{
		.key = "ro.boot.secureboot",
		.value = "1",
	},
	{
		.key = "ro.boot.serialno",
		.value = "28f519ad0604",
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
		.value = "Thu Aug 3 19:37:25 WIB 2017",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1501763845",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "xiaomi/mido/mido:7.0/NRD90M/V8.5.4.0.NCFMIED:user/release-keys",
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
		.key = "ro.build.characteristics",
		.value = "nosdcard",
	},
	{
		.key = "ro.build.date",
		.value = "Thu Aug  3 19:37:25 WIB 2017",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1501763845",
	},
	{
		.key = "ro.build.description",
		.value = "mido-user 7.0 NRD90M V8.5.4.0.NCFMIED release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "AL1512-mido-build-20170803193727",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "xiaomi/mido/mido:7.0/NRD90M/V8.5.4.0.NCFMIED:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "mido-user",
	},
	{
		.key = "ro.build.hardware.version",
		.value = "V1",
	},
	{
		.key = "ro.build.host",
		.value = "mi-server",
	},
	{
		.key = "ro.build.id",
		.value = "NRD90M",
	},
	{
		.key = "ro.build.product",
		.value = "mido",
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
		.value = "builder",
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
		.value = "V8.5.4.0.NCFMIED",
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
		.value = "2017-07-01",
	},
	{
		.key = "ro.carrier",
		.value = "unknown",
	},
	{
		.key = "ro.channelid.ucnewsintl",
		.value = "/system/etc/ucintlconfiginfo",
	},
	{
		.key = "ro.com.android.dataroaming",
		.value = "false",
	},
	{
		.key = "ro.com.android.mobiledata",
		.value = "false",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-xiaomi",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "7.0_r3",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Alarm_Classic.ogg",
	},
	{
		.key = "ro.config.elder-ringtone",
		.value = "Angel.mp3",
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
		.key = "ro.config.sms_delivered_sound",
		.value = "MessageComplete.ogg",
	},
	{
		.key = "ro.config.sms_received_sound",
		.value = "FadeIn.ogg",
	},
	{
		.key = "ro.core_ctl_max_cpu",
		.value = "4",
	},
	{
		.key = "ro.core_ctl_min_cpu",
		.value = "2",
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
		.key = "ro.cutoff_voltage_mv",
		.value = "3400",
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
		.key = "ro.emmc_size",
		.value = "16GB",
	},
	{
		.key = "ro.expect.recovery_id",
		.value = "0x1e60c0a1d98a20844b25d79bebd86d2774c5dbb3000000000000000000000000",
	},
	{
		.key = "ro.fm.transmitter",
		.value = "false",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/bootdevice/by-name/config",
	},
	{
		.key = "ro.gpu.available_frequencies",
		.value = "650000000 560000000 510000000 400000000 320000000 216000000 133330000 ",
	},
	{
		.key = "ro.hardware",
		.value = "qcom",
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
		.key = "ro.logdumpd.enabled",
		.value = "0",
	},
	{
		.key = "ro.mdtp.package_name2",
		.value = "com.qualcomm.qti.securemsm.mdtp.MdtpDemo",
	},
	{
		.key = "ro.memperf.enable",
		.value = "false",
	},
	{
		.key = "ro.memperf.lib",
		.value = "libmemperf.so",
	},
	{
		.key = "ro.miui.cust_variant",
		.value = "us",
	},
	{
		.key = "ro.miui.has_cust_partition",
		.value = "true",
	},
	{
		.key = "ro.miui.has_handy_mode_sf",
		.value = "1",
	},
	{
		.key = "ro.miui.has_real_blur",
		.value = "1",
	},
	{
		.key = "ro.miui.mcc",
		.value = "9310",
	},
	{
		.key = "ro.miui.mnc",
		.value = "9999",
	},
	{
		.key = "ro.miui.region",
		.value = "US",
	},
	{
		.key = "ro.miui.ui.version.code",
		.value = "6",
	},
	{
		.key = "ro.miui.ui.version.name",
		.value = "V8",
	},
	{
		.key = "ro.miui.version.code_time",
		.value = "1492621200",
	},
	{
		.key = "ro.opengles.version",
		.value = "196610",
	},
	{
		.key = "ro.product.board",
		.value = "msm8953",
	},
	{
		.key = "ro.product.brand",
		.value = "xiaomi",
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
		.key = "ro.product.cuptsm",
		.value = "XIAOMI|ESE|02|01",
	},
	{
		.key = "ro.product.device",
		.value = "mido",
	},
	{
		.key = "ro.product.first_api_level",
		.value = "23",
	},
	{
		.key = "ro.product.locale",
		.value = "en-GB",
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
		.value = "Xiaomi",
	},
	{
		.key = "ro.product.mod_device",
		.value = "mido_global",
	},
	{
		.key = "ro.product.model",
		.value = "Redmi Note 4",
	},
	{
		.key = "ro.product.name",
		.value = "mido",
	},
	{
		.key = "ro.qc.sdk.audio.fluencetype",
		.value = "fluence",
	},
	{
		.key = "ro.qc.sdk.audio.ssr",
		.value = "false",
	},
	{
		.key = "ro.qualcomm.bluetooth.ftp",
		.value = "true",
	},
	{
		.key = "ro.qualcomm.bluetooth.hfp",
		.value = "true",
	},
	{
		.key = "ro.qualcomm.bluetooth.hsp",
		.value = "true",
	},
	{
		.key = "ro.qualcomm.bluetooth.map",
		.value = "true",
	},
	{
		.key = "ro.qualcomm.bluetooth.nap",
		.value = "true",
	},
	{
		.key = "ro.qualcomm.bluetooth.opp",
		.value = "true",
	},
	{
		.key = "ro.qualcomm.bluetooth.pbap",
		.value = "true",
	},
	{
		.key = "ro.qualcomm.bt.hci_transport",
		.value = "smd",
	},
	{
		.key = "ro.qualcomm.cabl",
		.value = "0",
	},
	{
		.key = "ro.revision",
		.value = "0",
	},
	{
		.key = "ro.ril.miui.imei0",
		.value = "866471038915944",
	},
	{
		.key = "ro.ril.miui.imei1",
		.value = "866471038915951",
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
		.key = "ro.runtime.firstboot",
		.value = "17007685282",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.secureboot.devicelock",
		.value = "1",
	},
	{
		.key = "ro.secureboot.lockstate",
		.value = "locked",
	},
	{
		.key = "ro.serialno",
		.value = "28f519ad0604",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "OPTIONAL",
	},
	{
		.key = "ro.setupwizard.require_network",
		.value = "any",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "480",
	},
	{
		.key = "ro.ss.nohidden",
		.value = "true",
	},
	{
		.key = "ro.ss.version",
		.value = "5.1.111-004",
	},
	{
		.key = "ro.sys.fw.bservice_age",
		.value = "5000",
	},
	{
		.key = "ro.sys.fw.bservice_enable",
		.value = "true",
	},
	{
		.key = "ro.sys.fw.bservice_limit",
		.value = "5",
	},
	{
		.key = "ro.sys.fw.dex2oat_thread_count",
		.value = "4",
	},
	{
		.key = "ro.sys.fw.empty_app_percent",
		.value = "50",
	},
	{
		.key = "ro.sys.fw.trim_cache_percent",
		.value = "100",
	},
	{
		.key = "ro.sys.fw.trim_empty_percent",
		.value = "100",
	},
	{
		.key = "ro.sys.fw.trim_enable_memory",
		.value = "2147483648",
	},
	{
		.key = "ro.sys.fw.use_trim_settings",
		.value = "true",
	},
	{
		.key = "ro.telephony.call_ring.multiple",
		.value = "false",
	},
	{
		.key = "ro.telephony.default_network",
		.value = "20,20",
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
		.key = "ro.vendor.extension_library",
		.value = "libqti-perfd-client.so",
	},
	{
		.key = "ro.vendor.gt_library",
		.value = "libqti-gt.so",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
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
		.key = "sys.fp.goodix",
		.value = "0",
	},
	{
		.key = "sys.fp.vendor",
		.value = "searchf",
	},
	{
		.key = "sys.haptic.down.normal",
		.value = "0,24,20,46",
	},
	{
		.key = "sys.haptic.down.strong",
		.value = "0,36,20,64",
	},
	{
		.key = "sys.haptic.down.weak",
		.value = "0,12,24,32",
	},
	{
		.key = "sys.ims.QMI_DAEMON_STATUS",
		.value = "1",
	},
	{
		.key = "sys.is_keyguard_showing",
		.value = "1",
	},
	{
		.key = "sys.kernel.firstboot",
		.value = "17007695076",
	},
	{
		.key = "sys.keyguard.bleunlock",
		.value = "true",
	},
	{
		.key = "sys.keyguard.screen_off_by_lid",
		.value = "false",
	},
	{
		.key = "sys.keymaster.loaded",
		.value = "true",
	},
	{
		.key = "sys.listeners.registered",
		.value = "true",
	},
	{
		.key = "sys.miui.user_authenticated",
		.value = "true",
	},
	{
		.key = "sys.oem_unlock_allowed",
		.value = "0",
	},
	{
		.key = "sys.panel.color",
		.value = "black",
	},
	{
		.key = "sys.post_boot.parsed",
		.value = "1",
	},
	{
		.key = "sys.rpmb_state",
		.value = "0",
	},
	{
		.key = "sys.sysctl.extra_free_kbytes",
		.value = "24300",
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
		.key = "sys.usb.rps_mask",
		.value = "0",
	},
	{
		.key = "sys.usb.state",
		.value = "mtp,adb",
	},
	{
		.key = "telephony.lteOnCdmaDevice",
		.value = "1",
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
		.key = "vidc.dec.downscalar_height",
		.value = "1088",
	},
	{
		.key = "vidc.dec.downscalar_width",
		.value = "1920",
	},
	{
		.key = "vidc.disable.split.mode",
		.value = "1",
	},
	{
		.key = "vidc.enc.dcvs.extra-buff-count",
		.value = "2",
	},
	{
		.key = "vidc.enc.disable.pq",
		.value = "true",
	},
	{
		.key = "vidc.enc.disable_bframes",
		.value = "1",
	},
	{
		.key = "voice.conc.fallbackpath",
		.value = "deep-buffer",
	},
	{
		.key = "voice.playback.conc.disabled",
		.value = "true",
	},
	{
		.key = "voice.record.conc.disabled",
		.value = "false",
	},
	{
		.key = "voice.voip.conc.disabled",
		.value = "true",
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
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "wlan.driver.ath",
		.value = "0",
	},
	{
		.key = "wlan.driver.config",
		.value = "/data/misc/wifi/WCNSS_qcom_cfg.ini",
	},
	{
		.key = "wlan.driver.status",
		.value = "ok",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
