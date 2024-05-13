struct cpuinfo_mock_file filesystem[] = {
	{
		.path = "/proc/cpuinfo",
		.size = 2488,
		.content =
			"processor\t: 0\n"
			"model name\t: ARMv7 Processor rev 4 (v7l)\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm aes pmull sha1 sha2 crc32 \n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 7\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 1\n"
			"model name\t: ARMv7 Processor rev 4 (v7l)\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm aes pmull sha1 sha2 crc32 \n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 7\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 2\n"
			"model name\t: ARMv7 Processor rev 4 (v7l)\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm aes pmull sha1 sha2 crc32 \n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 7\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 3\n"
			"model name\t: ARMv7 Processor rev 4 (v7l)\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm aes pmull sha1 sha2 crc32 \n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 7\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 4\n"
			"model name\t: ARMv7 Processor rev 4 (v7l)\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm aes pmull sha1 sha2 crc32 \n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 7\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 5\n"
			"model name\t: ARMv7 Processor rev 4 (v7l)\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm aes pmull sha1 sha2 crc32 \n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 7\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 6\n"
			"model name\t: ARMv7 Processor rev 4 (v7l)\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm aes pmull sha1 sha2 crc32 \n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 7\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"processor\t: 7\n"
			"model name\t: ARMv7 Processor rev 4 (v7l)\n"
			"BogoMIPS\t: 38.00\n"
			"Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm aes pmull sha1 sha2 crc32 \n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 7\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xd03\n"
			"CPU revision\t: 4\n"
			"\n"
			"Hardware\t: Qualcomm Technologies, Inc MSM8937\n"
			"Revision\t: 8500\n"
			"Serial\t\t: d693e67e00000000\n"
			"Processor\t: ARMv7 Processor rev 4 (v7l)\n"
			"Device\t\t: cedric\n"
			"Radio\t\t: LATAM\n"
			"MSM Hardware\t: MSM8937\n",
	},
	{
		.path = "/system/build.prop",
		.size = 13726,
		.content =
			"\n"
			"#\n"
			"# PRODUCT_OEM_PROPERTIES\n"
			"#\n"
			"import /oem/oem.prop ro.product.brand1\n"
			"import /oem/oem.prop ro.product.name\n"
			"import /oem/oem.prop ro.product.model\n"
			"import /oem/oem.prop ro.product.display\n"
			"import /oem/oem.prop ro.config.ringtone\n"
			"import /oem/oem.prop ro.build.date\n"
			"import /oem/oem.prop ro.build.date.uts\n"
			"import /oem/oem.prop ro.build.id\n"
			"import /oem/oem.prop ro.build.display.id\n"
			"import /oem/oem.prop ro.build.version.incremental\n"
			"import /oem/oem.prop ro.mot.build.customerid\n"
			"import /oem/oem.prop ro.mot.build.product.increment\n"
			"import /oem/oem.prop ro.mot.build.oem.product\n"
			"import /oem/oem.prop persist.lte.pco_supported\n"
			"import /oem/oem.prop persist.radio.mode_pref_nv10\n"
			"import /oem/oem.prop persist.radio.sib16_support\n"
			"import /oem/oem.prop persist.radio.nw_mtu_enabled\n"
			"import /oem/oem.prop persist.radio.customer_mbns\n"
			"import /oem/oem.prop ro.carrier\n"
			"import /oem/oem.prop ro.carrier.oem\n"
			"import /oem/oem.prop ro.telephony.default_network\n"
			"import /oem/oem.prop ro.product.locale\n"
			"import /oem/oem.prop ro.config.alarm_alert\n"
			"import /oem/oem.prop ro.config.notification_sound\n"
			"import /oem/oem.prop ro.config.wallpaper\n"
			"import /oem/oem.prop ro.com.google.clientidbase.am\n"
			"import /oem/oem.prop ro.com.google.clientidbase.gmm\n"
			"import /oem/oem.prop ro.com.google.clientidbase.ms\n"
			"import /oem/oem.prop ro.com.google.clientidbase.yt\n"
			"import /oem/oem.prop ro.oem.*\n"
			"import /oem/oem.prop oem.*\n"
			"# begin build properties\n"
			"# autogenerated by buildinfo.sh\n"
			"ro.build.id=NPP25.137-15\n"
			"ro.build.version.incremental=13\n"
			"ro.build.version.sdk=24\n"
			"ro.build.version.preview_sdk=0\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.all_codenames=REL\n"
			"ro.build.version.release=7.0\n"
			"ro.build.version.security_patch=2017-01-01\n"
			"ro.build.date=Fri Jan 13 08:37:11 CST 2017\n"
			"ro.build.date.utc=1484318231\n"
			"ro.build.type=user\n"
			"ro.build.user=hudsoncm\n"
			"ro.build.host=ilclbld35\n"
			"ro.build.tags=release-keys\n"
			"ro.product.model=Moto G (5)\n"
			"ro.product.brand=motorola\n"
			"ro.product.name=cedric_retail\n"
			"ro.product.device=cedric\n"
			"ro.product.board=msm8937\n"
			"# ro.product.cpu.abi and ro.product.cpu.abi2 are obsolete,\n"
			"# use ro.product.cpu.abilist instead.\n"
			"ro.product.cpu.abi=armeabi-v7a\n"
			"ro.product.cpu.abi2=armeabi\n"
			"ro.product.cpu.abilist=armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist32=armeabi-v7a,armeabi\n"
			"ro.product.cpu.abilist64=\n"
			"ro.product.manufacturer=motorola\n"
			"ro.product.locale=en-US\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=msm8937\n"
			"# ro.build.product is obsolete; use ro.product.device\n"
			"ro.build.product=cedric\n"
			"# Do not try to parse description, fingerprint, or thumbprint\n"
			"ro.build.thumbprint=7.0/NPP25.137-15/13:user/release-keys\n"
			"ro.build.characteristics=default\n"
			"# end build properties\n"
			"#\n"
			"# from device/qcom/msm8937_32/system.prop\n"
			"#\n"
			"#\n"
			"# system.prop for msm8937_32\n"
			"#\n"
			"\n"
			"#rild.libpath=/system/lib/libreference-ril.so\n"
			"rild.libpath=/system/vendor/lib/libril-qc-qmi-1.so\n"
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
			"# Start in cdma mode\n"
			"#Moto zhangcj1 03/24/2016 IKSWM-27559\n"
			"#ro.telephony.default_network=22,20\n"
			"\n"
			"debug.sf.hw=1\n"
			"debug.egl.hw=1\n"
			"persist.hwc.mdpcomp.enable=true\n"
			"debug.mdpcomp.logs=0\n"
			"dalvik.vm.heapsize=36m\n"
			"dev.pm.dyn_samplingrate=1\n"
			"persist.demo.hdmirotationlock=false\n"
			"debug.enable.sglscale=1\n"
			"\n"
			"#ro.hdmi.enable=true\n"
			"#tunnel.decode=true\n"
			"#tunnel.audiovideo.decode=true\n"
			"#lpa.decode=false\n"
			"#lpa.use-stagefright=true\n"
			"persist.speaker.prot.enable=false\n"
			"qcom.hw.aac.encoder=true\n"
			"\n"
			"#\n"
			"# system props for the cne module\n"
			"#\n"
			"persist.cne.feature=1\n"
			"\n"
			"#system props for the MM modules\n"
			"media.msm8956hw=0\n"
			"mm.enable.smoothstreaming=true\n"
			"mmp.enable.3g2=true\n"
			"media.aac_51_output_enabled=true\n"
			"#codecs:(PARSER_)AAC AC3 AMR_NB AMR_WB ASF AVI DTS FLV 3GP 3G2 MKV MP2PS MP2TS MP3 OGG QCP WAV FLAC AIFF APE\n"
			"mm.enable.qcom_parser=1048575\n"
			"\n"
			"# system prop for UBWC\n"
			"video.disable.ubwc=1\n"
			"\n"
			"# system prop to disable split mode\n"
			"vidc.disable.split.mode=1\n"
			"\n"
			"# system property to accelerate Progressive Download using STA\n"
			"persist.mm.sta.enable=0\n"
			"\n"
			"#Audio voice concurrency related flags\n"
			"voice.playback.conc.disabled=true\n"
			"voice.record.conc.disabled=false\n"
			"voice.voip.conc.disabled=true\n"
			"#Decides the audio fallback path during voice call, deep-buffer and fast are the two allowed fallback paths now.\n"
			"voice.conc.fallbackpath=deep-buffer\n"
			"\n"
			"#parser input buffer size(256kb) in byte stream mode\n"
			"audio.parser.ip.buffer.size=262144\n"
			"\n"
			"#\n"
			"# system props for the camera\n"
			"#\n"
			"# preferred IS type for 8937 is IS_TYPE_DIS i.e, 1\n"
			"# IS_TYPE_NONE=0, IS_TYPE_DIS=1, IS_TYPE_GA_DIS=2, IS_TYPE_EIS_1_0=3, IS_TYPE_EIS_2_0=4 IS_TYPE_MAX=5\n"
			"#\n"
			"persist.camera.is_type=1\n"
			"persist.camera.gyro.android=1\n"
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
			"# 131072 is decimal for 0x20000 to report version 2\n"
			"# 196608 is decimal for 0x30000 to report major/minor versions as 3/0\n"
			"# 196609 is decimal for 0x30001 to report major/minor versions as 3/1\n"
			"#ro.opengles.version=196609\n"
			"\n"
			"# System property for cabl\n"
			"ro.qualcomm.cabl=0\n"
			"\n"
			"#\n"
			"# System props for telephony\n"
			"# System prop to turn on CdmaLTEPhone always\n"
			"telephony.lteOnCdmaDevice=1\n"
			"#\n"
			"# System props for bluetooh\n"
			"# System prop to turn on hfp client\n"
			"bluetooth.hfp.client=1\n"
			"\n"
			"#Simulate sdcard on /data/media\n"
			"#\n"
			"persist.fuse_sdcard=true\n"
			"\n"
			"#\n"
			"#snapdragon value add features\n"
			"#\n"
			"ro.qc.sdk.audio.ssr=false\n"
			"##fluencetype can be \"fluence\" or \"fluencepro\" or \"none\"\n"
			"ro.qc.sdk.audio.fluencetype=none\n"
			"persist.audio.fluence.voicecall=true\n"
			"persist.audio.fluence.voicerec=false\n"
			"persist.audio.fluence.speaker=true\n"
			"#Set for msm8937\n"
			"tunnel.audio.encode = false\n"
			"#Buffer size in kbytes for compress offload playback\n"
			"audio.offload.buffer.size.kb=64\n"
			"#Minimum duration for offload playback in secs\n"
			"audio.offload.min.duration.secs=30\n"
			"#Enable offload audio video playback by default\n"
			"audio.offload.video=true\n"
			"\n"
			"#Enable PCM offload by default\n"
			"audio.offload.pcm.16bit.enable=true\n"
			"audio.offload.pcm.24bit.enable=true\n"
			"\n"
			"#Enable audio track offload by default\n"
			"audio.offload.track.enable=true\n"
			"\n"
			"#Enable music through deep buffer\n"
			"audio.deep_buffer.media=true\n"
			"\n"
			"#disable voice path for PCM VoIP by default\n"
			"use.voice.path.for.pcm.voip=false\n"
			"ro.config.vc_call_vol_steps=8\n"
			"\n"
			"#enable downsampling for multi-channel content > 48Khz\n"
			"audio.playback.mch.downsample=true\n"
			"\n"
			"#\n"
			"#System property for FM transmitter\n"
			"#\n"
			"ro.fm.transmitter=false\n"
			"#enable dsp gapless mode by default\n"
			"audio.offload.gapless.enabled=true\n"
			"\n"
			"#multi offload\n"
			"audio.offload.multiple.enabled=false\n"
			"\n"
			"#enable software decoders for ALAC and APE.\n"
			"use.qti.sw.alac.decoder=true\n"
			"use.qti.sw.ape.decoder=true\n"
			"\n"
			"#enable pbe effects\n"
			"audio.safx.pbe.enabled=true\n"
			"#property for AudioSphere Post processing\n"
			"audio.pp.asphere.enabled=false\n"
			"\n"
			"\n"
			"# set max background services\n"
			"ro.config.max_starting_bg=8\n"
			"\n"
			"#property to enable user to access Google WFD settings\n"
			"#persist.debug.wfd.enable=1\n"
			"#propery to enable VDS WFD solution\n"
			"persist.hwc.enable_vds=1\n"
			"\n"
			"#selects CoreSight configuration to enable\n"
			"persist.debug.coresight.config=stm-events\n"
			"\n"
			"#property for vendor specific library\n"
			"ro.vendor.at_library=libqti-at.so\n"
			"ro.vendor.gt_library=libqti-gt.so\n"
			"\n"
			"#property for game detection feature\n"
			"debug.enable.gamed=0\n"
			"#property to enable narrow search range for video encoding\n"
			"vidc.enc.narrow.searchrange=1\n"
			"\n"
			"#property to enable fingerprint\n"
			"persist.qfp=false\n"
			"\n"
			"#property to enable DS2 dap\n"
			"audio.dolby.ds2.enabled=true\n"
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
			"ro.hwui.text_large_cache_height=2048\n"
			"\n"
			"#Enable B service adj transition by default\n"
			"ro.sys.fw.bservice_enable=true\n"
			"ro.sys.fw.bservice_limit=5\n"
			"ro.sys.fw.bservice_age=5000\n"
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
			"ro.config.zram=true\n"
			"\n"
			"# Create Swap disk, if below sys-prop enabled & also if device has lower (< 1 GB) RAM\n"
			"ro.config.swap=true\n"
			"\n"
			"# set cutoff voltage to 3200mV\n"
			"ro.cutoff_voltage_mv=3200\n"
			"\n"
			"#set device emmc size\n"
			"ro.emmc_size=16GB\n"
			"\n"
			"#force HAL1 for below packages\n"
			"camera.hal1.packagelist=com.skype.raider,com.google.android.talk\n"
			"\n"
			"#properties for limiting preview size in camera\n"
			"camera.display.umax=1920x1080\n"
			"camera.display.lmax=1280x720\n"
			"\n"
			"#set cutoff voltage to 3400mV\n"
			"ro.cutoff_voltage_mv=3400\n"
			"\n"
			"#\n"
			"# ADDITIONAL_BUILD_PROPERTIES\n"
			"#\n"
			"ro.build.version.qcom=LA.UM.5.6.r1-01900-89xx.0\n"
			"ro.mot.build.customerid=retail\n"
			"ro.mot.build.version.sdk_int=25\n"
			"ro.mot.build.product.increment=11\n"
			"ro.mot.build.version.release=25.11\n"
			"ro.product.first_api_level=24\n"
			"ro.telephony.default_network=10\n"
			"ro.radio.imei.sv=2\n"
			"ro.mot.ignore_csim_appid=true\n"
			"ro.config.ringtone=Moto.ogg\n"
			"ro.config.ringtone_2=Moto.ogg\n"
			"ro.config.notification_sound=Moto.ogg\n"
			"ro.config.alarm_alert=Oxygen.ogg\n"
			"ro.com.google.ime.theme_id=4\n"
			"persist.audio.calfile0=/etc/acdbdata/Bluetooth_cal.acdb\n"
			"persist.audio.calfile1=/etc/acdbdata/General_cal.acdb\n"
			"persist.audio.calfile2=/etc/acdbdata/Global_cal.acdb\n"
			"persist.audio.calfile3=/etc/acdbdata/Handset_cal.acdb\n"
			"persist.audio.calfile4=/etc/acdbdata/Hdmi_cal.acdb\n"
			"persist.audio.calfile5=/etc/acdbdata/Headset_cal.acdb\n"
			"persist.audio.calfile6=/etc/acdbdata/Speaker_cal.acdb\n"
			"persist.audio.dualmic.config=endfire\n"
			"persist.audio.fluence.voicecall=true\n"
			"persist.audio.fluence.voicecomm=true\n"
			"persist.audio.fluence.voicerec=false\n"
			"persist.audio.fluence.speaker=false\n"
			"qcom.bt.le_dev_pwr_class=1\n"
			"ro.bluetooth.hfp.ver=1.6\n"
			"ro.qualcomm.bt.hci_transport=smd\n"
			"persist.mot.gps.conf.from.sim=true\n"
			"persist.mot.gps.smart_battery=1\n"
			"ro.frp.pst=/dev/block/bootdevice/by-name/frp\n"
			"audio.offload.disable=false\n"
			"audio.offload.video=false\n"
			"av.offload.enable=false\n"
			"audio.offload.pcm.16bit.enable=false\n"
			"audio.offload.pcm.24bit.enable=false\n"
			"audio.offload.min.duration.secs=60\n"
			"audio.offload.gapless.enabled=false\n"
			"qcom.hw.aac.encoder=false\n"
			"mm.enable.sec.smoothstreaming=false\n"
			"mm.enable.smoothstreaming=false\n"
			"audio_hal.period_size=240\n"
			"ro.usb.bpt=2ec1\n"
			"ro.usb.bpt_adb=2ec5\n"
			"ro.usb.bpteth=2ec3\n"
			"ro.usb.bpteth_adb=2ec6\n"
			"persist.sys.ssr.restart_level=ALL_ENABLE\n"
			"persist.sys.qc.sub.rdump.on=1\n"
			"persist.sys.qc.sub.rdump.max=0\n"
			"ro.usb.mtp=2e82\n"
			"ro.usb.mtp_adb=2e76\n"
			"ro.usb.ptp=2e83\n"
			"ro.usb.ptp_adb=2e84\n"
			"persist.esdfs_sdcard=true\n"
			"ro.vendor.extension_library=libqti-perfd-client.so\n"
			"persist.radio.apm_sim_not_pwdn=1\n"
			"persist.radio.sib16_support=1\n"
			"af.fast_track_multiplier=1\n"
			"camera.disable_zsl_mode=1\n"
			"persist.radio.no_wait_for_card=1\n"
			"persist.radio.dfr_mode_set=1\n"
			"persist.radio.relay_oprt_change=1\n"
			"persist.radio.msgtunnel.start=true\n"
			"persist.radio.oem_ind_to_both=0\n"
			"persist.qcril_uim_vcc_feature=1\n"
			"persist.data.qmi.adb_logmask=0\n"
			"persist.radio.0x9e_not_callname=1\n"
			"persist.radio.mt_sms_ack=30\n"
			"persist.radio.force_get_pref=1\n"
			"persist.dpm.feature=0\n"
			"persist.radio.is_wps_enabled=true\n"
			"persist.radio.custom_ecc=1\n"
			"ro.bug2go.magickeys=24,26\n"
			"persist.radio.sw_mbn_update=1\n"
			"persist.radio.app_mbn_path=/fsg\n"
			"persist.vold.ecryptfs_supported=true\n"
			"dalvik.vm.heapstartsize=8m\n"
			"dalvik.vm.heapgrowthlimit=192m\n"
			"dalvik.vm.heapsize=384m\n"
			"dalvik.vm.heaptargetutilization=0.75\n"
			"dalvik.vm.heapminfree=512k\n"
			"dalvik.vm.heapmaxfree=8m\n"
			"persist.radio.sar_sensor=1\n"
			"ro.carrier=unknown\n"
			"ro.com.google.clientidbase=android-motorola\n"
			"ro.com.google.clientidbase.ms=android-motorola\n"
			"ro.com.google.clientidbase.am=android-motorola\n"
			"ro.com.google.clientidbase.gmm=android-motorola\n"
			"ro.com.google.clientidbase.yt=android-motorola\n"
			"ro.url.legal=http://www.google.com/intl/%s/mobile/android/basic/phone-legal.html\n"
			"ro.url.legal.android_privacy=http://www.google.com/intl/%s/mobile/android/basic/privacy.html\n"
			"ro.setupwizard.mode=OPTIONAL\n"
			"ro.com.google.gmsversion=7.0_r4\n"
			"persist.radio.apn_delay=5000\n"
			"persist.sys.media.use-awesome=false\n"
			"mm.enable.qcom_parser=4643\n"
			"persist.cne.rat.wlan.chip.oem=WCN\n"
			"keyguard.no_require_sim=true\n"
			"drm.service.enabled=true\n"
			"mdc_initial_max_retry=10\n"
			"ro.lenovo.single_hand=1\n"
			"ro.mot.security.enable=true\n"
			"persist.radio.call.audio.output=0\n"
			"persist.cne.feature=1\n"
			"persist.data.netmgrd.qos.enable=true\n"
			"persist.data.iwlan.enable=true\n"
			"persist.sys.cnd.iwlan=1\n"
			"persist.cne.logging.qxdm=3974\n"
			"persist.ims.disableADBLogs=2\n"
			"persist.ims.volte=true\n"
			"persist.ims.vt=false\n"
			"persist.ims.vt.epdg=false\n"
			"persist.ims.rcs=false\n"
			"persist.radio.calls.on.ims=true\n"
			"persist.radio.jbims=1\n"
			"persist.radio.domain.ps=0\n"
			"persist.radio.VT_ENABLE=1\n"
			"persist.radio.VT_HYBRID_ENABLE=1\n"
			"persist.radio.ROTATION_ENABLE=1\n"
			"persist.radio.REVERSE_QMI=0\n"
			"persist.radio.RATE_ADAPT_ENABLE=1\n"
			"persist.rmnet.mux=enabled\n"
			"persist.radio.VT_USE_MDM_TIME=0\n"
			"persist.radio.videopause.mode=0\n"
			"persist.vt.supported=0\n"
			"persist.eab.supported=0\n"
			"persist.rcs.supported=0\n"
			"persist.rcs.presence.provision=0\n"
			"persist.ims.disableDebugLogs=0\n"
			"persist.ims.disableQXDMLogs=0\n"
			"persist.ims.disableIMSLogs=0\n"
			"persist.sys.dalvik.vm.lib.2=libart.so\n"
			"dalvik.vm.isa.arm.variant=cortex-a53\n"
			"dalvik.vm.isa.arm.features=default\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"sys.mod.platformsdkversion=105\n"
			"ro.mdtp.package_name2=com.qualcomm.qti.securemsm.mdtp.MdtpDemo\n"
			"ro.expect.recovery_id=0x0a7c12044b30be9e8bdfafd85cfb45489f44e2a4000000000000000000000000\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/accessory_chip",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/soc0/build_id",
		.size = 25,
		.content = "8937A-FAASANAZA-40000000\n",
	},
	{
		.path = "/sys/devices/soc0/family",
		.size = 11,
		.content = "Snapdragon\n",
	},
	{
		.path = "/sys/devices/soc0/foundry_id",
		.size = 2,
		.content = "2\n",
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
		.size = 13,
		.content =
			"cedric-user\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/image_version",
		.size = 20,
		.content =
			"10:NPP25.137-15:13\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/images",
		.size = 570,
		.content =
			"0:\n"
			"\tCRM:\t\t00:BOOT.BF.3.3-00193\n"
			"\tVariant:\tFAASANAZA\n"
			"\tVersion:\tshws28\n"
			"\n"
			"1:\n"
			"\tCRM:\t\t01:TZ.BF.4.0.5-00030\n"
			"\tVariant:\t\n"
			"\tVersion:\tCRM\n"
			"\n"
			"3:\n"
			"\tCRM:\t\t03:RPM.BF.2.2-00209\n"
			"\tVariant:\tAAAAANAAR\n"
			"\tVersion:\tshws28\n"
			"\n"
			"10:\n"
			"\tCRM:\t\t10:NPP25.137-15:13\n"
			"\n"
			"\tVariant:\tcedric-user\n"
			"\n"
			"\tVersion:\tREL\n"
			"\n"
			"\n"
			"11:\n"
			"\tCRM:\t\t11:MPSS.JO.2.0.C1-00122\n"
			"\tVariant:\t8937.genns.prodQ\n"
			"\tVersion:\tilclbld116/hudsonmd/3106b73\n"
			"\n"
			"12:\n"
			"\tCRM:\t\t12:ADSP.8953.2.8.2-00042\n"
			"\tVariant:\tAAAAAAAAQ\n"
			"\tVersion:\tCRM\n"
			"\n"
			"13:\n"
			"\tCRM:\t\t13:CNSS.PR.4.0-80391\n"
			"\tVariant:\tSCAQJAZM\n"
			"\tVersion:\tCRM\n"
			"\n"
			"14:\n"
			"\tCRM:\t\t14:VIDEO.VE_ULT.3.1-00025\n"
			"\tVariant:\tPROD\n"
			"\tVersion:\t:CRM\n"
			"\n",
	},
	{
		.path = "/sys/devices/soc0/machine",
		.size = 8,
		.content = "MSM8937\n",
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
		.size = 6,
		.content = "65536\n",
	},
	{
		.path = "/sys/devices/soc0/pmic_model",
		.size = 6,
		.content = "65561\n",
	},
	{
		.path = "/sys/devices/soc0/raw_id",
		.size = 3,
		.content = "79\n",
	},
	{
		.path = "/sys/devices/soc0/raw_version",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/soc0/revision",
		.size = 4,
		.content = "1.0\n",
	},
	{
		.path = "/sys/devices/soc0/select_image",
		.size = 3,
		.content = "10\n",
	},
	{
		.path = "/sys/devices/soc0/serial_number",
		.size = 11,
		.content = "2129040342\n",
	},
	{
		.path = "/sys/devices/soc0/soc_id",
		.size = 4,
		.content = "294\n",
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
		.size = 513,
		.content =
			"freq\t\tcpu0\t\tcpu1\t\tcpu2\t\tcpu3\t\tcpu4\t\tcpu5\t\tcpu6\t\tcpu7\t\t\n"
			"768000\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t41124\t\t41124\t\t41124\t\t41124\t\t\n"
			"902400\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t220\t\t220\t\t220\t\t220\t\t\n"
			"960000\t\t41797\t\t41797\t\t41797\t\t41797\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"998400\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t32\t\t32\t\t32\t\t32\t\t\n"
			"1094400\t\t452\t\t452\t\t452\t\t452\t\t5481\t\t5481\t\t5481\t\t5481\t\t\n"
			"1209600\t\t70\t\t70\t\t70\t\t70\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1248000\t\t860\t\t860\t\t860\t\t860\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1344000\t\t24\t\t24\t\t24\t\t24\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n"
			"1401000\t\t3653\t\t3653\t\t3653\t\t3653\t\tN/A\t\tN/A\t\tN/A\t\tN/A\t\t\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpufreq/current_in_state",
		.size = 0,
		.content = "",
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
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1401000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "960000\n",
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
		.size = 48,
		.content = "960000 1094400 1209600 1248000 1344000 1401000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "960000\n",
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
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 72,
		.content =
			"960000 41917\n"
			"1094400 452\n"
			"1209600 70\n"
			"1248000 860\n"
			"1344000 24\n"
			"1401000 3653\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "273\n",
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
		.content = "1\n",
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
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1401000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "960000\n",
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
		.size = 48,
		.content = "960000 1094400 1209600 1248000 1344000 1401000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "960000\n",
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
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/time_in_state",
		.size = 72,
		.content =
			"960000 42135\n"
			"1094400 452\n"
			"1209600 70\n"
			"1248000 860\n"
			"1344000 24\n"
			"1401000 3653\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu1/cpufreq/stats/total_trans",
		.size = 4,
		.content = "273\n",
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
		.content = "1\n",
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
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1401000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "960000\n",
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
		.size = 48,
		.content = "960000 1094400 1209600 1248000 1344000 1401000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "960000\n",
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
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/time_in_state",
		.size = 72,
		.content =
			"960000 42339\n"
			"1094400 452\n"
			"1209600 70\n"
			"1248000 860\n"
			"1344000 24\n"
			"1401000 3653\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu2/cpufreq/stats/total_trans",
		.size = 4,
		.content = "273\n",
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
		.content = "1\n",
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
		.size = 8,
		.content = "0 1 2 3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1401000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "960000\n",
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
		.size = 48,
		.content = "960000 1094400 1209600 1248000 1344000 1401000 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "960000\n",
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
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/time_in_state",
		.size = 72,
		.content =
			"960000 42558\n"
			"1094400 452\n"
			"1209600 70\n"
			"1248000 860\n"
			"1344000 24\n"
			"1401000 3653\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu3/cpufreq/stats/total_trans",
		.size = 4,
		.content = "273\n",
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
		.content = "1\n",
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
		.size = 8,
		.content = "4 5 6 7\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1094400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "768000\n",
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
		.size = 30,
		.content = "768000 902400 998400 1094400 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "768000\n",
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
		.content = "1094400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "768000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/time_in_state",
		.size = 47,
		.content =
			"768000 42073\n"
			"902400 224\n"
			"998400 32\n"
			"1094400 5491\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu4/cpufreq/stats/total_trans",
		.size = 4,
		.content = "255\n",
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
		.content = "1094400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "768000\n",
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
		.size = 30,
		.content = "768000 902400 998400 1094400 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "768000\n",
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
		.content = "1094400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "768000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/time_in_state",
		.size = 47,
		.content =
			"768000 42293\n"
			"902400 224\n"
			"998400 32\n"
			"1094400 5491\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu5/cpufreq/stats/total_trans",
		.size = 4,
		.content = "255\n",
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
		.content = "1094400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "768000\n",
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
		.size = 30,
		.content = "768000 902400 998400 1094400 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "768000\n",
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
		.content = "1094400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "768000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/time_in_state",
		.size = 47,
		.content =
			"768000 42511\n"
			"902400 224\n"
			"998400 32\n"
			"1094400 5491\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu6/cpufreq/stats/total_trans",
		.size = 4,
		.content = "255\n",
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
		.content = "1094400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "768000\n",
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
		.size = 30,
		.content = "768000 902400 998400 1094400 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "768000\n",
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
		.content = "1094400\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "768000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state",
		.size = 47,
		.content =
			"768000 42736\n"
			"902400 224\n"
			"998400 32\n"
			"1094400 5491\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu7/cpufreq/stats/total_trans",
		.size = 4,
		.content = "255\n",
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
		.key = "audio.dolby.ds2.enabled",
		.value = "true",
	},
	{
		.key = "audio.offload.buffer.size.kb",
		.value = "64",
	},
	{
		.key = "audio.offload.disable",
		.value = "false",
	},
	{
		.key = "audio.offload.gapless.enabled",
		.value = "false",
	},
	{
		.key = "audio.offload.min.duration.secs",
		.value = "60",
	},
	{
		.key = "audio.offload.multiple.enabled",
		.value = "false",
	},
	{
		.key = "audio.offload.pcm.16bit.enable",
		.value = "false",
	},
	{
		.key = "audio.offload.pcm.24bit.enable",
		.value = "false",
	},
	{
		.key = "audio.offload.track.enable",
		.value = "true",
	},
	{
		.key = "audio.offload.video",
		.value = "false",
	},
	{
		.key = "audio.parser.ip.buffer.size",
		.value = "262144",
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
		.value = "240",
	},
	{
		.key = "av.offload.enable",
		.value = "false",
	},
	{
		.key = "bluetooth.hfp.client",
		.value = "1",
	},
	{
		.key = "camera.disable_zsl_mode",
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
		.key = "camera.mot.startup_probing",
		.value = "0",
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
		.value = "384m",
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
		.value = "cortex-a53",
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
		.value = "1",
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
		.value = "1",
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
		.value = "false",
	},
	{
		.key = "gsm.operator.numeric",
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
		.value = "M8937_8000.122.02.40R CEDRIC_LATAMDSDS_CUST",
	},
	{
		.key = "gsm.version.baseband1",
		.value = "M8937_8000.122.02.40R CEDRIC_LATAMDSDS_CUST",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "Qualcomm RIL 1.0",
	},
	{
		.key = "hw.aov.disable_hotword",
		.value = "0",
	},
	{
		.key = "hw.aov.hotword_dsp_path",
		.value = "0",
	},
	{
		.key = "hw.motosh.booted",
		.value = "1",
	},
	{
		.key = "hw.touch.status",
		.value = "ready",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.adspd",
		.value = "running",
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
		.key = "init.svc.capsense_reset",
		.value = "running",
	},
	{
		.key = "init.svc.clear-bcb",
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
		.key = "init.svc.config_bluetooth",
		.value = "stopped",
	},
	{
		.key = "init.svc.config_bt_addr",
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
		.key = "init.svc.dropboxd",
		.value = "running",
	},
	{
		.key = "init.svc.emmc_ffu",
		.value = "stopped",
	},
	{
		.key = "init.svc.esdpll",
		.value = "running",
	},
	{
		.key = "init.svc.esepmdaemon",
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
		.key = "init.svc.hw_revs",
		.value = "stopped",
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
		.key = "init.svc.init_wifi",
		.value = "stopped",
	},
	{
		.key = "init.svc.installd",
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
		.key = "init.svc.mbm_spy",
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
		.key = "init.svc.mmi-audio-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.mmi-block-perm",
		.value = "stopped",
	},
	{
		.key = "init.svc.mmi-boot-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.mmi-laser-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.mmi-touch-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.mmi-usb-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.motosh",
		.value = "stopped",
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
		.key = "init.svc.oem-hw-sh",
		.value = "stopped",
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
		.key = "init.svc.pstore_annotate",
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
		.key = "init.svc.qcomsysd",
		.value = "running",
	},
	{
		.key = "init.svc.qe",
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
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.setup_shutdown",
		.value = "stopped",
	},
	{
		.key = "init.svc.ss_ramdump",
		.value = "running",
	},
	{
		.key = "init.svc.ssr_setup",
		.value = "stopped",
	},
	{
		.key = "init.svc.surfaceflinger",
		.value = "running",
	},
	{
		.key = "init.svc.tcmd",
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
		.key = "init.svc.touch-ready-sh",
		.value = "stopped",
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
		.key = "init.svc.wcnss-service",
		.value = "running",
	},
	{
		.key = "init.svc.zygote",
		.value = "running",
	},
	{
		.key = "installd.post_fs_data_ready",
		.value = "1",
	},
	{
		.key = "keyguard.no_require_sim",
		.value = "true",
	},
	{
		.key = "mdc_initial_max_retry",
		.value = "10",
	},
	{
		.key = "media.aac_51_output_enabled",
		.value = "true",
	},
	{
		.key = "media.msm8956hw",
		.value = "0",
	},
	{
		.key = "mm.enable.qcom_parser",
		.value = "4643",
	},
	{
		.key = "mm.enable.sec.smoothstreaming",
		.value = "false",
	},
	{
		.key = "mm.enable.smoothstreaming",
		.value = "false",
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
		.value = "android-156d0fabd8cd70d8",
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
		.key = "net.tethering.on",
		.value = "0",
	},
	{
		.key = "partition.system.verified",
		.value = "2",
	},
	{
		.key = "persist.audio.calfile0",
		.value = "/etc/acdbdata/Bluetooth_cal.acdb",
	},
	{
		.key = "persist.audio.calfile1",
		.value = "/etc/acdbdata/General_cal.acdb",
	},
	{
		.key = "persist.audio.calfile2",
		.value = "/etc/acdbdata/Global_cal.acdb",
	},
	{
		.key = "persist.audio.calfile3",
		.value = "/etc/acdbdata/Handset_cal.acdb",
	},
	{
		.key = "persist.audio.calfile4",
		.value = "/etc/acdbdata/Hdmi_cal.acdb",
	},
	{
		.key = "persist.audio.calfile5",
		.value = "/etc/acdbdata/Headset_cal.acdb",
	},
	{
		.key = "persist.audio.calfile6",
		.value = "/etc/acdbdata/Speaker_cal.acdb",
	},
	{
		.key = "persist.audio.dualmic.config",
		.value = "endfire",
	},
	{
		.key = "persist.audio.fluence.speaker",
		.value = "false",
	},
	{
		.key = "persist.audio.fluence.voicecall",
		.value = "true",
	},
	{
		.key = "persist.audio.fluence.voicecomm",
		.value = "true",
	},
	{
		.key = "persist.audio.fluence.voicerec",
		.value = "false",
	},
	{
		.key = "persist.camera.debug.logfile",
		.value = "0",
	},
	{
		.key = "persist.camera.gyro.android",
		.value = "1",
	},
	{
		.key = "persist.camera.gyro.disable",
		.value = "0",
	},
	{
		.key = "persist.camera.is_type",
		.value = "1",
	},
	{
		.key = "persist.cne.feature",
		.value = "1",
	},
	{
		.key = "persist.cne.logging.qxdm",
		.value = "3974",
	},
	{
		.key = "persist.cne.rat.wlan.chip.oem",
		.value = "WCN",
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
		.key = "persist.data.qmi.adb_logmask",
		.value = "0",
	},
	{
		.key = "persist.debug.coresight.config",
		.value = "stm-events",
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
		.key = "persist.eab.supported",
		.value = "0",
	},
	{
		.key = "persist.esdfs_sdcard",
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
		.key = "persist.ims.disableADBLogs",
		.value = "2",
	},
	{
		.key = "persist.ims.disableDebugLogs",
		.value = "0",
	},
	{
		.key = "persist.ims.disableIMSLogs",
		.value = "0",
	},
	{
		.key = "persist.ims.disableQXDMLogs",
		.value = "0",
	},
	{
		.key = "persist.ims.rcs",
		.value = "false",
	},
	{
		.key = "persist.ims.volte",
		.value = "true",
	},
	{
		.key = "persist.ims.vt",
		.value = "false",
	},
	{
		.key = "persist.ims.vt.epdg",
		.value = "false",
	},
	{
		.key = "persist.mm.sta.enable",
		.value = "0",
	},
	{
		.key = "persist.mot.gps.conf.from.sim",
		.value = "true",
	},
	{
		.key = "persist.mot.gps.smart_battery",
		.value = "1",
	},
	{
		.key = "persist.mot.usb.config",
		.value = "mtp,adb",
	},
	{
		.key = "persist.net.doxlat",
		.value = "true",
	},
	{
		.key = "persist.qcril_uim_vcc_feature",
		.value = "1",
	},
	{
		.key = "persist.qe",
		.value = "qe 0/0",
	},
	{
		.key = "persist.qfp",
		.value = "false",
	},
	{
		.key = "persist.radio.0x9e_not_callname",
		.value = "1",
	},
	{
		.key = "persist.radio.RATE_ADAPT_ENABLE",
		.value = "1",
	},
	{
		.key = "persist.radio.REVERSE_QMI",
		.value = "0",
	},
	{
		.key = "persist.radio.ROTATION_ENABLE",
		.value = "1",
	},
	{
		.key = "persist.radio.VT_ENABLE",
		.value = "1",
	},
	{
		.key = "persist.radio.VT_HYBRID_ENABLE",
		.value = "1",
	},
	{
		.key = "persist.radio.VT_USE_MDM_TIME",
		.value = "0",
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
		.key = "persist.radio.apn_delay",
		.value = "5000",
	},
	{
		.key = "persist.radio.app_mbn_path",
		.value = "/fsg",
	},
	{
		.key = "persist.radio.call.audio.output",
		.value = "0",
	},
	{
		.key = "persist.radio.calls.on.ims",
		.value = "true",
	},
	{
		.key = "persist.radio.custom_ecc",
		.value = "1",
	},
	{
		.key = "persist.radio.dfr_mode_set",
		.value = "1",
	},
	{
		.key = "persist.radio.domain.ps",
		.value = "0",
	},
	{
		.key = "persist.radio.eons.enabled",
		.value = "false",
	},
	{
		.key = "persist.radio.force_get_pref",
		.value = "1",
	},
	{
		.key = "persist.radio.is_wps_enabled",
		.value = "true",
	},
	{
		.key = "persist.radio.jbims",
		.value = "1",
	},
	{
		.key = "persist.radio.mcfg_ver_num",
		.value = "0,0",
	},
	{
		.key = "persist.radio.mcfg_version",
		.value = "null,null",
	},
	{
		.key = "persist.radio.msgtunnel.start",
		.value = "true",
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
		.value = "30",
	},
	{
		.key = "persist.radio.multisim.config",
		.value = "dsds",
	},
	{
		.key = "persist.radio.no_wait_for_card",
		.value = "1",
	},
	{
		.key = "persist.radio.oem_ind_to_both",
		.value = "0",
	},
	{
		.key = "persist.radio.relay_oprt_change",
		.value = "1",
	},
	{
		.key = "persist.radio.ril_payload_on",
		.value = "0",
	},
	{
		.key = "persist.radio.sar_sensor",
		.value = "1",
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
		.key = "persist.radio.stack_id_0",
		.value = "0",
	},
	{
		.key = "persist.radio.stack_id_1",
		.value = "1",
	},
	{
		.key = "persist.radio.sw_mbn_update",
		.value = "1",
	},
	{
		.key = "persist.radio.videopause.mode",
		.value = "0",
	},
	{
		.key = "persist.rcs.presence.provision",
		.value = "0",
	},
	{
		.key = "persist.rcs.supported",
		.value = "0",
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
		.key = "persist.rmnet.mux",
		.value = "enabled",
	},
	{
		.key = "persist.speaker.prot.enable",
		.value = "false",
	},
	{
		.key = "persist.sys.bootupvolume",
		.value = "11",
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
		.key = "persist.sys.locale",
		.value = "en-US",
	},
	{
		.key = "persist.sys.media.use-awesome",
		.value = "false",
	},
	{
		.key = "persist.sys.phonelock.mode",
		.value = "0",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
	},
	{
		.key = "persist.sys.qc.sub.rdump.max",
		.value = "0",
	},
	{
		.key = "persist.sys.qc.sub.rdump.on",
		.value = "1",
	},
	{
		.key = "persist.sys.ssr.enable_ramdumps",
		.value = "1",
	},
	{
		.key = "persist.sys.ssr.restart_level",
		.value = "ALL_ENABLE",
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
		.value = "104857600",
	},
	{
		.key = "persist.timed.enable",
		.value = "true",
	},
	{
		.key = "persist.vold.ecryptfs_supported",
		.value = "true",
	},
	{
		.key = "persist.vt.supported",
		.value = "0",
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
		.key = "qcom.bt.le_dev_pwr_class",
		.value = "1",
	},
	{
		.key = "qcom.hw.aac.encoder",
		.value = "false",
	},
	{
		.key = "qemu.hw.mainkeys",
		.value = "0",
	},
	{
		.key = "ril.baseband.config.ver_num",
		.value = "02.59",
	},
	{
		.key = "ril.baseband.config.version",
		.value = "CEDRIC_LATAMDSDS_CUST",
	},
	{
		.key = "ril.baseband.rfcable.status",
		.value = "0",
	},
	{
		.key = "ril.ecclist",
		.value = "000,08,110,999,118,119,190,066,060,911,112",
	},
	{
		.key = "ril.ecclist1",
		.value = "000,08,110,999,118,119,190,066,060,911,112",
	},
	{
		.key = "ril.ims.supported_services",
		.value = "0",
	},
	{
		.key = "ril.lte.bc.config",
		.value = "134285407",
	},
	{
		.key = "ril.qcril_pre_init_lock_held",
		.value = "0",
	},
	{
		.key = "ril.radio.ctbk_inst",
		.value = "733",
	},
	{
		.key = "ril.radio.ctbk_val",
		.value = "1,0,0,0,0,0,0,0,0,0,1,0,0,0,0",
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
		.value = "/system/vendor/lib/libril-qc-qmi-1.so",
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
		.key = "ro.bluetooth.hfp.ver",
		.value = "1.6",
	},
	{
		.key = "ro.board.platform",
		.value = "msm8937",
	},
	{
		.key = "ro.boot.baseband",
		.value = "msm",
	},
	{
		.key = "ro.boot.bl_state",
		.value = "1",
	},
	{
		.key = "ro.boot.bootdevice",
		.value = "7824900.sdhci",
	},
	{
		.key = "ro.boot.bootloader",
		.value = "0xB809",
	},
	{
		.key = "ro.boot.bootreason",
		.value = "reboot",
	},
	{
		.key = "ro.boot.btmacaddr",
		.value = "D0:77:14:C3:2D:B3",
	},
	{
		.key = "ro.boot.build_vars",
		.value = "NA",
	},
	{
		.key = "ro.boot.carrier",
		.value = "retla",
	},
	{
		.key = "ro.boot.cid",
		.value = "0x32",
	},
	{
		.key = "ro.boot.device",
		.value = "cedric",
	},
	{
		.key = "ro.boot.dualsim",
		.value = "true",
	},
	{
		.key = "ro.boot.emmc",
		.value = "true",
	},
	{
		.key = "ro.boot.flash.locked",
		.value = "1",
	},
	{
		.key = "ro.boot.fsg-id",
		.value = "",
	},
	{
		.key = "ro.boot.hardware",
		.value = "qcom",
	},
	{
		.key = "ro.boot.hardware.sku",
		.value = "XT1671",
	},
	{
		.key = "ro.boot.hwrev",
		.value = "0x8500",
	},
	{
		.key = "ro.boot.mode",
		.value = "normal",
	},
	{
		.key = "ro.boot.poweroff_alarm",
		.value = "0",
	},
	{
		.key = "ro.boot.powerup_reason",
		.value = "0x00004000",
	},
	{
		.key = "ro.boot.radio",
		.value = "LATAM",
	},
	{
		.key = "ro.boot.secure_hardware",
		.value = "1",
	},
	{
		.key = "ro.boot.serialno",
		.value = "ZY322M49RD",
	},
	{
		.key = "ro.boot.uid",
		.value = "D693E67E00000000000000000000",
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
		.key = "ro.boot.wifimacaddr",
		.value = "D0:77:14:C3:2D:B4,D0:77:14:C3:2D:B5",
	},
	{
		.key = "ro.boot.write_protect",
		.value = "1",
	},
	{
		.key = "ro.bootimage.build.date",
		.value = "Fri Jan 13 08:37:11 CST 2017",
	},
	{
		.key = "ro.bootimage.build.date.utc",
		.value = "1484318231",
	},
	{
		.key = "ro.bootimage.build.fingerprint",
		.value = "motorola/cedric_retail/cedric:7.0/NPP25.137-15/13:user/release-keys",
	},
	{
		.key = "ro.bootloader",
		.value = "0xB809",
	},
	{
		.key = "ro.bootmode",
		.value = "normal",
	},
	{
		.key = "ro.bootreason",
		.value = "reboot",
	},
	{
		.key = "ro.bug2go.magickeys",
		.value = "24,26",
	},
	{
		.key = "ro.build.characteristics",
		.value = "default",
	},
	{
		.key = "ro.build.date",
		.value = "Fri Jan 13 08:37:11 CST 2017",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1484318231",
	},
	{
		.key = "ro.build.description",
		.value = "cedric-user 7.0 NPP25.137-15 13 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "NPP25.137-15",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "motorola/cedric/cedric:7.0/NPP25.137-15/13:user/release-keys",
	},
	{
		.key = "ro.build.flavor",
		.value = "cedric-user",
	},
	{
		.key = "ro.build.host",
		.value = "ilclbld35",
	},
	{
		.key = "ro.build.id",
		.value = "NPP25.137-15",
	},
	{
		.key = "ro.build.product",
		.value = "cedric",
	},
	{
		.key = "ro.build.tags",
		.value = "release-keys",
	},
	{
		.key = "ro.build.thumbprint",
		.value = "7.0/NPP25.137-15/13:user/release-keys",
	},
	{
		.key = "ro.build.type",
		.value = "user",
	},
	{
		.key = "ro.build.user",
		.value = "hudsoncm",
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
		.key = "ro.build.version.full",
		.value = "Blur_Version.25.11.13.cedric.retail.en.US",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "13",
	},
	{
		.key = "ro.build.version.preview_sdk",
		.value = "0",
	},
	{
		.key = "ro.build.version.qcom",
		.value = "LA.UM.5.6.r1-01900-89xx.0",
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
		.value = "2017-01-01",
	},
	{
		.key = "ro.carrier",
		.value = "retla",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-motorola",
	},
	{
		.key = "ro.com.google.clientidbase.am",
		.value = "android-motorola",
	},
	{
		.key = "ro.com.google.clientidbase.gmm",
		.value = "android-motorola",
	},
	{
		.key = "ro.com.google.clientidbase.ms",
		.value = "android-motorola",
	},
	{
		.key = "ro.com.google.clientidbase.yt",
		.value = "android-motorola",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "7.0_r4",
	},
	{
		.key = "ro.com.google.ime.theme_id",
		.value = "4",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Oxygen.ogg",
	},
	{
		.key = "ro.config.max_starting_bg",
		.value = "8",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "Moto.ogg",
	},
	{
		.key = "ro.config.ringtone",
		.value = "Moto.ogg",
	},
	{
		.key = "ro.config.ringtone_2",
		.value = "Moto.ogg",
	},
	{
		.key = "ro.config.swap",
		.value = "true",
	},
	{
		.key = "ro.config.vc_call_vol_steps",
		.value = "8",
	},
	{
		.key = "ro.config.zram",
		.value = "true",
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
		.value = "3200",
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
		.value = "0x0a7c12044b30be9e8bdfafd85cfb45489f44e2a4000000000000000000000000",
	},
	{
		.key = "ro.fm.transmitter",
		.value = "false",
	},
	{
		.key = "ro.frp.pst",
		.value = "/dev/block/bootdevice/by-name/frp",
	},
	{
		.key = "ro.gpu.available_frequencies",
		.value = "450000000 400000000 375000000 300000000 216000000 ",
	},
	{
		.key = "ro.hardware",
		.value = "qcom",
	},
	{
		.key = "ro.hw.boardversion",
		.value = "PVT1",
	},
	{
		.key = "ro.hw.device",
		.value = "cedric",
	},
	{
		.key = "ro.hw.dtv",
		.value = "false",
	},
	{
		.key = "ro.hw.dualsim",
		.value = "true",
	},
	{
		.key = "ro.hw.ecompass",
		.value = "false",
	},
	{
		.key = "ro.hw.fps",
		.value = "true",
	},
	{
		.key = "ro.hw.frontcolor",
		.value = "(null)",
	},
	{
		.key = "ro.hw.hwrev",
		.value = "0x8500",
	},
	{
		.key = "ro.hw.imager",
		.value = "13MP",
	},
	{
		.key = "ro.hw.nfc",
		.value = "false",
	},
	{
		.key = "ro.hw.radio",
		.value = "LATAM",
	},
	{
		.key = "ro.hw.ram",
		.value = "2GB",
	},
	{
		.key = "ro.hw.revision",
		.value = "p5",
	},
	{
		.key = "ro.hw.storage",
		.value = "32GB",
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
		.value = "2048",
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
		.key = "ro.kick.logcopy",
		.value = "reboot",
	},
	{
		.key = "ro.lenovo.single_hand",
		.value = "1",
	},
	{
		.key = "ro.logdumpd.enabled",
		.value = "0",
	},
	{
		.key = "ro.manufacturedate",
		.value = "3/6/2018",
	},
	{
		.key = "ro.mdtp.package_name2",
		.value = "com.qualcomm.qti.securemsm.mdtp.MdtpDemo",
	},
	{
		.key = "ro.mot.build.customerid",
		.value = "retail",
	},
	{
		.key = "ro.mot.build.oem.product",
		.value = "cedric",
	},
	{
		.key = "ro.mot.build.product.increment",
		.value = "11",
	},
	{
		.key = "ro.mot.build.version.release",
		.value = "25.11",
	},
	{
		.key = "ro.mot.build.version.sdk_int",
		.value = "25",
	},
	{
		.key = "ro.mot.ignore_csim_appid",
		.value = "true",
	},
	{
		.key = "ro.mot.security.enable",
		.value = "true",
	},
	{
		.key = "ro.oem_unlock_supported",
		.value = "1",
	},
	{
		.key = "ro.opengles.version",
		.value = "196609",
	},
	{
		.key = "ro.product.board",
		.value = "msm8937",
	},
	{
		.key = "ro.product.brand",
		.value = "motorola",
	},
	{
		.key = "ro.product.cpu.abi",
		.value = "armeabi-v7a",
	},
	{
		.key = "ro.product.cpu.abi2",
		.value = "armeabi",
	},
	{
		.key = "ro.product.cpu.abilist",
		.value = "armeabi-v7a,armeabi",
	},
	{
		.key = "ro.product.cpu.abilist32",
		.value = "armeabi-v7a,armeabi",
	},
	{
		.key = "ro.product.cpu.abilist64",
		.value = "",
	},
	{
		.key = "ro.product.device",
		.value = "cedric",
	},
	{
		.key = "ro.product.display",
		.value = "Moto GxE2x81xB5",
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
		.value = "motorola",
	},
	{
		.key = "ro.product.model",
		.value = "Moto G (5)",
	},
	{
		.key = "ro.product.name",
		.value = "cedric",
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
		.key = "ro.qualcomm.bt.hci_transport",
		.value = "smd",
	},
	{
		.key = "ro.qualcomm.cabl",
		.value = "0",
	},
	{
		.key = "ro.radio.imei.sv",
		.value = "2",
	},
	{
		.key = "ro.revision",
		.value = "p500",
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
		.value = "1485562460740",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "ZY322M49RD",
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
		.value = "9",
	},
	{
		.key = "ro.url.legal",
		.value = "http://www.google.com/intl/%s/mobile/android/basic/phone-legal.html",
	},
	{
		.key = "ro.url.legal.android_privacy",
		.value = "http://www.google.com/intl/%s/mobile/android/basic/privacy.html",
	},
	{
		.key = "ro.usb.bpt",
		.value = "2ec1",
	},
	{
		.key = "ro.usb.bpt_adb",
		.value = "2ec5",
	},
	{
		.key = "ro.usb.bpteth",
		.value = "2ec3",
	},
	{
		.key = "ro.usb.bpteth_adb",
		.value = "2ec6",
	},
	{
		.key = "ro.usb.mtp",
		.value = "2e82",
	},
	{
		.key = "ro.usb.mtp_adb",
		.value = "2e76",
	},
	{
		.key = "ro.usb.ptp",
		.value = "2e83",
	},
	{
		.key = "ro.usb.ptp_adb",
		.value = "2e84",
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
		.key = "service.bootanim.exit",
		.value = "1",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
	},
	{
		.key = "sys.bootbroadcast_completed",
		.value = "1",
	},
	{
		.key = "sys.ims.QMI_DAEMON_STATUS",
		.value = "1",
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
		.key = "sys.mod.platformsdkversion",
		.value = "105",
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
		.key = "sys.rmnet_vnd.rps_mask",
		.value = "0",
	},
	{
		.key = "sys.runtime.restart.times",
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
		.value = "40",
	},
	{
		.key = "sys.usb.state",
		.value = "mtp,adb",
	},
	{
		.key = "tcmd.blan.interface",
		.value = "usb0",
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
		.value = "false",
	},
	{
		.key = "vidc.disable.split.mode",
		.value = "1",
	},
	{
		.key = "vidc.enc.narrow.searchrange",
		.value = "1",
	},
	{
		.key = "video.disable.ubwc",
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
		.key = "vold.emulated_mount_state",
		.value = "true",
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
