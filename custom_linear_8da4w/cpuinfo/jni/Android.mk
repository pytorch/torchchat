LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)
LOCAL_MODULE := cpuinfo
LOCAL_SRC_FILES := \
	src/init.c \
	src/api.c \
	src/cache.c \
	src/linux/processors.c \
	src/linux/smallfile.c \
	src/linux/multiline.c \
	src/linux/cpulist.c
ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),armeabi-v7a arm64-v8a))
LOCAL_SRC_FILES += \
	src/arm/uarch.c \
	src/arm/cache.c \
	src/arm/linux/init.c \
	src/arm/linux/cpuinfo.c \
	src/arm/linux/clusters.c \
	src/arm/linux/chipset.c \
	src/arm/linux/midr.c \
	src/arm/linux/hwcap.c \
	src/arm/android/properties.c
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
LOCAL_SRC_FILES += src/arm/linux/aarch32-isa.c
endif # armeabi-v7a
ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
LOCAL_SRC_FILES += src/arm/linux/aarch64-isa.c
endif # arm64-v8a
endif # armeabi-v7a, or arm64-v8a
ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),x86 x86_64))
LOCAL_SRC_FILES += \
	src/x86/init.c \
	src/x86/info.c \
	src/x86/name.c \
	src/x86/isa.c \
	src/x86/vendor.c \
	src/x86/uarch.c \
	src/x86/topology.c \
	src/x86/cache/init.c \
	src/x86/cache/descriptor.c \
	src/x86/cache/deterministic.c \
	src/x86/linux/cpuinfo.c \
	src/x86/linux/init.c
endif # x86 or x86_64
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
LOCAL_C_INCLUDES := $(LOCAL_EXPORT_C_INCLUDES) $(LOCAL_PATH)/src
LOCAL_CFLAGS := -std=c99 -Wall -D_GNU_SOURCE=1
ifeq (,$(findstring 4.9,$(NDK_TOOLCHAIN)))
# Clang compiler supports -Oz
LOCAL_CFLAGS += -Oz
else
# gcc-4.9 compiler supports only -Os
LOCAL_CFLAGS += -Os
endif
ifeq ($(NDK_DEBUG),1)
LOCAL_CFLAGS += -DCPUINFO_LOG_LEVEL=5
else
LOCAL_CFLAGS += -DCPUINFO_LOG_LEVEL=0
endif
LOCAL_STATIC_LIBRARIES := clog
include $(BUILD_STATIC_LIBRARY)

$(call import-add-path,$(LOCAL_PATH)/deps)

$(call import-module,clog/jni)
