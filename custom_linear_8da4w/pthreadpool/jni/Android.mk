LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)
LOCAL_MODULE := pthreadpool_interface
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pthreadpool
LOCAL_SRC_FILES := src/threadpool-pthreads.c
LOCAL_CFLAGS := -std=c99 -Wall
LOCAL_STATIC_LIBRARIES := pthreadpool_interface fxdiv
include $(BUILD_STATIC_LIBRARY)

$(call import-add-path,$(LOCAL_PATH)/deps)

$(call import-module,fxdiv/jni)
