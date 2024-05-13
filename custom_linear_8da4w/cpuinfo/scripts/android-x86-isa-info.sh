#!/usr/bin/env bash

set -e

adb push build/android/x86/isa-info /data/local/tmp/isa-info
adb shell /data/local/tmp/isa-info
