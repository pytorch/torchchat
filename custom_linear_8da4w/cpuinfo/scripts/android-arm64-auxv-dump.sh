#!/usr/bin/env bash

set -e

adb push build/android/arm64-v8a/auxv-dump /data/local/tmp/auxv-dump
adb shell /data/local/tmp/auxv-dump
