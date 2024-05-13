#!/usr/bin/env bash

set -e

adb push build/android/x86/cpuid-dump /data/local/tmp/cpuid-dump
adb shell /data/local/tmp/cpuid-dump
