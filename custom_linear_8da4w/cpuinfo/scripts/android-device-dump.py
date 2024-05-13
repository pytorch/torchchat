#!/usr/bin/env python

import os
import sys
import string
import argparse
import subprocess
import tempfile


root_dir = os.path.abspath(os.path.dirname(__file__))


parser = argparse.ArgumentParser(description='Android system files extractor')
parser.add_argument("-p", "--prefix", metavar="NAME", required=True,
                    help="Prefix for stored files, e.g. galaxy-s7-us")


# System files which need to be read with `adb shell cat filename`
# instead of `adb pull filename`
SHELL_PREFIX = [
    "/sys/class/kgsl/kgsl-3d0/",
]

SYSTEM_FILES = [
    "/proc/cpuinfo",
    "/system/build.prop",
    "/sys/class/kgsl/kgsl-3d0/bus_split",
    "/sys/class/kgsl/kgsl-3d0/clock_mhz",
    "/sys/class/kgsl/kgsl-3d0/deep_nap_timer",
    "/sys/class/kgsl/kgsl-3d0/default_pwrlevel",
    "/sys/class/kgsl/kgsl-3d0/dev",
    "/sys/class/kgsl/kgsl-3d0/devfreq/available_frequencies",
    "/sys/class/kgsl/kgsl-3d0/devfreq/available_governors",
    "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
    "/sys/class/kgsl/kgsl-3d0/devfreq/governor",
    "/sys/class/kgsl/kgsl-3d0/devfreq/gpu_load",
    "/sys/class/kgsl/kgsl-3d0/devfreq/max_freq",
    "/sys/class/kgsl/kgsl-3d0/devfreq/min_freq",
    "/sys/class/kgsl/kgsl-3d0/devfreq/polling_interval",
    "/sys/class/kgsl/kgsl-3d0/devfreq/suspend_time",
    "/sys/class/kgsl/kgsl-3d0/devfreq/target_freq",
    "/sys/class/kgsl/kgsl-3d0/devfreq/trans_stat",
    "/sys/class/kgsl/kgsl-3d0/device/op_cpu_table",
    "/sys/class/kgsl/kgsl-3d0/freq_table_mhz",
    "/sys/class/kgsl/kgsl-3d0/ft_fast_hang_detect",
    "/sys/class/kgsl/kgsl-3d0/ft_hang_intr_status",
    "/sys/class/kgsl/kgsl-3d0/ft_long_ib_detect",
    "/sys/class/kgsl/kgsl-3d0/ft_pagefault_policy",
    "/sys/class/kgsl/kgsl-3d0/ft_policy",
    "/sys/class/kgsl/kgsl-3d0/gpu_available_frequencies",
    "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
    "/sys/class/kgsl/kgsl-3d0/gpu_clock_stats",
    "/sys/class/kgsl/kgsl-3d0/gpu_llc_slice_enable",
    "/sys/class/kgsl/kgsl-3d0/gpu_model",
    "/sys/class/kgsl/kgsl-3d0/gpubusy",
    "/sys/class/kgsl/kgsl-3d0/gpuclk",
    "/sys/class/kgsl/kgsl-3d0/gpuhtw_llc_slice_enable",
    "/sys/class/kgsl/kgsl-3d0/hwcg",
    "/sys/class/kgsl/kgsl-3d0/idle_timer",
    "/sys/class/kgsl/kgsl-3d0/lm",
    "/sys/class/kgsl/kgsl-3d0/max_gpuclk",
    "/sys/class/kgsl/kgsl-3d0/max_pwrlevel",
    "/sys/class/kgsl/kgsl-3d0/min_clock_mhz",
    "/sys/class/kgsl/kgsl-3d0/min_pwrlevel",
    "/sys/class/kgsl/kgsl-3d0/num_pwrlevels",
    "/sys/class/kgsl/kgsl-3d0/pmqos_active_latency",
    "/sys/class/kgsl/kgsl-3d0/popp",
    "/sys/class/kgsl/kgsl-3d0/preempt_count",
    "/sys/class/kgsl/kgsl-3d0/preempt_level",
    "/sys/class/kgsl/kgsl-3d0/preemption",
    "/sys/class/kgsl/kgsl-3d0/pwrscale",
    "/sys/class/kgsl/kgsl-3d0/reset_count",
    "/sys/class/kgsl/kgsl-3d0/skipsaverestore",
    "/sys/class/kgsl/kgsl-3d0/sptp_pc",
    "/sys/class/kgsl/kgsl-3d0/thermal_pwrlevel",
    "/sys/class/kgsl/kgsl-3d0/throttling",
    "/sys/class/kgsl/kgsl-3d0/usesgmem",
    "/sys/class/kgsl/kgsl-3d0/wake_nice",
    "/sys/class/kgsl/kgsl-3d0/wake_timeout",
    "/sys/devices/soc0/accessory_chip",
    "/sys/devices/soc0/build_id",
    "/sys/devices/soc0/chip_family",
    "/sys/devices/soc0/chip_name",
    "/sys/devices/soc0/family",
    "/sys/devices/soc0/foundry_id",
    "/sys/devices/soc0/hw_platform",
    "/sys/devices/soc0/image_crm_version",
    "/sys/devices/soc0/image_variant",
    "/sys/devices/soc0/image_version",
    "/sys/devices/soc0/images",
    "/sys/devices/soc0/machine",
    "/sys/devices/soc0/ncluster_array_offset",
    "/sys/devices/soc0/ndefective_parts_array_offset",
    "/sys/devices/soc0/nmodem_supported",
    "/sys/devices/soc0/nproduct_id",
    "/sys/devices/soc0/num_clusters",
    "/sys/devices/soc0/num_defective_parts",
    "/sys/devices/soc0/platform_subtype",
    "/sys/devices/soc0/platform_subtype_id",
    "/sys/devices/soc0/platform_version",
    "/sys/devices/soc0/pmic_die_revision",
    "/sys/devices/soc0/pmic_model",
    "/sys/devices/soc0/raw_device_family",
    "/sys/devices/soc0/raw_device_number",
    "/sys/devices/soc0/raw_id",
    "/sys/devices/soc0/raw_version",
    "/sys/devices/soc0/revision",
    "/sys/devices/soc0/select_image",
    "/sys/devices/soc0/serial_number",
    "/sys/devices/soc0/soc_id",
    "/sys/devices/soc0/vendor",
    "/sys/devices/system/b.L/big_threads",
    "/sys/devices/system/b.L/boot_cluster",
    "/sys/devices/system/b.L/core_status",
    "/sys/devices/system/b.L/little_threads",
    "/sys/devices/system/b.L/down_migrations",
    "/sys/devices/system/b.L/up_migrations",
    "/sys/devices/system/cpu/isolated",
    "/sys/devices/system/cpu/kernel_max",
    "/sys/devices/system/cpu/modalias",
    "/sys/devices/system/cpu/offline",
    "/sys/devices/system/cpu/online",
    "/sys/devices/system/cpu/possible",
    "/sys/devices/system/cpu/present",
    "/sys/devices/system/cpu/sched_isolated",
    "/sys/devices/system/cpu/clusterhotplug/cur_hstate",
    "/sys/devices/system/cpu/clusterhotplug/down_freq",
    "/sys/devices/system/cpu/clusterhotplug/down_tasks",
    "/sys/devices/system/cpu/clusterhotplug/down_threshold",
    "/sys/devices/system/cpu/clusterhotplug/sampling_rate",
    "/sys/devices/system/cpu/clusterhotplug/time_in_state",
    "/sys/devices/system/cpu/clusterhotplug/up_freq",
    "/sys/devices/system/cpu/clusterhotplug/up_tasks",
    "/sys/devices/system/cpu/clusterhotplug/up_threshold",
    "/sys/devices/system/cpu/cpufreq/all_time_in_state",
    "/sys/devices/system/cpu/cpufreq/current_in_state",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/big_cpu_num",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/big_max_freq",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/big_min_freq",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/hmp_boost_type",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/hmp_prev_boost_type",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/ltl_cpu_num",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/ltl_divider",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/ltl_max_freq",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/ltl_min_freq",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/ltl_min_lock",
    "/sys/devices/system/cpu/cpufreq/cpufreq_limit/requests",
    "/sys/devices/system/cpu/cpuidle/current_driver",
    "/sys/devices/system/cpu/cpuidle/current_governor_ro",
    "/sys/devices/system/cpu/cputopo/cpus_per_cluster",
    "/sys/devices/system/cpu/cputopo/big_cpumask",
    "/sys/devices/system/cpu/cputopo/glbinfo",
    "/sys/devices/system/cpu/cputopo/is_big_little",
    "/sys/devices/system/cpu/cputopo/is_multi_cluster",
    "/sys/devices/system/cpu/cputopo/little_cpumask",
    "/sys/devices/system/cpu/cputopo/nr_clusters",
    "/sys/devices/system/sched/idle_prefer",
    "/sys/devices/system/sched/sched_boost",
]

CPU_FILES = [
    "core_ctl/active_cpus",
    "core_ctl/busy_up_thres",
    "core_ctl/busy_down_thres",
    "core_ctl/enable",
    "core_ctl/global_state",
    "core_ctl/is_big_cluster",
    "core_ctl/max_cpus",
    "core_ctl/min_cpus",
    "core_ctl/need_cpus",
    "core_ctl/not_preferred",
    "core_ctl/offline_delay_ms",
    "core_ctl/task_thres",
    "current_driver",
    "current_governor_ro",
    "cpuidle/driver/name",
    "cpufreq/affected_cpus",
    "cpufreq/cpuinfo_max_freq",
    "cpufreq/cpuinfo_min_freq",
    "cpufreq/cpuinfo_transition_latency",
    "cpufreq/related_cpus",
    "cpufreq/scaling_available_frequencies",
    "cpufreq/scaling_available_governors",
    "cpufreq/scaling_cur_freq",
    "cpufreq/scaling_driver",
    "cpufreq/scaling_governor",
    "cpufreq/scaling_max_freq",
    "cpufreq/scaling_min_freq",
    "cpufreq/sched/down_throttle_nsec",
    "cpufreq/sched/up_throttle_nsec",
    "cpufreq/stats/time_in_state",
    "cpufreq/stats/total_trans",
    "cpufreq/stats/trans_table",
    "isolate",
    "regs/identification/midr_el1",
    "regs/identification/revidr_el1",
    "sched_load_boost",
    "topology/core_id",
    "topology/core_siblings",
    "topology/core_siblings_list",
    "topology/cpu_capacity",
    "topology/max_cpu_capacity",
    "topology/physical_package_id",
    "topology/thread_siblings",
    "topology/thread_siblings_list",
]

CACHE_FILES = [
    "allocation_policy",
    "coherency_line_size",
    "level",
    "number_of_sets",
    "shared_cpu_list",
    "shared_cpu_map",
    "size",
    "type",
    "ways_of_associativity",
    "write_policy",
]

def c_escape(string):
    c_string = ""
    for c in string:
        if c == "\\":
            c_string += "\\\\"
        elif c == "\"":
            c_string += "\\\""
        elif c == "\t":
            c_string += "\\t"
        elif c == "\n":
            c_string += "\\n"
        elif c == "\r":
            c_string += "\\r"
        elif ord(c) == 0:
            c_string += "\\0"
        elif 32 <= ord(c) < 127:
            c_string += c
        else:
            c_string += "x%02X" % ord(c)
    return c_string

def adb_shell(commands):
    env = os.environ.copy()
    env["LC_ALL"] = "C"

    adb = subprocess.Popen(["adb", "shell"] + commands, env=env, stdout=subprocess.PIPE)
    stdout, _ = adb.communicate()
    if adb.returncode == 0:
        return stdout

def adb_push(local_path, device_path):
    env = os.environ.copy()
    env["LC_ALL"] = "C"

    adb = subprocess.Popen(["adb", "push", local_path, device_path], env=env)
    adb.communicate()
    return adb.returncode == 0

def adb_pull(device_path, local_path):
    if any(device_path.startswith(prefix) for prefix in SHELL_PREFIX):
        content = adb_shell(["cat", device_path])
        if content is not None:
            if not content.rstrip().endswith("No such file or directory"):
                with open(local_path, "wb") as local_file:
                    local_file.write(content)
                    return True
    else:
        env = os.environ.copy()
        env["LC_ALL"] = "C"

        adb = subprocess.Popen(["adb", "pull", device_path, local_path], env=env)
        adb.communicate()
        return adb.returncode == 0

def adb_getprop():
    properties = adb_shell(["getprop"])
    properties_list = list()
    while properties:
        assert properties.startswith("[")
        properties = properties[1:]
        key, properties = properties.split("]", 1)
        properties = properties.strip()
        assert properties.startswith(":")
        properties = properties[1:].strip()
        assert properties.startswith("[")
        properties = properties[1:]
        value, properties = properties.split("]", 1)
        properties = properties.strip()
        properties_list.append((key, value))
    return properties_list

def add_mock_file(stream, path, content):
    assert content is not None
    stream.write("\t{\n")
    stream.write("\t\t.path = \"%s\",\n" % path)
    stream.write("\t\t.size = %d,\n" % len(content))
    if len(content.splitlines()) > 1:
        stream.write("\t\t.content =")
        for line in content.splitlines(True):
            stream.write("\n\t\t\t\"%s\"" % c_escape(line))
        stream.write(",\n")
    else:
        stream.write("\t\t.content = \"%s\",\n" % c_escape(content))
    stream.write("\t},\n")


def dump_device_file(stream, path, prefix_line=None):
    temp_fd, temp_path = tempfile.mkstemp()
    os.close(temp_fd)
    try:
        if adb_pull(path, temp_path):
            with open(temp_path, "rb") as temp_file:
                content = temp_file.read()
                if prefix_line is not None:
                    stream.write(prefix_line)
                add_mock_file(stream, path, content)
                return content
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main(args):
    options = parser.parse_args(args)

    dmesg_content = adb_shell(["dmesg"])
    if dmesg_content is not None and dmesg_content.strip() == "klogctl: Operation not permitted":
        dmesg_content = None
    if dmesg_content is not None:
        with open(os.path.join("test", "dmesg", options.prefix + ".log"), "w") as dmesg_dump:
            dmesg_dump.write(dmesg_content)

    build_prop_content = None
    proc_cpuinfo_content = None
    proc_cpuinfo_content32 = None
    kernel_max = 0
    with open(os.path.join("test", "mock", options.prefix + ".h"), "w") as file_header:
        file_header.write("struct cpuinfo_mock_file filesystem[] = {\n")
        android_props = adb_getprop()
        abi = None
        for key, value in android_props:
            if key == "ro.product.cpu.abi":
                abi = value
        for path in SYSTEM_FILES:
            arm64_prefix = None
            if path == "/proc/cpuinfo" and abi == "arm64-v8a":
                arm64_prefix = "#if CPUINFO_ARCH_ARM64\n"
            content = dump_device_file(file_header, path, prefix_line=arm64_prefix)
            if content is not None:
                if path == "/proc/cpuinfo":
                    proc_cpuinfo_content = content
                elif path == "/system/build.prop":
                    build_prop_content = content
                elif path == "/sys/devices/system/cpu/kernel_max":
                    kernel_max = int(content.strip())
            if arm64_prefix:
                cpuinfo_dump_binary = os.path.join(root_dir, "..", "build", "android", "armeabi-v7a", "cpuinfo-dump")
                assert os.path.isfile(cpuinfo_dump_binary)
                adb_push(cpuinfo_dump_binary, "/data/local/tmp/cpuinfo-dump")
                proc_cpuinfo_content32 = adb_shell(["/data/local/tmp/cpuinfo-dump"])
                if proc_cpuinfo_content32:
                    proc_cpuinfo_content32 = "\n".join(proc_cpuinfo_content32.splitlines())
                    file_header.write("#elif CPUINFO_ARCH_ARM\n")
                    add_mock_file(file_header, "/proc/cpuinfo", proc_cpuinfo_content32)
                file_header.write("#endif\n")

        for cpu in range(kernel_max + 1):
            for filename in CPU_FILES:
                path = "/sys/devices/system/cpu/cpu%d/%s" % (cpu, filename)
                dump_device_file(file_header, path)
            for index in range(5):
                for filename in CACHE_FILES:
                    path = "/sys/devices/system/cpu/cpu%d/cache/index%d/%s" % (cpu, index, filename)
                    dump_device_file(file_header, path)
        file_header.write("\t{ NULL },\n")
        file_header.write("};\n")
        file_header.write("#ifdef __ANDROID__\n")
        file_header.write("struct cpuinfo_mock_property properties[] = {\n")
        for key, value in android_props:
            file_header.write("\t{\n")
            file_header.write("\t\t.key = \"%s\",\n" % c_escape(key))
            file_header.write("\t\t.value = \"%s\",\n" % c_escape(value))
            file_header.write("\t},\n")
        file_header.write("\t{ NULL },\n")
        file_header.write("};\n")
        file_header.write("#endif /* __ANDROID__ */\n")

    if proc_cpuinfo_content is not None:
        with open(os.path.join("test", "cpuinfo", options.prefix + ".log"), "w") as proc_cpuinfo_dump:
            proc_cpuinfo_dump.write(proc_cpuinfo_content)
    if proc_cpuinfo_content32 is not None:
        with open(os.path.join("test", "cpuinfo", options.prefix + ".armeabi.log"), "w") as proc_cpuinfo_dump32:
            proc_cpuinfo_dump32.write(proc_cpuinfo_content32)
    if build_prop_content is not None:
        with open(os.path.join("test", "build.prop", options.prefix + ".log"), "w") as build_prop_dump:
            build_prop_dump.write(build_prop_content)

if __name__ == "__main__":
    main(sys.argv[1:])
