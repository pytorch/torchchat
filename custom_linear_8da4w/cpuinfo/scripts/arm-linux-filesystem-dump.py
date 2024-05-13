#!/usr/bin/env python

import os
import sys
import argparse
import shutil


parser = argparse.ArgumentParser(description='Android system files extractor')
parser.add_argument("-p", "--prefix", metavar="NAME", required=True,
                    help="Prefix for stored files, e.g. galaxy-s7-us")


SYSTEM_FILES = [
    "/proc/cpuinfo",
    "/sys/devices/system/cpu/kernel_max",
    "/sys/devices/system/cpu/possible",
    "/sys/devices/system/cpu/present",
]

CPU_FILES = [
    "cpufreq/cpuinfo_max_freq",
    "cpufreq/cpuinfo_min_freq",
    "topology/physical_package_id",
    "topology/core_siblings_list",
    "topology/core_id",
    "topology/thread_siblings_list",
]

CACHE_FILES = [
    "allocation_policy",
    "coherency_line_size",
    "level",
    "number_of_sets",
    "shared_cpu_list",
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


def dump_system_file(stream, path):
    try:
        with open(path, "rb") as device_file:
            content = device_file.read()
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
            return True
    except IOError:
        pass


def main(args):
    options = parser.parse_args(args)

    # with open(os.path.join("test", "dmesg", options.prefix + ".log"), "w") as dmesg_log:
    #     dmesg_log.write(device.Shell("dmesg"))
    with open(os.path.join("test", options.prefix + ".h"), "w") as file_header:
        file_header.write("struct cpuinfo_mock_file filesystem[] = {\n")
        for path in SYSTEM_FILES:
            dump_system_file(file_header, path)
        for cpu in range(16):
            for filename in CPU_FILES:
                path = "/sys/devices/system/cpu/cpu%d/%s" % (cpu, filename)
                dump_system_file(file_header, path)
            for index in range(10):
                for filename in CACHE_FILES:
                    path = "/sys/devices/system/cpu/cpu%d/cache/index%d/%s" % (cpu, index, filename)
                    dump_system_file(file_header, path)
        file_header.write("\t{ NULL },\n")
        file_header.write("};\n")
    shutil.copy("/proc/cpuinfo",
        os.path.join("test", "cpuinfo", options.prefix + ".log"))

if __name__ == "__main__":
    main(sys.argv[1:])
