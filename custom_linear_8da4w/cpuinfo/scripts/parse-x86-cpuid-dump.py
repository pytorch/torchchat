#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys
import re


parser = argparse.ArgumentParser(description='x86 CPUID dump parser')
parser.add_argument("input", metavar="INPUT", nargs=1,
                    help="Path to CPUID dump log")


def main(args):
    options = parser.parse_args(args)

    cpuid_dump = list()
    for line in open(options.input[0]).read().splitlines():
        match = re.match(r"CPUID ([\dA-F]{8}): ([\dA-F]{8})-([\dA-F]{8})-([\dA-F]{8})-([\dA-F]{8})", line)
        if match is not None:
            input_eax, eax, ebx, ecx, edx = tuple(int(match.group(i), 16) for i in [1, 2, 3, 4, 5])
            line = line[match.end(0):].strip()
            input_ecx = None
            match = re.match(r"\[SL (\d{2})\]", line)
            if match is not None:
                input_ecx = int(match.group(1), 16)
            cpuid_dump.append((input_eax, input_ecx, eax, ebx, ecx, edx))


    print("struct cpuinfo_mock_cpuid cpuid_dump[] = {")
    for input_eax, input_ecx, eax, ebx, ecx, edx in cpuid_dump:
        print("\t{")
        print("\t\t.input_eax = 0x%08X," % input_eax)
        if input_ecx is not None:
            print("\t\t.input_ecx = 0x%08X," % input_ecx)
        print("\t\t.eax = 0x%08X," % eax)
        print("\t\t.ebx = 0x%08X," % ebx)
        print("\t\t.ecx = 0x%08X," % ecx)
        print("\t\t.edx = 0x%08X," % edx)
        print("\t},")
    print("};")
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
