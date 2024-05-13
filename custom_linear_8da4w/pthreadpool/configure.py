#!/usr/bin/env python


import confu
parser = confu.standard_parser("pthreadpool configuration script")


def main(args):
    options = parser.parse_args(args)
    build = confu.Build.from_options(options)

    build.export_cpath("include", ["pthreadpool.h"])

    with build.options(source_dir="src", extra_include_dirs="src", deps=build.deps.fxdiv):
        sources = ["legacy-api.c", "portable-api.c"]
        if build.target.is_emscripten:
            sources.append("shim.c")
        elif build.target.is_macos:
            sources.append("gcd.c")
        elif build.target.is_windows:
            sources.append("windows.c")
        else:
            sources.append("pthreads.c")
        build.static_library("pthreadpool", [build.cc(src) for src in sources])

    with build.options(source_dir="test", deps=[build, build.deps.googletest]):
        build.unittest("pthreadpool-test", build.cxx("pthreadpool.cc"))

    with build.options(source_dir="bench", deps=[build, build.deps.googlebenchmark]):
        build.benchmark("latency-bench", build.cxx("latency.cc"))
        build.benchmark("throughput-bench", build.cxx("throughput.cc"))

    return build


if __name__ == "__main__":
    import sys
    main(sys.argv[1:]).generate()
