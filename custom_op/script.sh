#!/usr/bin/env bash

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function configure() {
    cd $script_dir
    rm -rf build
    mkdir -p build
    cd build
    cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
}


function build() {
    cd $script_dir/build
    make -j
}

function run() {
    cd $script_dir
    for i in 0 1 2 3; do
        python ./linear.py
    done
}

$@
