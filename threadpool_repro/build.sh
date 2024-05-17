rm -rf build
mkdir build


CMAKE_PREFIX_PATH="/Users/scroy/repos/pytorch/torch/share/cmake"
# CMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"

cmake -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" -DTORCHCHAT_ROOT="${PWD}/.." -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake -DCMAKE_PREFIX_PATH= -DTORCHCHAT_ROOT="${PWD}/.." -S . -B build -GNinja
cmake --build build
