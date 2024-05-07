rm -rf build
mkdir build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" -S . -B build
# cmake -DCMAKE_PREFIX_PATH="/Users/scroy/repos/pytorch/torch/share/cmake" -S . -B build
cmake --build build
