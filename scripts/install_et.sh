cd ${LLAMA_FAST_ROOT}
echo "Inside: $LLAMA_FAST_ROOT"

echo "Cloning executorch to ${LLAMA_FAST_ROOT}/build/src"
rm -rf ${LLAMA_FAST_ROOT}/build
mkdir -p ${LLAMA_FAST_ROOT}/build/src
cd ${LLAMA_FAST_ROOT}/build/src
git clone https://github.com/pytorch/executorch.git
cd executorch
echo "Install executorch: submodule update"
git submodule sync
git submodule update --init

echo "Applying fixes"
cp ${LLAMA_FAST_ROOT}/scripts/fixes_et/module.cpp ${LLAMA_FAST_ROOT}/build/src/executorch/extension/module/module.cpp # ET uses non-standard C++ that does not compile in GCC
cp ${LLAMA_FAST_ROOT}/scripts/fixes_et/managed_tensor.h ${LLAMA_FAST_ROOT}/build/src/executorch/extension/runner_util/managed_tensor.h # ET is missing headers for vector/memory.  This causes downstream issues when building runner-et.

echo "Building and installing python libraries"
if [ z "${ET_NO_PYBIND}" ]; then
    echo "Installing pybind"
    ./install_requirements.sh --pybind xnnpack
else
    echo "Not installing pybind"
    ./install_requirements.sh
fi

echo "Building and installing C++ libraries"
echo "Inside: ${PWD}"
mkdir cmake-out
cmake -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_OPTIMIZED=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_XNNPACK=ON -S . -B cmake-out -G Ninja
cmake --build cmake-out
cmake --install cmake-out --prefix ${LLAMA_FAST_ROOT}/build/install
