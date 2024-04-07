echo "Install executorch: cloning"
rm -rf ${LLAMA_FAST_DIR}/build
mkdir ${LLAMA_FAST_DIR}/build
cd ${LLAMA_FAST_DIR}/build
echo "Inside: ${PWD}"
git clone https://github.com/pytorch/executorch.git
cd executorch
echo "Install executorch: submodule update"
git submodule sync
git submodule update --init

export ET_DIR=${LLAMA_FAST_DIR}/build

echo "Applying fixes"
echo "Inside: ${PWD}"
cp ${LLAMA_FAST_DIR}/scripts/fixes_et/module.cpp ${ET_DIR}/executorch/extension/module/module.cpp # ET uses non-standard C++ that does not compile in GCC
cp ${LLAMA_FAST_DIR}/scripts/fixes_et/managed_tensor.h ${ET_DIR}/executorch/extension/runner_util/managed_tensor.h # ET is missing headers for vector/memory

echo "Install executorch: running pip install"
./install_requirements.sh --pybind xnnpack

echo "Install executorch: building C++ libraries"
echo "Inside: ${PWD}"
mkdir cmake-out
cmake -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_XNNPACK=ON -S . -B cmake-out -G Ninja
cmake --build cmake-out

cd ${LLAMA_FAST_DIR}
echo "Inside: ${PWD}"
mkdir -p build/cmake-out
cmake -DET_DIR:STRING=${ET_DIR} -DCMAKE_BUILD_TYPE=Release -S runner-et -B build/cmake-out
cmake --build build/cmake-out
