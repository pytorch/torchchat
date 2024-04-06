echo "Install executorch: cloning"

mkdir build
cd build
git clone https://github.com/pytorch/executorch.git
cd executorch

echo "Install executorch: submodule update"
git submodule sync
git submodule update --init

echo "Applying fixes"
export SCRIPT_DIR=$(dirname $(realpath $0))
echo "Script dir: ${SCRIPT_DIR}"
echo "Current dir: ${PWD}"
cp ${SCRIPT_DIR}/fixes_et/module.h ${PWD}/extension/module/module.h
cp ${SCRIPT_DIR}/fixes_et/module.cpp ${PWD}/extension/module/module.cpp

echo "Install executorch: running pip install"
sh ./install_requirements.sh --pybind xnnpack

echo "Install executorch: building C++ libraries"
mkdir cmake-out
cmake -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_XNNPACK=ON -S . -B cmake-out -G Ninja
cmake --build cmake-out
