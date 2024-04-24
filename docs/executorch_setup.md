# Set-up ExecuTorch

Before running any commands in torchchat that require ExecuTorch, you must first install ExecuTorch.

To install ExecuTorch, run the following commands *from the torchchat root directory*.

```
export TORCHCHAT_ROOT=${PWD}
export ENABLE_ET_PYBIND=true
./scripts/install_et.sh $ENABLE_ET_PYBIND
```

This will download the ExecuTorch repo to ./et-build/src and install various ExecuTorch libraries to ./et-build/install.
