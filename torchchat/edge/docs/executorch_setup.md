> [!WARNING]
> Files in this directory may be outdated, incomplete, scratch notes, or a WIP. torchchat provides no guarantees on these files as references. Please refer to the root README for stable features and documentation.

# Set-up ExecuTorch

Before running any commands in torchchat that require ExecuTorch, you
must first install ExecuTorch.

To install ExecuTorch, run the following commands *from the torchchat
root directory*.

```
export TORCHCHAT_ROOT=$PWD
./torchchat/utils/scripts/install_et.sh
```

This will download the ExecuTorch repo to ./et-build/src and install
various ExecuTorch libraries to ./et-build/install.

[end default]: end
