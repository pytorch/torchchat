#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

MODEL_REPO="$1"
RESOURCES_STRING="$2"
CHECKPOINT_NAME="${MODEL_REPO##*/}"

# Create the directory for the checkpoint
mkdir -p "checkpoints/${MODEL_REPO}"
pushd "checkpoints/${MODEL_REPO}" || exit

# Download all resources
IFS=',' # Set the field separator to comma
for resource in $RESOURCES_STRING; do
  echo "Downloading: $resource"
  if ! wget "$resource" 2>&1; then
    echo "Error: Failed to download $resource" >&2
    exit 1
  fi
done

popd || exit
