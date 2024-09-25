# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch.distributed.device_mesh import init_device_mesh

from torchchat.distributed.logging_utils import SingletonLogger
logger = SingletonLogger.get_logger()

@dataclass
class ParallelDims:
    tp: int
    pp: int
    world_size: int

    def __post_init__(self):
        self._validate()

    def _validate(self):
        tp, pp = self.tp, self.pp
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert (
            tp * pp == self.world_size
        ), f"Invalid parallel dims: tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.tp], ["pp", "tp"], strict=True
        ):
            if d > 1:
                dims.append(d)
                names.append(name)
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        return init_device_mesh(device_type, dims, mesh_dim_names=names)

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1
