# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist

from build.model import TransformerArgs
from build.model_dist import Transformer

# Model config
config = TransformerArgs.from_name("Transformer-2-7b-chat-hf")
print(config)

# Construct a device mesh with available devices (multi-host or single host)
device_mesh = dist.init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Create parallel model with device_mesh context
with device:
    with device_mesh:
        model = Transformer(config)
        model.setup_caches(1, 4096)

print(model)

# Distributed run
input_ids = torch.randint(0, config.vocab_size, (1, 4096), device=device)
input_pos = torch.arange(0, 4096, device=device)
output = model(input_ids, input_pos)
dist.destroy_process_group()
print(f"Rank {rank} completes.")
