# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import time

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from timeit import default_timer as timer
from typing import Any, Dict, List

import numpy as np

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import loss_parallel

from daylight.config_manager import JobConfig
#from daylight.datasets import build_hf_data_loader, create_tokenizer
#from daylight.float8_linear import build_fp8_linear
from daylight.logging_utils import init_logger, logger

#from daylight.metrics import build_gpu_memory_monitor, build_metric_logger
#from daylight.models import model_name_to_cls, model_name_to_tokenizer, models_config
#from daylight.parallelisms import (
#    models_parallelize_fns,
#    models_pipelining_fns,
#    ParallelDims,
#)

#from daylight.parallelisms.pipelining_utils import build_pipeline_schedule
#from daylight.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from daylight.utils import (
    Color,
    init_distributed,
    NoColor,
    set_pg_timeouts,
)

def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = Color if job_config.metrics.enable_color_printing else NoColor

    #logger.info(f"{Color.blue}('[INFO]')} Starting job: {job_config}")
    logger.info(
        f"{color.blue}Model {job_config.model.name} {job_config.model.flavor} {Color.reset} ")

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=job_config.training.data_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    init_distributed(job_config)


if __name__ == "__main__":
    print(f"Daylight starting...")
    config = JobConfig()
    config.parse_args()
    main(config)
    destroy_process_group()
