# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from datetime import datetime

def millisecond_timestamp(*args):
    return datetime.now().strftime('%m-%d %H:%M:%S.%f')[:-3]

def setup_logging(name=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
        formatter.formatTime = millisecond_timestamp

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # suppress verbose torch.profiler logging
        os.environ["KINETO_LOG_LEVEL"] = "5"

    return logger
