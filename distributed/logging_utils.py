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


class SingletonLogger:
    """Singleton (global) logger to avoid logging duplication"""
    # Usage:
    # from logging_utils import SingletonLogger
    # logger = SingletonLogger.get_logger()
    
    _instance = None

    @classmethod
    def get_logger(cls, name='global_logger', level=logging.INFO):
        if cls._instance is None:
            cls._instance = cls._setup_logger(name, level)
        return cls._instance

    @staticmethod
    def _setup_logger(name, level):
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            logger.setLevel(level)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
            formatter.formatTime = millisecond_timestamp

            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # suppress verbose torch.profiler logging
            os.environ["KINETO_LOG_LEVEL"] = "5"
        
        logger.propagate = False
        return logger



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

    logger.propagate = False

    return logger
