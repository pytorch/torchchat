# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from datetime import datetime
from typing import Optional


def millisecond_timestamp(include_year: bool = False) -> str:
    format_string = "%Y-%m-%d %H:%M:%S.%f" if include_year else "%m-%d %H:%M:%S.%f"
    return datetime.now().strftime(format_string)[:-3]


class CompactFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        validate: bool = True,
        *,
        defaults: Optional[dict] = None,
        show_lower_levels: bool = True,
    ):
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.show_lower_levels = show_lower_levels
        self.original_fmt = fmt

    def format(self, record: logging.LogRecord) -> str:
        # Remove .py extension from filename
        record.filename = os.path.splitext(record.filename)[0]

        if self.show_lower_levels or record.levelno > logging.INFO:
            return super().format(record)
        else:
            # Create a copy of the record and modify it
            new_record = logging.makeLogRecord(record.__dict__)
            new_record.levelname = ""
            # Temporarily change the format string
            temp_fmt = self.original_fmt.replace(" - %(levelname)s", "")
            self._style._fmt = temp_fmt
            formatted_message = super().format(new_record)
            # Restore the original format string
            self._style._fmt = self.original_fmt
            return formatted_message


class SingletonLogger:
    """Singleton (global) logger to avoid logging duplication"""

    _instance = None

    @classmethod
    def get_logger(
        cls,
        name: str = "global_logger",
        level: int = logging.INFO,
        include_year: bool = False,
        show_lower_levels: bool = False,
    ) -> logging.Logger:
        """
        Get or create a singleton logger instance.

        :param name: Name of the logger
        :param level: Logging level
        :param include_year: Whether to include the year in timestamps
        :param show_lower_levels: Whether to show level names for INFO and DEBUG messages
        :return: Logger instance
        """
        if cls._instance is None:
            cls._instance = cls._setup_logger(
                name, level, include_year, show_lower_levels
            )
        return cls._instance

    @staticmethod
    def _setup_logger(
        name: str,
        level: int,
        include_year: bool = False,
        show_lower_levels: bool = False,
    ) -> logging.Logger:
        logger = logging.getLogger(name)

        if not logger.handlers:
            logger.setLevel(level)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            formatter = CompactFormatter(
                "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
                show_lower_levels=show_lower_levels,
            )
            formatter.formatTime = lambda record, datefmt=None: millisecond_timestamp(
                include_year
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Suppress verbose torch.profiler logging
            os.environ["KINETO_LOG_LEVEL"] = "5"

        logger.propagate = False
        return logger
