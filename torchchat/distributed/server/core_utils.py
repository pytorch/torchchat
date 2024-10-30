import random
import subprocess as sp
import uuid
from enum import Enum
from typing import List, TypeAlias

import numpy as np
import psutil
import torch

_GB = 1 << 30
_MB = 1 << 20


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0
