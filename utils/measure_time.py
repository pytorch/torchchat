from time import perf_counter
from typing import Optional


class measure_time:
    def __init__(self, message: Optional[str] = "Time: {time:.3f} seconds"):
        self.message = message

    def __enter__(
        self,
    ):
        self.start = perf_counter()
        self.message
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        if self.message is not None:
            print(self.message.format(time=self.time))

    def get_time(self):
        return self.time
