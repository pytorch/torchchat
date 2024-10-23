from enum import auto, Enum
from typing import Optional

from torchchat.distributed.logging_utils import SingletonLogger

logger = SingletonLogger.get_logger()


class SequenceStatus(Enum):
    """Status of a sequence."""

    WAITING = auto()
    RUNNING = auto()
    PAUSED = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    FINISHED_IGNORED = auto()

    @property
    def is_finished(self) -> bool:
        return self in (
            self.FINISHED_STOPPED,
            self.FINISHED_LENGTH_CAPPED,
            self.FINISHED_IGNORED,
        )

    @property
    def is_executing(self) -> bool:
        return self in (self.RUNNING, self.PAUSED)

    @property
    def is_waiting(self) -> bool:
        return self == self.WAITING

    @property
    def is_paused(self) -> bool:
        return self == self.PAUSED

    @property
    def is_running(self) -> bool:
        return self == self.RUNNING

    @property
    def finish_reason(self) -> Optional[str]:
        return {
            self.FINISHED_STOPPED: "stop",
            self.FINISHED_LENGTH_CAPPED: "length",
            self.FINISHED_IGNORED: "length",
        }.get(self)
