import time
from dataclasses import dataclass, field
from typing import Optional

from torchchat.distributed.adaptive.datatypes.sequence_status import SequenceStatus


@dataclass
class SequenceState:
    id: str
    arrived_at: float
    num_prompt_tokens: int
    num_output_tokens: int = field(default=0, init=False)
    status: SequenceStatus = field(default=SequenceStatus.WAITING, init=False)
    is_scheduled: bool = field(default=False, init=False)
    is_completed: bool = field(default=False, init=False)
    scheduled_at: Optional[float] = field(default=None, init=False)
    completed_at: Optional[float] = field(default=None, init=False)
    prompt_processing_completed_at: Optional[float] = field(default=None, init=False)
    last_restart_at: Optional[float] = field(default=None, init=False)
    last_pause_at: Optional[float] = field(default=None, init=False)
    execution_time: float = field(default=0.0, init=False)
    preempted_time: float = field(default=0.0, init=False)
    last_execution_start_at: Optional[float] = field(default=None, init=False)
    num_restarts: int = field(default=0, init=False)
    num_pauses: int = field(default=0, init=False)
    is_ignore_finished: bool = field(default=False, init=False)
    last_token_generated_at: Optional[float] = field(default=None, init=False)
    last_token_generation_time: float = field(default=0.0, init=False)

    @property
    def num_total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def e2e_time(self) -> Optional[float]:
        return self.completed_at - self.arrived_at if self.completed_at else None

    @property
    def e2e_time_piecewise_normalized(self) -> float:
        return self.scheduling_delay + (
            self.execution_plus_preemption_time / self.num_output_tokens
        )

    @property
    def e2e_time_normalized(self) -> Optional[float]:
        return self.e2e_time / self.num_output_tokens if self.e2e_time else None

    @property
    def e2e_prefill_time(self) -> Optional[float]:
        return (
            self.prompt_processing_completed_at - self.arrived_at
            if self.prompt_processing_completed_at
            else None
        )

    @property
    def e2e_prefill_time_normalized(self) -> Optional[float]:
        return (
            self.e2e_prefill_time / self.num_prompt_tokens
            if self.e2e_prefill_time
            else None
        )

    @property
    def e2e_prefill_time_piecewise_normalized(self) -> Optional[float]:
        if not self.prompt_processing_completed_at:
            return None
        return self.scheduling_delay + (
            self.prefill_execution_plus_preemption_time / self.num_prompt_tokens
        )

    @property
    def prefill_execution_plus_preemption_time(self) -> Optional[float]:
        if not self.prompt_processing_completed_at or not self.scheduled_at:
            return None
        return self.prompt_processing_completed_at - self.scheduled_at

    @property
    def decode_execution_plus_preemption_time(self) -> Optional[float]:
        if not self.completed_at or not self.prompt_processing_completed_at:
            return None
        return self.completed_at - self.prompt_processing_completed_at

    @property
    def prefill_execution_plus_preemption_time_normalized(self) -> Optional[float]:
        prefill_time = self.prefill_execution_plus_preemption_time
        return prefill_time / self.num_prompt_tokens if prefill_time else None

    @property
    def decode_execution_plus_preemption_time_normalized(self) -> Optional[float]:
        decode_time = self.decode_execution_plus_preemption_time
        return decode_time / self.num_output_tokens if decode_time else None

    @property
    def scheduling_delay(self) -> Optional[float]:
        return self.scheduled_at - self.arrived_at if self.scheduled_at else None

    @property
    def execution_time_normalized(self) -> float:
        return self.execution_time / self.num_output_tokens

    @property
    def execution_plus_preemption_time(self) -> float:
        return self.execution_time + self.preempted_time

    @property
    def execution_plus_preemption_time_normalized(self) -> float:
        return self.execution_plus_preemption_time / self.num_output_tokens

    def set_status(self, new_status: SequenceStatus) -> None:
        current_time = time.monotonic()

        if self.status == SequenceStatus.WAITING:
            self._handle_transitions_from_waiting(current_time, new_status)
        elif self.status == SequenceStatus.RUNNING:
            self._handle_transitions_from_running(current_time, new_status)
        elif self.status == SequenceStatus.PAUSED:
            self._handle_transitions_from_paused(current_time, new_status)
        elif self.status.is_finished:
            raise ValueError(
                f"Cannot transition from finished state {self.status} to {new_status} for request {self.id}."
            )
        else:
            raise ValueError(
                f"Invalid state transition from {self.status} to {new_status} for request {self.id}."
            )

        self.status = new_status

    def _handle_transitions_from_waiting(
        self, current_time: float, new_status: SequenceStatus
    ) -> None:
        if new_status == SequenceStatus.RUNNING:
            if not self.scheduled_at:
                self.is_scheduled = True
                self.scheduled_at = current_time
            else:
                self.preempted_time += current_time - self.last_restart_at
            self.last_execution_start_at = current_time
        elif new_status == SequenceStatus.FINISHED_IGNORED:
            self.is_ignore_finished = True
            self.is_completed = True
            self.completed_at = self.scheduled_at = current_time
        else:
            raise ValueError(
                f"Invalid transition from WAITING to {new_status} for request {self.id}."
            )

    def _handle_transitions_from_running(
        self, current_time: float, new_status: SequenceStatus
    ) -> None:
        self.execution_time += current_time - self.last_execution_start_at
        if new_status == SequenceStatus.PAUSED:
            self.num_pauses += 1
            self.last_pause_at = current_time
        elif new_status == SequenceStatus.WAITING:
            self.num_restarts += 1
            self.last_restart_at = current_time
        else:
            raise ValueError(
                f"Invalid transition from RUNNING to {new_status} for request {self.id}."
            )

    def _handle_transitions_from_paused(
        self, current_time: float, new_status: SequenceStatus
    ) -> None:
        self.preempted_time += current_time - self.last_pause_at
        if new_status.is_finished:
            self.is_completed = True
            self.completed_at = current_time
        elif new_status == SequenceStatus.RUNNING:
            self.last_execution_start_at = current_time
        elif new_status == SequenceStatus.WAITING:
            self.num_restarts += 1
            self.last_restart_at = current_time
        else:
            raise ValueError(
                f"Invalid transition from PAUSED to {new_status} for request {self.id}."
            )

    def on_prompt_processing_completed(self) -> None:
        self.prompt_processing_completed_at = time.monotonic()

    def on_token_generated(self) -> None:
        current_time = time.monotonic()
        self.num_output_tokens += 1
        if self.last_token_generated_at:
            self.last_token_generation_time = (
                current_time - self.last_token_generated_at
            )
        self.last_token_generated_at = current_time
