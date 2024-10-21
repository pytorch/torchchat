from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence

@dataclass
class StepInputs:
    """
    Input data for a single step of the language model.

    Attributes:
        scheduler_outputs: The outputs from the scheduler for this step.
        new_seqs: A list of new sequences to add to the engine.
        pending_step_outputs: A list of tuples of scheduler outputs and sampler outputs
                              from previous steps that need to be processed.
    """
    scheduler_outputs: SchedulerOutputs
    new_seqs: List[Sequence] = field(default_factory=list)
    pending_step_outputs: List[Tuple[SchedulerOutputs, SamplerOutputs]] = field(default_factory=list)

    def __post_init__(self):
        if self.new_seqs is None:
            self.new_seqs = []
        if self.pending_step_outputs is None:
            self.pending_step_outputs = []

    @property
    def has_new_sequences(self) -> bool:
        """Check if there are any new sequences to process."""
        return bool(self.new_seqs)

    @property
    def has_pending_outputs(self) -> bool:
        """Check if there are any pending outputs to process."""
        return bool(self.pending_step_outputs)

    def __str__(self) -> str:
        return (f"StepInputs(scheduler_outputs_id={self.scheduler_outputs.id}, "
                f"new_sequences={len(self.new_seqs)}, "
                f"pending_outputs={len(self.pending_step_outputs)})")

    def __repr__(self) -> str:
        return (f"StepInputs(\n"
                f"  scheduler_outputs={self.scheduler_outputs!r},\n"
                f"  new_seqs={self.new_seqs!r},\n"
                f"  pending_step_outputs={self.pending_step_outputs!r}\n"
                f")")
