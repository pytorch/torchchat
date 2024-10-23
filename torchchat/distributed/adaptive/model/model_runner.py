from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed

from sarathi.config import SchedulerType, SystemConfig
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.sequence import Sequence, SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import CpuOperationMetrics
from sarathi.metrics.cpu_timer import CpuTimer
from sarathi.model_executor import get_model, set_random_seed
from sarathi.model_executor.attention.attention_backend_registry import (
    AttentionBackendRegistry,
)
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.model_executor.layers.sampler import Sampler
from sarathi.model_executor.utils import pad_to_alignment
from sarathi.utils import get_gpu_memory

from torchchat.distributed.logging_utils import SingletonLogger

logger = SingletonLogger.get_logger()


@dataclass
class ModelInput:
    tokens: torch.Tensor
    positions: torch.Tensor


class ModelRunner:
    def __init__(self, config: SystemConfig, device: torch.device, rank: int):
        self.config = config
        self.device = device
        self.rank = rank

        self.attention_backend_wrapper = self._initialize_attention_backend()
        self.model = self._initialize_model()
        self.sampler = self._initialize_sampler()

        self.timers = self._initialize_timers()

    def _initialize_attention_backend(self) -> BaseAttentionWrapper:
        return AttentionBackendRegistry.get(
            self.config.worker_config.attention_backend,
            self.config.model_config,
            self.config.parallel_config,
            self.config.cache_config,
            self.device,
        )

    def _initialize_model(self):
        return get_model(self.config.model_config)

    def _initialize_sampler(self) -> Optional[Sampler]:
        if self.model.lm_head:
            return Sampler(self.model.lm_head.weight, self.model.config.vocab_size)
        return None

    def _initialize_timers(self) -> dict:
        return {
            "prepare_inputs": CpuTimer(
                CpuOperationMetrics.PREPARE_INPUTS_E2E, rank=self.rank
            ),
            "sampler": CpuTimer(CpuOperationMetrics.SAMPLER_E2E, rank=self.rank),
            "model_execution": CpuTimer(
                CpuOperationMetrics.MODEL_EXECUTION_E2E, rank=self.rank
            ),
        }

    def init_kv_cache(self, num_gpu_blocks: int):
        self.attention_backend_wrapper.init_gpu_cache(num_gpu_blocks)

    def _prepare_inputs(self, seq_metadata_list: List[SequenceMetadata]) -> ModelInput:
        input_tokens, input_positions = [], []
        current_prompt_chunk_lens = []

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                self._process_prompt_sequence(
                    seq_metadata,
                    input_tokens,
                    input_positions,
                    current_prompt_chunk_lens,
                )
            else:
                self._process_generation_sequence(
                    seq_metadata, input_tokens, input_positions
                )

        input_tokens = pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = pad_to_alignment(input_positions, multiple_of=8)

        return ModelInput(
            tokens=torch.tensor(input_tokens, dtype=torch.long, device=self.device),
            positions=torch.tensor(
                input_positions, dtype=torch.long, device=self.device
            ),
        )

    def _process_prompt_sequence(
        self, seq_metadata, input_tokens, input_positions, current_prompt_chunk_lens
    ):
        prompt_chunk_len = seq_metadata.prompt_chunk_len
        current_prompt_chunk_tokens = seq_metadata.seq.get_next_prompt_chunk_token_ids(
            prompt_chunk_len
        )
        current_prompt_chunk_len = len(current_prompt_chunk_tokens)
        current_prompt_chunk_lens.append(current_prompt_chunk_len)
        processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_stage_processed()

        current_total_len = processed_prompt_len + current_prompt_chunk_len

        input_tokens.extend(current_prompt_chunk_tokens)
        input_positions.extend(range(processed_prompt_len, current_total_len))

    def _process_generation_sequence(self, seq_metadata, input_tokens, input_positions):
        generation_token = seq_metadata.seq.get_last_token_id()
        input_tokens.append(generation_token)

        context_len = seq_metadata.seq.get_len()
        input_positions.append(context_len - 1)

    @torch.inference_mode()
    def profile_num_available_blocks(
        self, block_size: int, gpu_memory_utilization: float
    ) -> Tuple[int, int]:
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        sampling_params = SamplingParams(
            top_p=0.99, top_k=self.model.config.vocab_size - 1
        )
        seq_metadata_list = self._create_profiling_sequences(
            block_size, sampling_params
        )

        model_input = self._prepare_inputs(seq_metadata_list)
        self.attention_backend_wrapper.begin_forward(seq_metadata_list)

        self._execute_model(model_input)

        num_gpu_blocks = self._calculate_available_blocks(gpu_memory_utilization)

        self.attention_backend_wrapper.end_forward()
        set_random_seed(self.config.model_config.seed)

        return num_gpu_blocks

    def _create_profiling_sequences(
        self, block_size: int, sampling_params: SamplingParams
    ) -> List[SequenceMetadata]:
        if self.config.scheduler_config.get_type() in {
            SchedulerType.SARATHI,
            SchedulerType.SIMPLE_CHUNKING,
        }:
            return self._create_chunking_profiling_sequence(block_size, sampling_params)
        else:
            return self._create_standard_profiling_sequences(
                block_size, sampling_params
            )

    def _create_chunking_profiling_sequence(
        self, block_size: int, sampling_params: SamplingParams
    ) -> List[SequenceMetadata]:
        chunk_size = min(
            self.config.scheduler_config.chunk_size,
            self.config.model_config.max_model_len,
        )
        seq = Sequence(
            seq_id=0,
            prompt=None,
            prompt_token_ids=[0] * self.config.model_config.max_model_len,
            block_size=block_size,
            eos_token_id=1,
            arrival_time=None,
            sampling_params=sampling_params,
        )
        return [
            SequenceMetadata(seq=seq, block_table=None, prompt_chunk_len=chunk_size)
        ]

    def _create_standard_profiling_sequences(
        self, block_size: int, sampling_params: SamplingParams
    ) -> List[SequenceMetadata]:
        max_num_batched_tokens = (
            self.config.scheduler_config.get_max_num_batched_tokens(
                self.config.model_config.max_model_len
            )
        )
        max_num_seqs = self.config.scheduler_config.max_num_seqs
        return [
            SequenceMetadata(
                seq=Sequence(
                    seq_id=str(seq_id),
                    prompt=None,
                    prompt_token_ids=[0]
                    * (
                        max_num_batched_tokens // max_num_seqs
                        + (seq_id < max_num_batched_tokens % max_num_seqs)
                    ),
                    block_size=block_size,
                    eos_token_id=1,
                    arrival_time=None,
                    sampling_params=sampling_params,
                ),
                block_table=None,
                prompt_chunk_len=max_num_batched_tokens // max_num_seqs
                + (seq_id < max_num_batched_tokens % max_num_seqs),
            )
            for seq_id in range(max_num_seqs)
        ]

    def _execute_model(self, model_input: ModelInput):
        self.model(
            hidden_states=model_input.tokens,
            positions=model_input.positions,
            attention_backend_wrapper=self.attention_backend_wrapper,
        )

    def _calculate_available_blocks(self, gpu_memory_utilization: float) -> int:
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        cache_block_size = self.attention_backend_wrapper.get_cache_block_size()
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory)
            // cache_block_size
        )
        return max(num_gpu_blocks, 0)

    def run(self, seq_metadata_list: List[SequenceMetadata]) -> torch.Tensor:
        with self.timers["prepare_inputs"]:
            model_input = self._prepare_inputs(seq_metadata_list)

        self.attention_backend_wrapper.begin_forward(seq_metadata_list)

        with self.timers["model_execution"]:
            try:
                output = self.model(
                    hidden_states=model_input.tokens,
                    positions=model_input.positions,
                    attention_backend_wrapper=self.attention_backend_wrapper,
                )
            except RuntimeError as e:
                logger.error(
                    f"RuntimeError: {e} for seq_metadata_list: {seq_metadata_list}"
                )
                raise

        with self.timers["sampler"]:
            if self.sampler is not None:
                output = self.sampler(output, seq_metadata_list)

        self.attention_backend_wrapper.end_forward()

        return output
