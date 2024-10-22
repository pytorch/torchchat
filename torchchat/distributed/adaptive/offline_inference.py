"""Offline inference system for large language model text generation.

This module provides functionality for batch text generation using a large language model
with configurable parameters and metrics collection.
"""

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import logging

from sarathi import (
    LLMEngine,
    SamplingParams,
    RequestOutput,
    SystemConfig,
    ModelConfig,
    ParallelConfig,
    SarathiSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for inference settings."""
    base_output_dir: Path
    model_name: str
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 100
    chunk_size: int = 100
    max_num_seqs: int = 10
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 4
    enable_metrics: bool = True
    enable_chrome_trace: bool = True

    @property
    def output_dir(self) -> Path:
        """Generate timestamped output directory path."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return self.base_output_dir / timestamp

    def create_system_config(self) -> SystemConfig:
        """Create SystemConfig from inference settings."""
        return SystemConfig(
            replica_config=ReplicaConfig(
                output_dir=str(self.output_dir),
            ),
            model_config=ModelConfig(
                model=self.model_name,
            ),
            parallel_config=ParallelConfig(
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
            ),
            scheduler_config=SarathiSchedulerConfig(
                chunk_size=self.chunk_size,
                max_num_seqs=self.max_num_seqs,
            ),
            metrics_config=MetricsConfig(
                write_metrics=self.enable_metrics,
                enable_chrome_trace=self.enable_chrome_trace,
            ),
        )

class TextGenerator:
    """Handles text generation using LLM engine."""

    def __init__(self, config: InferenceConfig):
        """Initialize TextGenerator with configuration.

        Args:
            config: Configuration settings for inference.
        """
        self.config = config
        self._ensure_output_directory()
        self.engine = self._initialize_engine()
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )

    def _ensure_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_engine(self) -> LLMEngine:
        """Initialize the LLM engine with system configuration."""
        try:
            return LLMEngine.from_system_config(self.config.create_system_config())
        except Exception as e:
            logger.error(f"Failed to initialize LLM engine: {e}")
            raise

    def generate(self, prompts: List[str]) -> List[RequestOutput]:
        """Generate text completions for given prompts.

        Args:
            prompts: List of input prompts for text generation.

        Returns:
            List of RequestOutput objects containing generated texts.

        Raises:
            ValueError: If prompts list is empty.
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")

        self._add_requests(prompts)
        return self._process_requests()

    def _add_requests(self, prompts: List[str]) -> None:
        """Add generation requests to the engine.

        Args:
            prompts: List of prompts to process.
        """
        for prompt in prompts:
            self.engine.add_request(prompt, self.sampling_params)

    def _process_requests(self) -> List[RequestOutput]:
        """Process all pending requests and collect outputs.

        Returns:
            List of RequestOutput objects sorted by sequence ID.
        """
        outputs: List[RequestOutput] = []
        num_requests = self.engine.get_num_unfinished_requests()

        with tqdm(total=num_requests, desc="Processing prompts") as pbar:
            while self.engine.has_unfinished_requests():
                step_outputs = self.engine.step()
                outputs.extend(
                    output for output in step_outputs if output.finished
                )
                pbar.update(len([o for o in step_outputs if o.finished]))

        return sorted(outputs, key=lambda x: int(x.seq_id))

    def collect_metrics(self) -> None:
        """Collect and visualize engine metrics."""
        try:
            self.engine.pull_worker_metrics()
            self.engine.plot_metrics()
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")

    @staticmethod
    def format_output(output: RequestOutput) -> str:
        """Format generation output for display.

        Args:
            output: RequestOutput object containing generation results.

        Returns:
            Formatted string representation of the output.
        """
        return (
            "=" * 60 + "\n"
            f"Prompt: {output.prompt!r}\n"
            "-" * 60 + "\n"
            f"Generated text: {output.text!r}\n"
            "=" * 60 + "\n"
        )

def main():
    """Main execution function."""
    # Example prompts
    prompts = [
        "The immediate reaction in some circles...",  # Add your prompts here
        "The breakthrough technique developed...",
        "Hydrogen ions are the key component...",
    ]

    config = InferenceConfig(
        base_output_dir=Path("./offline_inference_output"),
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    try:
        generator = TextGenerator(config)
        outputs = generator.generate(prompts)

        # Print outputs
        for output in outputs:
            print(generator.format_output(output))

        # Collect metrics
        generator.collect_metrics()

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
