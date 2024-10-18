# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
from torchchat.cli.builder import BuilderArgs, TokenizerArgs


import importlib.util
import subprocess


def run_script(script_path, *args):
    # Construct the command to run the script
    cmd = [sys.executable, script_path] + list(args)

    # Run the script as a subprocess
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    # Stream the output in real-time
    for line in process.stdout:
        print(line, end="")
    for line in process.stderr:
        print(line, end="", file=sys.stderr)

    # Wait for the process to complete and get the return code
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _launch_distributed_inference(builder_args: BuilderArgs) -> None:
    # create programmatic elastic launch
    print("Launching distributed inference ...")

    num_processes_per_node = 4  # builder_args.num_gpus + 1

    lc = launcher.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=num_processes_per_node,
        # run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:29401",
        max_restarts=0,
        monitor_interval=1,
    )

    # train_file_path = Path(__file__).parent.parent.parent / "dist_run.py"
    # print(f"train_file_path: {train_file_path}")
    # import argparse

    # parser2 = argparse.ArgumentParser()

    # args = parser2.parse_args()
    args = []
    print(f"args: {args}")

    from dist_run import main

    elastic_launch(
        config=lc,
        entrypoint=run_script,
    )(main, *args)
    print(
        f"Done launching distributed inference on **4 ** {builder_args.num_gpus} GPUs."
    )
    #  role=role, *args, **kwargs)

    # assert False, "distributed inference is not supported yet"
    # pass

@dataclass
class Output:
    request_id: int
    is_finished: bool = False
    output: Optional[str] = None

class Generator(object):

    @abstractmethod
    def add_request(self, request_id: int, prompt: str):
        raise NotImplementedError()

    def step(self) -> List[Output]:
        raise NotImplementedError()


class DistributedGenerator(Generator):
    def __init__(
        self,
        builder_args: BuilderArgs,
        speculative_builder_args: BuilderArgs,
        tokenizer_args: TokenizerArgs,
        #TODO: move GeneratorArgs into a different module
        # generator_args: GeneratorArgs,
        profile: Optional[Path],
        quantize: bool,
        draft_quantize: bool,
        ):
        self.requests = {}
        # if builder_args.distributed:
        # # we part ways here with torchchat cli and move into dist inference
        _launch_distributed_inference(builder_args)
        # return None


    def add_request(self, request_id: int, prompt: str):
        assert request_id not in self.requests
        self.requests[request_id] = prompt


    def step(self) -> List[Output]:
        outputs = []
        for request_id, prompt in self.requests.items():
            outputs.append(Output(request_id, is_finished=True, output=prompt))
        
        for output in outputs:
            if output.is_finished:
                del self.requests[output.request_id]

        return outputs
