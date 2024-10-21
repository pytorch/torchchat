# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
from os import environ
from torchchat.cli.builder import BuilderArgs, TokenizerArgs
from functools import partial

import atexit
import torch.multiprocessing as mp
import importlib.util
import subprocess


def _setup_env(world_size:int, rank:int, target: callable, *args, **kwargs):
    environ["MASTER_ADDR"] = "localhost"
    environ["MASTER_PORT"] = "29500"
    environ["RDZV_BACKEND"] = "c10d"
    environ["WORLD_SIZE"] = str(world_size)
    environ["RANK"] = str(rank)
    environ["LOCALRANK"] = str(rank)

    return target(*args, **kwargs)


def _launch_distributed_inference(builder_args: BuilderArgs) -> None:
    # create programmatic elastic launch
    print("Launching distributed inference ...")

    num_processes_per_node = 4  # builder_args.num_gpus + 1

    from torchchat.distributed.dist_run import main
    mp.set_start_method('spawn')

    pipes = []
    procs = []
    for rank in range(num_processes_per_node):
        server_pipe, client_pipe = mp.Pipe(duplex=True)
        pipes.append(server_pipe)
        proc = mp.Process(
            target=partial(_setup_env, num_processes_per_node, rank, main),
            args=(builder_args, client_pipe)
        )
        proc.start()


    for pipe in pipes:
        response = pipe.recv()
        print(f"Received: {response=}")

    print(
        f"Done launching distributed inference on **4 ** {builder_args.num_gpus} GPUs."
    )
    return procs, pipes

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
        self.builder_args = builder_args
        self.requests = {}
        self.in_flight_requests = {}
        # For now we have a static batch order we save separately
        self.in_flight_batch_order = []
        # if builder_args.distributed:
        # # we part ways here with torchchat cli and move into dist inference
        self.procs, self.pipes = _launch_distributed_inference(builder_args)
        self.current_step = 0

        atexit.register(self.shutdown)

    def shutdown(self):
        for p in self.pipes:
                p.send("stop")
        for p in self.procs:
            p.kill()

    #TODO: Replace against (async) generate
    def add_request(self, request_id: int, prompt: str):
        assert request_id not in self.requests
        self.requests[request_id] = prompt


    def step(self) -> List[Output]:
        responses = []
        #TODO: Implement a scheduler to handle the requests
        if len(self.in_flight_requests) > 0:
            #Receive decoded token
            for p in self.pipes:
                p.send("step")
            for p in self.pipes:
                responses.append(p.recv())
            
        else:
            # Send requests to backend
            self.in_flight_batch_order = list(self.requests.keys())
            prompts = [self.requests[k] for k in self.in_flight_batch_order]
            for p in self.pipes:
                p.send(prompts)
            self.in_flight_requests = self.requests
            self.requests = {}
            self.current_step = 0
            #Receive first token
            for p in self.pipes:
                responses.append(p.recv())

        responses = responses[0]
        outputs = []
        for k, v in zip(self.in_flight_batch_order, responses):
            outputs.append(Output(k, is_finished=self.current_step>=self.builder_args.ntokens, output=v))
        
        self.current_step += 1

        return outputs
