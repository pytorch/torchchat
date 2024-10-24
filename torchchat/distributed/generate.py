# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import atexit
import importlib.util
import subprocess
import threading
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from functools import partial
from os import environ
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import torch.multiprocessing as mp
from torchchat.cli.builder import BuilderArgs, TokenizerArgs
from torchchat.distributed.dist_run import NAME_TO_DISTRIBUTION_AND_DTYPE
from torchchat.distributed.logging_utils import SingletonLogger

logger = SingletonLogger.get_logger()


def _setup_env(world_size: int, rank: int, target: callable, *args, **kwargs):
    environ["MASTER_ADDR"] = "localhost"
    environ["MASTER_PORT"] = "29500"
    environ["RDZV_BACKEND"] = "c10d"
    environ["WORLD_SIZE"] = str(world_size)
    environ["RANK"] = str(rank)
    environ["LOCALRANK"] = str(rank)

    return target(*args, **kwargs)


def _launch_distributed_inference(
    model_name: str, builder_args: BuilderArgs, tokenizer_args: TokenizerArgs
) -> tuple[List]:
    # launch distributed inference worker, each worker gets a pipe to communicate with the main process
    logger.info("Launching distributed inference ...")

    num_processes_per_node = builder_args.pp * builder_args.tp

    from torchchat.distributed.dist_run import main

    mp.set_start_method("spawn")

    pipes = []
    procs = []
    try:
        for rank in range(num_processes_per_node):
            server_pipe, client_pipe = mp.Pipe(duplex=True)
            pipes.append(server_pipe)
            procs.append(
                mp.Process(
                    target=partial(_setup_env, num_processes_per_node, rank, main),
                    args=(model_name, builder_args, tokenizer_args, client_pipe),
                )
            )
            procs[-1].start()

        for pipe in pipes:
            assert pipe.recv() == "ready", "Starting the worker failed"
    except Exception as e:
        logger.error(f"Error during distributed inference: {str(e)}")
        for p in procs:
            p.kill()
        raise e

    logger.info(
        f"Done launching distributed inference on {num_processes_per_node} GPUs."
    )
    return procs, pipes


@dataclass
class Output:
    is_finished: bool = False
    text: Optional[str] = None
    token: Optional[list] = None


@dataclass
class Request:
    request_id: int
    prompt: str

    @classmethod
    def new_request(cls, prompt):
        return cls(request_id=uuid4().int, prompt=prompt)


class Scheduler(object):
    def __init__(
        self,
        builder_args,
        generator_args,
        pipes,
        loop,
    ):
        self.builder_args = builder_args
        self.generator_args = generator_args
        self.requests = {}
        self.in_flight_requests = {}
        self.in_flight_batch_order = []
        self.pipes = pipes
        self.req_to_states = {}
        self.req_to_results = {}
        self.request_queue = mp.Queue()
        self.loop = loop

    def schedule_request(self, req: Request):
        # add request to queue and create deque and async event for response
        self.req_to_states[req.request_id] = asyncio.Event()
        self.req_to_results[req.request_id] = deque()
        self.request_queue.put(req)

    def process_requests_loop(self):
        # Continuously process requests (one at a time for now), results are routed into the requests deque
        while True:
            req = self.request_queue.get()
            if req == "stop":
                break
            self.requests = {req.request_id: req.prompt}

            responses = {}
            running = True
            while running:
                outputs = self.step()
                self.req_to_results[req.request_id].append(outputs[0])

                self.loop.call_soon_threadsafe(self.req_to_states[req.request_id].set)

                running &= not outputs[0].is_finished

    async def wait_for_request(self, req: Request) -> Output:
        # Wait for request to deliver result, uses event to trigger and reads from left side of deque
        is_finished = False
        while not is_finished:
            await self.req_to_states[req.request_id].wait()
            while len(self.req_to_results[req.request_id]):
                output = self.req_to_results[req.request_id].popleft()
                is_finished |= output.is_finished
                yield output
        del self.req_to_states[req.request_id]
        del self.req_to_results[req.request_id]

    def step(self) -> List[Output]:
        # Make a prefill or decoding step and receive results
        responses = []
        # TODO: Implement a scheduler to handle the requests
        if len(self.in_flight_requests) > 0:
            # Receive decoded token
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
            # Receive first token
            for p in self.pipes:
                responses.append(p.recv())
        # Filter out None responses from in-between stages
        responses = [r for r in responses if r is not None][0]
        outputs = []
        for k, v in zip(self.in_flight_batch_order, zip(responses[0], responses[1])):
            text, token_ids = v
            outputs.append(
                Output(
                    # TODO: Look for tokenizer.eos_id as well
                    is_finished=self.current_step >= self.generator_args.max_new_tokens,
                    text=text,
                    token=token_ids,
                )
            )
        if self.current_step >= self.generator_args.max_new_tokens:
            for p in self.pipes:
                p.send("stop")
            self.in_flight_requests = []

        self.current_step += 1

        return outputs


class DistributedGenerator(object):
    def __init__(
        self,
        # TODO: switch this to torchchat method
        model_name: str,
        builder_args: BuilderArgs,
        tokenizer_args: TokenizerArgs,
        # TODO: move GeneratorArgs into a different module
        generator_args,
        profile: Optional[Path],
        quantize: bool,
        draft_quantize: bool,
    ):
        self.model_name = model_name
        self.builder_args = builder_args
        self.generate_args = generator_args

        self.check_args()

        self.procs, self.pipes = _launch_distributed_inference(
            model_name, builder_args, tokenizer_args
        )

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.scheduler = Scheduler(builder_args, generator_args, self.pipes, self.loop)

        # TODO: Mode into process and use pipe or queue for comm
        self.scheduler_thread = threading.Thread(
            target=self.scheduler.process_requests_loop
        )
        self.scheduler_thread.start()

        atexit.register(self.shutdown)

    def shutdown(self):
        # Stop all processes and threads
        self.scheduler.request_queue.put("stop")
        self.scheduler_thread.join()

        for p in self.pipes:
            p.send("stop")
        for p in self.procs:
            p.kill()

    def generate(self, text):
        # Function to generate text from prompt
        req = Request.new_request(text)
        self.scheduler.schedule_request(req)

        generator = self.scheduler.wait_for_request(req)

        running = True
        while running:
            output = self.loop.run_until_complete(generator.__anext__())
            running &= not output.is_finished

            yield output

    def check_args(self):
        if self.generate_args.chat_mode:
            raise NotImplementedError(
                "Currently we only support generate with --distributed"
            )
        elif self.builder_args.tp < 2:
            raise ValueError("TP degree must be at least 2 for distributed inference")
        elif self.model_name not in NAME_TO_DISTRIBUTION_AND_DTYPE.keys():
            raise ValueError(
                f"Distributed inference currently only supports then following models: {list(NAME_TO_DISTRIBUTION_AND_DTYPE.keys())}"
            )
        elif self.builder_args.chpt_from == "torchchat":
            raise ValueError(
                f"Distributed inference currently only supports HF checkpoints"
            )
