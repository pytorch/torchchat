# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import json

import logging

logger = logging.getLogger(__name__)

from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from os import environ
from typing import Dict, List, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from concurrent import futures
from flask import Flask, request, Response

from torchchat.cli.builder import BuilderArgs, TokenizerArgs
from torchchat.distributed.utils import run_in_dist_env
from torchchat.generate import GeneratorArgs

from torchchat.usages.openai_api import (
    CompletionRequest,
    get_model_info_list,
    create_openai_api_generator,
    retrieve_model_info,
)

OPENAI_API_VERSION = "v1"


def run_worker(
    args,
    rank,
    queue,
    ):
    """
    This function creates and executes a generator 
    """
    gen = initialize_generator(args)
    
    while True:
        try:
            req = queue.get()
        except KeyboardInterrupt:
            break

        if req == "stop":
            break
        
        for _ in gen.chunked_completion(req):
            pass

def create_app(args):  # noqa: C901
    """
    Creates a flask app that can be used to serve the model as a chat API.
    """
    app = Flask(__name__)

    builder_args = BuilderArgs.from_args(args)
    procs = []
    queue = None
    if builder_args.distributed:
        world_size = builder_args.tp * builder_args.pp
        mp_context = mp.get_context('spawn')
        queue = mp_context.Queue()
    
        for i in range(1, world_size):
            fn = partial(run_worker, args, i, queue)
            mp_context = mp.get_context('spawn')
            procs.append(mp_context.Process(target=run_in_dist_env, args=(world_size, i, fn)))
            procs[-1].start()

        environ["MASTER_ADDR"] = "localhost"
        environ["MASTER_PORT"] = "29500"
        environ["RDZV_BACKEND"] = "c10d"
        environ["WORLD_SIZE"] = str(world_size)
        environ["RANK"] = str(0)
        environ["LOCALRANK"] = str(0)

    gen = initialize_generator(args)

    def _del_none(d: Union[Dict, List]) -> Union[Dict, List]:
        """Recursively delete None values from a dictionary."""
        if type(d) is dict:
            return {k: _del_none(v) for k, v in d.items() if v}
        elif type(d) is list:
            return [_del_none(v) for v in d if v]
        return d

    @app.route(f"/{OPENAI_API_VERSION}/chat/completions", methods=["POST"])
    def chat_endpoint():
        """
        Endpoint for the Chat API. This endpoint is used to generate a response to a user prompt.
        This endpoint emulates the behavior of the OpenAI Chat API. (https://platform.openai.com/docs/api-reference/chat)

        ** Warning ** : Not all arguments of the CompletionRequest are consumed.

        See https://github.com/pytorch/torchchat/issues/973 and the OpenAiApiGenerator class for more details.

        If stream is set to true, the response will be streamed back as a series of CompletionResponseChunk objects. Otherwise,
        a single CompletionResponse object will be returned.
        """

        print(" === Completion Request ===")
        if seed := request.args.get("seed"):
            torch.manual_seed(int(seed))

        # Parse the request in to a CompletionRequest object
        data = request.get_json()
        req = CompletionRequest(**data)

        if req.stream:

            if builder_args.distributed:
                for _ in range(world_size-1):
                    queue.put(req)

            def chunk_processor(chunked_completion_generator):
                """Inline function for postprocessing CompletionResponseChunk objects.

                Here, we just jsonify the chunk and yield it as a string.
                """
                for chunk in chunked_completion_generator:
                    if (next_tok := chunk.choices[0].delta.content) is None:
                        next_tok = ""
                    print(next_tok, end="", flush=True)
                    yield f"data:{json.dumps(_del_none(asdict(chunk)))}\n\n"

            resp = Response(
                chunk_processor(gen.chunked_completion(req)),
                mimetype="text/event-stream",
            )
            return resp
        else:
            if builder_args.distributed:
                for _ in range(world_size-1):
                    queue.put(req)

            response = gen.sync_completion(req)

            return json.dumps(_del_none(asdict(response)))

    @app.route(f"/{OPENAI_API_VERSION}/models", methods=["GET"])
    def models_endpoint():
        return json.dumps(asdict(get_model_info_list(args)))

    @app.route(f"/{OPENAI_API_VERSION}/models/<model_id>", methods=["GET"])
    def models_retrieve_endpoint(model_id):
        if response := retrieve_model_info(args, model_id):
            return json.dumps(asdict(response))
        else:
            return "Model not found", 404

    return app, (procs, queue)


def initialize_generator(args) -> type:
    builder_args = BuilderArgs.from_args(args)
    speculative_builder_args = BuilderArgs.from_speculative_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)
    generator_args.chat_mode = False

    OpenAiApiGenerator = create_openai_api_generator(builder_args.distributed)

    return OpenAiApiGenerator(
        builder_args=builder_args,
        speculative_builder_args=speculative_builder_args,
        tokenizer_args=tokenizer_args,
        generator_args=generator_args,
        profile=args.profile,
        quantize=args.quantize,
        draft_quantize=args.draft_quantize,
    )


def main(args):
    app, (procs, queue) = create_app(args)

    def shutdown_worker():
        while not queue.empty():
            queue.get(block=False)
        for p in procs:
            queue.put("stop")
        for p in procs:
            p.join(timeout=0.5)
        for p in procs:
            if p.is_alive():
                p.kill()

    atexit.register(shutdown_worker)

    app.run()
