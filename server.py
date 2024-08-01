# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

from dataclasses import asdict
from typing import Dict, List, Union

from api.api import AssistantMessage, CompletionRequest, OpenAiApiGenerator, UserMessage

from build.builder import BuilderArgs, TokenizerArgs
from flask import Flask, request, Response
from generate import GeneratorArgs


"""
Creates a flask app that can be used to serve the model as a chat API.
"""
app = Flask(__name__)
# Messages and gen are kept global so they can be accessed by the flask app endpoints.
messages: list = []
gen: OpenAiApiGenerator = None


def _del_none(d: Union[Dict, List]) -> Union[Dict, List]:
    """Recursively delete None values from a dictionary."""
    if type(d) is dict:
        return {k: _del_none(v) for k, v in d.items() if v}
    elif type(d) is list:
        return [_del_none(v) for v in d if v]
    return d


@app.route("/chat", methods=["POST"])
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

    # Parse the request in to a CompletionRequest object
    data = request.get_json()
    req = CompletionRequest(**data)

    # Add the user message to our internal message history.
    messages.append(UserMessage(**req.messages[-1]))

    if data.get("stream") == "true":

        def chunk_processor(chunked_completion_generator):
            """Inline function for postprocessing CompletionResponseChunk objects.

            Here, we just jsonify the chunk and yield it as a string.
            """
            messages.append(AssistantMessage(content=""))
            for chunk in chunked_completion_generator:
                if (next_tok := chunk.choices[0].delta.content) is None:
                    next_tok = ""
                messages[-1].content += next_tok
                print(next_tok, end="")
                yield json.dumps(_del_none(asdict(chunk)))

        return Response(
            chunk_processor(gen.chunked_completion(req)), mimetype="text/event-stream"
        )
    else:
        response = gen.sync_completion(req)

        messages.append(response.choices[0].message)
        print(messages[-1].content)

        return json.dumps(_del_none(asdict(response)))


def initialize_generator(args) -> OpenAiApiGenerator:
    builder_args = BuilderArgs.from_args(args)
    speculative_builder_args = BuilderArgs.from_speculative_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)
    generator_args.chat_mode = False

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
    global gen
    gen = initialize_generator(args)
    app.run()
