# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from api.api import AssistantMessage, CompletionRequest, OpenAiApiGenerator

from build.builder import BuilderArgs, TokenizerArgs
from flask import Flask, jsonify, request, Response
from generate import GeneratorArgs

app = Flask(__name__)
# Messages and gen are kept global so they can be accessed by the flask app endpoints.
messages: list = []
gen: OpenAiApiGenerator = None


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Endpoint for the Chat API. This endpoint is used to generate a response to a user prompt.
    This endpoint emulates the behavior of the OpenAI Chat API. (https://platform.openai.com/docs/api-reference/chat)

    ** Warning ** : Not all arguments of the CompletionRequest are consumed.

    See https://github.com/pytorch/torchchat/issues/973 and the OpenAiApiGenerator class for more details.

    """
    data = request.get_json()

    # Add user message to chat history
    messages.append(data["messages"][-1])
    prompt = messages[-1]["content"]

    # Generate the assistant response
    req = CompletionRequest(
        model=gen.builder_args.checkpoint_path,
        prompt=prompt,
        temperature=0,
        messages=[],
    )

    response = ""

    def unwrap(completion_generator):
        token_count = 0
        for chunk_response in completion_generator:
            content = chunk_response.choices[0].delta.content
            if not gen.is_llama3_model or content not in set(
                gen.tokenizer.special_tokens.keys()
            ):
                yield content if content is not None else ""
            if content == gen.tokenizer.eos_id():
                yield "."
            token_count += 1

    if data.get("stream") == "true":
        return Response(unwrap(gen.completion(req)), mimetype="text/event-stream")
    else:
        for content in unwrap(gen.completion(req)):
            response += content

    # Add assistant response to chat history
    messages.append(AssistantMessage(content=response))

    return jsonify({"response": response})


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
