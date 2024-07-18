# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time

import streamlit as st
from api.api import CompletionRequest, OpenAiApiGenerator

from build.builder import BuilderArgs, TokenizerArgs

from generate import GeneratorArgs


def main(args):
    builder_args = BuilderArgs.from_args(args)
    speculative_builder_args = BuilderArgs.from_speculative_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)
    generator_args.chat_mode = False

    @st.cache_resource
    def initialize_generator() -> OpenAiApiGenerator:
        return OpenAiApiGenerator(
            builder_args,
            speculative_builder_args,
            tokenizer_args,
            generator_args,
            args.profile,
            args.quantize,
            args.draft_quantize,
        )

    gen = initialize_generator()

    st.title("torchchat")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"), st.status(
            "Generating... ", expanded=True
        ) as status:

            req = CompletionRequest(
                model=gen.builder_args.checkpoint_path,
                prompt=prompt,
                temperature=generator_args.temperature,
                messages=[],
            )

            def unwrap(completion_generator):
                start = time.time()
                tokcount = 0
                for chunk_response in completion_generator:
                    content = chunk_response.choices[0].delta.content
                    if not gen.is_llama3_model or content not in set(
                        gen.tokenizer.special_tokens.keys()
                    ):
                        yield content
                    if content == gen.tokenizer.eos_id():
                        yield "."
                    tokcount += 1
                status.update(
                    label="Done, averaged {:.2f} tokens/second".format(
                        tokcount / (time.time() - start)
                    ),
                    state="complete",
                )

            response = st.write_stream(unwrap(gen.completion(req)))

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
