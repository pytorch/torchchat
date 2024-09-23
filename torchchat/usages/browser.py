import base64
import logging
import time
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from openai import OpenAI

st.title("torchchat")

start_state = [
    {
        "role": "system",
        "content": "You're a helpful assistant - have fun.",
    },
    {"role": "assistant", "content": "How can I help you?"},
]

st.session_state.uploader_key = 0


def reset_per_message_state():
    # Catch all function for anything that should be reset between each message.
    _update_uploader_key()


def _update_uploader_key():
    # Increment the uploader key to reset the file uploader after each message.
    st.session_state.uploader_key = int(time.time())


with st.sidebar:
    # API Configuration
    api_base_url = st.text_input(
        label="API Base URL",
        value="http://127.0.0.1:5000/v1",
        help="The base URL for the OpenAI API to connect to",
    )

    st.divider()
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.01
    )

    response_max_tokens = st.slider(
        "Max Response Tokens", min_value=10, max_value=1000, value=250, step=10
    )
    if st.button("Reset Chat", type="primary"):
        st.session_state["messages"] = start_state

    image_prompts = st.file_uploader(
        "Image Prompts",
        type=["jpeg"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
    )

    for image in image_prompts:
        st.image(image)


client = OpenAI(
    base_url=api_base_url,
    api_key="813",  # The OpenAI API requires an API key, but since we don't consume it, this can be any non-empty string.
)

if "messages" not in st.session_state:
    st.session_state["messages"] = start_state


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if type(msg["content"]) is list:
            for content in msg["content"]:
                if content["type"] == "image_url":
                    extension = (
                        content["image_url"].split(";base64")[0].split("image/")[1]
                    )
                    base64_repr = content["image_url"].split("base64,")[1]
                    st.image(base64.b64decode(base64_repr))
                else:
                    st.write(content["text"])
        elif type(msg["content"]) is dict:
            if msg["content"]["type"] == "image_url":
                st.image(msg["content"]["image_url"])
            else:
                st.write(msg["content"]["text"])
        elif type(msg["content"]) is str:
            st.write(msg["content"])
        else:
            st.write(f"Unhandled content type: {type(msg['content'])}")


if prompt := st.chat_input():
    user_message = {"role": "user", "content": [{"type": "text", "text": prompt}]}

    if image_prompts:
        for image_prompt in image_prompts:
            extension = Path(image_prompt.name).suffix.strip(".")
            image_bytes = image_prompt.getvalue()
            base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
            user_message["content"].append(
                {
                    "type": "image_url",
                    "image_url": f"data:image/{extension};base64,{base64_encoded}",
                }
            )
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.write(prompt)
        for img in image_prompts:
            st.image(img)

    image_prompts = None
    reset_per_message_state()

    with st.chat_message("assistant"), st.status(
        "Generating... ", expanded=True
    ) as status:

        def get_streamed_completion(completion_generator):
            start = time.time()
            tokcount = 0
            for chunk in completion_generator:
                tokcount += 1
                yield chunk.choices[0].delta.content

            status.update(
                label="Done, averaged {:.2f} tokens/second".format(
                    tokcount / (time.time() - start)
                ),
                state="complete",
            )

        try:
            response = st.write_stream(
                get_streamed_completion(
                    client.chat.completions.create(
                        model="llama3",
                        messages=st.session_state.messages,
                        max_tokens=response_max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
                )
            )[0]
        except Exception as e:
            response = st.error(f"Error: {e}")
            print(e)

    st.session_state.messages.append({"role": "assistant", "content": response})
