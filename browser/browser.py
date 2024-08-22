import time

import streamlit as st
from openai import OpenAI

st.title("torchchat")

client = OpenAI(
    base_url="http://127.0.0.1:5000/v1",
    api_key="813",  # The OpenAI API requires an API key, but since we don't consume it, this can be any non-empty string.
)


with st.sidebar:
    response_max_tokens = st.slider(
        "Max Response Tokens", min_value=10, max_value=1000, value=250, step=10
    )
    st.divider()

    # Build model list and allow user to change the model running on the server.
    try:
        models = client.models.list().data
        model_keys = [model.id for model in models]
    except:
        models = []
        model_keys = []
    selected_model = st.selectbox(
        label="Model",
        options=model_keys,
    )
    is_instruct_model = "instruct" in selected_model.lower()

    st.divider()

    # Change system prompt and default chat message.
    system_prompt = st.text_area(
        label="System Prompt",
        value=(
            "You're an assistant. Answer questions directly, be brief, and have fun."
            if is_instruct_model
            else f'Selected model "{selected_model}" doesn\'t support chat.'
        ),
        disabled=not is_instruct_model,
    )
    assistant_prompt = st.text_area(
        label="Assistant Prompt",
        value=(
            "How can I help you?"
            if is_instruct_model
            else f'Selected model "{selected_model}" doesn\'t support chat.'
        ),
        disabled=not is_instruct_model,
    )

    st.divider()

    # Manage chat histoory and prompts.
    start_state = (
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "assistant", "content": assistant_prompt},
        ]
        if is_instruct_model
        else []
    )
    if st.button("Reset Chat", type="primary"):
        st.session_state["messages"] = start_state


st.session_state["messages"] = start_state


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

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

        response = st.write_stream(
            get_streamed_completion(
                client.chat.completions.create(
                    model=selected_model,
                    messages=st.session_state.messages,
                    max_tokens=response_max_tokens,
                    stream=True,
                )
            )
        )[0]

    st.session_state.messages.append({"role": "assistant", "content": response})
