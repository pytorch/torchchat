"""
Unit tests for chat formatters
"""

# Third Party
import pytest

# Local
from torchchat.generate import (
    HFTokenizerChatFormatter,
    Llama2ChatFormatter,
    Llama3ChatFormatter,
)

## Helpers #####################################################################

class DummyTokenizer:
    """Dummy tokenizer that encodes as strings so it's easy to check formatting"""
    def encode(self, text, *_, **__):
        return text


class DummySPTokenizer(DummyTokenizer):
    """Emulated Sentencepiece tokenizer with bos/eos"""
    bos = "<s>"
    eos = "</s>"


class DummyLlama3Tokenizer(DummyTokenizer):
    class _IdentityDict:
        def __getitem__(self, key):
            return key
    special_tokens = _IdentityDict()


class DummyHFTokenizer(DummyTokenizer):
    """Dummy made up chat template scheme"""
    # Sequence
    bos = "<bos>"
    # Turn
    bot = "<bot>"
    eot = "<eot>"
    # Role
    bor = "<bor>"
    eor = "<eor>"
    def apply_chat_template(self, messages, add_generation_prompt):
        out = [self.bos]
        role = None
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            out.append(f"{self.bot}{self.bor}{role}{self.eor}{content}{self.eot}")
        if add_generation_prompt and role != "assistant":
            out.append(f"{self.bot}{self.bor}assistant{self.eor}")
        return "\n".join(out)


def check_rendering(fmt, messages, expected, add_generation_prompt):
    """Render messages and compare to expected output"""
    assert "".join(fmt.encode_dialog_prompt(messages, add_generation_prompt)) == expected


def make_message(role, text):
    return {"role": role, "content": text}


SYSTEM_PROMPT = "You are a helpful assistant, feel free to ask me anything."
USER1 = "Hello world!"
ASSISTANT1 = "Greetings! How can I help you?"
USER2 = "Why is the sky blue?"
ASSISTANT2 = "The sky appears blue because of a phenomenon called Rayleigh scattering."


# Stock sets of messages to test
MSGS_NO_SYS= [
    make_message("user", USER1),
]
MSGS_SYS_USR = [
    make_message("system", SYSTEM_PROMPT),
    make_message("user", USER1),
]
MSGS_SYS_USR_ASST = [
    make_message("system", SYSTEM_PROMPT),
    make_message("user", USER1),
    make_message("assistant", ASSISTANT1),
]
MSGS_MULTI_TURN = [
    make_message("system", SYSTEM_PROMPT),
    make_message("user", USER1),
    make_message("assistant", ASSISTANT1),
    make_message("user", USER2),
    make_message("assistant", ASSISTANT2),
]

## Llama2ChatFormatter #########################################################

@pytest.mark.parametrize(
    ["messages", "expected"],
    [
        # single user message (no system prompt)
        (MSGS_NO_SYS, f"<s>[INST] {USER1} [/INST]"),
        # sys, usr
        (MSGS_SYS_USR, f"""<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

{USER1} [/INST]"""),
        # sys, usr, asst
        (MSGS_SYS_USR_ASST, f"""<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

{USER1} [/INST] {ASSISTANT1} </s>
"""),
        # sys, usr, asst, usr, asst
        (MSGS_MULTI_TURN, f"""<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

{USER1} [/INST] {ASSISTANT1} </s>
<s>[INST] {USER2} [/INST] {ASSISTANT2} </s>
"""),
    ]
)
def test_llama2_chat_formatter(messages, expected):
    """Tests for Llama2 following the official guide
    https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-2/
    """
    tok = DummySPTokenizer()
    fmt = Llama2ChatFormatter(tok)
    # NOTE: add_generation_prompt not used by Llama2
    check_rendering(fmt, messages, expected, True)

## Llama3ChatFormatter #########################################################

@pytest.mark.parametrize(
    ["messages", "expected"],
    [
        # single user message (no system prompt)
        (MSGS_NO_SYS, f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{USER1}<|eot_id|>"""),
        # sys, usr
        (MSGS_SYS_USR, f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{USER1}<|eot_id|>"""),
        # sys, usr, asst
        (MSGS_SYS_USR_ASST, f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{USER1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{ASSISTANT1}<|eot_id|>"""),
        # sys, usr, asst, usr, asst
        (MSGS_MULTI_TURN, f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{USER1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{ASSISTANT1}<|eot_id|><|start_header_id|>user<|end_header_id|>

{USER2}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{ASSISTANT2}<|eot_id|>"""),
    ]
)
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_llama3_chat_formatter(messages, expected, add_generation_prompt):
    """Tests for Llama3 following the official guide
    https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    """
    tok = DummyLlama3Tokenizer()
    fmt = Llama3ChatFormatter(tok)
    # No assistant prompt added if the last message is from the assistant
    if add_generation_prompt and messages[-1]["role"] != "assistant":
        expected += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    check_rendering(fmt, messages, expected, add_generation_prompt)

## HFTokenizerChatFormatter ####################################################

@pytest.mark.parametrize(
    ["messages", "expected"],
    [
        # single user message (no system prompt)
        (MSGS_NO_SYS, f"""<bos>
<bot><bor>user<eor>{USER1}<eot>"""),
        # sys, usr
        (MSGS_SYS_USR, f"""<bos>
<bot><bor>system<eor>{SYSTEM_PROMPT}<eot>
<bot><bor>user<eor>{USER1}<eot>"""),
        # sys, usr, asst
        (MSGS_SYS_USR_ASST, f"""<bos>
<bot><bor>system<eor>{SYSTEM_PROMPT}<eot>
<bot><bor>user<eor>{USER1}<eot>
<bot><bor>assistant<eor>{ASSISTANT1}<eot>"""),
        # sys, usr, asst, usr, asst
        (MSGS_MULTI_TURN, f"""<bos>
<bot><bor>system<eor>{SYSTEM_PROMPT}<eot>
<bot><bor>user<eor>{USER1}<eot>
<bot><bor>assistant<eor>{ASSISTANT1}<eot>
<bot><bor>user<eor>{USER2}<eot>
<bot><bor>assistant<eor>{ASSISTANT2}<eot>"""),
    ]
)
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_hf_chat_formatter(messages, expected, add_generation_prompt):
    tok = DummyHFTokenizer()
    fmt = HFTokenizerChatFormatter(tok)
    # No assistant prompt added if the last message is from the assistant
    if add_generation_prompt and messages[-1]["role"] != "assistant":
        expected += f"\n{tok.bot}{tok.bor}assistant{tok.eor}"
    check_rendering(fmt, messages, expected, add_generation_prompt)
