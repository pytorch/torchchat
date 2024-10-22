from torchchat.cli.builder import _initialize_tokenizer, TokenizerArgs


try:
    from tokenizer.tiktoken import Tokenizer as TiktokenTokenizer
except ImportError:
    TiktokenTokenizer = None
try:
    from sentencepiece import SentencePieceProcessor
except ImportError:
    SentencePieceProcessor = None


from torchchat.distributed.logging_utils import SingletonLogger

_tokenizer_type = None  # global variable to store the tokenizer type

logger = SingletonLogger.get_logger()


class TokenizerType(Enum):
    Tiktoken = auto()
    SentencePiece = auto()


def _build_chat_tokenizer(
    model_name: str,
    model_base_name: Optional[str] = None,
) -> SentencePieceProcessor | TiktokenTokenizer:
    """Builds a tokenizer for the given model name, and sets the global tokenizer type variable"""

    global _tokenizer_type

    # Try to infer the model base name from the model name:
    # e.g. "llama2-7b-chat" -> "llama2"
    if model_base_name is None:
        model_base_name = model_name.split("-")[0]
        logger.info(
            f"Using model base name '{model_base_name}' to build tokenizer. "
            "If not found, please specify it using the `model_base_name` argument."
        )

    # Create base args for tokenizer
    default_model_dir = Path(
        os.getenv("TORCHCHAT_MODELDIR", "~/.torchchat/model-cache")
    ).expanduser()

    tokenconfig = {
        "model_directory": default_model_dir,
        "model": model_base_name,
        "tokenizer_path": None,
    }
    args = dict_to_args(tokenconfig)
    tokenizer_args = TokenizerArgs.from_args(args)
    tokenizer = _initialize_tokenizer(tokenizer_args)
    assert tokenizer is not None, f"Failed to get tokenizer using {tokenconfig=}"
    logger.info(
        f"using tokenizer = {tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
    )
    # set global variable _tokenizer_type
    if isinstance(tokenizer, TiktokenTokenizer):
        _tokenizer_type = TokenizerType.Tiktoken
    elif isinstance(tokenizer, SentencePieceProcessor):
        _tokenizer_type = TokenizerType.SentencePiece
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer.__class__}")

    logger.info(f"tokenizer type = {_tokenizer_type}")
    return tokenizer, _tokenizer_type
