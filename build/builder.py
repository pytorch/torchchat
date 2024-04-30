# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch._dynamo.config
import torch._inductor.config

from config.model_config import resolve_model_config
from quantize import quantize_model

from build.model import Transformer
from build.utils import device_sync, name_to_dtype


@dataclass
class BuilderArgs:
    checkpoint_path: Optional[Union[Path, str]] = None
    checkpoint_dir: Optional[Union[Path, str]] = None
    params_path: Optional[Union[Path, str]] = None
    params_table: Optional[str] = None
    gguf_path: Optional[Union[Path, str]] = None
    gguf_kwargs: Optional[Dict[str, Any]] = None
    dso_path: Optional[Union[Path, str]] = None
    pte_path: Optional[Union[Path, str]] = None
    device: Optional[str] = None
    precision: torch.dtype = torch.float32
    setup_caches: bool = False
    use_tp: bool = False
    is_chat_model: bool = False
    prefill_possible: bool = False

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not (
            (self.checkpoint_path and self.checkpoint_path.is_file())
            or (self.checkpoint_dir and self.checkpoint_dir.is_dir())
            or (self.gguf_path and self.gguf_path.is_file())
            or (self.dso_path and Path(self.dso_path).is_file())
            or (self.pte_path and Path(self.pte_path).is_file())
        ):
            raise RuntimeError(
                "need to specified a valid checkpoint path, checkpoint dir, gguf path, DSO path, or PTE path"
            )

        if self.dso_path and self.pte_path:
            raise RuntimeError("specify either DSO path or PTE path, but not both")

        if self.checkpoint_path and (self.dso_path or self.pte_path):
            print(
                "Warning: checkpoint path ignored because an exported DSO or PTE path specified"
            )
        if self.checkpoint_dir and (self.dso_path or self.pte_path):
            print(
                "Warning: checkpoint dir ignored because an exported DSO or PTE path specified"
            )
        if self.gguf_path and (self.dso_path or self.pte_path):
            print(
                "Warning: GGUF path ignored because an exported DSO or PTE path specified"
            )
        if not (self.dso_path) and not (self.pte_path):
            self.prefill_possible = True

    @classmethod
    def from_args(cls, args):  # -> BuilderArgs:
        # Handle disabled checkpoint_dir option
        checkpoint_dir = None
        if hasattr(args, "checkpoint_dir"):
            checkpoint_dir = args.checkpoint_dir

        checkpoint_path = args.checkpoint_path
        params_table = args.params_table
        if args.model:  # Using a named, well-known model
            model_config = resolve_model_config(args.model)

            checkpoint_path = (
                Path(args.model_directory)
                / model_config.name
                / model_config.checkpoint_file
            )
            # The transformers config is keyed on the last section
            # of the name/path.
            params_table = (
                model_config.transformer_params_key or model_config.name.split("/")[-1]
            )

        is_chat_model = False
        if args.is_chat_model:
            is_chat_model = True
        else:
            for path in [
                checkpoint_path,
                checkpoint_dir,
                args.dso_path,
                args.pte_path,
                args.gguf_path,
            ]:
                if path is not None:
                    path = str(path)
                    if path.endswith("/"):
                        path = path[:-1]
                    if os.path.isfile(path):
                        path = os.path.dirname(path)

                    path_basename = os.path.basename(path).lower()
                    if "chat" in path_basename or "instruct" in path_basename:
                        is_chat_model = True

        if args.output_pte_path and args.dtype.startswith("fast"):
            if args.dtype == "fast":
                dtype = torch.float32
            else:
                dtype = torch.float16
        else:
            dtype = name_to_dtype(args.dtype)

        return cls(
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path,
            params_path=args.params_path,
            params_table=params_table,
            gguf_path=args.gguf_path,
            gguf_kwargs=None,
            dso_path=args.dso_path,
            pte_path=args.pte_path,
            device=args.device,
            precision=dtype,
            setup_caches=(args.output_dso_path or args.output_pte_path),
            use_tp=False,
            is_chat_model=is_chat_model,
        )

    @classmethod
    def from_speculative_args(cls, args):  # -> BuilderArgs:
        speculative_builder_args = BuilderArgs.from_args(args)
        # let's limit multi-checkpoint to checker
        speculative_builder_args.checkpoint_dir = None
        speculative_builder_args.checkpoint_path = args.draft_checkpoint_path
        speculative_builder_args.gguf_path = None
        speculative_builder_args.dso_path = None
        speculative_builder_args.pte_path = None
        return speculative_builder_args


@dataclass
class TokenizerArgs:
    tokenizer_path: Optional[Union[Path, str]] = None
    is_sentencepiece: bool = False
    is_tiktoken: bool = False
    t: Optional[Any] = None

    def __post_init__(self):
        try:
            from tokenizer.tiktoken import Tokenizer as TiktokenTokenizer

            self.t = TiktokenTokenizer(model_path=str(self.tokenizer_path))
            self.is_tiktoken = True
            self.is_sentencepiece = False
            return
        except:
            pass

        try:
            from sentencepiece import SentencePieceProcessor

            self.t = SentencePieceProcessor(model_file=str(self.tokenizer_path))
            self.is_tiktoken = False
            self.is_sentencepiece = True
            return
        except:
            pass

        self.is_tiktoken = False
        self.is_sentencepiece = False
        self.t = None
        return

    def validate_model(
        self,
        model: Transformer,
        model_description: str = "model",
    ) -> None:
        if model is None:
            return

        is_tiktoken = self.is_tiktoken
        is_sentencepiece = self.is_sentencepiece
        use_tiktoken = model.config.use_tiktoken

        if not (is_tiktoken == use_tiktoken) or not (is_sentencepiece != use_tiktoken):
            raise RuntimeError(
                f"model-specified tokenizer ({tokenizer_setting_to_name(use_tiktoken)}) does not match provided tokenizer ({tokenizer_setting_to_name(is_tiktoken)} for {model_description}"
            )

        return

    @classmethod
    def from_args(cls, args):  # -> TokenizerArgs:
        is_sentencepiece = False
        is_tiktoken = False

        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
        elif args.model:  # Using a named, well-known model
            model_config = resolve_model_config(args.model)
            tokenizer_path = (
                Path(args.model_directory)
                / model_config.name
                / model_config.tokenizer_file
            )

        elif args.checkpoint_path:
            tokenizer_path = args.checkpoint_path.parent / "tokenizer.model"
        elif hasattr(args, "checkpoint_dir") and args.checkpoint_dir:
            tokenizer_path = args.checkpoint_dir / "tokenizer.model"
        else:
            raise RuntimeError("cannot find tokenizer model")

        if not tokenizer_path.is_file():
            raise RuntimeError(f"did not find tokenizer at {tokenizer_path}")

        return cls(
            tokenizer_path=tokenizer_path,
            is_sentencepiece=is_sentencepiece,
            is_tiktoken=is_tiktoken,
            t=None,
        )


def _initialize_tokenizer(tokenizer_args: TokenizerArgs):
    return tokenizer_args.t


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


# TODO: remove these once ET supports _weight_int4pack_mm
def _set_gguf_kwargs(builder_args, is_et, context: str):
    assert context in ["export", "generate"]
    assert builder_args.gguf_kwargs is None

    if builder_args.gguf_path is None:
        print("No gguf_path provided, so ignoring set_gguf_kwargs.")
        return

    builder_args.gguf_kwargs = {}
    if is_et:
        builder_args.gguf_kwargs["load_as_quantized"] = False


def _unset_gguf_kwargs(builder_args):
    builder_args.gguf_kwargs = None


def _load_model_gguf(builder_args, only_config=False):
    assert builder_args.gguf_path
    if builder_args.gguf_kwargs is None:
        kwargs = {}
    else:
        kwargs = builder_args.gguf_kwargs
    model = Transformer.from_gguf(builder_args.gguf_path, **kwargs)
    return model


def _load_model_default(builder_args, only_config=False):
    assert not builder_args.gguf_path

    with torch.device("meta"):
        if builder_args.params_path:
            model = Transformer.from_params(builder_args.params_path)
        elif builder_args.params_table:
            model = Transformer.from_table(builder_args.params_table)
        else:
            model = Transformer.from_name(builder_args.checkpoint_path.parent.name)

    # checkpoint = torch.load(str(builder_args.checkpoint_path), mmap=True, weights_only=True)
    cps = []
    if builder_args.checkpoint_dir is not None:
        # Load multiple checkpoint; ignore the single path.
        builder_args.checkpoint_path = None
        for i in range(4):
            cp_name = f"consolidated.{i}.pth"
            print(f"Loading {cp_name}")
            cps.append(
                torch.load(
                    os.path.join(builder_args.checkpoint_dir, cp_name),
                    map_location=builder_args.device,
                    mmap=True,
                )
            )

        checkpoint = {}
        for key in cps[0].keys():
            if not torch.allclose(cps[0][key], cps[1][key]):
                values = (cps[0][key], cps[1][key], cps[2][key], cps[3][key])
                if key.endswith("wo.weight") or key.endswith("w2.weight"):
                    checkpoint[key] = torch.cat(values, dim=1)
                else:
                    checkpoint[key] = torch.cat(values, dim=0)
            else:
                checkpoint[key] = cps[0][key]
    else:
        checkpoint = torch.load(
            str(builder_args.checkpoint_path),
            map_location=builder_args.device,
            mmap=True,
            weights_only=True,
        )

    if "model" in checkpoint and "stories" in str(builder_args.checkpoint_path):
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint, assign=True, strict=False)

    return model


def _load_model(builder_args, only_config=False):
    if builder_args.gguf_path:
        model = _load_model_gguf(builder_args)
    else:
        model = _load_model_default(builder_args)

    if builder_args.use_tp:
        from tp import apply_tp

        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=builder_args.device, dtype=builder_args.precision)
    return model.eval()


def _initialize_model(
    builder_args,
    quantize,
    tokenizer=None,
):
    print("Loading model...")

    if builder_args.gguf_path and (builder_args.dso_path or builder_args.pte_path):
        print("Setting gguf_kwargs for generate.")
        is_dso = builder_args.dso_path is not None
        is_pte = builder_args.pte_path is not None
        assert not (is_dso and is_pte)
        assert builder_args.gguf_kwargs is None
        # TODO: make GGUF load independent of backend
        # currently not working because AVX int_mm broken
        #   (no unpack available)
        _set_gguf_kwargs(builder_args, is_et=is_pte, context="generate")

    if builder_args.dso_path:
        # assert (
        #     quantize is None or quantize == "{ }"
        # ), "quantize not valid for exported DSO model. Specify quantization during export."

        t0 = time.time()
        model = _load_model(builder_args, only_config=True)
        device_sync(device=builder_args.device)
        print(f"Time to load model: {time.time() - t0:.02f} seconds")

        try:
            # Replace model forward with the AOT-compiled forward
            # This is a hacky way to quickly demo AOTI's capability.
            # model is still a Python object, and any mutation to its
            # attributes will NOT be seen on by AOTI-compiled forward
            # function, e.g. calling model.setup_cache will NOT touch
            # AOTI compiled and maintained model buffers such as kv_cache.
            model.forward = torch._export.aot_load(
                str(builder_args.dso_path.absolute()), builder_args.device
            )
        except:
            raise RuntimeError(f"Failed to load AOTI compiled {builder_args.dso_path}")
    elif builder_args.pte_path:
        # assert (
        #     quantize is None or quantize == "{ }"
        # ), "quantize not valid for exported PTE model. Specify quantization during export."

        t0 = time.time()
        model = _load_model(builder_args, only_config=True)
        device_sync(device=builder_args.device)
        print(f"Time to load model: {time.time() - t0:.02f} seconds")

        try:
            from build.model_et import PTEModel

            model = PTEModel(model.config, builder_args.pte_path)
        except Exception:
            raise RuntimeError(f"Failed to load ET compiled {builder_args.pte_path}")
    else:
        t0 = time.time()
        model = _load_model(builder_args)
        device_sync(device=builder_args.device)
        print(f"Time to load model: {time.time() - t0:.02f} seconds")

        if quantize:
            t0q = time.time()
            print(f"Quantizing the model with: {quantize}")
            quantize_model(model, builder_args.device, quantize, tokenizer)
            device_sync(device=builder_args.device)
            print(f"Time to quantize model: {time.time() - t0q:.02f} seconds")

        if builder_args.setup_caches:
            with torch.device(builder_args.device):
                model.setup_caches(
                    max_batch_size=1, max_seq_length=model.config.max_seq_length
                )

        model.to(dtype=builder_args.precision)

    return model


def tokenizer_setting_to_name(tiktoken: bool = False) -> str:
    return "TikToken" if tiktoken else "SentencePiece"
