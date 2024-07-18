# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
import torch._dynamo.config
import torch._inductor.config

from config.model_config import resolve_model_config
from distributed import init_distributed, ParallelDims, parallelize_llama
from quantization.quantize import quantize_model
from utils.measure_time import measure_time

from build.model import Transformer
from build.utils import device_sync, is_cpu_device, is_cuda_or_cpu_device, name_to_dtype
from distributed import launch_distributed


@dataclass
class BuilderArgs:
    checkpoint_path: Optional[Union[Path, str]] = None
    checkpoint_dir: Optional[Union[Path, str]] = None
    dcp_dir: Optional[Union[Path, str]] = None
    params_path: Optional[Union[Path, str]] = None
    params_table: Optional[str] = None
    gguf_path: Optional[Union[Path, str]] = None
    gguf_kwargs: Optional[Dict[str, Any]] = None
    dso_path: Optional[Union[Path, str]] = None
    aoti_package_path: Optional[Union[Path, str]] = None
    pte_path: Optional[Union[Path, str]] = None
    device: Optional[str] = None
    precision: torch.dtype = torch.float32
    setup_caches: bool = False
    use_distributed: bool = False
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
            or (self.aoti_package_path and Path(self.aoti_package_path).is_file())
            or (self.pte_path and Path(self.pte_path).is_file())
        ):
            raise RuntimeError(
                "need to specified a valid checkpoint path, checkpoint dir, gguf path, DSO path, or PTE path"
            )

        if self.pte_path and self.aoti_package_path:
            raise RuntimeError("specify either AOTI Package path or PTE path, but not more than one")

        if self.checkpoint_path and (self.pte_path or self.aoti_package_path):
            print(
                "Warning: checkpoint path ignored because an exported AOTI or PTE path specified"
            )
        if self.checkpoint_dir and (self.pte_path or self.aoti_package_path):
            print(
                "Warning: checkpoint dir ignored because an exported AOTI or PTE path specified"
            )
        if self.gguf_path and (self.pte_path or self.aoti_package_path):
            print(
                "Warning: GGUF path ignored because an exported AOTI or PTE path specified"
            )
        if not (self.dso_path) and not (self.aoti_package_path):
            self.prefill_possible = True

    @classmethod
    def from_args(cls, args):  # -> BuilderArgs:
        # Handle disabled checkpoint_dir option
        checkpoint_dir = None
        if hasattr(args, "checkpoint_dir"):
            checkpoint_dir = args.checkpoint_dir
        if hasattr(args, "dcp_dir"):
            dcp_dir = args.dcp_dir

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
                args.aoti_package_path,
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
                # As per Kimish, float32 should be faster on ET XNNPACK
                # (because fp16 is implemented as upcast to fp32 for several
                # operators, and in particular a8w4dq and ET's sdpa+kv)
                dtype = torch.float32
            else:
                dtype = torch.float16
        else:
            dtype = name_to_dtype(args.dtype, args.device)

        return cls(
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path,
            dcp_dir=dcp_dir,
            params_path=args.params_path,
            params_table=params_table,
            gguf_path=args.gguf_path,
            gguf_kwargs=None,
            dso_path=args.dso_path,
            aoti_package_path=args.aoti_package_path,
            pte_path=args.pte_path,
            device=args.device,
            precision=dtype,
            setup_caches=(args.output_dso_path or args.output_pte_path or args.output_aoti_package_path),
            use_distributed=args.distributed,
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
        speculative_builder_args.aoti_package_path = None
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

        if self.is_tiktoken == self.is_sentencepiece:
            raise RuntimeError(f"no tokenizer was found at {self.tokenizer_path}")

        is_tiktoken = self.is_tiktoken
        is_sentencepiece = self.is_sentencepiece
        use_tiktoken = model.config.use_tiktoken

        if not (is_tiktoken == use_tiktoken) or not (is_sentencepiece != use_tiktoken):
            raise RuntimeError(
                f"model-specified tokenizer ({tokenizer_setting_to_name(use_tiktoken)}) does not match provided tokenizer ({tokenizer_setting_to_name(is_tiktoken)}) for {model_description}"
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


def _init_model_on_meta_device(builder_args):
    with torch.device("meta"):
        if builder_args.params_path:
            return Transformer.from_params(builder_args.params_path)
        elif builder_args.params_table:
            return Transformer.from_table(builder_args.params_table)
        else:
            return Transformer.from_name(builder_args.checkpoint_path.parent.name)


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

    model = _init_model_on_meta_device(builder_args)
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


def _maybe_init_distributed(
    builder_args: BuilderArgs,
) -> Tuple[Optional[DeviceMesh], Optional[ParallelDims]]:
    """
    Initialize distributed related setups if the user specified 
    using distributed inference. If not, this is a no-op.

    Args:
        builder_args (:class:`BuilderArgs`):
            Command args for model building.
    Returns:
        Tuple[Optional[DeviceMesh], Optional[ParallelDims]]: 
            - The first element is an optional DeviceMesh object, 
            which which describes the mesh topology of devices for the DTensor.
            - The second element is an optional ParallelDims object, 
            which represents the parallel dimensions configuration.
    """
    if not builder_args.use_distributed:
        return None, None
    dist_config = 'llama3_8B.toml'  # TODO - integrate with chat cmd line
    
    world_mesh, parallel_dims = launch_distributed(dist_config) 
    
    assert world_mesh is not None and parallel_dims is not None, f"failed to launch distributed using {dist_config}"
    
    return world_mesh, parallel_dims


def _maybe_parellelize_model(
    model: nn.Module,
    builder_args: BuilderArgs,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
) -> nn.Module:
    """
    We parallelize the module and load the distributed checkpoint to the model
    if the user specifies using distributed inference. If not, this is a no-op.

    Args:
        module (:class:`nn.Module`):
            Module to be parallelized.
        builder_args (:class:`BuilderArgs`):
            Command args for model building.
        world_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        parallel_dims (:class:`ParallelDims`):
            Object which represents the parallel dimensions configuration.
    Returns:
        A :class:`nn.Module` object which is parallelized and checkpoint loaded
        if the user specifies using distributed inference.
    """
    if world_mesh is None:
        return model
    assert parallel_dims is not None
    print("Applying model parallel to model ...")
    parallelize_llama(model, world_mesh, parallel_dims)
    return load_checkpoints_to_model(model, builder_args, world_mesh)


def _load_model(builder_args, only_config=False):
    world_mesh, parallel_dims = _maybe_init_distributed(builder_args)
    if builder_args.gguf_path:
        model = _load_model_gguf(builder_args)
    elif builder_args.use_distributed:
        model = _init_model_on_meta_device(builder_args)
    else:
        model = _load_model_default(builder_args)
    model = _maybe_parellelize_model(model, builder_args, world_mesh, parallel_dims)

    model = model.to(device=builder_args.device, dtype=builder_args.precision)
    return model.eval()


def _initialize_model(
    builder_args,
    quantize,
    tokenizer=None,
):
    print("Loading model...")

    if builder_args.gguf_path and (builder_args.dso_path or builder_args.aoti_package_path or builder_args.pte_path):
        print("Setting gguf_kwargs for generate.")
        is_dso = builder_args.dso_path is not None
        is_aoti_package = builder_args.aoti_package_path is not None
        is_pte = builder_args.pte_path is not None
        assert not (is_dso and is_aoti_package and is_pte)
        assert builder_args.gguf_kwargs is None
        # TODO: make GGUF load independent of backend
        # currently not working because AVX int_mm broken
        #   (no unpack available)
        _set_gguf_kwargs(builder_args, is_et=is_pte, context="generate")

    if builder_args.dso_path:
        if not is_cuda_or_cpu_device(builder_args.device):
            print(
                f"Cannot load specified DSO to {builder_args.device}. Attempting to load model to CPU instead"
            )
            builder_args.device = "cpu"

        # assert (
        #     quantize is None or quantize == "{ }"
        # ), "quantize not valid for exported DSO model. Specify quantization during export."

        with measure_time("Time to load model: {time:.02f} seconds"):
            model = _load_model(builder_args, only_config=True)
            device_sync(device=builder_args.device)

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
    
    elif builder_args.aoti_package_path:
        if not is_cuda_or_cpu_device(builder_args.device):
            print(
                f"Cannot load specified PT2 to {builder_args.device}. Attempting to load model to CPU instead"
            )
            builder_args.device = "cpu"

        # assert (
        #     quantize is None or quantize == "{ }"
        # ), "quantize not valid for exported PT2 model. Specify quantization during export."

        with measure_time("Time to load model: {time:.02f} seconds"):
            model = _load_model(builder_args, only_config=True)
            device_sync(device=builder_args.device)

        try:
            # Replace model forward with the AOT-compiled forward
            # This is a hacky way to quickly demo AOTI's capability.
            # model is still a Python object, and any mutation to its
            # attributes will NOT be seen on by AOTI-compiled forward
            # function, e.g. calling model.setup_cache will NOT touch
            # AOTI compiled and maintained model buffers such as kv_cache.
            from torch._inductor.package import load_package
            model.forward = load_package(
                str(builder_args.aoti_package_path.absolute()), builder_args.device
            )
        except:
            raise RuntimeError(f"Failed to load AOTI compiled {builder_args.aoti_package_path}")

    elif builder_args.pte_path:
        if not is_cpu_device(builder_args.device):
            print(
                f"Cannot load specified PTE to {builder_args.device}. Attempting to load model to CPU instead"
            )
            builder_args.device = "cpu"

        # assert (
        #     quantize is None or quantize == "{ }"
        # ), "quantize not valid for exported PTE model. Specify quantization during export."

        with measure_time("Time to load model: {time:.02f} seconds"):
            model = _load_model(builder_args, only_config=True)
            device_sync(device=builder_args.device)

        try:
            from build.model_et import PTEModel

            model = PTEModel(model.config, builder_args.pte_path)
        except Exception:
            raise RuntimeError(f"Failed to load ET compiled {builder_args.pte_path}")
    else:
        with measure_time("Time to load model: {time:.02f} seconds"):
            model = _load_model(builder_args)
            device_sync(device=builder_args.device)

        if quantize:
            print(f"Quantizing the model with: {quantize}")
            with measure_time("Time to quantize model: {time:.02f} seconds"):
                quantize_model(model, builder_args.device, quantize, tokenizer)
                device_sync(device=builder_args.device)

        if builder_args.setup_caches:
            with torch.device(builder_args.device):
                model.setup_caches(
                    max_batch_size=1, max_seq_length=model.config.max_seq_length
                )

        model.to(dtype=builder_args.precision)

    print("-----------------------------------------------------------")
    return model


def tokenizer_setting_to_name(tiktoken: bool = False) -> str:
    return "TikToken" if tiktoken else "SentencePiece"
