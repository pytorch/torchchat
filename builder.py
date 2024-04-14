# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config

from quantize import (
    quantize_model, name_to_dtype, set_precision, get_precision
)
from cli import cli_args
from dataclasses import dataclass
from typing import Union, Optional

from sentencepiece import SentencePieceProcessor
from model import Transformer


@dataclass
class BuilderArgs:
    checkpoint_path: Optional[Union[Path, str]] = None
    checkpoint_dir: Optional[Union[Path, str]] = None
    params_path: Optional[Union[Path, str]] = None
    params_table: Optional[str] = None
    gguf_path: Optional[Union[Path, str]] = None
    dso_path: Optional[Union[Path, str]] = None
    pte_path: Optional[Union[Path, str]] = None
    device: str = "cpu"
    precision: torch.dtype = torch.float32
    setup_caches: bool = False
    use_tp: bool = False

    @classmethod
    def from_args(cls, args): # -> BuilderArgs:
        return cls(
            checkpoint_path = args.checkpoint_path,
            checkpoint_dir = args.checkpoint_dir,
            params_path = args.params_path,
            params_table = args.params_table,
            gguf_path = args.gguf_path,
            dso_path = args.dso_path,
            pte_path = args.pte_path,
            device = args.device,
            precision = name_to_dtype(args.precision),
            setup_caches = (args.output_dso_path or args.output_pte_path),
            use_tp = False,
        )
    
@dataclass
class TokenizerArgs:
    tokenizer_path: Optional[Union[Path, str]] = None
    is_SentencePiece: bool = True
    is_TikToken: bool = False

    @classmethod
    def from_args(cls, args): # -> TokenizerArgs:
        is_Sentencepiece = True
        is_TikToken = False
        
        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
        elif argscheckpoint_path:
            tokenizer_path = args.checkpoint_path.parent / "tokenizer.model"
        elif checkpoint_dir:
            tokenizer_path = args.checkpoint_dir / "tokenizer.model"
        else:
            raise RuntimeError(f"cannot find tokenizer model")
            
        if not tokenizer_path.is_file():
                raise RuntimeError(f"did not find tokenizer at {tokenizer_path}")

        if args.toktoken:
            is_Sentencepiece = False
            is_TikToken = True

        return cls(
            tokenizer_path=tokenizer_path,
            is_SentencePiece=is_SentencePiece,
            is_TikToken=is_TikToken
        )

def _initialize_tokenizer(config: TokenizerArgs):
    if is_SentencePiece:
        return SentencePieceProcessor(model_file=str(tokenizer_path))
    elif is_TikToken:
        raise RUntimeError("TikToken not implemented yet!")
    else:
        raise RUntimeError("must specify a valid tokenizer in TokenizerArgs")
        

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={ device } is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

def _load_model(
        checkpoint_path,
        checkpoint_dir,
        params_path,
        params_table,
        gguf_path,
        device,
        precision,
        use_tp # =False
):
    use_cuda = "cuda" in device
    with torch.device("meta"):
        if params_path:
            model = Transformer.from_params(params_path)
        elif params_table:
            model = Transformer.from_table(params_path)
        elif gguf_path:
            model = Transformer.from_gguf(gguf_path)            
        else:
            model = Transformer.from_name(checkpoint_path.parent.name)

    # checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    cps = []
    if checkpoint_dir is not None:
        # Load multiple checkpoint; ignore the single path.
        checkpoint_path = None
        for i in range(4):
            cp_name = f"consolidated.{i}.pth"
            print(f"Loading {cp_name}")
            cps.append(
                torch.load(
                    os.path.join(checkpoint_dir, cp_name),
                    map_location=device,
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
        checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True, weights_only=True)

    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp

        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def _initialize_model(
        checkpoint_path,
        checkpoint_dir,
        params_path,
        params_table,
        gguf_path,
        dso_path,
        pte_path,
        quantize,
        device,
        precision,
        setup_caches,
        use_tp # =False
):
    assert (
        (checkpoint_path and checkpoint_path.is_file()) or
        (checkpoint_dir and checkpoint_path.is_dir()) or
        (gguf_path and gguf_path.is_file()) or
        (dso_path and Path(dso_path).is_file()) or
        (pte_path and Path(pte_path).is_file())
    ), "need to specified a valid checkpoint path, checkpoint dir, gguf path, DSO path, or PTE path"
    assert not (dso_path and pte_path), "specify either DSO path or PTE path, but not both"

    if (checkpoint_path and (dso_path or pte_path)):
        print("Warning: checkpoint path ignored because an exported DSO or PTE path specified")
    if (checkpoint_dir and (dso_path or pte_path)):
        print("Warning: checkpoint dir ignored because an exported DSO or PTE path specified")
    if (gguf_path and (dso_path or pte_path)):
        print("Warning: GGUF path ignored because an exported DSO or PTE path specified")

    print("Loading model ...")
    t0 = time.time()    
    model_ = _load_model(
        checkpoint_path,
        checkpoint_dir,
        params_path,
        params_table,
        gguf_path,
        device,
        precision,
        use_tp
    )
    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    if dso_path:
        # make sure user did not try to set dtype
        # assert model_dtype == "float32", f"dtype setting not valid for a DSO model. Specify dtype during export."
        assert quantize is None or quantize == "{ }", f"quantize not valid for exported DSO model. Specify quantization during export."
        try:
            model = model_
            # Replace model forward with the AOT-compiled forward
            # This is a hacky way to quickly demo AOTI's capability.
            # model is still a Python object, and any mutation to its
            # attributes will NOT be seen on by AOTI-compiled forward
            # function, e.g. calling model.setup_cache will NOT touch
            # AOTI compiled and maintained model buffers such as kv_cache.
            model.forward = torch._export.aot_load(str(dso_path.absolute()), device)
        except:
            raise RuntimeError(f"Failed to load AOTI compiled {dso_path}")
    elif pte_path:
        # make sure user did not try to set dtype
        # assert model_dtype == "float32", f"dtype setting not valid for a DSO model. Specify dtype during export."
        assert quantize is None or quantize == "{ }", f"quantize not valid for exported PTE model. Specify quantization during export."
        try:
            from model_et import PTEModel
            model = PTEModel(model_.config, pte_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load ET compiled {pte_path}")
    else:
        model = model_

        if quantize:
            t0q = time.time()
            quantize_model(model, quantize)
            device_sync(device=device)  # MKG
            print(f"Time to quantize model: {time.time() - t0q:.02f} seconds")

        if setup_caches:
            max_seq_length = 350
            with torch.device(device):
                model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        model.to(dtype=precision)

    return model


