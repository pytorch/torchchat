# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import Dim, export

from generate import _load_model, decode_one_token
from quantize import quantize_model

from model import Transformer
# from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
#    XnnpackDynamicallyQuantizedPartitioner,
#)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch_portable_utils import export_to_edge # TODO: change back to executorch.examples.portable.utils when executorch installs correctly

from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from generate import _load_model

from model import Transformer
from torch._export import capture_pre_autograd_graph

default_device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


def materialze_broadcast_of_rope_freq_cis(
    module: torch.nn.Module,
):
    assert isinstance(module, Transformer)
    assert module.freqs_cos.dim() == 2
    dim0 = module.freqs_cos.size(0)
    dim1 = module.freqs_cos.size(1)
    assert (
        module.layers[0].attention.n_local_kv_heads
        == module.layers[0].attention.n_local_heads
    ), f"For rope freqs to be materialzed for broadcast q, k, v num heads must match. For q got {module.attention.n_kv_heads} for k got {module.attention.n_local_heads} and v got {module.attention.n_local_kv_heads}"
    num_heads = module.layers[0].attention.n_local_heads
    module.freqs_cos = module.freqs_cos.view(dim0, 1, dim1)
    module.freqs_cos = module.freqs_cos.expand(dim0, num_heads, dim1).contiguous()
    assert module.freqs_sin.dim() == 2
    assert dim0 == module.freqs_sin.size(
        0
    ), f"sin and cos freq table sizes must match. Mismatch found at dim 0: {dim0} vs {module.freqs_sin.size(0)}"
    assert dim1 == module.freqs_sin.size(
        1
    ), f"sin and cos freq table sizes must match. Mismatch found at dim 1: {dim1} vs {module.freqs_sin.size(1)}"
    module.freqs_sin = module.freqs_sin.view(dim0, 1, dim1)
    module.freqs_sin = module.freqs_sin.expand(dim0, num_heads, dim1).contiguous()
    return module


def canonical_path(path):
    return path


def export_model(model, input, device, output_path, args=None) -> str:  # noqa: C901

    # applied wrapper already in export.
    # export_model = model_wrapper(model, device=device)
    export_model = model
    print(export_model)

    #input = (
    #    torch.tensor([[1]], dtype=torch.long, device=device),
    #    torch.tensor([0], dtype=torch.long, device=device),
    #)

    state_dict = model.state_dict()
    state_dict_dtype = state_dict[next(iter(state_dict))].dtype

    # need to use kv sdpa?
    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_type_promotion=bool(args.dtype == "fp16"),
    )

    dynamic_shapes = None

    if args.dtype is not None:
        if args.dtype == "fp16": # or args.quantization_mode == "int4":
            if state_dict_dtype != torch.float16:
                print("model.to torch.float16")
                model = model.to(dtype=torch.float16)
                state_dict_dtype = torch.float16
        elif args.dtype == "fp32":
            if state_dict_dtype != torch.float32:
                print("model.to torch.float32")
                model = model.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported dtype: {args.dtype}")

    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]), torch.no_grad():
        m = capture_pre_autograd_graph(
            export_model,
            input,
            dynamic_shapes=dynamic_shapes
        )

        edge_manager = export_to_edge(
            m,
            input,
            dynamic_shapes=dynamic_shapes,
            edge_compile_config=edge_config,
        )

    edge_manager = edge_manager.to_backend(XnnpackPartitioner())
    export_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_constant_segment=True,
            extract_delegate_segments=True,
            passes=[
                QuantFusionPass(),
            ],
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    print("The methods are: ", export_program.methods)
    with open(output_path, "wb") as f:
        export_program.write_to_file(f)
    # save_pte_program(export_program, output_path)

    return output_path
