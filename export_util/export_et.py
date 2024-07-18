# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from build.model import Transformer
from build.utils import get_precision

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)

# TODO: change back to executorch.examples.portable.utils
# when executorch installs correctly

from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from export_util.executorch_portable_utils import export_to_edge
from export_util.export_et_util import replace_attention_with_custom_sdpa_attention
from torch._export import capture_pre_autograd_graph

default_device = "cpu"


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


def export_model(model, device, output_path, args=None) -> str:  # noqa: C901

    input = (
        torch.tensor([[1]], dtype=torch.long, device=device),
        torch.tensor([0], dtype=torch.long, device=device),
    )

    state_dict = model.state_dict()
    state_dict_dtype = state_dict[next(iter(state_dict))].dtype
    target_precision = get_precision()
    dynamic_shapes = None

    # TODO: need to use kv sdpa?
    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_type_promotion=bool(target_precision == torch.float16),
    )

    if target_precision == torch.float16 or target_precision == torch.bfloat16:
        if state_dict_dtype != torch.float16:
            print("model.to torch.float16")
            model = model.to(dtype=torch.float16)
            state_dict_dtype = torch.float16
    elif target_precision == torch.float32:
        if state_dict_dtype != torch.float32:
            print("model.to torch.float32")
            model = model.to(dtype=torch.float32)
    elif target_precision == torch.bfloat16:
        print("model.to torch.bfloat16")
        model = model.to(dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported dtype for ET export: {target_precision}")

    replace_attention_with_custom_sdpa_attention(model)
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.MATH]
    ), torch.no_grad():
        m = capture_pre_autograd_graph(model, input, dynamic_shapes=dynamic_shapes)

        edge_manager = export_to_edge(
            m,
            input,
            dynamic_shapes=dynamic_shapes,
            edge_compile_config=edge_config,
        )
    edge_manager = edge_manager.to_backend(XnnpackDynamicallyQuantizedPartitioner())
    # Delegation visualization APIs: https://pytorch.org/executorch/main/debug-backend-delegate.html
    # from executorch.exir.backend.utils import get_delegation_info, format_delegated_graph
    # from tabulate import tabulate
    # graph_module = edge_manager.exported_program().graph_module
    # delegation_info = get_delegation_info(graph_module)
    # print(delegation_info.get_summary())
    # print(format_delegated_graph(graph_module))
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

    return output_path
