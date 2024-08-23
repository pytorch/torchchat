# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

from typing import Any, Dict, Optional, Tuple, Union

import executorch.exir as exir

import torch
from build.model import apply_rotary_emb, Attention, Transformer
from build.utils import get_precision

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)
from executorch.exir import EdgeProgramManager, to_edge

from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.tracer import Value

from torch import nn
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram

default_device = "cpu"

_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=True,
)


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


class CustomKVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype):
        super().__init__()

        dtype = torch.float

        # This is flipped around from what is in build.model's KVCache
        cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )

    def update(self, input_pos, k_val, v_val):
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val.float()
        v_out[:, :, input_pos] = v_val.float()

        return k_out, v_out


class CustomSDPAAttention(nn.Module):
    def __init__(self, attention: Attention):
        super().__init__()

        self.wq = attention.wq
        self.wk = attention.wk
        self.wv = attention.wv

        self.wo = attention.wo

        max_batch_size, n_heads, max_seq_length, head_dim = (
            attention.kv_cache.k_cache.shape
        )
        cache_dtype = attention.kv_cache.k_cache.dtype
        self.kv_cache = CustomKVCache(
            max_batch_size, max_seq_length, n_heads, head_dim, cache_dtype
        )

        self.n_heads = attention.n_heads
        self.head_dim = attention.head_dim
        self.n_local_heads = attention.n_local_heads
        self.dim = attention.dim

    def forward(self, x, freqs_cis, mask, input_pos=None):
        bsz, seqlen, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis).to(dtype=torch.float)
        k = apply_rotary_emb(k, freqs_cis).to(dtype=torch.float)
        v = v.to(dtype=torch.float)

        # KV cache should always be enabled
        assert self.kv_cache is not None
        output = torch.ops.llama.sdpa_with_kv_cache(
            q,
            k,
            v,
            self.kv_cache.k_cache,
            self.kv_cache.v_cache,
            input_pos[-1].item(),
            seqlen,
        )
        output = output.view(bsz, seqlen, self.dim).to(dtype=q.dtype)
        return self.wo(output)


def replace_attention_with_custom_sdpa_attention(module: nn.Module):
    from executorch.examples.models.llama2.custom_ops import sdpa_with_kv_cache  # noqa

    for name, child in module.named_children():
        if isinstance(child, Attention):
            setattr(module, name, CustomSDPAAttention(child))
        else:
            replace_attention_with_custom_sdpa_attention(child)


def _to_core_aten(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    verbose=True,
) -> ExportedProgram:
    # post autograd export. eventually this will become .to_core_aten
    if not isinstance(model, torch.fx.GraphModule) and not isinstance(
        model, torch.nn.Module
    ):
        raise ValueError(
            f"Expected passed in model to be an instance of fx.GraphModule, got {type(model)}"
        )
    core_aten_ep = export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    if verbose:
        logging.info(f"Core ATen graph:\n{core_aten_ep.graph}")
    return core_aten_ep


def _core_aten_to_edge(
    core_aten_exir_ep: ExportedProgram,
    edge_constant_methods: Optional[Dict[str, Any]] = None,
    edge_compile_config=None,
    verbose=True,
) -> EdgeProgramManager:
    if not edge_compile_config:
        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False,  # quant ops currently break ir verification
        )
    edge_manager: EdgeProgramManager = to_edge(
        core_aten_exir_ep,
        constant_methods=edge_constant_methods,
        compile_config=edge_compile_config,
    )
    if verbose:
        logging.info(f"Exported graph:\n{edge_manager.exported_program().graph}")
    return edge_manager


def export_to_edge(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    edge_constant_methods: Optional[Dict[str, Any]] = None,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
    verbose=True,
) -> EdgeProgramManager:
    core_aten_ep = _to_core_aten(model, example_inputs, dynamic_shapes, verbose=verbose)
    return _core_aten_to_edge(
        core_aten_ep, edge_constant_methods, edge_compile_config, verbose=verbose
    )


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
