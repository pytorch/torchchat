# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Optional

import torch
import torch._inductor
import torch.nn as nn

from torch.export import Dim

from torchchat.cli.builder import (
    _initialize_model,
    _initialize_tokenizer,
    _set_gguf_kwargs,
    _unset_gguf_kwargs,
    BuilderArgs,
    TokenizerArgs,
)

from torchchat.utils.build_utils import set_backend, set_precision


default_device = "cpu"


"""
Export Snapshot
"""


def export_snapshot(
    model: nn.Module,
    device: Optional[str] = None,
    output_path: str = "model-snapshot.tc",
) -> str:
    """
    Export the model as snapshot.

    Args:
        model: The model to be exported.
        device: The device to run the model on.
        output_path: The path to save the exported model.
    Returns:
        The path to the exported model.
    """
    assert output_path.endswith(".tc"), "use .tc extension for snapshots"
    torch.save(model, output_path)
    return output_path


"""
Export for Server
"""


def export_for_server(
    model: nn.Module,
    device: Optional[str] = "cpu",
    output_path: str = "model.pt2",
    dynamic_shapes: bool = False,
    package: bool = True,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Export the model using AOT Compile to get a .dso for server use cases.

    Args:
        model: The model to be exported.
        device: The device to run the model on.
        output_path: The path to save the exported model.
    Returns:
        The path to the exported model.
    """
    if dynamic_shapes:
        example_inputs = (
            torch.tensor([[1, 9038, 2501, 263, 931]], dtype=torch.int, device=device),
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int, device=device),
        )

        seq = Dim("seq", min=1, max=model.text_transformer_args.max_seq_length)
        # Specify that the first dimension of each input is that batch size
        dynamic_shapes = {"tokens": {1: seq}, "input_pos": {0: seq}}
    else:
        example_inputs = (
            torch.tensor([[1]], dtype=torch.int, device=device),
            torch.tensor([0], dtype=torch.int, device=device),
        )
        dynamic_shapes = None

    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        options = {
            "aot_inductor.package": package,
            "aot_inductor.metadata": metadata or {},
        }

        if not package:
            options = {"aot_inductor.output_path": output_path}

        ep = torch.export.export(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

        if package:
            path = torch._inductor.aoti_compile_and_package(
                ep, package_path=output_path, inductor_configs=options
            )
        else:
            path = torch._inductor.aot_compile(
                ep.module(), example_inputs, options=options
            )

    print(f"The generated packaged model can be found at: {path}")
    return path


"""
Export for ExecuTorch

TODO (https://github.com/pytorch/torchchat/issues/1058): Replace
replace_attention_with_custom_sdpa_attention with ET's implementation
"""

try:
    executorch_export_available = True

    import logging

    from typing import Any, Dict, Tuple, Union

    import executorch.exir as exir
    from executorch.backends.xnnpack._passes.convert_to_linear import (
        ConvertToLinearPass,
    )

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner,
    )
    from executorch.exir import EdgeProgramManager, to_edge

    from executorch.exir.capture._config import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
    )
    from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
    from executorch.exir.passes.sym_shape_eval_pass import (
        ConstraintBasedSymShapeEvalPass,
    )
    from executorch.exir.tracer import Value

    from torch.export import export, export_for_training, ExportedProgram

    from torchchat.model import apply_rotary_emb, Attention
    from torchchat.utils.build_utils import get_precision

    default_device = "cpu"

    _EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
        _check_ir_validity=True,
    )

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

            max_batch_size, n_heads, max_seq_length, head_dim = attention.kv_cache[
                0
            ].k_cache.shape
            cache_dtype = attention.kv_cache[0].k_cache.dtype
            # The `Attention` module being replaced can have multiple KV caches
            # (denoted by `cache_lanes`).  Thus we follow the same setup format
            # as in `Attention.setup_cache`.
            cache_lanes = len(attention.kv_cache)
            self.kv_cache = nn.ModuleList(
                [
                    CustomKVCache(
                        max_batch_size, max_seq_length, n_heads, head_dim, cache_dtype
                    )
                    for _ in range(cache_lanes)
                ]
            )

            self.n_heads = attention.n_heads
            self.head_dim = attention.head_dim
            self.n_local_heads = attention.n_local_heads
            self.dim = attention.dim

        def forward(self, x, freqs_cis, mask, input_pos=None, cache_lane: int = 0):
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
            kv_cache = self.kv_cache[cache_lane]
            output = torch.ops.llama.sdpa_with_kv_cache(
                q,
                k,
                v,
                kv_cache.k_cache,
                kv_cache.v_cache,
                input_pos[-1].item(),
                seqlen,
            )
            output = output.view(bsz, seqlen, self.dim).to(dtype=x.dtype)
            return self.wo(output)

    def replace_attention_with_custom_sdpa_attention(module: nn.Module):
        from executorch.extension.llm.custom_ops import custom_ops  # noqa

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
        core_aten_ep = export_for_training(
            model, example_inputs, dynamic_shapes=dynamic_shapes
        )
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
        core_aten_ep = _to_core_aten(
            model, example_inputs, dynamic_shapes, verbose=verbose
        )
        return _core_aten_to_edge(
            core_aten_ep, edge_constant_methods, edge_compile_config, verbose=verbose
        )

    def export_for_et(model, device, output_path) -> str:

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

        if target_precision not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError(f"Unsupported dtype for ET export: {target_precision}")

        if state_dict_dtype != target_precision:
            print(f"model.to {target_precision}")
            model = model.to(dtype=target_precision)
            state_dict_dtype = target_precision

        replace_attention_with_custom_sdpa_attention(model)

        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.MATH]
        ), torch.no_grad():
            m = export_for_training(model, input, dynamic_shapes=dynamic_shapes).module()

            edge_manager = export_to_edge(
                m,
                input,
                dynamic_shapes=dynamic_shapes,
                edge_compile_config=edge_config,
            )
        edge_manager = edge_manager.to_backend(XnnpackDynamicallyQuantizedPartitioner())
        export_program = edge_manager.to_executorch(
            ExecutorchBackendConfig(
                extract_delegate_segments=True,
                passes=[
                    ConvertToLinearPass(),
                    QuantFusionPass(),
                ],
                sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
            )
        )

        print("The methods are: ", export_program.methods)
        with open(output_path, "wb") as f:
            export_program.write_to_file(f)

        return output_path

except Exception as e:
    executorch_exception = f"ET EXPORT EXCEPTION: {e}"
    executorch_export_available = False


"""
Exporting Flow
"""


def main(args):
    builder_args = BuilderArgs.from_args(args)
    quantize = args.quantize

    print(f"Using device={builder_args.device}")
    set_precision(builder_args.precision)
    set_backend(
        dso=args.output_dso_path,
        pte=args.output_pte_path,
        aoti_package=args.output_aoti_package_path,
    )

    builder_args.dso_path = None
    builder_args.pte_path = None
    builder_args.aoti_package_path = None
    builder_args.setup_caches = True

    output_pte_path = args.output_pte_path
    output_dso_path = args.output_dso_path
    output_snapshot_path = args.output_snapshot_path
    output_aoti_package_path = args.output_aoti_package_path

    if output_pte_path and builder_args.device != "cpu":
        print(
            f"Warning! ExecuTorch export target is controlled by export recipe, not device setting. Ignoring device={builder_args.device} setting."
        )
        builder_args.device = "cpu"
    elif (output_pte_path or output_dso_path or output_aoti_package_path) and "mps" in builder_args.device:
        print("Warning! Device MPS not supported for export. Exporting for device CPU.")
        builder_args.device = "cpu"

    # TODO: clean this up
    # This mess is because ET does not support _weight_int4pack_mm right now
    tokenizer_args = None
    if not builder_args.gguf_path:
        # tokenizer needed for quantization so get that here,
        try:
            tokenizer_args = TokenizerArgs.from_args(args)
            tokenizer = _initialize_tokenizer(tokenizer_args)
        except:
            tokenizer = None

        if builder_args.max_seq_length is None:
            if (
                output_dso_path is not None or output_aoti_package_path is not None
            ) and not builder_args.dynamic_shapes:
                print("Setting max_seq_length to 300 for DSO export.")
                builder_args.max_seq_length = 300
            elif output_pte_path is not None:
                # The value of 128 was chosen to match the ExecuTorch Llama example setup.
                print("Setting max_seq_length to 128 for ExecuTorch export.")
                builder_args.max_seq_length = 128

        model = _initialize_model(
            builder_args,
            quantize,
            tokenizer,
            max_seq_length=builder_args.max_seq_length,
            support_tensor_subclass=output_dso_path is None
            and output_aoti_package_path is None,
        )
        model_to_pte = model
        model_to_dso = model
        model_to_aoti_package = model
        model_to_snapshot = model
    else:
        if output_pte_path:
            _set_gguf_kwargs(builder_args, is_et=True, context="export")
            model_to_pte = _initialize_model(
                builder_args,
                quantize,
            )
            _unset_gguf_kwargs(builder_args)

        if output_dso_path or output_aoti_package_path:
            _set_gguf_kwargs(builder_args, is_et=False, context="export")
            model_to_aoti_package = _initialize_model(
                builder_args,
                quantize,
                support_tensor_subclass=False,
            )
            model_to_dso = model_to_aoti_package
            _unset_gguf_kwargs(builder_args)

        if output_snapshot_path:
            _set_gguf_kwargs(builder_args, is_et=False, context="export")
            model_to_snapshot = _initialize_model(
                builder_args,
                quantize,
                support_tensor_subclass=False,
            )
            _unset_gguf_kwargs(builder_args)
 
    with torch.no_grad():
        if output_pte_path:
            output_pte_path = str(os.path.abspath(output_pte_path))
            if executorch_export_available:
                print(f"Exporting model using ExecuTorch to {output_pte_path}")
                export_for_et(model_to_pte, builder_args.device, args.output_pte_path)
            else:
                print(
                    "Export with executorch requested but ExecuTorch could not be loaded"
                )
                print(executorch_exception)
        if output_dso_path:
            output_dso_path = str(os.path.abspath(output_dso_path))
            print(f"Exporting model using AOT Inductor to {output_dso_path}")
            print(
                "WARNING!! The path of compiling a dso is deprecated. Please use --output-aoti-package-path to create a .pt2 artifact instead."
            )
            export_for_server(
                model_to_dso,
                builder_args.device,
                output_dso_path,
                builder_args.dynamic_shapes,
                package=False,
            )

        if output_aoti_package_path:
            output_aoti_package_path = str(os.path.abspath(output_aoti_package_path))

            if tokenizer_args is None:
                tokenizer_type = "0"
            elif tokenizer_args.is_sentencepiece:
                tokenizer_type = "2"  # Corresponding to llama2
            else:
                tokenizer_type = "3"  # Corresponding to llama3

            metadata = {"tokenizer_type": tokenizer_type}
            print(
                "Exporting model using AOT Inductor to " f"{output_aoti_package_path}."
            )
            export_for_server(
                model_to_aoti_package,
                builder_args.device,
                output_aoti_package_path,
                builder_args.dynamic_shapes,
                package=True,
                metadata=metadata,
            )

        if output_snapshot_path:
            output_snapshot_path = str(os.path.abspath(output_snapshot_path))
            print(f"Exporting model using Snapshot to {output_snapshot_path}")
            export_snapshot(
                model_to_snapshot,
                builder_args.device,
                output_snapshot_path,
            )

