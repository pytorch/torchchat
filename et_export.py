# Export using Executorch, the PyTorch 2 mobile solution coming soon
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.export import Dim, export

from generate import _load_model, decode_one_token

from model import Transformer

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


class model_wrapper(nn.Module):
    def __init__(self, model, device):
        super().__init__()

        max_seq_length = 350
        with torch.device(device):
            model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        self.model = model
        # init model here if necessary

    def forward(self, x, input_pos):
        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1
        logits = self.model(x, input_pos)
        return logits  # sample(logits, **sampling_kwargs)


def canonical_path(path):
    return path


def preprocess(args) -> Tuple:
    """
    returns a list of quantizers, and the dtype to use
    """

    # load model from checkpoint and params.json
    checkpoint_path = canonical_path(args.checkpoint)
    params_path = canonical_path(args.params)

    if args.quantization_mode in ["8da4w", "8da4w-gptq"]:
        dtype_override = torch.float16
    else:
        dtype_override = torch.float32
        
    # source transforms
    transforms = []
    if args.quantization_mode:
        transforms.append(
            partial(
                quantize,
                qmode=args.quantization_mode,
                activation_dtype=dtype_override,
                checkpoint_path=(
                    Path(path) if (path := args.checkpoint) is not None else None
                ),
                tokenizer_path=(
                    Path(path) if (path := args.tokenizer_path) is not None else None
                ),
                group_size=args.group_size,
                calibration_tasks=args.calibration_tasks,
                calibration_limit=args.calibration_limit,
                calibration_seq_length=args.calibration_seq_length,
            )
        )

    if args.embedding_quantize:
        bitwidth, group_size = args.embedding_quantize.split(",")
        if group_size == "none" or group_size == "None" or group_size == "0":
            group_size = None
        else:
            group_size = int(group_size)
        bitwidth = int(bitwidth)
        transforms.append(
            lambda model: EmbeddingOnlyInt8QuantHandler(
                model, bitwidth=bitwidth, group_size=group_size
            ).quantized_model()
        )

    if args.expand_rope_table:
        transforms.append(materialze_broadcast_of_rope_freq_cis)

    return transforms, dtype_override


def export_model(model: nn.Module, device, output_path):

    export_model = model_wrapper(model, device=device)
    print(export_model)

    input = (
        torch.tensor([[1]], dtype=torch.long, device=device),
        torch.tensor([0], dtype=torch.long, device=device),
    )

    print(f"len(input)={len(input)}")

    batch = Dim("batch")
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"idx": {0: batch}, "input_pos": {0: batch}}

    quantizers, dtype_override = preprocess_quantizers(args)
    dtype_override = preprocess_dtype(args)

    # so = torch._export.aot_compile(
    #     export_model,
    #     args=input,
    #     options={"aot_inductor.output_path": output_path},
    # )
    # print(f"The generated DSO model can be found at: {so}")
    # return so

    # process model here 
    _prepare_for_llama_export(
        args
    ).export_to_edge()
    checkpoint=checkpoint_path,

    # we always use kv cache.  Let's discuss sdpa_w_kv
    # maybe always on???? ... discuss pros and cons @kimish
    use_kv_cache=args.use_kv_cache,
    use_sdpa_with_kv_cache=args.use_sdpa_with_kv_cache,

    # this doesn't use params - just hard code the models we support
    see what gpt-fast does (and yes, params is better, but dict
                            is simpler. This shows ET is SIMPLE!)

    # I think we set that to a semi fixed value, check aoti export
    max_seq_len=args.max_seq_length,

    # maybe inline calling transforms here... 
    .source_transform(transforms)
    # FIXME: we should get rid of the separate "quantize" step for AOTI
    # as well and do it the same way

    # this 
    .to_dtype(dtype_override)
    )

    # to_backend

    #### We don't do this one?
    partitioner = None
    if pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None:
        partitioner = XnnpackDynamicallyQuantizedPartitioner()
        modelname = f"xnnpack_dq_{modelname}"

    if args.xnnpack:
        # Following changes due to.
        # 1. We need dynamically quantized partitioner for both pt2e_quantize options
        #    as well as "qmode 8da4w" which is also dynamic quantizes linear layers.
        # 2. XNNPACK partitioner seems to result in seg fault for non dqlinear ops.
        partitioner = XnnpackDynamicallyQuantizedPartitioner()
        # partitioner = XnnpackPartitioner()

    if args.vulkan:
        assert (
            args.dtype_override == "fp32" or args.dtype_override is None
        ), "Vulkan backend does not support non fp32 dtypes at the moment"
        assert (
            args.quantization_mode is None
        ), "Vulkan backend does not support quantization at the moment"

        partitioner = VulkanPartitioner()

    if args.mps:
        assert (
            args.use_kv_cache is True
        ), "MPS backend currently only supports static shape and use_kv_cache=True is the only way to support it at the moment"
        try:
            # pyre-ignore Undefined import [21]: Could not find a module corresponding to import `executorch.backends.apple.mps.partition.mps_partitioner`.
            from executorch.backends.apple.mps.partition.mps_partitioner import (
                MPSPartitioner,
            )
        except ImportError:
            raise ImportError(
                "Please install the MPS backend follwing https://pytorch.org/executorch/main/build-run-mps.html"
            )

        compile_specs = [CompileSpec("use_fp16", bytes([True]))]
        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `apple`.
        partitioner = MPSPartitioner(compile_specs)

    if args.coreml:
        assert (
            args.use_kv_cache is True
        ), "CoreML backend currently only supports static shape and use_kv_cache=True is the only way to support it at the moment"
        try:
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.apple.coreml.partition.coreml_partitioner`.
            import coremltools as ct

            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.apple.coreml.compiler`
            from executorch.backends.apple.coreml.compiler import CoreMLBackend

            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.apple.coreml.partition.coreml_partitioner`
            from executorch.backends.apple.coreml.partition.coreml_partitioner import (
                CoreMLPartitioner,
            )
        except ImportError:
            raise ImportError(
                "Please install the CoreML backend follwing https://pytorch.org/executorch/main/build-run-coreml.html"
            )

        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `apple`.
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=ct.precision(ct.precision.FLOAT16.value),
            compute_unit=ct.ComputeUnit[ct.ComputeUnit.ALL.name.upper()],
            # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `apple`
            model_type=CoreMLBackend.MODEL_TYPE.MODEL,
        )
        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `apple`
        partitioner = CoreMLPartitioner(
            skip_ops_for_coreml_delegation=None, compile_specs=compile_specs
        )

    #### QNN requires pt2e quantization.
    #### Don't use.  This subverts the simplicity message.
    #### Future update when we align quantization
    
    if args.qnn:
        assert (
            args.use_kv_cache is True
        ), "Qualcomm backend currently only supports static shape and use_kv_cache=True is the only way to support it at the moment"
        try:
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.partition.qnn_partitioner`
            from executorch.backends.qualcomm.partition.qnn_partitioner import (
                QnnPartitioner,
            )

            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.serialization.qnn_compile_spec_schema`
            from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
                QcomChipset,
            )

            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.utils.utils`
            from executorch.backends.qualcomm.utils.utils import (
                _transform,
                generate_htp_compiler_spec,
                generate_qnn_executorch_compiler_spec,
            )
        except ImportError:
            raise ImportError(
                "Please install the Qualcomm backend follwing https://pytorch.org/executorch/main/build-run-qualcomm.html"
            )

        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`
        partitioner = QnnPartitioner(
            # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`
            generate_qnn_executorch_compiler_spec(
                # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`.
                soc_model=QcomChipset.SM8650,  # default to SM8650
                backend_options=backend_options,
                debug=False,
                saver=False,
            ),
            skip_node_id_set={},
            skip_node_op_set={},
        )
        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`
        _transform(builder_exported_to_edge.export_program())

    #### no builder?
    output_path = args.output_path
    builder.save_to_pte(output_path)

    return output_path

    #################################################################
    ### FIXME.... used for separate export & compile
    #################################################################
    
    exported_program: torch.export.ExportedProgram = export(
        export_model, args=input
    )  # , dynamic_shapes=dynamic_shapes)

    torch.export.save(exported_program, "exported_gpt-fast.pt2")

    print(exported_program)

    return

    so = torch._export.aot_compile(exported_program, args=input)
    print(f"{so=}")
    assert so is not None


def main(checkpoint_path, device, output_path):
    assert checkpoint_path.is_file(), checkpoint_path

    print(f"Using device={device}")
    precision = torch.float  # bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, False)

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    with torch.no_grad():
        export_model(model, device, output_path)


def cli():
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is", help="Input prompt."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to launch in interactive mode",
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--compile_prefill",
        action="store_true",
        help="Whether to compile the prefill (improves prefill perf, but higher compile times)",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--speculate_k", type=int, default=5, help="Speculative execution depth."
    )
    parser.add_argument(
        "--draft_checkpoint_path",
        type=Path,
        default=None,
        help="Draft checkpoint path.",
    )
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )
    parser.add_argument(
        "--out-path", type=str, default="model.so", help="Filename"
    )

    args = parser.parse_args()
    main(args.checkpoint_path, args.device, args.out_path)

if __name__ == "__main__":
    cli()
