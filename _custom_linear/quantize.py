# https://www.internalfb.com/code/fbsource/[f1458254b3caba86fb497abbfe15c74c4e8ca38d]/fbcode/executorch/backends/xnnpack/test/ops/linear.py?lines=348

import torch
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig


# Note: not using from torchao.quantization.quant_primitives because it will run into op registraion issues
def get_group_qparams_symmetric(w, n_bit, groupsize, precision):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    max_val_abs = torch.max(-min_val_neg, max_val_pos)
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))

    # max_int - min_int is just 2**(n_bit) - 1

    scales = max_val_abs / (
        float(max_int - min_int) / 2
    )  # This is just 2 * max(abs(x)) / (int range)
    scales = torch.max(scales, torch.full_like(scales, torch.finfo(torch.float32).eps))
    # TODO: make sure abs(scales) is not too small?
    zeros = torch.full_like(scales, 0)
    return scales.to(precision).reshape(w.shape[0], -1), zeros.to(precision).reshape(
        w.shape[0], -1
    )


# Note: not using from torchao.quantization.quant_primitives because it will run into op registraion issues
# Does 4-bit quantization
def group_quantize_tensor_symmetric(w, group_size, precision):
    n_bit = 4
    scales, zeros = get_group_qparams_symmetric(w, n_bit, group_size, precision)
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))
    # TODO: currently we don't know how to express torch.int4, we'll
    # add torch.int4 to core later
    w_int8 = torch.ops.quantized_decomposed.quantize_per_channel_group(
        w, scales, zeros, min_int, max_int, torch.int8, group_size
    )

    return w_int8, scales, zeros


# https://www.internalfb.com/code/fbsource/[f1458254b3caba86fb497abbfe15c74c4e8ca38d]/fbcode/executorch/backends/xnnpack/operators/node_visitor.py?lines=451
def convert_to_qc4w(inp: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor to a quantized channelwise tensor 4bit tensor
    """

    import torch.nn.functional as F

    # Assert we got a properly quantized tensor.
    min, max = inp.min().item(), inp.max().item()
    assert (
        max <= 7 and min >= -8
    ), f"convert_to_qc4w: [min,max] out of [-8, 7] range, got [{min}, {max}]"

    # Assuming we have a 2d tensor
    if inp.ndim != 2:
        inp = inp.squeeze()
    assert (
        inp.ndim == 2
    ), f"convert_to_qc4w: expecting input tensor to be 2d, got {inp.ndim}"

    # pad ic
    if inp.shape[-1] % 2 != 0:
        inp = F.pad(input=inp, pad=(0, 1, 0, 0), mode="constant", value=0)

    # Shape after padding
    oc, ic = inp.shape
    assert ic % 2 == 0, "convert_to_qc4w: expecting ic to be even"

    # Adjust inp tensor for zp
    inp = inp.to(dtype=torch.uint8) + 8

    # Prepare the Result tensor
    inp = inp.contiguous().view(-1)
    return (inp[1::2] << 4 | inp[::2]).view(oc, int(ic / 2))
