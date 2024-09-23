import torch
from typing import Any, Dict, Optional, Tuple
from torchchat.utils.build_utils import find_multiple, get_precision

# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L77
def hf_precompute_freqs_cis(dim: int, end: int, theta: float):
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device="cpu", dtype=torch.int64).float() / dim)
    )
    # pyre-ignore Undefined attribute [16]: `float` has no attribute `device`.
    t = torch.arange(end, device=freqs.device, dtype=torch.int64).type_as(
        freqs  # pyre-ignore
    )
    freqs = torch.outer(t, freqs).float()  # pyre-ignore
    emb = torch.cat((freqs, freqs), dim=-1)
    freqs_cos = torch.cos(emb)
    freqs_sin = torch.sin(emb)
    return freqs_cos, freqs_sin


def precompute_freqs_cis(
    n_elem: int,
    seq_len: int,
    base: int = 10000,
    dtype=None,
    rope_scaling: Optional[Dict[str, Any]] = None,
):
    if not dtype:
        dtype = get_precision()
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    if rope_scaling is not None:
        freqs = apply_scaling(freqs, rope_scaling)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)

# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L135
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def hf_apply_rotary_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_emb(x, freqs_cis):
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


# 比较函数
def compare_methods():
    torch.manual_seed(0)
    x = torch.randn(1, 636, 32, 128)

    # 设置参数
    n_elem = 128
    seq_len =  1536
    base =  10000
    dtype = None
    rope_scaling = None

    all_freq_cis = precompute_freqs_cis(n_elem, seq_len, base, dtype, rope_scaling)
    input_pos = torch.arange(
                x.shape[1],
                device=x.device,
                dtype=torch.int,
            )
    freq_cis = all_freq_cis[input_pos]
    x_out1 = apply_rotary_emb(x, freq_cis)


    dim =  128
    end =  1536
    theta =  10000.0
    freqs_cos, freqs_sin = hf_precompute_freqs_cis(dim, end, theta)
    fc, fs = freqs_cos[:x.shape[1]], freqs_sin[:x.shape[1]]
    x_out2, _ = hf_apply_rotary_emb(x, x, fc, fs)

    print(x_out1)
    print("************************")
    print(x_out2)


if __name__ == "__main__":
    compare_methods()
