from executorch.examples.models.llama2.custom_ops import sdpa_with_kv_cache
from build.model import Attention, apply_rotary_emb
from torch import nn
import torch

class CustomSDPAAttention(nn.Module):
    def __init__(self, attention: Attention):
        super().__init__()


        self.wq = attention.wq
        self.wk = attention.wk
        self.wv = attention.wv

        self.wo = attention.wo
        self.kv_cache = attention.kv_cache

        self.n_heads = attention.n_heads
        self.head_dim = attention.head_dim
        self.n_local_heads = attention.n_local_heads
        self.dim = attention.dim

    def forward(self, x, freqs_cis, mask, input_pos = None):
        bsz, seqlen, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

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
        output = output.view(bsz, seqlen, self.dim)
        return self.wo(output)


def replace_attention_with_custom_sdpa_attention(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, Attention):
            setattr(module, name, CustomSDPAAttention(child))
        else:
            replace_attention_with_sdpa_attention(child)
