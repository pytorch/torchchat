from ctypes import c_void_p

import torch
import torch.nn as nn
from torch import empty
from torch._dynamo.testing import rand_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.utils import print_performance
from torch._inductor.wrapper_benchmark import compiled_module_main

# with open("./dso_model.h", "rb") as f:
#     dso_src = f.read().decode("utf-8")

dso_src =""

async_compile = AsyncCompile()

src = """
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/torch.h>

#define MODELPATH "./stories15M.so"

torch::inductor::AOTIModelContainerRunnerCpu *transformer_dso =
new torch::inductor::AOTIModelContainerRunnerCpu(MODELPATH, 1);

extern "C" void kernel(long *tokens, long *pos, float *logits)
{
    torch::Tensor token_tensor = torch::from_blob(
             tokens, {1, 1}, torch::kLong);
    torch::Tensor pos_tensor = torch::from_blob(pos, { 1 }, torch::kLong);
    std::vector<torch::Tensor> inputs{token_tensor, pos_tensor};

    std::vector<at::Tensor> result = transformer_dso -> run(inputs);
    std::memcpy(logits, result[0].data_ptr<float>(), result[0].numel()*sizeof(float));
}

"""

transformer_model = async_compile.cpp_pybinding(
    ["long *", "long *", "float *"], dso_src + src
)

async_compile.wait(globals())
del async_compile


class DSOModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, x, input_pos):
        vocab_size = self.config.vocab_size # 32000
        assert x.dim() == 2 and x.size(0) == 1 and x.size(1) == 1
        logits = torch.empty(1, 1, vocab_size)
        x = x.to(torch.long)
        input_pos = input_pos.to(torch.long)
        transformer_model(x, input_pos, logits)
        return logits

    def setup_caches(self, max_batch_size, max_seq_length):
        pass
