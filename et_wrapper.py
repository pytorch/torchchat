from ctypes import c_void_p

import torch
import torch.nn as nn
from torch import empty
from executorch.extension.pybindings import portable_lib as exec_lib

class PTEModel(nn.Module):
    def __init__(self, config, path) -> None:
        super().__init__()
        self.config = config
        self.model_ = exec_lib._load_for_executorch(path)

    defccorward(self, x, input_pos):
        logits = module.forward(
            x.to(torch.long),
            input_pos.to(torch.long),
        )
        return logits

    def setup_caches(self, max_batch_size, max_seq_length):
        pass
