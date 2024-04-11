from ctypes import c_void_p

import torch
import torch.nn as nn
from torch import empty
from executorch.extension.pybindings import portable_lib as exec_lib

class PTEModel(nn.Module):
    def __init__(self, config, path) -> None:
        super().__init__()
        self.config = config
        self.model_ = exec_lib._load_for_executorch(str(path))

    def forward(self, x, input_pos):
        # model_.forward expects inputs to be wrapped in a tuple
        forward_inputs = (x.to(torch.long), input_pos.to(torch.long))
        logits = self.model_.forward(forward_inputs)

        # After wrapping in a tuple, we get a list back, so we need to grab
        # the first element to get the tensor
        assert len(logits) == 1
        logits = logits[0]
        return logits

    def setup_caches(self, max_batch_size, max_seq_length, dtype=torch.float):
        pass
