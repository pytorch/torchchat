# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch
from torch import float16, bfloat16, float32
from torch import Tensor
from torch import nn
import numpy as np
from .quantize import HQQLinear, Quantizer
from .utils import cleanup


# Return trainable weight matrix
def _get_dense_param(
    in_features: int,
    out_features: int,
    device="cuda",
    trainable: bool = True,
    dtype: torch.dtype = float32,
) -> Tensor:
    W = (
        nn.Linear(in_features, out_features, bias=None)
        .weight.data.t()
        .to(dtype)
        .to(device)
        .contiguous()
    )
    return nn.Parameter(W, requires_grad=trainable)


# Applies LoRA to a linear module (include HQQLinear)
class HQQLinearLoRA(nn.Module):
    def __init__(self, linear_layer: nn.Module, peft_config: dict):
        super().__init__()

        # Device
        self.device = (
            linear_layer.device
            if hasattr(linear_layer, "device")
            else next(linear_layer.parameters()).device
        )
        self.train_dtype = (
            peft_config["train_dtype"]
            if ("train_dtype" in peft_config)
            else torch.float
        )

        # Linear layer
        self.linear_layer = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # Bias
        self.bias = None if (linear_layer.bias is None) else linear_layer.bias.clone()
        self.linear_layer.bias = None
        self.train_bias = (
            peft_config["train_bias"] if ("train_bias" in peft_config) else False
        )

        if self.bias is not None:
            self.bias = nn.Parameter(self.bias, requires_grad=self.train_bias)
            if self.train_bias:
                self.bias.data = self.bias.data.to(self.train_dtype)

        if (self.bias is None) and self.train_bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    (self.out_features,), device=self.device, dtype=self.train_dtype
                ),
                requires_grad=True,
            )

        # Dropout
        if "dropout" in peft_config:
            self.peft_drop = (
                nn.Dropout(p=peft_config["dropout"])
                if (peft_config["dropout"] > 0.0)
                else nn.Identity()
            )
        else:
            self.peft_drop = nn.Identity()

        # LoRA A/B
        self.peft_config = peft_config
        self.lora_alpha = peft_config["lora_alpha"]
        self.r = peft_config["r"]
        self.scaling = self.lora_alpha / self.r

        self.lora_A = _get_dense_param(
            self.in_features,
            self.r,
            device=self.device,
            trainable=True,
            dtype=self.train_dtype,
        )

        self.lora_B = _get_dense_param(
            self.r,
            self.out_features,
            device=self.device,
            trainable=True,
            dtype=self.train_dtype,
        )

        # LoRA weights init
        if "lora_init" in peft_config:
            # Set lora init
            assert (
                peft_config["lora_init"]["lora_A"].shape[0],
                peft_config["lora_init"]["lora_B"].shape[1],
            ) == (self.in_features, self.out_features), (
                "Invalid init LoRA weight shapes. Expected: lora_A: "
                + str(self.in_features)
                + " x r , lora_B: r x "
                + str(self.out_features)
                + ")"
            )
            self.lora_A.data = peft_config["lora_init"]["lora_A"].to(
                device=self.device, dtype=self.train_dtype
            )
            self.lora_B.data = peft_config["lora_init"]["lora_B"].to(
                device=self.device, dtype=self.train_dtype
            )
        else:
            # Init weights, as as the original LoRA implementation
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward_lora(self, x: Tensor) -> Tensor:  # output is self.train_dtype
        return (
            torch.matmul(
                torch.matmul(self.peft_drop(x.to(self.lora_A.dtype)), self.lora_A),
                self.lora_B,
            )
            * self.scaling
        )

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype

        # Forward with the base linear layer
        out = self.linear_layer(x)

        # LoRA
        if self.train_bias:
            out += (self.forward_lora(x) + self.bias).to(x_dtype)

        else:
            out += self.forward_lora(x).to(x_dtype)
            if self.bias is not None:
                out += self.bias

        return out

    def merge_and_quantize(self, quant_config: dict) -> nn.Module:
        # not easy to figure out the type for "any" potential linear layer (nn.Linear, HQQLinear, etc.)
        for _dtype in [float16, bfloat16, float32]:
            try:
                # Get initial weights
                W = self.linear_layer(
                    torch.eye(self.in_features, device=self.device, dtype=_dtype)
                ).t()  # == self.linear_layer.dequantize()
            except Exception:
                pass

        # Merge weights
        W += (
            (torch.matmul(self.lora_A.data, self.lora_B.data) * self.scaling)
            .t()
            .to(W.dtype)
        )

        # New HQQ layer
        new_hqq_layer = HQQLinear(None, quant_config)
        new_hqq_layer.bias = None if (self.bias is None) else self.bias.clone()
        new_hqq_layer.quantize(W, **quant_config)

        return new_hqq_layer

    def cast(self, dtype: torch.dtype = float16):
        self.lora_A.data = self.lora_A.data.to(dtype)
        self.lora_B.data = self.lora_B.data.to(dtype)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype)
        if isinstance(self.scaling, nn.Parameter):
            self.scaling.data = self.scaling.data.to(dtype)
        if isinstance(self.scaling, Tensor):
            self.scaling = self.scaling.to(dtype)
        return self

    def state_dict(self):
        return {
            "lora_A": self.lora_A.data,
            "lora_B": self.lora_B.data,
            "scaling": self.scaling,
            "bias": self.bias,
        }

    def load_state_dict(self, state_dict):
        self.lora_A.data = state_dict["lora_A"].data.to(self.device)
        self.lora_B.data = state_dict["lora_B"].data.to(self.device)

        if state_dict["bias"] is not None:
            self.bias.data = state_dict["bias"].data.to(self.device)

        # Handle different use-cases of scaling
        if isinstance(state_dict["scaling"], (int, float)):
            self.scaling = state_dict["scaling"]
        if isinstance(state_dict["scaling"], nn.Parameter):
            self.scaling.data = state_dict["scaling"].data.to(self.device)
        if isinstance(state_dict["scaling"], Tensor):
            self.scaling = state_dict["scaling"].to(self.device)


# LoRA with fake quantization
class HQQLinearLoRAWithFakeQuant(HQQLinearLoRA):
    def __init__(self, linear_layer: nn.Module, peft_config: dict):
        super(HQQLinearLoRAWithFakeQuant, self).__init__(linear_layer, peft_config)
        self.quant_param = peft_config["quant_param"]

    # @torch.no_grad()
    # @torch.compile()
    def fake_quant(self, weight):
        if self.quant_param:
            W_q, meta = Quantizer.quantize(
                weight, **self.quant_param, bitpack=False
            )  # todo: clone() tensor
            weight_est = Quantizer.dequantize(W_q, meta)
        else:
            weight_est = weight
        return weight_est

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype

        # Get initial weights
        W = self.linear_layer(
            torch.eye(self.in_features, device=self.device, dtype=x_dtype)
        ).t()  # == self.linear_layer.dequantize()

        # Merge weights
        W += (torch.matmul(self.lora_A, self.lora_B) * self.scaling).t().to(W.dtype)

        # Fake quant
        W = self.fake_quant(W).to(x_dtype)

        # Matmul
        out = torch.matmul(x, W.t())

        # Bias
        if self.bias is not None:
            out += self.bias

        out = out.to(x_dtype)

        return out


# Experimental
class HQQLinearGroupedProj(nn.Module):
    def __init__(self, linear_layer: nn.Module, peft_config: dict):
        super().__init__()

        # Device
        self.device = (
            linear_layer.device
            if hasattr(linear_layer, "device")
            else next(linear_layer.parameters()).device
        )
        self.train_dtype = (
            peft_config["train_dtype"]
            if ("train_dtype" in peft_config)
            else torch.float
        )

        # Linear layer
        self.linear_layer = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.bias = None if (linear_layer.bias is None) else linear_layer.bias.clone()

        # Turn-off bias in the linear layer
        self.linear_layer.bias = None

        # Group proj
        self.peft_config = peft_config
        self.proj_size = peft_config["proj_size"]
        self.proj_num = peft_config["proj_num"]
        self.proj = nn.Parameter(
            torch.stack(
                [torch.eye(self.proj_size, dtype=self.train_dtype, device=self.device)]
                * self.proj_num
            )
        )
        if peft_config["zero_trainable"]:
            self.linear_layer.meta["zero"] = nn.Parameter(
                self.linear_layer.meta["zero"].to(self.train_dtype), requires_grad=True
            )

    @torch.compile()
    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype

        # Forward with base linear
        with torch.no_grad():
            W = self.linear_layer.dequantize().clone()
        # W = self.linear_layer(torch.eye(self.in_features, device=self.device, dtype=x_dtype)).t()
        shape = W.shape

        # Grouped proj
        proj_b, gs = self.proj.shape[0], self.proj.shape[1]
        W = (
            torch.matmul(self.proj, W.reshape((proj_b, gs, -1)).to(self.proj.dtype))
            .to(x_dtype)
            .reshape(shape)
        )

        # Matmul
        out = torch.matmul(x, W.t())

        # Bias
        if self.bias is not None:
            out += self.bias

        out = out.to(x_dtype)

        return out

    def cast(self, dtype: torch.dtype = float16):
        self.proj.data = self.proj.data.to(dtype)
        self.linear_layer.meta["zero"] = self.linear_layer.meta["zero"].to(dtype)
        if self.bias is not None:
            if self.bias.requires_grad:
                self.bias.data = self.bias.data.to(dtype)
            else:
                self.bias = self.bias.to(dtype)
        return self

    def state_dict(self):
        return {
            "proj": self.proj.data,
            "bias": self.bias,
            "peft_config": self.peft_config,
        }

    def load_state_dict(self, state_dict):
        self.proj.data = state_dict["proj"].data.to(self.device)
        self.bias = state_dict["bias"] if ("bias" in state_dict) else None
        self.bias = self.bias.to(self.device) if (self.bias is not None) else None
        self.peft_config = state_dict["peft_config"]


_HQQ_LORA_CLASSES = [HQQLinearLoRA, HQQLinearLoRAWithFakeQuant, HQQLinearGroupedProj]
_HQQ_LORA_MAPPING = {
    "default": HQQLinearLoRA,
    "lora_with_fakequant": HQQLinearLoRAWithFakeQuant,
    "grouped_proj": HQQLinearGroupedProj,
}


def is_hqq_lora_layer(layer):
    return type(layer) in _HQQ_LORA_CLASSES


##################################################################################################################
def autoname_modules(model):
    for name, module in model.named_modules():
        module.name = name


# Patching functions
def patch_linear_add_peft(layer: nn.Module, patch_params: dict | None) -> nn.Module:
    _peft_config = patch_params
    if _peft_config:
        lora_type = (
            _peft_config["lora_type"] if ("lora_type" in _peft_config) else "default"
        )
        new_layer = _HQQ_LORA_MAPPING[lora_type](layer, _peft_config)
    else:
        new_layer = layer
    return new_layer


def patch_linear_merge_peft(layer: nn.Module, patch_params: dict | None) -> nn.Module:
    _quant_config = patch_params
    if _quant_config:
        new_layer = layer.merge_and_quantize(_quant_config)
        del layer
        cleanup()
    else:
        new_layer = layer
    return new_layer


def patch_linear_cast_peft(layer: nn.Module, patch_params: dict | None) -> nn.Module:
    if is_hqq_lora_layer(layer):
        layer.cast(patch_params)
    return layer


# Putting it all together
class PeftUtils:
    @classmethod
    def get_base_class(cls, model, base_class):
        # Get base class
        if (base_class is None) and hasattr(model, "base_class"):
            base_class = model.base_class

        assert (
            base_class is not None
        ), "You need to provide the base HQQ class (LlamaHQQ, MixtralHQQ, etc.) as model.base_class or as an argument base_class=LlamaHQQ"
        return base_class

    @classmethod
    def add_lora(
        cls, model, peft_config, base_class: bool = None, verbose: bool = True
    ) -> None:
        # Base classs
        base_class = cls.get_base_class(model, base_class)

        # Freeze
        for param in model.parameters():
            param.requires_grad = False

        # Patch
        base_class.patch_linearlayers(
            model, patch_linear_add_peft, peft_config, verbose=verbose
        )

        # Rename modules
        autoname_modules(model)

        # add config to model
        model.peft_config = peft_config

    @classmethod
    def merge_lora(
        cls, model, merge_lora_params, base_class: bool = None, verbose: bool = True
    ) -> None:
        # Base classs
        base_class = cls.get_base_class(model, base_class)

        # Patch
        base_class.patch_linearlayers(
            model, patch_linear_merge_peft, merge_lora_params, verbose=verbose
        )

    @classmethod
    def cast_lora_weights(
        cls, model, dtype: torch.dtype, base_class: bool = None, verbose: bool = True
    ) -> None:
        # Base classs
        base_class = cls.get_base_class(model, base_class)

        # Linear tags
        linear_tags = base_class.get_linear_tags()

        # Patch
        base_class.patch_linearlayers(
            model,
            patch_linear_cast_peft,
            dict([(linear_tag, dtype) for linear_tag in linear_tags]),
            verbose=verbose,
        )

    @classmethod
    def save_lora_weights(
        cls, model, filename: str, base_class: bool = None, verbose: bool = True
    ) -> None:
        # Base classs
        base_class = cls.get_base_class(model, base_class)

        lora_global_params = {}

        def _patch_linear_save_weights(layer, patch_params, return_layer=True):
            if is_hqq_lora_layer(layer):
                lora_global_params[layer.name] = layer.state_dict()
            if return_layer:
                return layer

        # Linear tags
        linear_tags = base_class.get_linear_tags()

        # Patch
        base_class.patch_linearlayers(
            model,
            _patch_linear_save_weights,
            dict([(linear_tag, None) for linear_tag in linear_tags]),
            verbose=verbose,
        )

        # save
        torch.save(
            {"peft_config": model.peft_config, "parameters": lora_global_params},
            filename,
        )

    @classmethod
    def load_lora_weights(
        cls, model, filename: str, base_class: bool = None, verbose: bool = True
    ) -> None:
        # Base classs
        base_class = cls.get_base_class(model, base_class)

        lora_data = torch.load(filename, map_location="cpu")

        # V0.2 format: automatically creates lora modules if the model doesn't contain it
        if ("peft_config" in lora_data) and ("parameters" in lora_data):
            peft_config = lora_data["peft_config"]
            lora_global_params = lora_data["parameters"]
            if not hasattr(model, "peft_config"):
                cls.add_lora(model=model, peft_config=peft_config)

        # v0.1 format
        else:
            if not hasattr(model, "peft_config"):
                raise Exception(
                    "Using older version of lora weights. LoRa modules should be manually added in this case."
                )
            lora_global_params = lora_data

        def _patch_linear_load_weights(layer, patch_params, return_layer=True):
            if is_hqq_lora_layer(layer):
                layer.load_state_dict(lora_global_params[layer.name])
            if return_layer:
                return layer

        # Linear tags
        linear_tags = base_class.get_linear_tags()

        # Patch
        base_class.patch_linearlayers(
            model,
            _patch_linear_load_weights,
            dict([(linear_tag, None) for linear_tag in linear_tags]),
            verbose=verbose,
        )
