from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import torch
import torch.nn as nn
from typing import Callable
import torch.fx as fx
from torch._subclasses import FakeTensorruff
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@contextmanager
def init_on_meta_device(device: torch.device):
    """
    Device initialization context manager.
    A context manager under which parameters are initialized on meta device,
    butbuffers init on actual device,
    The goal here is to ensure that buffer initialization is done on the actual device,
    preserving generated buffers esp RopE embeddings.

    """

    def register_empty_parameter(
        module: nn.Module, name: str, param: nn.Parameter
    ) -> None:
        old_register_parameter(module, name, param)
        if param is not None:
            param_class = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_class(
                module._parameters[name].to(device), **kwargs
            )

    def reroute_device_tensor_constructor(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    old_register_parameter = nn.Module.register_parameter
    tensor_constructors_to_proxy: Dict[str, Callable] = {
        torch_function_name: getattr(torch, torch_function_name)
        for torch_function_name in ["empty", "zeros", "ones", "full"]
    }

    try:
        nn.Module.register_parameter = register_empty_parameter
        for (
            torch_function_name,
            old_torch_function,
        ) in tensor_constructors_to_proxy.items():
            setattr(
                torch,
                torch_function_name,
                reroute_device_tensor_constructor(old_torch_function),
            )
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        for (
            torch_function_name,
            old_torch_function,
        ) in tensor_constructors_to_proxy.items():
            setattr(torch, torch_function_name, old_torch_function)


def verify_graph_tensor_properties(
    graph_module: fx.GraphModule,
) -> Tuple[bool, List[str]]:
    """
    Verify that all tensors in the given fx.GraphModule have either float32 or bfloat16 dtypes,
    and are not fake or meta tensors.

    Args:
        graph_module (fx.GraphModule): The graph module to verify.

    Returns:
        Tuple[bool, List[str]]: A tuple containing:
            - A boolean indicating whether all tensors meet the criteria.
            - A list of error messages for any tensors that don't meet the criteria.
    """
    allowed_dtypes = {torch.float32, torch.bfloat16}
    all_correct = True
    error_messages = []

    def check_tensor(tensor: torch.Tensor, name: str):
        logger.info(f"checking tensor {name} {tensor=} with {tensor.dtype=}")
        nonlocal all_correct, error_messages
        if tensor.dtype not in allowed_dtypes:
            all_correct = False
            error_messages.append(
                f"Tensor '{name}' has dtype {tensor.dtype}, which is not allowed."
            )

        if isinstance(tensor, FakeTensor):
            all_correct = False
            error_messages.append(f"Tensor '{name}' is a fake tensor.")

        if tensor.is_meta:
            all_correct = False
            error_messages.append(f"Tensor '{name}' is a meta tensor.")

    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            attr_value = getattr(graph_module, node.target)
            if isinstance(attr_value, torch.Tensor):
                check_tensor(attr_value, f"{node.op}:{node.target}")
        elif node.op in ["call_function", "call_method", "call_module"]:
            # Check input tensors
            for arg in node.args:
                if isinstance(arg, torch.Tensor):
                    check_tensor(arg, f"{node.op}:{node.target} input")
            for kwarg in node.kwargs.values():
                if isinstance(kwarg, torch.Tensor):
                    check_tensor(kwarg, f"{node.op}:{node.target} input")

            # Check output tensor
            if isinstance(node.meta.get("val"), torch.Tensor):
                check_tensor(node.meta["val"], f"{node.op}:{node.target} output")
        elif node.op == "output":
            # Check output tensors
            if isinstance(node.args[0], (tuple, list)):
                for i, arg in enumerate(node.args[0]):
                    if isinstance(arg, torch.Tensor):
                        check_tensor(arg, f"output[{i}]")
            elif isinstance(node.args[0], torch.Tensor):
                check_tensor(node.args[0], "output")

    return all_correct, error_messages


def inspect_module_tensors(
    module: nn.Module,
) -> Dict[str, List[Tuple[str, torch.dtype]]]:
    """
    Inspect an nn.Module and return information about its tensor types.

    Args:
        module (nn.Module): The module to inspect.

    Returns:
        Dict[str, List[Tuple[str, torch.dtype]]]: A dictionary with keys 'parameters', 'buffers', and 'submodules',
        each containing a list of (name, dtype) tuples for the respective tensors.
    """
    result = defaultdict(list)

    def get_tensor_info(tensor: torch.Tensor, name: str) -> Tuple[str, str]:
        tensor_type = "Regular"
        if isinstance(tensor, FakeTensor):
            tensor_type = "Fake"
        elif tensor.is_meta:
            tensor_type = "Meta"
        return f"{name} ({tensor_type})", str(tensor.dtype)

    # parameters
    for name, param in module.named_parameters(recurse=False):
        result["parameters"].append(get_tensor_info(param, name))

    # buffers
    for name, buffer in module.named_buffers(recurse=False):
        result["buffers"].append(get_tensor_info(buffer, name))

    # Recursively inspect submodules
    for name, submodule in module.named_children():
        submodule_info = inspect_module_tensors(submodule)
        for key, value in submodule_info.items():
            result["submodules"].extend(
                [(f"{name}.{item[0]}", item[1]) for item in value]
            )

    return dict(result)


def find_main_llama_rope_embeddings(model):
    rope_embeddings = []

    for name, module in model.named_children():
        if isinstance(module, LlamaRotaryEmbedding):
            rope_embeddings.append((name, module))

    if not rope_embeddings:
        print("No LlamaRotaryEmbedding found at the main level of the model.")
    elif len(rope_embeddings) == 1:
        print(
            f"Found one LlamaRotaryEmbedding at the main level: {rope_embeddings[0][0]}"
        )
        return rope_embeddings[0][1]
    else:
        print(
            f"Found multiple LlamaRotaryEmbeddings at the main level: {[name for name, _ in rope_embeddings]}"
        )
        return rope_embeddings


def print_model_structure(model):
    """prints tab indented model structure"""

    def _search(module, depth=0):
        for name, child in module.named_children():
            print(f"{'  '*(depth+1)}{name}")
            search(child, depth + 1)

    _search(model)


def reinit_layers(
    model, target_type=LlamaRotaryEmbedding, config_file: Optional[str] = None
):
    """Reinitializes all layers of a given type in the model."""
    reinitialized_count = 0

    def recursive_reinit(module, depth=0):
        nonlocal reinitialized_count
        for name, child in module.named_children():
            if isinstance(child, target_type):
                # if hasattr(child, 'reset_parameters'):
                print(f"{depth=}, Reinitializing {name} of type {type(child).__name__}")
                if depth == 1:
                    return child
                # child.__init__(config=config_file)
                reinitialized_count += 1
                # else:
                # print(f"Warning: {name} of type {type(child).__name__} does not have a reset_parameters method")
                # If there's no reset_parameters method, we can implement a custom initialization here

            else:
                recursive_reinit(child, depth + 1)

    recursive_reinit(model)
    print(f"Total reinitialized modules: {reinitialized_count}")


def enumerate_transformer_llm(model, prefix="", output_file=None):
    """Prints information about the model's modules and parameters."""

    def print_info(*args):
        print(*args)
        if output_file:
            print(*args, file=output_file)

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        print_info(f"Module: {full_name}, Type: {type(module).__name__}")

        if list(module.parameters()):
            for param_name, param in module.named_parameters():
                print_info(
                    f"  Parameter: {full_name}.{param_name}, Shape: {param.shape}"
                )
                if isinstance(param, FakeTensor):
                    print_info(f" ***** Fake Tensor: {param_name}")

        if list(module.buffers()):
            for buffer_name, buffer in module.named_buffers():
                print_info(
                    f"  Buffer: {full_name}.{buffer_name}, Shape: {buffer.shape}"
                )
                if isinstance(buffer, FakeTensor):
                    print_info(f" ***** Fake Tensor: {buffer_name}")

        if list(module.children()):
            enumerate_transformer_llm(module, full_name, output_file)


def check_rope_embedding(stage_module, layer_to_check: str = None):
    """generic check on rope embedding"""

    # stage_module.model.graph.print_tabular()
    if not layer_to_check:
        layer_to_check = "model.layers.0.self_attn.rotary_emb"

    rotary = stage_module.get_submodule(layer_to_check)
    print(f"checkmate {rotary=}")
    buffer = rotary._buffers["inv_freq"]
    print(f"inv freq checkmate {buffer=}")
