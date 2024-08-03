from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import redirect_stdout, contextmanager
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import torch
import torch.nn as nn

@contextmanager
def init_on_meta_device(device: torch.device):
    """Device initialization context manager for meta init. Keeps buffers on current device"""

    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    tensor_constructors_to_patch = {torch_function_name: getattr(torch, torch_function_name) 
                                    for torch_function_name in ['empty', 'zeros', 'ones', 'full']}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs['device'] = device
            return fn(*args, **kwargs)
        return wrapper
    try:
        nn.Module.register_parameter = register_empty_parameter
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, patch_tensor_constructor(old_torch_function))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)



def find_main_llama_rope_embeddings(model):
    rope_embeddings = []
    
    for name, module in model.named_children():
        if isinstance(module, LlamaRotaryEmbedding):
            rope_embeddings.append((name, module))
    
    if not rope_embeddings:
        print("No LlamaRotaryEmbedding found at the main level of the model.")
    elif len(rope_embeddings) == 1:
        print(f"Found one LlamaRotaryEmbedding at the main level: {rope_embeddings[0][0]}")
        return rope_embeddings[0][1]
    else:
        print(f"Found multiple LlamaRotaryEmbeddings at the main level: {[name for name, _ in rope_embeddings]}")
        return rope_embeddings

def print_model_structure(model):
    """prints tab indented model structure"""
    def _search(module, depth=0):
            for name, child in module.named_children():
                print(f"{'  '*(depth+1)}{name}")
                search(child, depth+1)
    _search(model)

def reinit_layers(model, target_type=LlamaRotaryEmbedding, config_file: Optional[str] = None):
    """Reinitializes all layers of a given type in the model."""
    reinitialized_count = 0

    def recursive_reinit(module, depth=0):
        nonlocal reinitialized_count
        for name, child in module.named_children():
            if isinstance(child, target_type):
                #if hasattr(child, 'reset_parameters'):
                print(f"{depth=}, Reinitializing {name} of type {type(child).__name__}")
                if depth==1:
                    return child
                # child.__init__(config=config_file)
                reinitialized_count += 1
                #else:
                #print(f"Warning: {name} of type {type(child).__name__} does not have a reset_parameters method")
                # If there's no reset_parameters method, we can implement a custom initialization here
                    
            else:
                recursive_reinit(child, depth+1)

    recursive_reinit(model)
    print(f"Total reinitialized modules: {reinitialized_count}")


def enumerate_transformer_llm(model, prefix='', output_file=None):
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
                print_info(f"  Parameter: {full_name}.{param_name}, Shape: {param.shape}")
        
        if list(module.buffers()):
            for buffer_name, buffer in module.named_buffers():
                print_info(f"  Buffer: {full_name}.{buffer_name}, Shape: {buffer.shape}")
        
        if list(module.children()):
            enumerate_transformer_llm(module, full_name, output_file)
