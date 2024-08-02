
from contextlib import redirect_stdout

def reinit_layers(model, target_type=LlamaRotaryEmbedding, config_file: Optional[str] = None):
    """Reinitializes all layers of a given type in the model."""
    reinitialized_count = 0

    def recursive_reinit(module):
        nonlocal reinitialized_count
        for name, child in module.named_children():
            if isinstance(child, target_type):
                #if hasattr(child, 'reset_parameters'):
                print(f"Reinitializing {name} of type {type(child).__name__}")
                child.__init__(config=config_file)
                reinitialized_count += 1
                #else:
                #print(f"Warning: {name} of type {type(child).__name__} does not have a reset_parameters method")
                # If there's no reset_parameters method, we can implement a custom initialization here
                    
            else:
                recursive_reinit(child)

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
