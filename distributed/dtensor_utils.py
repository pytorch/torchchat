import torch
import safetensors
from torch.distributed._tensor import DeviceMesh, DTensor, Shard, Replicate

from distributed.logging_utils import setup_logging
import torch.nn as nn
from collections import defaultdict

logger = setup_logging(__name__)

def is_dtensor(tensor):
    """ Check if a tensor is a DTensor by class or has a placements attribute (not sure if we want to use attr check) """
    return isinstance(tensor, DTensor) or hasattr(tensor, 'placements')

def find_cpu_tensors(module):
    """ Find all CPU tensors in a module and return a list of their names """
    cpu_tensors = []

    def recurse(mod):
        for name, param in mod.named_parameters(recurse=False):
            if not param.is_cuda:
                cpu_tensors.append(f"{mod.__class__.__name__}.{name}")
        
        for name, buf in mod.named_buffers(recurse=False):
            if not buf.is_cuda:
                cpu_tensors.append(f"{mod.__class__.__name__}.{name}")
        
        # Check for self.weights in RMSNorm
        if mod.__class__.__name__ == 'RMSNorm' and hasattr(mod, 'weights'):
            if not mod.weights.is_cuda:
                cpu_tensors.append(f"{mod.__class__.__name__}.weights")
        

        for name, child in mod.named_children():
            recurse(child)

    recurse(module)
    return cpu_tensors

def record_module_dtypes(module):
    """ Record the dtypes of all parameters and buffers in a module and return a dictionary of dtype -> list of names"""
    dtype_count = defaultdict(int)
    dtype_locations = defaultdict(list)

    def recurse(mod, prefix=''):
        for name, param in mod.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            dtype = param.dtype
            dtype_count[dtype] += 1
            dtype_locations[dtype].append(full_name)

        for name, buf in mod.named_buffers(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            dtype = buf.dtype
            dtype_count[dtype] += 1
            dtype_locations[dtype].append(full_name)

        for name, child in mod.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            recurse(child, child_prefix)

    recurse(module)
    return dtype_count, dtype_locations


def load_into_dtensor(weight_tensor, model_dtensor, debug=False):
    """ Adjust a loaded tensor to match the shape/placement of the model DTensor and copy the data into it """
    weight_tensor = weight_tensor.to(model_dtensor.device)
    if debug:
        logger.info(f"Loading into DTensor: {weight_tensor.shape=} {model_dtensor.shape=}")
    
    if weight_tensor.shape != model_dtensor.shape:
        raise ValueError(f"Shape mismatch: weight tensor shape {weight_tensor.shape} "
                         f"doesn't match DTensor shape {model_dtensor.shape}")
    
    placements = model_dtensor.placements
    mesh = model_dtensor.device_mesh
    mesh_dims = mesh.ndim
    
    if debug:
        logger.info(f"DTensor: Placements: {placements}")
        logger.info(f"DTensor: Mesh: {mesh}, mesh_dims={mesh_dims}")

    for placement in placements:
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            if debug:
                logger.info(f"DTensor Sharding dimension: {shard_dim}")
            
            if shard_dim >= weight_tensor.dim():
                raise ValueError(f"Shard dimension {shard_dim} is out of range for tensor with {weight_tensor.dim()} dimensions.")
            
            num_shards = mesh.size(0) # Assuming sharding is always along the first mesh dimension
            shard_size = weight_tensor.size(shard_dim) // num_shards
            shard_index = mesh.get_coordinate()[0]
            
            start_idx = shard_index * shard_size
            end_idx = start_idx + shard_size
            
            slice_list = [slice(None)] * weight_tensor.dim()
            slice_list[shard_dim] = slice(start_idx, end_idx)
            weight_tensor = weight_tensor[tuple(slice_list)]
            
            if debug:
                logger.info(f"Sharded tensor shape: {weight_tensor.shape}")
        
        elif isinstance(placement, Replicate):
            if debug:
                logger.info("Placement is Replicate, no sharding needed.")
        else:
            raise ValueError(f"Unsupported placement type: {type(placement)}")
    
    new_dtensor = DTensor.from_local(weight_tensor, mesh, placements)
    
    # Debug information
    if debug:
        local_tensor = new_dtensor.to_local()
        local_shard_shape = local_tensor.shape
        global_shape = new_dtensor.shape
        logger.info("=" * 50)
        logger.info(f"New DTensor: {global_shape=}, Local shard shape: {local_shard_shape}, Placements: {placements}")
        model_shard_shape = model_dtensor.to_local().shape
        model_global_shape = model_dtensor.shape
        logger.info(f"Model DTensor: {model_global_shape=}, Local shard shape: {model_shard_shape}, Placements {model_dtensor.placements}")
        assert local_shard_shape == model_shard_shape, f"Local shard shape {local_shard_shape} does not match model shard shape {model_shard_shape}"
        logger.info("=" * 50)
    
    model_dtensor.copy_(new_dtensor)
    return model_dtensor
