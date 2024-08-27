import torch
import safetensors
from torch.distributed._tensor import DeviceMesh, DTensor, Shard, Replicate

from distributed.logging_utils import setup_logging

logger = setup_logging(__name__)


def is_dtensor(tensor):
    """ Check if a tensor is a DTensor by class or has a placements attribute (not sure if we want to use attr check) """
    return isinstance(tensor, DTensor) or hasattr(tensor, 'placements')

def load_into_dtensor(weight_tensor, model_dtensor):
    """ Adjust a loaded tensor to match the shape/placement of the model DTensor and copy the data into it """

    weight_tensor = weight_tensor.to(model_dtensor.device)
    logger.info(f"Loading into DTensor: {weight_tensor.shape=} {model_dtensor.shape=}")

    if weight_tensor.shape != model_dtensor.shape:
        raise ValueError(f"Shape mismatch: weight tensor shape {weight_tensor.shape} "
                         f"doesn't match DTensor shape {model_dtensor.shape}")

    placements = model_dtensor.placements
    mesh = model_dtensor.device_mesh
    # Get the mesh dimensions
    mesh_dims = mesh.ndim

    # Handle sharding 
    for dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            if shard_dim >= mesh_dims:
                print(f"Warning: Shard dimension {shard_dim} is out of range for mesh with {mesh_dims} dimensions. "
                      f"Treating as replicated.")
                continue  # Treat as replicated

            num_shards = mesh.size(shard_dim)
            shard_size = weight_tensor.size(dim) // num_shards
            shard_index = mesh.get_coordinate()[shard_dim]
            
            # Calculate start and end indices for this shard
            start_idx = shard_index * shard_size
            end_idx = start_idx + shard_size

            # Create a slice object for this dimension
            dim_slice = slice(start_idx, end_idx)

            # Create a list of slice objects, with ':' for all dims except the sharded one
            slice_list = [slice(None)] * weight_tensor.dim()
            slice_list[dim] = dim_slice

            # Apply the slice to get the shard for this device
            weight_tensor = weight_tensor[tuple(slice_list)]
        elif isinstance(placement, Replicate):
            # No action needed for replicated dimensions
            pass
        else:
            raise ValueError(f"Unsupported placement type: {type(placement)}")

    # Create a new DTensor from the weight tensor
    new_dtensor = DTensor.from_local(weight_tensor, mesh, placements)

    # debug - TODO remove
    local_tensor = new_dtensor.to_local()
    local_shard_shape = local_tensor.shape
    placements = new_dtensor.placements
    global_shape = new_dtensor.shape
    logger.info("================================================\n")
    logger.info(f"New DTensor: {global_shape=}, Local shard shape: {local_shard_shape}, Placements: {placements}")
    model_shard_shape = model_dtensor.to_local().shape
    model_global_shape = model_dtensor.shape
    logger.info(f"Model DTensor: {model_global_shape=}, Local shard shape: {model_shard_shape}, Placements {model_dtensor.placements}")
    assert local_shard_shape == model_shard_shape, f"Local shard shape {local_shard_shape} does not match model shard shape {model_shard_shape}"
    logger.info("================================================\n")

    # Copy data from new_dtensor to model_dtensor
    model_dtensor.copy_(new_dtensor)

    return model_dtensor
