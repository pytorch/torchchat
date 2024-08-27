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

    if weight_tensor.shape != model_dtensor.shape:
        raise ValueError(f"Shape mismatch: weight tensor shape {weight_tensor.shape} "
                         f"doesn't match DTensor shape {model_dtensor.shape}")

    placements = model_dtensor.placements
    mesh = model_dtensor.device_mesh

    # Handle sharding...nothing to do for replicated tensors
    # Handle sharding if necessary
    for dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            num_shards = mesh.size(shard_dim)
            shard_size = weight_tensor.size(shard_dim) // num_shards
            shard_index = mesh.get_coordinate()[shard_dim]
            
            # Calculate start and end indices for this shard
            start_idx = shard_index * shard_size
            end_idx = start_idx + shard_size

            # Create a slice object for this dimension
            dim_slice = slice(start_idx, end_idx)

            # Create a list of slice objects, with ':' for all dims except the sharded one
            slice_list = [slice(None)] * weight_tensor.dim()
            slice_list[shard_dim] = dim_slice

            # Apply the slice to get the shard for this device
            weight_tensor = weight_tensor[tuple(slice_list)]
        elif isinstance(placement, Replicate):
            # No action needed for replicated dimensions
            pass
        else:
            raise ValueError(f"Unsupported placement type: {type(placement)}")

    # Create a new DTensor from the (potentially sharded) weight tensor
    new_dtensor = DTensor.from_local(weight_tensor, mesh, placements)

    # Copy data from new_dtensor to model_dtensor
    model_dtensor.copy_(new_dtensor)

    return model_dtensor
