import torch
from torch.distributed._tensor import DTensor, Shard, Replicate


from collections import defaultdict

from distributed.logging_utils import SingletonLogger
logger = SingletonLogger.get_logger()



def is_dtensor(tensor):
    """Check if a tensor is a DTensor by class or has a placements attribute (not sure if we want to use attr check)"""
    return isinstance(tensor, DTensor) or hasattr(tensor, "placements")


def load_into_dtensor(weight_tensor, model_dtensor):
    """Adjust a loaded tensor to match the shape/placement of the model DTensor and copy the data into it"""
    weight_tensor = weight_tensor.to(model_dtensor.device)

    if weight_tensor.shape != model_dtensor.shape:
        raise ValueError(
            f"Shape mismatch: weight tensor shape {weight_tensor.shape} "
            f"doesn't match DTensor shape {model_dtensor.shape}"
        )

    placements = model_dtensor.placements
    mesh = model_dtensor.device_mesh
    mesh_dims = mesh.ndim

    for placement in placements:
        if isinstance(placement, Shard):
            shard_dim = placement.dim

            if shard_dim >= weight_tensor.dim():
                raise ValueError(
                    f"Shard dimension {shard_dim} is out of range for tensor with {weight_tensor.dim()} dimensions."
                )

            num_shards = mesh.size(
                0
            )  # Assuming sharding is always along the first mesh dimension
            shard_size = weight_tensor.size(shard_dim) // num_shards
            shard_index = mesh.get_coordinate()[0]

            start_idx = shard_index * shard_size
            end_idx = start_idx + shard_size

            slice_list = [slice(None)] * weight_tensor.dim()
            slice_list[shard_dim] = slice(start_idx, end_idx)
            weight_tensor = weight_tensor[tuple(slice_list)]

        elif isinstance(placement, Replicate):
            continue
        else:
            raise ValueError(f"Unsupported placement type: {type(placement)}")

    new_dtensor = DTensor.from_local(weight_tensor, mesh, placements)

    return new_dtensor


def inspect_dtensor_sharding(dtensor):
    """hepful debug util for inspecting DTensor sharding"""
    if not is_dtensor(dtensor):
        logger.info(f"This tensor {dtensor} is not a DTensor")
        return

    placements = dtensor.placements
    logger.info(f"DTensor shape: {dtensor.shape}")
    logger.info(f"Number of dimensions: {len(placements)}")

    for dim, placement in enumerate(placements):
        logger.info(f"Dimension {dim}:")
        logger.info(f"  Placement type: {placement.type}")
        if placement.type == "shard":
            logger.info(f"  Sharding spec: {placement.sharding_spec}")
        elif placement.type == "replicate":
            logger.info("  Replicated across devices")
        else:
            logger.info(f"  Other placement type: {placement.type}")

    logger.info(f"Device mesh shape: {dtensor.device_mesh.shape}")
    logger.info(f"Device mesh devices: {dtensor.device_mesh.device_type}")
