import torch
from torch.distributed._tensor import DTensor, Shard, Replicate

from distributed.logging_utils import setup_logging
from collections import defaultdict

logger = setup_logging(__name__)


def is_dtensor(tensor):
    """Check if a tensor is a DTensor by class or has a placements attribute (not sure if we want to use attr check)"""
    return isinstance(tensor, DTensor) or hasattr(tensor, "placements")


# TODO - this should move to a model/debug utils file
def find_cpu_tensors(module):
    """Find all CPU tensors in a module and return a list of their names"""
    cpu_tensors = []

    def recurse(mod):
        for name, param in mod.named_parameters(recurse=False):
            if not param.is_cuda:
                cpu_tensors.append(f"{mod.__class__.__name__}.{name}")

        for name, buf in mod.named_buffers(recurse=False):
            if not buf.is_cuda:
                cpu_tensors.append(f"{mod.__class__.__name__}.{name}")

        # Check for self.weights in RMSNorm
        if mod.__class__.__name__ == "RMSNorm" and hasattr(mod, "weights"):
            if not mod.weights.is_cuda:
                cpu_tensors.append(f"{mod.__class__.__name__}.weights")

        for name, child in mod.named_children():
            recurse(child)

    recurse(module)
    return cpu_tensors


def record_module_dtypes(module):
    """Record the dtypes of all parameters and buffers in a module and return a dictionary of dtype -> list of names"""
    dtype_count = defaultdict(int)
    dtype_locations = defaultdict(list)
    fp32_locations = defaultdict(list)

    def recurse(mod, prefix=""):
        for name, param in mod.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            dtype = param.dtype
            dtype_count[dtype] += 1
            dtype_locations[dtype].append(full_name)
            if dtype == torch.float32:
                fp32_locations[full_name] = param

        for name, buf in mod.named_buffers(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            dtype = buf.dtype
            dtype_count[dtype] += 1
            dtype_locations[dtype].append(full_name)
            if dtype == torch.float32:
                fp32_locations[full_name] = buf

        for name, child in mod.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            recurse(child, child_prefix)

    recurse(module)
    return dtype_count, dtype_locations, fp32_locations


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
