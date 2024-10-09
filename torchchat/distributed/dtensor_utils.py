import torch
from torch.distributed import DeviceMesh
from torch.distributed._tensor import DTensor, Shard, Replicate, Placement
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from collections import defaultdict
from typing import Optional, Sequence

from torchchat.distributed.logging_utils import SingletonLogger
logger = SingletonLogger.get_logger()


def convert_to_dtensor(
    full_tensor: torch.Tensor,
    dtensor_template: DTensor,
) -> DTensor:
    """
    Converts a full tensor to a DTensor with the same placements as the given
    DTensor template.
    """
    if full_tensor.shape != dtensor_template.shape:
        raise ValueError(
            f"Shape mismatch: weight tensor shape {full_tensor.shape} "
            f"doesn't match DTensor shape {dtensor_template.shape}"
        )

    new_dtensor = shard(
        full_tensor,
        dtensor_template.placements,
        dtensor_template.device_mesh
    )
    return new_dtensor


def shard(
    full_tensor: torch.Tensor,
    placements: Sequence[Placement],
    device_mesh: Optional[DeviceMesh] = None,
) -> DTensor:
    """
    Shards a full tensor based on indicated placements, and returns a
    DTensor containing the shard.
    Args:
        full_tensor (torch.Tensor): the full tensor to be sharded.
        placements (Sequence[:class:`Placement`]): the placements that
            describes how to place the local tensor on DeviceMesh.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
            DTensor.  Must have same dimension as the number of placements.
            If not specified, would be retrieve from current context.
    Returns:
        A :class:`DTensor` object with the shard as its local tensor.
    Examples:
        >>> # xdoctest: +SKIP("need world_size and rank")
        >>> device_mesh = dist.init_device_mesh("cuda", (world_size,))
        >>> full_tensor = torch.arange(world_size, device=f"cuda:{rank}")
        >>> placements = [Shard(1)]
        >>> dtensor = shard(full_tensor, placements, device_mesh)
    """
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()

    shape, offset = compute_local_shape_and_global_offset(
        full_tensor.shape, device_mesh, placements
    )
    slices = [
        slice(cur_offset, cur_offset + cur_shape)
        for cur_shape, cur_offset in zip(shape, offset)
    ]
    local_tensor = full_tensor[slices]
    return DTensor.from_local(local_tensor, device_mesh, placements)
