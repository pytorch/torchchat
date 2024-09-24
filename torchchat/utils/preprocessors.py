import torch
import torchvision as tv
from torchvision import transforms as tvT
from PIL import Image
import os

from typing import List


def llava_image_preprocess(
        image: Image,
        *,
        target_h: int = 336,
        target_w: int = 336,
        rescale_factor: float = 0.00392156862745098, 
        image_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        image_std: List[float] = [0.26862954, 0.26130258, 0.27577711],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
    """
    Preprocess an image by resizing it to fit a target height and width, 
    padding with median RGB value to make a square, scaling, and normalizing.

    Args:
        img_address (str): Address of the local image file will be forwarded to the model.
        target_h (int): Target height.
        target_w (int): Target width.
        rescale_factor (float): Rescaling factor.
        image_mean (list): Mean values for normalization.
        image_std (list): Standard deviation values for normalization.

    Returns:
        torch.Tensor: Preprocessed image tensor.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the target height or width is not positive.
    """

    # Check if the target height and width are positive
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Target height and width must be positive")

    # Convert the image to a tensor
    img = tvT.functional.pil_to_tensor(image)

    # Calculate the height and width ratios
    ratio_h = img.shape[1] / target_h
    ratio_w = img.shape[2] / target_w

    # Resize the image to fit in a target_h x target_w canvas
    ratio = max(ratio_h, ratio_w)
    output_size = (int(img.shape[1] / ratio), int(img.shape[2] / ratio))
    img = tvT.Resize(size=output_size)(img)

    # Pad the image with median RGB value to make a square
    l_pad = (target_w - img.shape[2]) // 2
    t_pad = (target_h - img.shape[1]) // 2
    r_pad = -((target_w - img.shape[2]) // -2)
    b_pad = -((target_h - img.shape[1]) // -2)

    torch._check(l_pad >= 0)
    torch._check(t_pad >= 0)
    torch._check(r_pad >= 0)
    torch._check(b_pad >= 0)

    # Pad the image
    resized = torch.nn.functional.pad(
        img,
        (l_pad, r_pad, t_pad, b_pad),
    )

    # Scale the image
    scaled = resized * rescale_factor

    # Normalize the image
    normed = tvT.Normalize(image_mean, image_std)(scaled)

    return normed.unsqueeze(0).to(device).to(dtype)
