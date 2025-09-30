import numpy as np
from PIL import Image


def batch_to_images(batch_tensor, mean=None, std=None):
    """Convert a batch tensor to a list of denormalized images.

    Args:
        batch_tensor: PyTorch tensor of shape [B, C, H, W]
        mean: Normalization mean values [C] (if None, uses default values)
        std: Normalization std values [C] (if None, uses default values)

    Returns:
        List of numpy arrays, each of shape [H, W, C] with uint8 values
    """
    # Use default normalization values if not provided
    if mean is None:
        mean = np.array([228.5544, 178.8584, 219.8793])
    if std is None:
        std = np.array([27.8285, 51.4639, 26.4458])

    # Convert to numpy and move to CPU if needed
    batch_np = batch_tensor.cpu().numpy()

    # Process each image in the batch
    images = []
    for i in range(batch_np.shape[0]):
        # Get single image: [C, H, W]
        image = batch_np[i]

        # Transpose from CHW to HWC
        image_hwc = image.transpose(1, 2, 0)

        # Denormalize: (normalized * std) + mean
        image_denorm = (image_hwc * std) + mean

        # Clip to valid pixel range and convert to uint8
        image_denorm = np.clip(image_denorm, 0, 255).astype(np.uint8)

        images.append(Image.fromarray(image_denorm))

    return images
