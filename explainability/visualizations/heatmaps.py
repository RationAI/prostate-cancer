import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, UInt8

from explainability.visualizations.image_transforms import get_inverse_norm_transform


def visualize_cams(x_batch, cams, alpha=0.4, show_colorbar=False, title_prefix=""):
    """Visualize LayerCAM for a whole batch.

    Args:
        x_batch: Tensor [B, C, H, W]
        cams: Numpy array or tensor [B, H_l, W_l]
        alpha: Overlay strength
        show_colorbar: Whether to show a colorbar for each heatmap
        title_prefix: Optional prefix for subplot titles
    """
    assert isinstance(x_batch, torch.Tensor), "x_batch must be a torch.Tensor [B,C,H,W]"

    # Denormalize all images using existing helper

    # list of PIL images
    # denorm_imgs = batch_to_images(x_batch.detach())  # list of PIL
    inv_norm = get_inverse_norm_transform()
    denorm_imgs = [
        inv_norm(img).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
        for img in x_batch
    ]
    images = [np.asarray(img) / 255.0 for img in denorm_imgs]  # list of HxWx3 in [0,1]
    num_of_images = len(images)

    # Ensure cams is a torch tensor [B, H_l, W_l]
    if isinstance(cams, np.ndarray):
        cams_t = torch.from_numpy(cams).float()
    elif isinstance(cams, torch.Tensor):
        cams_t = cams.float()
    else:
        raise TypeError("cams must be numpy array or torch tensor of shape [B,H_l,W_l]")

    assert cams_t.dim() == 3 and cams_t.size(0) == num_of_images, (
        f"cams must be [B,H_l,W_l], got {tuple(cams_t.shape)} vs B={num_of_images}"
    )

    # Prepare figure: B rows, 3 cols
    ncols = 3 + (1 if show_colorbar else 0)
    width_ratios = [1, 1, 1] + ([0.05] if show_colorbar else [])
    figsize = (16 if show_colorbar else 15, 5 * num_of_images)
    fig, axs = plt.subplots(
        num_of_images,
        ncols,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )

    cmap = plt.get_cmap("jet")

    for i in range(num_of_images):
        img = images[i]
        height, width = img.shape[:2]

        # Resize CAM i to exactly HxW using bilinear interpolation
        cam_i = cams_t[i].unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        cam_resized = torch.nn.functional.interpolate(
            cam_i, size=(height, width), mode="bilinear", align_corners=False
        )[0, 0]
        cam_resized = cam_resized - cam_resized.min()
        cam_resized = cam_resized / (cam_resized.max() + 1e-8)
        cam_np = cam_resized.cpu().numpy()

        # Colorize and overlay
        colored_heatmap = cmap(cam_np)[..., :3]
        overlay_img = np.clip((1 - alpha) * img + alpha * colored_heatmap, 0.0, 1.0)

        # Plot row i
        ax_orig = axs[i, 0]
        ax_hm = axs[i, 1]
        ax_ovl = axs[i, 2]
        ax_orig.imshow(img)
        ax_orig.set_title(f"{title_prefix}Image {i}")
        ax_orig.axis("off")

        im = ax_hm.imshow(cam_np, cmap="jet", vmin=0.0, vmax=1.0)
        ax_hm.set_title(f"{title_prefix}Heatmap {i}")
        ax_hm.axis("off")

        ax_ovl.imshow(overlay_img)
        ax_ovl.set_title(f"{title_prefix}Overlay {i}")
        ax_ovl.axis("off")

        if show_colorbar:
            ax_cbar = axs[i, 3]
            fig.colorbar(im, cax=ax_cbar)

    plt.tight_layout()
    plt.show()


def plot_overlays_side_by_image(
    x_batch: Float[torch.Tensor, "B 3 H W"],  # [B,3,H,W]
    overlays: UInt8[torch.Tensor, "B 3 H W"],  # [B,3,H,W]
    alpha: float = 0.6,
):
    """This function blends plots B lines of side-by-side triples of images: original, blended, and overlay."""
    overlaid = superimpose(
        opacity=alpha, images=x_batch.to(torch.uint8), overlays=overlays
    )
    B = x_batch.shape[0]
    fig, axes = plt.subplots(B, 3, figsize=(12, 4 * B))
    if B == 1:
        axes = axes[None, :]  # Make it 2D for consistency
    for i in range(B):
        axes[i, 0].imshow(x_batch[i].permute(1, 2, 0).cpu().numpy().astype("uint8"))
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlaid[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 1].set_title("Blended Image")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlays[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.show()


def superimpose(
    images: UInt8[torch.Tensor, "B 3 H1 W1"],
    overlays: UInt8[torch.Tensor, "B 3 H2 W2"],
    strategy: str = "additive_overflow_press",
    opacity: float = 0.5,
) -> UInt8[torch.Tensor, "B 3 H W"]:
    """This function blends the overlay images on top of the original images using the specified strategy, producing a batch of blended images.

    Args:
        images: [B, 3, H, W] tensor of original images (uint8)
        overlays: [B, 3, H, W] tensor of overlay images (uint8)
        strategy: blending strategy
        opacity: blending opacity
    """
    assert images.shape == overlays.shape, (
        "Images and overlays must have the same shape"
    )

    if strategy == "additive_overflow_press":
        # tries to add the overlay without any modifications. Just slaps it on top of the image.
        # But then, when pixels overflows, it compresses them back to range [0..255]
        # First add the overlay to the image with a scaling factor
        res = images + overlays * opacity  # [B, 3, H, W]

        # get the maximum values across channels and stack them to get 3 channels again
        channel_maximums = res.amax(dim=1, keepdim=True)  # [B, 1, H, W]

        # get the locations of overflowing pixels using the maximum channel value
        overflow_locations = channel_maximums > 255  # [B, 1, H, W]
        overflow_locations = overflow_locations.expand(-1, 3, -1, -1)  # [B, 3, H, W]

        # downscale the overflowing pixels so that the maximum value is 255. Channels are broadcasted automatically and colors are downscaled together
        # the maximums are not broadcasted to the 3 channels automatically, we need to repeat the dimension
        print("SHAPES:", res.shape, channel_maximums.shape, overflow_locations.shape)
        res[overflow_locations] = res[overflow_locations] * (
            255 / channel_maximums.expand(-1, 3, -1, -1)[overflow_locations]
        )

    elif strategy == "sub":
        res = torch.clamp(images - overlays * opacity, min=0)
    elif strategy == "linear_combination":
        res = images * (1.0 - opacity) + overlays * opacity
    elif strategy == "outline":
        raise NotImplementedError("Outline strategy not implemented in torch version")
    elif strategy == "segment_contour":
        raise NotImplementedError(
            "Segment contour strategy not implemented in torch version"
        )
    elif strategy == "black_alpha_blend":
        alpha = overlays.amax(dim=1, keepdim=True) / 255.0  # [B, 1, H, W]
        res = images * (1 - alpha) + overlays  # *alpha
    else:
        raise NotImplementedError(f"There is no strategy with name {strategy}!")

    return res.clamp(0, 255).to(torch.uint8)
