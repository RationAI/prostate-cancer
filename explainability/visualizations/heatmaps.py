import matplotlib.pyplot as plt
import numpy as np
import torch

from explainability.visualizations.to_image import batch_to_images


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
    denorm_imgs = batch_to_images(x_batch.detach())  # list of PIL
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
