import numpy as np
import pyvips
from numpy.typing import NDArray

from app.empaia import Client
from app.empaia.typing import WSI


async def get_background_mask(
    client: Client, wsi: WSI, level: int
) -> tuple[NDArray[np.float32], int]:
    level = min(level, wsi["num_levels"] - 1)
    extent = wsi["levels"][level]["extent"]

    data = await client.get_region(wsi["id"], level, 0, 0, extent["x"], extent["y"])
    slide = pyvips.Image.new_from_buffer(data, "")
    background_mask = _create_thresholded_bg_mask(slide) / 255
    return background_mask.astype(np.float32), level


def _create_thresholded_bg_mask(
    vi_slide_rgb: pyvips.Image, disk_size: int = 10
) -> NDArray[np.uint8]:
    """Draw binary background mask using image processing techniques.

    Returns:
        pyvips.Image : Binary background mask.

    Raises:
        Value Error: When bg level is higher than level count.
    """
    # Extract saturation channel
    vi_slide_hsv = vi_slide_rgb.sRGB2HSV()
    _, vi_s, *_ = vi_slide_hsv.bandsplit()

    # Otsu Thresholding
    vi_res = vi_s.colourspace("b-w").hist_find().numpy().squeeze()
    vi_threshold = _otsu(vi_res)
    vi_hi_s = vi_s > vi_threshold

    # Morph Object
    vi_disk_object = pyvips.Image.black(2 * disk_size + 1, 2 * disk_size + 1) + 128
    vi_disk_object = vi_disk_object.draw_circle(
        255, disk_size, disk_size, disk_size, fill=True
    )

    # Closing
    vi_mask = vi_hi_s.morph(vi_disk_object, pyvips.enums.OperationMorphology.DILATE)
    vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.ERODE)

    # Opening
    vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.ERODE)
    vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.DILATE)

    return vi_mask.numpy()


def _otsu(hist: NDArray[np.uint32]) -> int:
    """Calculates Otsu's threshold from a histogram.

    Short version: find intensity value that minimizes per-class variance
                    weighted by cummulative distro

    https://en.wikipedia.org/wiki/Otsu%27s_method
    """
    bins = np.arange(256)
    otsu_th, otsu_crit = None, np.inf

    for th in bins:
        total_px = hist.sum()

        # Calculate CDF
        w0, w1 = hist[:th].sum() / total_px, hist[th:].sum() / total_px
        if w1 == 0 or w0 == 0:
            continue

        # Calculate per-class variances
        mean_0 = np.average(bins[:th], weights=hist[:th])
        mean_1 = np.average(bins[th:], weights=hist[th:])
        var_0 = np.average((bins[:th] - mean_0) ** 2, weights=hist[:th])
        var_1 = np.average((bins[th:] - mean_1) ** 2, weights=hist[th:])

        # Find minimum
        crit = w0 * var_0 + w1 * var_1
        if crit < otsu_crit:
            otsu_crit = crit
            otsu_th = th

    return otsu_th
