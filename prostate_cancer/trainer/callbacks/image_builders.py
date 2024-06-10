# Copyright (c) The RationAI team.

import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pyvips
import scipy
import torch
from numpy.typing import NDArray

from prostate_cancer.trainer.callbacks.vis_mode import VisualisationMode


logger = logging.getLogger("callbacks/image_builder")


class ImageBuilder(ABC):
    visualization_mode: VisualisationMode
    save_dir: Path | str
    filename: str

    def __init__(
        self, vis_mode: VisualisationMode, save_dir: Path | str, filename: str
    ) -> None:
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.visualization_mode = vis_mode

        self.save_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def update(self, data: torch.Tensor, metadata: dict) -> None: ...

    @abstractmethod
    def save(self) -> str: ...

    @staticmethod
    def _to_numpy(data: torch.Tensor) -> NDArray:
        if isinstance(data, np.ndarray):
            return data

        if isinstance(data, torch.Tensor):
            logger.debug("Converting tiles.")
            return data.detach().cpu().numpy()

        raise TypeError(f"Type '{type(data)}' is not supported.")

    @staticmethod
    def _to_nwhc(data: NDArray) -> NDArray:
        """Converts data to [N, W, H, C] format.

        [N, W, H, C]:
            N - batch size
            W - width in pixels
            H - height in pixels
            C - number of channels
        """
        if data.ndim == 4:  # tile is image
            logger.debug("Shuffling dimensions.")
            return data.transpose(
                (0, 2, 3, 1)
            )  # [N, C, W, H] into sensible [N, W, H, C]
        if data.ndim == 2:  # tile is classification
            logger.debug("Adding dimensions.")
            return data[:, None, None, :]  # [N, C] into [N, W, H, C]
        raise ValueError(
            f"Incorrect data format. Expected 4-dim [N, C, W, H] or 2-dim [N, C]"
            f" but found {data.ndim}-dim {data.shape}."
        )

    def _preprocess_data(self, data: torch.Tensor) -> NDArray:
        """Converts data to numpy, scales it to [0,255] and resizes it to tile_size if segmentation task is visualized.

        Args:
            data: A tensor of shape [N, C, W, H] or [N, C].
        """
        data = self._to_numpy(data)
        data = self._to_nwhc(data)
        data = self.visualization_mode.scale(data)  # Scale from [0,1] to [0,255]
        return data

    def _finalize_image(self, vips_im: pyvips.Image) -> pyvips.Image:
        # Restore default value
        if self.visualization_mode.zero_offset != 0:
            vips_im += self.visualization_mode.zero_offset
        return vips_im


class ImageAssembler(ImageBuilder, ABC):
    image_size: tuple[int, int, int]
    tile_size: int
    interpolation: str

    def __init__(
        self,
        image_size: tuple[int, int, int],
        tile_size: int,
        vis_mode: VisualisationMode,
        save_dir: Path | str,
        filename: str,
        interpolation: str,
    ) -> None:
        super().__init__(vis_mode=vis_mode, save_dir=save_dir, filename=filename)
        self.image_size = image_size
        self.tile_size = tile_size
        self.interpolation = interpolation

    def _resize_to_tile_size(self, data: NDArray) -> NDArray:
        interpolation_map = {"nearest": 0, "bilinear": 1, "cubic": 2}
        width, height = data.shape[1], data.shape[2]

        if self.tile_size != width or self.tile_size != height:
            logger.debug("Resizing the tiles to fit tile_size.")
            zoom = (1, self.tile_size / width, self.tile_size / height, 1)
            data = scipy.ndimage.zoom(
                input=data, zoom=zoom, order=interpolation_map[self.interpolation]
            )

        return data

    def _save_xopat_compatible(self, vips_im: pyvips.Image) -> str:
        vips_im = vips_im.cast("uchar")
        save_path = (self.save_dir / self.filename).with_suffix(".tiff")

        vips_im.tiffsave(
            save_path,
            bigtiff=True,
            compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
        )

        return save_path


class DiskMappedPatchAssembler(ImageAssembler):
    """A class to compose any masks into a segmentation overlay matching the input image.

    Stores data in a disk-mapped arrays to prevent RAM overflow.
    Is generally slower than in-memory patch assemblers.
    """

    scale_factor: int
    image: np.memmap
    count: np.memmap

    def __init__(
        self,
        metadata: dict,
        vis_mode: VisualisationMode,
        save_dir: Path | str,
        interpolation: str = "nearest",
    ) -> None:
        filename = metadata["slide_name"]
        image_size = (
            int(metadata["slide_width"]),
            int(metadata["slide_height"]),
            int(metadata["slide_channels"]),
        )
        tile_size = int(metadata["tile_size"])
        super().__init__(
            image_size=image_size,
            tile_size=tile_size,
            vis_mode=vis_mode,
            save_dir=save_dir,
            filename=filename,
            interpolation=interpolation,
        )

        self.scale_factor = 2 ** int(metadata["sample_level"])

        save_path = (self.save_dir / self.filename).as_posix()

        self.image = np.memmap(
            save_path + "_mask.nmp",
            dtype=np.float32,
            mode="w+",
            shape=(
                self.image_size[1],
                self.image_size[0],
                self.image_size[2],
            ),  # row-first format (H, W, C)
        )
        self.count = np.memmap(
            save_path + "_count.nmp",
            dtype=np.int32,
            mode="w+",
            shape=(
                self.image_size[1],
                self.image_size[0],
                1,
            ),  # row-first format (H, W, C)
        )

    def update(self, data: torch.Tensor, metadata: dict) -> None:
        logger.debug("Pasting tiles.")
        xs, ys = (
            metadata["coord_x"] // self.scale_factor,
            metadata["coord_y"] // self.scale_factor,
        )
        data = self._preprocess_data(data)
        data = self._resize_to_tile_size(data)

        # Paste tiles onto mask
        for x, y, tile in zip(xs, ys, data, strict=False):
            mm_h, mm_w, mm_c = self.image[
                y : y + self.tile_size, x : x + self.tile_size, :
            ].shape
            self.image[y : y + self.tile_size, x : x + self.tile_size, :] += tile[
                :mm_h, :mm_w, :mm_c
            ]
            self.count[y : y + self.tile_size, x : x + self.tile_size, :] += 1

        self.image.flush()
        self.count.flush()

    def save(self) -> str:
        # Converting to pyVips
        vips_im = pyvips.Image.new_from_array(self.image)
        count_im = pyvips.Image.new_from_array(self.count)

        # Resolve overlaps
        vips_im /= count_im  # invokes zero-safe division from vips

        # TODO: Add options to save each band as separate image?
        # Sub: [vips_im_i.tiffsave() for vips_im_i in vips_im.bandsplit()]

        vips_im = self._finalize_image(vips_im)
        save_path = self._save_xopat_compatible(vips_im)

        return save_path


class InMemoryHeatmapAssembler(ImageAssembler):
    """A class to accumulate predictions into a prediction heatmap.

    Minimizes required RAM by calculating minimum loss-less resolution
    from tile size and step size and accumulates predictions into compressed arrays.
    Compression is meaningful only for scalar predictions,
    To control compression, use `compress_accumulator_array` parameter.
    """

    heatmap_accumulator: np.ndarray
    patch_overlap_counter: np.ndarray
    gcd_size_factor: int
    accumulator_tile_size: int
    overlap_counter_tile_size: int
    compress_accumulator_array: bool

    def __init__(
        self,
        metadata: dict,
        vis_mode: VisualisationMode,
        save_dir: Path | str,
        interpolation: str,
        compress_accumulator_array: bool,
    ) -> None:
        filename = metadata["slide_name"]
        image_size = (
            int(metadata["slide_width"]),
            int(metadata["slide_height"]),
            int(metadata["slide_channels"]),
        )
        tile_size = int(metadata["tile_size"])
        step_size = int(metadata["step_size"])
        super().__init__(
            image_size=image_size,
            tile_size=tile_size,
            vis_mode=vis_mode,
            save_dir=save_dir,
            filename=filename,
            interpolation=interpolation,
        )
        # setup compression parameters
        self.compress_accumulator_array = compress_accumulator_array
        self.gcd_size_factor = math.gcd(tile_size, step_size)
        self.accumulator_tile_size = tile_size
        self.overlap_counter_tile_size = tile_size // self.gcd_size_factor

        # set accumulator tile size to compressed size if enabled
        if self.compress_accumulator_array:
            self.accumulator_tile_size = self.overlap_counter_tile_size
        self.level_coord_multiplier = 2 ** int(metadata["sample_level"])

        # Calculate sizes for accumulator and overlap counter
        self.w, self.h, self.c = self.image_size
        compressed_w = self.w // self.gcd_size_factor
        compressed_h = self.h // self.gcd_size_factor

        # set accumulator size to compressed size if enabled
        if self.compress_accumulator_array:
            accum_h, accum_w = compressed_h, compressed_w
        else:
            accum_h, accum_w = self.h, self.w

        self.heatmap_accumulator = np.zeros(
            shape=(accum_h, accum_w, self.c),  # row-first format (H, W, C)
            dtype=np.float32,
        )
        # overlap counter is always compressed
        self.patch_overlap_counter = np.zeros(
            shape=(compressed_h, compressed_w, 1),  # row-first format (H, W, C)
            dtype=np.uint8,
        )

    def update(self, data: torch.Tensor, metadata: dict) -> None:
        logger.debug("Pasting tiles.")

        # Get base tile coordinates for uncompressed accumulator
        xs_accum = metadata["coord_x"] // self.level_coord_multiplier
        ys_accum = metadata["coord_y"] // self.level_coord_multiplier
        data = self._preprocess_data(data)

        # compress overlap counter coordinates
        xs_count = xs_accum // self.gcd_size_factor
        ys_count = ys_accum // self.gcd_size_factor

        # compress accumulator coordinates if enabled
        if self.compress_accumulator_array:
            xs_accum = xs_count
            ys_accum = ys_count

        # Paste tiles onto mask
        for xa, ya, xc, yc, tile in zip(
            xs_accum, ys_accum, xs_count, ys_count, data, strict=False
        ):
            mm_h, mm_w, mm_c = self.heatmap_accumulator[
                ya : ya + self.accumulator_tile_size,
                xa : xa + self.accumulator_tile_size,
                :,
            ].shape
            self.heatmap_accumulator[
                ya : ya + self.accumulator_tile_size,
                xa : xa + self.accumulator_tile_size,
                :,
            ] += tile[:mm_h, :mm_w, :mm_c]
            self.patch_overlap_counter[
                yc : yc + self.overlap_counter_tile_size,
                xc : xc + self.overlap_counter_tile_size,
                :,
            ] += 1

    def save(self) -> str:
        # Converting to pyVips
        vips_im = pyvips.Image.new_from_array(self.heatmap_accumulator)
        count_im = pyvips.Image.new_from_array(self.patch_overlap_counter)

        if not self.compress_accumulator_array:
            # resize overlap counter to full size
            count_im = count_im.resize(
                self.w / count_im.width,
                vscale=self.h / count_im.height,
                kernel=self.interpolation,
            )

        # Resolve overlaps
        vips_im /= count_im

        vips_im = self._finalize_image(vips_im)

        # Resize to full size
        if self.compress_accumulator_array:
            vips_im = vips_im.resize(
                self.w / vips_im.width,
                vscale=self.h / vips_im.height,
                kernel=self.interpolation,
            )

        save_path = self._save_xopat_compatible(vips_im)

        return save_path
