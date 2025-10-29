from numpy.lib.format import open_memmap
from pathlib import Path
import math
import numpy as np
import torch

import matplotlib.pyplot as plt
import contextlib

import logging
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def safe_file_op_ctxm(target_file: Path, unlink_on_exception: bool = True):
    """A context manager which provides you with a temporary filepath to write to, and then renames it to the target file on successful completion of the block. If an exception occurs, the temp file is deleted and the target file is left unchanged."""
    suffix = target_file.suffix
    temp_file_path = target_file.with_suffix(f".tmp{suffix}")
    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file_path.unlink(missing_ok=True)  # Ensure temp file does not exist
    try:
        yield temp_file_path
        temp_file_path.rename(target_file)
    except Exception as e:
        if unlink_on_exception:
            temp_file_path.unlink(missing_ok=True)
            logger.debug(f"Deleted temporary file {temp_file_path} due to exception inside the managed block.")
        raise e


class MultichannelHeatmapAssembler:
    def __init__(
        self,
        heatmap_width: int,
        heatmap_height: int,
        heatmap_channels: int,
        heatmap_npy_fp: Path,
    ):
        self.npy_file_path = heatmap_npy_fp
       
        # init using lib.format.open_memmap
        self.heatmap_accumulator = open_memmap(
            heatmap_npy_fp,
            mode="w+",
            dtype="float32",
            shape=(heatmap_channels, heatmap_height, heatmap_width),
        )

        self.patch_overlap_counter = np.zeros(
            (1, heatmap_height, heatmap_width),  # row-first format (C, H, W)
            dtype=np.uint8,
        )

    def update_batch_torch(self, data: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
        """Expects the data in the form (B, C, H, W)"""
        for xa, ya, tile in zip(
            xs, ys, data, strict=True
        ):
            tile_w, tile_h = tile.shape[1], tile.shape[2]
            mm_c, mm_h, mm_w = self.heatmap_accumulator[
                :,
                ya : ya + tile_h,
                xa : xa + tile_w,
            ].shape

            self.heatmap_accumulator[
                :,
                ya : ya + tile_h,
                xa : xa + tile_w,
            ] += tile[:mm_c, :mm_h, :mm_w]

            self.patch_overlap_counter[
                :,
                ya : ya + tile_h,
                xa : xa + tile_w,
            ] += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Finalize the heatmap assembly by normalizing the accumulated heatmap by the overlap counts."""
        # Normalize heatmap by patch overlap counts but do not expand the counter
        self.heatmap_accumulator /= self.patch_overlap_counter.clip(min=1)
        self.heatmap_accumulator.flush()
        # Return the final heatmap, cropped to the original heatmap extent
        return self.heatmap_accumulator, self.patch_overlap_counter.squeeze(0)
    
# # perform sanity checks to make sure the assembler works as expected
# def test_heatmap_assembler():
#     heatmap_width = 256 + 7
#     heatmap_height = 128 + 15
#     heatmap_channels = 3
#     tile_extent = 32
#     npy_file_path = Path("test_heatmap.npy")

#     assembler = MultichannelHeatmapAssembler(
#         heatmap_width,
#         heatmap_height,
#         heatmap_channels,
#         heatmap_npy_fp=npy_file_path
#     )
#     # create dummy data
#     B = 4
#     example_tile_batch = np.ones((B, heatmap_channels, tile_extent, tile_extent), dtype=np.float32)
#     increment = example_tile_batch.sum()
#     assert increment == B*heatmap_channels*tile_extent*tile_extent, f"Checksum mismatch in example tile batch: Should be {B*heatmap_channels*tile_extent*tile_extent}, is {example_tile_batch.sum()}"
#     xs = []
#     ys = []
#     # add the tiles randomly to cover the heatmap
#     for i in range(7):
#         x, y = (np.random.rand(B,)*(heatmap_width - tile_extent)).astype(int), (np.random.rand(B,)*(heatmap_height - tile_extent)).astype(int)
#         assembler.update_batch_torch(example_tile_batch, x, y)
#         plt.imshow(assembler.heatmap_accumulator[0], cmap='hot', interpolation='nearest')
#         plt.axis('off')
#         plt.show()
#         assert assembler.heatmap_accumulator.sum() == (i+1)*increment, f"Checksum mismatch in heatmap accumulator after update {i+1}. Is {assembler.heatmap_accumulator.sum()}, but expected {(i+1)*increment}"

#     assert assembler.heatmap_accumulator.sum() == 7*B*tile_extent*tile_extent*heatmap_channels, f"Checksum mismatch in heatmap accumulator after updates. Is {assembler.heatmap_accumulator.sum()}, but expected {7*B*tile_extent*tile_extent*heatmap_channels}"

#     assembled_heatmap, overlap_counter = assembler.finalize()
#     assert assembled_heatmap.shape == (heatmap_channels, heatmap_height, heatmap_width), f"Assembled heatmap has incorrect shape: {assembled_heatmap.shape}"
#     assert overlap_counter.shape == (1, heatmap_height, heatmap_width), f"Overlap counter has incorrect shape: {overlap_counter.shape}"
#     assert overlap_counter.sum() == 7*B*tile_extent*tile_extent, f"Checksum mismatch in overlap counter after finalization. Is {overlap_counter.sum()}, but expected {7*B*tile_extent*tile_extent}"
#     assert assembled_heatmap.max() == 1.0, f"Assembled heatmap values should be normalized to 1.0 after finalization: {assembled_heatmap.max()}"

# test_heatmap_assembler()



def npy_data_offset(filename: Path):
    """Return (offset, dtype, shape, fortran_order) for a .npy file."""
    with open(filename, "rb") as f:
        # read magic number and version
        version = np.lib.format.read_magic(f)
        # read the header (this leaves the file pointer after it)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    return offset, dtype, shape, fortran_order, version


def append_data_to_a_memmap_npy_file(
    npy_file_path: Path,
    data_to_append: np.ndarray,
) -> None:
    """Append data to a memmap .npy file.

    The first dimension of the array is assumed to be the one which will increase, if the array is C-contiguous,
    otherwise the last dimension is assumed to be the one which will increase.
    Remaining dimensions must match.

    Args:
        npy_file_path (str): Path to the .npy file.
        data_to_append (np.ndarray): Data to append.
    """
    if not npy_file_path.exists():
        shape = data_to_append.shape
        dtype = data_to_append.dtype
        fortran_order = False  # default to C-order
        version = (1, 0)
        np.lib.format.write_array(
            open(npy_file_path, "wb"),
            data_to_append,
            version=version,
        )
        return

    offset, dtype, shape, fortran_order, version = npy_data_offset(npy_file_path)
    if fortran_order:
        *rest_shape_existing, N_existing = shape
        *rest_shape_appended, N_appended = data_to_append.shape
        new_shape = (*rest_shape_existing, N_existing + N_appended)
    else:
        N_existing, *rest_shape_existing = shape
        N_appended, *rest_shape_appended = data_to_append.shape
        new_shape = (N_existing + N_appended, *rest_shape_existing)

    assert rest_shape_existing == rest_shape_appended, (
        f"Shape of the data to append {data_to_append.shape} does not match the shape of existing data {shape}."
    )
    
    # Open the memmap in read-write mode
    memmap_array = np.memmap(
        filename=npy_file_path,
        mode="r+",
        dtype=dtype,
        shape=new_shape,
        offset=offset,
        order='F' if fortran_order else 'C'
    )

    # Append the new data
    if fortran_order:
        memmap_array[..., N_existing:] = data_to_append
    else:   
        memmap_array[N_existing:, ...] = data_to_append

    # Flush changes to disk
    memmap_array.flush()
    del memmap_array

    # Update the header to reflect the new shape
    np.lib.format._write_array_header(
        open(npy_file_path, "r+b"),
        dict(descr=dtype.str, fortran_order=fortran_order, shape=new_shape),
        version=version
    )
