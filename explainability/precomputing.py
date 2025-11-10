from pathlib import Path
import contextlib
import math
import numpy as np
import logging

from numpy.lib.format import open_memmap
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import torch

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
        if temp_file_path.exists():
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


class EdgeClippingMultichannelHeatmapAssembler(MultichannelHeatmapAssembler):
    """A heatmap assembler that clips edges of the input tiles before assembling them into the heatmap."""
    def __init__(self, clip_top: int, clip_bottom: int, clip_left: int, clip_right: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_top = clip_top
        self.clip_bottom = clip_bottom
        self.clip_left = clip_left
        self.clip_right = clip_right

    def update_batch_torch(self, data: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
        # Clip the edges of the input tiles
        extent_x, extent_y = data.shape[3], data.shape[2]
        data = data[..., self.clip_top:extent_y-self.clip_bottom, self.clip_left:extent_x-self.clip_right]
        xs = xs + self.clip_left
        ys = ys + self.clip_top
        super().update_batch_torch(data, xs, ys)


# # perform sanity checks to make sure the assembler works as expected
# def test_edge_clipping_heatmap_assembler():
#     heatmap_width = 256
#     heatmap_height = 128
#     heatmap_channels = 3
#     tile_extent = 32
#     clip = 12
#     npy_file_path = Path("test_edge_clipping_heatmap.npy")
#     num_batches = 16
#     B = 32

#     assembler = EdgeClippingMultiscaleHeatmapAssembler(
#         clip_top=clip,
#         clip_bottom=clip,
#         clip_left=clip,
#         clip_right=clip,
#         heatmap_width=heatmap_width,
#         heatmap_height=heatmap_height,
#         heatmap_channels=heatmap_channels,
#         heatmap_npy_fp=npy_file_path
#     )
#     # create dummy data
#     example_tile_batch = np.ones((B, heatmap_channels, tile_extent, tile_extent), dtype=np.float32)
    
#     increment = example_tile_batch[..., clip:tile_extent-clip, clip:tile_extent-clip].sum()
#     assert increment == B*heatmap_channels*(tile_extent - 2*clip)*(tile_extent - 2*clip), f"Checksum mismatch in example tile batch: Should be {B*heatmap_channels*(tile_extent - 2*clip)*(tile_extent - 2*clip)}, is {example_tile_batch[..., clip:tile_extent-clip, clip:tile_extent-clip].sum()}"
#     xs = []
#     ys = []
#     # add the tiles randomly to cover the heatmap
    
#     for i in range(num_batches):
#         x, y = (np.random.rand(B,)*(heatmap_width - (tile_extent - clip))).astype(int), (np.random.rand(B,)*(heatmap_height - (tile_extent - clip))).astype(int)
#         assembler.update_batch_torch(example_tile_batch, x, y)
#         plt.imshow((assembler.heatmap_accumulator.transpose(1, 2, 0)*32).astype(np.uint8))
#         plt.show()
#         assert assembler.heatmap_accumulator.sum() == (i+1)*increment, f"Checksum mismatch in heatmap accumulator after update {i+1}. Is {assembler.heatmap_accumulator.sum()}, but expected {(i+1)*increment}"

#     assert assembler.heatmap_accumulator.sum() == num_batches*B*(tile_extent - 2*clip)*(tile_extent - 2*clip)*heatmap_channels, f"Checksum mismatch in heatmap accumulator after updates. Is {assembler.heatmap_accumulator.sum()}, but expected {num_batches*B*(tile_extent - 2*clip)*(tile_extent - 2*clip)*heatmap_channels}"

#     assembled_heatmap, overlap_counter = assembler.finalize()
#     assert assembled_heatmap.shape == (heatmap_channels, heatmap_height, heatmap_width), f"Assembled heatmap has incorrect shape: {assembled_heatmap.shape}, shall be {(heatmap_channels, heatmap_height, heatmap_width)}"
#     assert overlap_counter.shape == (heatmap_height, heatmap_width), f"Overlap counter has incorrect shape: {overlap_counter.shape}, shall be {(heatmap_height, heatmap_width)}"
#     assert overlap_counter.sum() == num_batches*B*(tile_extent - 2*clip)*(tile_extent - 2*clip), f"Checksum mismatch in overlap counter after finalization. Is {overlap_counter.sum()}, but expected {num_batches*B*(tile_extent - 2*clip)*(tile_extent - 2*clip)}"
#     assert assembled_heatmap.max() == 1.0, f"Assembled heatmap values should be normalized to 1.0 after finalization: {assembled_heatmap.max()}"    
# test_edge_clipping_heatmap_assembler()


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



class ClusteringManager:
    """Utility class to create, load, save, and extract clustering models."""

    SUPPORTED = {"NMF", "KMeans"}

    @staticmethod
    def create_model(algorithm: str, **kwargs):
        """Return a new clustering model instance based on algorithm name."""
        if algorithm == "NMF":
            return NMF(n_components=kwargs.get("num_clusters"), init="nndsvd", random_state=42, max_iter=500)
        elif algorithm == "KMeans":
            n_clusters = kwargs.get("num_clusters")
            assert n_clusters is not None, "num_clusters must be provided for KMeans"
            return KMeans(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    @staticmethod
    def load_model(algorithm: str, path: Path, **kwargs):
        """Load clustering model from saved numpy array."""
        model = ClusteringManager.create_model(algorithm, **kwargs)
        data = np.load(path)

        if algorithm == "NMF":
            model.components_ = data
        elif algorithm == "KMeans":
            model.cluster_centers_ = data
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

        return model

    @staticmethod
    def save_model(algorithm: str, model, path: Path):
        """Save model centroids/components to a numpy file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if algorithm == "NMF":
            np.save(path, model.components_)
        elif algorithm == "KMeans":
            np.save(path, model.cluster_centers_)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    @staticmethod
    def get_components(algorithm: str, model):
        """Return the centroid/component matrix for the given model."""
        if algorithm == "NMF":
            return model.components_
        elif algorithm == "KMeans":
            return model.cluster_centers_
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
