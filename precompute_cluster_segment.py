# %%
import os
import sys

# %%
# reload module
import importlib
import logging

import hydra
import mlflow

import openslide  #IMPORT BEFORE TORCH
import pyvips  #IMPORT BEFORE TORCH

import torch
from lightning import seed_everything

from explainability.cams import (
    grad_cam,
)
from explainability.clustering import NMFClusteringMethod
from explainability.prototypes import find_top_prototypes
from explainability.visualizations import (
    visualize_prototypes_row,
)
from prostate_cancer.data import DataModule
from prostate_cancer.prostate_cancer_model import ProstateCancerModel


# importlib.reload(sys.modules["prostate_cancer.prostate_cancer_model"])

# %%
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
seed_everything(42, workers=True)
torch.set_float32_matmul_precision(precision="medium")

# %%
# Configuration overrides for prediction
overrides = ["experiment=predict/images/vgg16", "mode=predict"]

# Initialize Hydra configuration
with hydra.initialize(config_path="conf", version_base=None):
    config = hydra.compose(config_name="default", overrides=overrides)

print("Configuration loaded successfully!")
print(f"Mode: {config.mode}")
print(f"Checkpoint: {config.checkpoint}")
print(f"Batch size: {config.data.batch_size}")

# %%
# Instantiate data module and model
data: DataModule = hydra.utils.instantiate(
    config.data,
    _recursive_=False,  # to avoid instantiating all the datasets
    _target_=DataModule,
)

# %%
chkcpt_path = mlflow.artifacts.download_artifacts(config.checkpoint)
model: ProstateCancerModel = hydra.utils.instantiate(config.model)
checkpoint = torch.load(chkcpt_path, map_location="cuda:0")
model.load_state_dict(checkpoint["state_dict"], strict=True)

model = model.to("cuda:0")


print("Data module and model instantiated successfully!")
print(f"Model type: {type(model)}")
print(f"Data module type: {type(data)}")

# %%
model.eval()

# %%
model.device

# %%
target_layer = "backbone.29"

# %%
from explainability.cams.abstract import HookedModule, modify_ReLU_inplace


print(model)
hooked_model = HookedModule(model, layer_names=[target_layer])
modify_ReLU_inplace(hooked_model, inplace=False)

# %%
# Get one batch from validation dataset
data.batch_size = 10
data.setup("test")

# %%
dataloaders = data.test_dataloader()
print(type(data.test_dataloader()))


# %%
import contextlib
import math
from pathlib import Path
from tqdm.auto import tqdm
from jaxtyping import Float, Int
import torch
import numpy as np
from numpy.lib.format import open_memmap
from sklearn.decomposition import NMF

from explainability.visualizations.clusters import get_overlay_from_clustering


class MultichannelHeatmapAssembler:
    def __init__(
        self,
        heatmap_width: int,
        heatmap_height: int,
        heatmap_channels: int,
        tile_extent: int,
        step_size: int,
        npy_file_path: Path,
    ):
        self.original_heatmap_width = heatmap_width
        self.original_heatmap_height = heatmap_height
        # self.heatmap_channels = heatmap_channels

        # check that the tile_extent with the step size can tile the heatmap fully,
        # and if not adjust the heatmap size to be fully tiled
        remainder_x = (heatmap_width - tile_extent) % step_size
        remainder_y = (heatmap_height - tile_extent) % step_size
        if remainder_x != 0:
            heatmap_width += step_size - remainder_x
            print(f"Adjusted heatmap width to {heatmap_width} to fit tiles")
        if remainder_y != 0:
            heatmap_height += step_size - remainder_y
            print(f"Adjusted heatmap height to {heatmap_height} to fit tiles")

        

        self.gcd_size_factor = math.gcd(tile_extent, step_size)
        self.overlap_counter_tile_size = tile_extent // self.gcd_size_factor
        self.accumulator_tile_size = tile_extent
        # self.level_coord_multiplier = 2 ** sample_level

        # Calculate sizes for accumulator and overlap counter
        counter_w = heatmap_width // self.gcd_size_factor
        counter_h = heatmap_height // self.gcd_size_factor



        # self.heatmap_accumulator = torch.zeros(
        #     (heatmap_channels, heatmap_height, heatmap_width),  # row-first format (C, H, W)
        #     dtype=torch.float32,
        # )
        # # overlap counter is always compressed
        # self.patch_overlap_counter = torch.zeros(
        #     (1, counter_h, counter_w),  # row-first format (C, H, W)
        #     dtype=torch.uint8,
        # )
        # print("SHAPES:", self.patch_overlap_counter.shape, self.heatmap_accumulator.shape)

        
        # init using lib.format.open_memmap
        self.heatmap_accumulator = open_memmap(
            npy_file_path,
            mode="w+",
            dtype="float32",
            shape=(heatmap_channels, heatmap_height, heatmap_width),
        )

        # overlap counter should fit in memory
        self.patch_overlap_counter = np.zeros(
            (1, counter_h, counter_w),  # row-first format (C, H, W)
            dtype=np.uint8,
        )
        print("SHAPES:", self.patch_overlap_counter.shape, self.heatmap_accumulator.shape)

    def update_batch_torch(self, data: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> None:
        """Expects the data in the form (B, C, H, W)!"""
        # Get base tile coordinates for uncompressed accumulator
        # xs_accum = x // self.level_coord_multiplier
        # ys_accum = y // self.level_coord_multiplier
        # data = self._preprocess_data(data)  # unnecesary, compose raw data

        # compress overlap counter coordinates
        xs_count = xs // self.gcd_size_factor
        ys_count = ys // self.gcd_size_factor

        # # compress accumulator coordinates if enabled
        # if self.compress_accumulator_array:
        #     xs_accum = xs_count
        #     ys_accum = ys_count

        # Paste tiles onto mask
        # do not try to optimize this, not worth it. You would have to resolve issues with overlaps preventing direct broadcasting,
        # which would require a lot of extra logic and probably end up not being significantly faster anyway.
        for xa, ya, xc, yc, tile in zip(
            xs, ys, xs_count, ys_count, data, strict=True
        ):
            # mm_c, mm_h, mm_w = self.heatmap_accumulator[
            #     :,
            #     ya : ya + self.accumulator_tile_size,
            #     xa : xa + self.accumulator_tile_size,
            # ].shape
            # self.heatmap_accumulator[
            #     :,
            #     ya : ya + self.accumulator_tile_size,
            #     xa : xa + self.accumulator_tile_size,
            # ] += tile[:mm_c, :mm_h, :mm_w]
            self.heatmap_accumulator[
                :,
                ya : ya + self.accumulator_tile_size,
                xa : xa + self.accumulator_tile_size,
            ] += tile.numpy()
            self.patch_overlap_counter[
                :,
                yc : yc + self.overlap_counter_tile_size,
                xc : xc + self.overlap_counter_tile_size,
            ] += 1
    
    def finalize(self) -> torch.Tensor:
        # Normalize heatmap by patch overlap counts
        # self.heatmap_accumulator /= (self.patch_overlap_counter.clamp(min=1)  # clamp division wins over masked division
        # .repeat_interleave( self.gcd_size_factor, dim=2)
        # .repeat_interleave( self.gcd_size_factor, dim=1)
        # )

        # Normalize heatmap by patch overlap counts but do not expand the counter
        for i in range(self.gcd_size_factor):
            for j in range(self.gcd_size_factor):
                self.heatmap_accumulator[
                    :,
                    i::self.gcd_size_factor,
                    j::self.gcd_size_factor,
                ] /= self.patch_overlap_counter.clip(min=1)
        # self.heatmap_accumulator.flush()
        # Return the final heatmap, cropped to the original heatmap extent
        return self.heatmap_accumulator


def reshape_for_clustering_universal(
    embeddings: torch.Tensor | np.ndarray,
    channel_dim_index: int    
) -> torch.Tensor | np.ndarray:
    if isinstance(embeddings, np.ndarray):
        n_dims = embeddings.ndim
    elif isinstance(embeddings, torch.Tensor):
        embeddings.dim()
        
    permute_tuple = tuple(
        [i for i in range(n_dims) if i != channel_dim_index] + [channel_dim_index]
    )
    if isinstance(embeddings, np.ndarray):
        return embeddings.transpose(permute_tuple).reshape(-1, embeddings.shape[channel_dim_index])
    elif isinstance(embeddings, torch.Tensor):
        return embeddings.permute(permute_tuple).reshape(-1, embeddings.shape[channel_dim_index])


def save_image_xopat_compatible(image: torch.Tensor | np.ndarray, save_path: Path, target_extent_x: int, target_extent_y: int):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    level_size_multiplier_x = target_extent_x / image.shape[1] 
    level_size_multiplier_y = target_extent_y / image.shape[0]
    vips_im = pyvips.Image.new_from_array(image).affine(
        (level_size_multiplier_x, 0, 0, level_size_multiplier_y), interpolate=pyvips.Interpolate.new("nearest")
    ).cast("uchar")


    vips_im.tiffsave(
        save_path,
        bigtiff=True,
        compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
    )

@contextlib.contextmanager
def safe_file_op_ctxm(target_file: Path):
    """A context manager which provides you with a temporary filepath to write to, and then renames it to the target file on successful completion of the block. If an exception occurs, the temp file is deleted and the target file is left unchanged."""
    temp_file = target_file.with_suffix(".tmp")
    try:
        yield temp_file
        temp_file.rename(target_file)
    except Exception as e:
        temp_file.unlink(missing_ok=True)
        raise e


OUT_DIR = Path("/mnt/data/rationai/data/XAICNNEmbeddings/")
PRECOMPUTE_DIR = OUT_DIR / "PRECOMPUTED"
EXP_DIR = PRECOMPUTE_DIR / "VGG16_Prostate"
EXP_DIR.mkdir(exist_ok=True)

# single dataloader is single slide
_slide_pbar = tqdm(enumerate(dataloaders), desc="Slides")
for i, dataloader in _slide_pbar:
    slide_metadata = data.test.slides.iloc[i]
    # print(slide_metadata)
    # print(slide_metadata.extent_x)
    # extent_x                                                     52592
    # extent_y                                                    110886
    # tile_extent_x                                                  512
    # tile_extent_y                                                  512
    # stride_x                                                       256
    # stride_y                                                       256
    # mpp_x                                                     0.467751
    # mpp_y                                                     0.468661
    # path             /mnt/data/Projects/prostate_cancer/cancer/test...
    # level                                                            1
    # carcinoma         
    
    # level = slide_metadata.level
    slide_path = slide_metadata.path
    slide_name = Path(slide_path).stem
    _slide_pbar.set_description(f"Started processing slide {slide_name} ({i})")
    OUT_FILE_PATH_ACTS = EXP_DIR / f"activations_slide-ggregated_{i}_{slide_name}.npy"
    if OUT_FILE_PATH_ACTS.exists():
        _slide_pbar.write(f"Slide {slide_name} exists, skipping.")
        activations_assembled_wsi = np.load(OUT_FILE_PATH_ACTS, mmap_mode='r+')
        _slide_pbar.write(f"Activations shape: {activations_assembled_wsi.shape}")
    else:
    
        first_input, first_label, first_m = dataloader.dataset[0]
        hooked_model(first_input.unsqueeze(0).to("cuda:0"))
        first_activations = hooked_model.get_activations(target_layer)
        _slide_pbar.write(f"First activations shape: {first_activations.shape}")
        _, heatmap_channels, heatmap_tile_height, heatmap_tile_width = first_activations.shape

        
        slide_width = slide_metadata.extent_x  # the width of the slide at the level used for sampling 
        slide_height = slide_metadata.extent_y
        tile_width = slide_metadata.tile_extent_x
        tile_height = slide_metadata.tile_extent_y
        tile_stride_x = slide_metadata.stride_x
        tile_stride_y = slide_metadata.stride_y

        slide_to_heatmap_ratio_x = heatmap_tile_width / tile_width
        slide_to_heatmap_ratio_y = heatmap_tile_height / tile_height

        heatmap_width = int(slide_width * slide_to_heatmap_ratio_x)
        heatmap_height = int(slide_height * slide_to_heatmap_ratio_y)
        heatmap_tile_stride_x = int(tile_stride_x * slide_to_heatmap_ratio_x)
        heatmap_tile_stride_y = int(tile_stride_y * slide_to_heatmap_ratio_y)

        # assert square heatmap tiles and strides
        assert heatmap_tile_width == heatmap_tile_height
        assert heatmap_tile_stride_x == heatmap_tile_stride_y

        _slide_pbar.write(f"Tile extent and stride: {heatmap_tile_width} | {heatmap_tile_stride_x}")
        with safe_file_op_ctxm(OUT_FILE_PATH_ACTS) as act_numpy_file:
            with safe_file_op_ctxm(Path("acts_accumulator_rounded.npy")) as tmp_numpy_file:
            
                heatmap_assembler = MultichannelHeatmapAssembler(
                    heatmap_width=heatmap_width,
                    heatmap_height=heatmap_height,
                    heatmap_channels=heatmap_channels,
                    tile_extent=heatmap_tile_width,  # assuming square tiles
                    step_size=heatmap_tile_stride_x,  # assuming square tiles and equal stride
                    npy_file_path=tmp_numpy_file
                )

                for batch in tqdm(dataloader, desc="Batch"):
                    inputs, labels, metadata = batch
                    # print(metadata)  # slide, x, y
                    
                    inputs = inputs.to("cuda:0")
                    hooked_model(inputs)
                    A = hooked_model.get_activations(target_layer)
                    X = (metadata['x'] * slide_to_heatmap_ratio_x).to(torch.int64)
                    Y = (metadata['y'] * slide_to_heatmap_ratio_y).to(torch.int64)
                    heatmap_assembler.update_batch_torch(A.cpu(), X.cpu(), Y.cpu())

                activations_assembled_wsi_rounded = heatmap_assembler.finalize()
                activations_assembled_wsi_narrowed = open_memmap(
                    act_numpy_file,
                    mode="w+",
                    dtype="float32",
                    shape=(heatmap_height, heatmap_width, heatmap_channels),
                )

                # first slide activations takes up ~44 GB
                _slide_pbar.write(f"Finalized activations shape (before narrowing): {activations_assembled_wsi_rounded.shape}")
                # activations_assembled_wsi_narrowed[:] = activations_assembled_wsi_rounded.narrow(1, 0, heatmap_height).narrow(2, 0, heatmap_width).transpose(1, 2, 0)
                activations_assembled_wsi_narrowed[:] = activations_assembled_wsi_rounded[:,:heatmap_height, :heatmap_width].transpose(1, 2, 0)
                _slide_pbar.write(f"Narrowed activations shape: {activations_assembled_wsi_narrowed.shape}")
                
                activations_assembled_wsi_narrowed.flush()
                
            del activations_assembled_wsi_narrowed
            del activations_assembled_wsi_rounded
            del heatmap_assembler

        activations_assembled_wsi = np.load(OUT_FILE_PATH_ACTS, mmap_mode='r+')
        _slide_pbar.write(f"Final activations shape: {activations_assembled_wsi.shape}")
    

        # print("Heatmap shape:", heatmap.shape)
        # the shape is (C, H, W)

        # _slide_pbar.set_description(f"Saving slide {slide_name} to {OUT_FILE_PATH_ACTS}")

    # =====================================================================
    # Cluster the heatmap pixels


    num_clusters = 6
    OUT_FILE_PATH_INDS = EXP_DIR / f"cluster-indices_slide-ggregated_{i}_{slide_name}.pt"
    if OUT_FILE_PATH_INDS.exists():
        _slide_pbar.write(f"Segmentation for slide {slide_name} exists, skipping.")
        indices = torch.load(OUT_FILE_PATH_INDS)
        # print(indices.shape, indices)
        # continue
    else:

        original_shape = activations_assembled_wsi.shape
        _slide_pbar.write(f"Original shape: {original_shape}")
        embeddings = reshape_for_clustering_universal(activations_assembled_wsi, channel_dim_index=2)
        _slide_pbar.write(f"Clustering-reshaped embeddings shape: {embeddings.shape}")
        
        # nmf_clustering = NMFClusteringMethod()
        # clustering = nmf_clustering.cluster(
        #     embeddings, k=num_clusters
        # )

        # clustering = nmf_clustering.cluster(embeddings, num_clusters)

        # # B, C, H, W = 10, 512, 32, 32 #original_shape
        # W, H, C = original_shape
        # indices = clustering.inference(embeddings).reshape((1, W, H))
        clustering_model = NMF(n_components=num_clusters, init='nndsvd', random_state=42, max_iter=500)
        clustering_model.fit(embeddings)

        # open memmapped array for clustering results
        clustering_indices_memmap = open_memmap(
            OUT_FILE_PATH_INDS,
            mode="w+",
            dtype="int8",
            shape=(1, original_shape[1], original_shape[2]),
        )

        # weights = clustering_model.transform(embeddings)  # [B, k]
        # clusters = np.argmax(weights, axis=1).astype(np.int8)
        clustering_indices_memmap[:] = np.argmax(clustering_model.transform(embeddings), axis=1).astype(np.int8).reshape((1, original_shape[1], original_shape[2]))
        clustering_indices_memmap.flush()
        # torch.save(indices, OUT_FILE_PATH_INDS)



    # =====================================================================
    # Visualize the clustering results as overlay on the WSI
    OUT_FILE_PATH_SEGS = EXP_DIR / f"segmentation_slide-ggregated_{i}_{slide_name}.tiff"
    if OUT_FILE_PATH_SEGS.exists():
        _slide_pbar.write(f"Segmentation for slide {slide_name} exists, skipping.")
        continue

    overlays = get_overlay_from_clustering(indices, n_indices=num_clusters)
    overlay = overlays.squeeze(0)
    # get WSI full extent
    slide_handle = openslide.OpenSlide(slide_path)
    level_zero_extent_x, level_zero_extent_y = slide_handle.level_dimensions[0]
    

    _slide_pbar.set_description(f"Saving segmentation overlay for slide {slide_name} at level 0 dimensions {level_zero_extent_x}x{level_zero_extent_y}")
    save_image_xopat_compatible( 
        overlay.cpu(), 
        OUT_FILE_PATH_SEGS, 
        target_extent_x=level_zero_extent_x, 
        target_extent_y=level_zero_extent_y
    )


    # save overlay as tiff using pyvips


    



    



# %%
pls = np.load("acts_accumulator_rounded.npy")
print(pls.shape)

# %%
print("BYTES:", pls.nbytes)

target_f = open_memmap(
    "/mnt/data/rationai/data/XAICNNEmbeddings/PRECOMPUTED/VGG16_Prostate/activations_slide-ggregated_0_TP-2019_7207-11-0.npy",
    mode="w+",
    dtype="float32",
    shape=(6930, 3287, 512),
)
print("COPY")
target_f[:] = pls[:,:6930, :3287].transpose(1, 2, 0)
print("FLUSH")
target_f.flush()
print("DONE")

# %%



