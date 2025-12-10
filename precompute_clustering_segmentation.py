# %%
from functools import partial
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.special import expit

import hydra
import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_RUN_NAME
import openslide  #IMPORT BEFORE TORCH
import pyvips  #IMPORT BEFORE TORCH

import torch
import torch.nn as nn
from lightning import seed_everything

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
import click

# from explainability.cams import grad_cam_pp_numpy, layer_cam_numpy, grad_cam_raw_numpy
from rationai.explainability.cams import grad_cam_numpy, layer_cam_numpy
from explainability.mlflow_persistence.storing import artifact_exists, ensure_mlflow_run, upload_image_if_missing
from prostate_cancer.data import DataModule
from prostate_cancer.prostate_cancer_model import ProstateCancerModel
# from explainability.precomputing import MultichannelHeatmapAssembler, EdgeClippingMultichannelHeatmapAssembler, safe_file_op_ctxm, ClusteringManager
from rationai.explainability.embedding_decomposition.clustering import ClusteringManager
from rationai.masks.mask_builders import AveragingClippingNumpyMemMapMaskBuilder 
from rationai.explainability.precomputing import safe_file_op_ctxm
from explainability.clustering.tensor_shaping import reshape_for_clustering_universal
from rationai.explainability.visualizations.clusters import get_overlay_from_clustering_numpy, plot_cluster_distance_matrix
from rationai.explainability.visualizations.image_transforms import save_image_xopat_compatible
from rationai.explainability.visualizations.color_palettes import COLOR_PALETTE_ADAM, ColorPalette
from rationai.explainability.model_probing import HookedModule, modify_ReLU_inplace


# %%
logger = logging.getLogger(__name__)


def tile_operation_to_wsi(operation, tile_size: int, inputs: dict, output_array: np.memmap):
    HEIGHT, WIDTH = output_array.shape
    for y_start in tqdm(range(0, HEIGHT, tile_size), desc="Tiled operation Y Tiles"):
        y_end = min(y_start + tile_size, HEIGHT)
        for x_start in tqdm(range(0, WIDTH, tile_size), desc="Tiled operation X Tiles", leave=False):
            x_end = min(x_start + tile_size, WIDTH)
            tile_inputs = {k: v[:, y_start:y_end, x_start:x_end] for k, v in inputs.items()}
            output_array[y_start:y_end, x_start:x_end] = operation(**tile_inputs)


CLUSTERING_ALGO_MAPPING = {
    "NMF": NMF,
    "KMeans": KMeans,
}


@click.command()
@click.option('--num-clusters', type=int, default=6, help='Number of clusters for NMF clustering.')
@click.option('--experiment-directory', type=str, help='Name of the experiment directory to store reusable results and artifacts.')
@click.option('--clustering-algorithm-name', type=click.Choice(list(CLUSTERING_ALGO_MAPPING.keys()), case_sensitive=False), default=list(CLUSTERING_ALGO_MAPPING.keys())[0], help='Clustering algorithm to use.')
@click.option('--clustering-instance-fp', type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), default=None, help='Path to precomputed clustering instance (.npy file). If provided, this instance will be used instead of fitting a new one.')
@click.option('-pdd', '--precomputed-data-directory', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=Path("/mnt/projects/explainability/XAICNNEmbeddings/"), help='Output directory for experiment results. If not provided, a default directory will be used.')
@click.option('-cut', '--cut-edge-subtiles', type=int, default=0, help='Number of subtiles to cut from each edge to avoid border artifacts.')
@click.option('-sid', '--slide-id-to-process', type=str, default=None, multiple=True, help='ID of the slide to process. If not provided, all slides will be processed.')
@click.option('--mlf-runid', type=str, default=None, help='MLflow run ID to continue experiment run.')
@click.option('--debug', is_flag=True, help='Enable debug logging.')
def main(num_clusters: int, experiment_directory: str, clustering_algorithm_name: str, clustering_instance_fp: Path | None, precomputed_data_directory: Path | None, cut_edge_subtiles: int, slide_id_to_process: list[str] | None, mlf_runid: str | None, debug: bool):


    if debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)

    clustering_algorithm = CLUSTERING_ALGO_MAPPING[clustering_algorithm_name]

    if precomputed_data_directory is None:
        raise ValueError("precomputed_data_directory must be provided.")
    PRECOMPUTED_DATA_DIR = precomputed_data_directory 
    # CLUSTERING_DIR = OUT_DIR / experiment_name
    TMP_DIR = Path("/tmp/XAI") / experiment_directory

    TMP_DIR.mkdir(exist_ok=True, parents=True)
    PRECOMPUTED_DATA_DIR.mkdir(exist_ok=True, parents=True)
    
    WSI_LEVEL_TO_MATCH_OUTPUTS_TO = 3
    NUM_CLUSTERS = num_clusters

    CUT_EDGE_SUBTILES = cut_edge_subtiles # number of subtiles to cut from each edge to avoid border artifacts

    # Set random seed for reproducibility
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision(precision="medium")

    # %%
    # Configuration overrides for prediction
    overrides = ["experiment=predict/images/vgg16", "mode=predict", "carcinoma_roi_t=0", "+stratified_filter=null"]

    # Initialize Hydra configuration
    with hydra.initialize(config_path="conf", version_base=None):
        config = hydra.compose(config_name="default", overrides=overrides)

    logger.info("Configuration loaded successfully!")
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Checkpoint: {config.checkpoint}")
    logger.info(f"Batch size: {config.data.batch_size}")
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


    logger.info("Data module and model instantiated successfully!")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Data module type: {type(data)}")

    # %%
    model.eval()


    # %%
    target_layer = "backbone.29"

    # %%
    


    print(model)
    hooked_model = HookedModule(model, layer_names=[target_layer])
    modify_ReLU_inplace(hooked_model, inplace=False)
    # get loss and backprop to get gradients
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    # %%
    # Get one batch from validation dataset
    data.batch_size = 4
    data.setup("test")

    # %%
    dataloaders = data.test_dataloader()


    # %%

    mlflow.set_tracking_uri("http://mlflow.rationai-mlflow:5000/")
    mlflow_experiment_name = "Testing"
    mlflow_client, mlflow_exp_id, mlflow_run_id = ensure_mlflow_run(mlflow_experiment_name, mlf_runid)
    logger.info(f"MLflow run ID: {mlflow_run_id}")
    # set run name
    mlflow_client.set_tag(mlflow_run_id, MLFLOW_RUN_NAME, f"ClusterSeg {clustering_algorithm_name} {NUM_CLUSTERS} clusters{' - pre-clustered' if clustering_instance_fp is not None else ''} - edgeClip {CUT_EDGE_SUBTILES} subtiles")
   

 

    # %%


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
        logger.debug("Slide metadata: %s", slide_metadata)
       
        slide_path = slide_metadata.path.replace("/mnt/data/Projects/prostate_cancer/cancer/test_data/", "/mnt/data/MOU/prostate/tile_level_annotations_test/")  # TODO: fix hardcoding
        slide_name = Path(slide_path).stem
        logger.info(f"Slides to process: {slide_id_to_process if slide_id_to_process is not None else 'All slides'}")
        _shall_be_processed = True
        if slide_id_to_process is not None:
            _shall_be_processed = False
            for sid in slide_id_to_process:
                if sid in slide_name:
                    logger.info(f"Skipping slide {slide_name} as it does not match the specified slide IDs to process.")
                    _shall_be_processed = True
                    break
        if not _shall_be_processed:
            continue
        _slide_pbar.set_description(f"Started processing slide {slide_name} ({i})")

        # =====================================================================
        # Assemble activations for the whole slide
        OUT_FILE_PATH_ACTS = PRECOMPUTED_DATA_DIR / f"activations_slide-aggregated_{i}_{slide_name}.npy"
        OUT_FILE_PATH_GRADS = PRECOMPUTED_DATA_DIR / f"gradients_slide-aggregated_{i}_{slide_name}.npy"
        OUT_FILE_PATH_ACT_OVERLAPS = OUT_FILE_PATH_ACTS.with_suffix(f".nzi{OUT_FILE_PATH_ACTS.suffix}")
        logger.debug(f"DEBUG: Initializes paths: {OUT_FILE_PATH_ACTS},\n {OUT_FILE_PATH_GRADS},\n {OUT_FILE_PATH_ACT_OVERLAPS}")

        _bool_acts_exists = OUT_FILE_PATH_ACTS.exists()
        _bool_grads_exists = OUT_FILE_PATH_GRADS.exists()
        _bool_act_overlaps_exists = OUT_FILE_PATH_ACT_OVERLAPS.exists()

        if _bool_acts_exists:
            _slide_pbar.write(f"Slide {slide_name} exists, skipping.")
            activations_assembled_wsi =     np.load(OUT_FILE_PATH_ACTS,     mmap_mode='r+')
        if _bool_grads_exists:
            _slide_pbar.write(f"Gradients for slide {slide_name} exist, skipping.")
            gradients_assembled_wsi = np.load(OUT_FILE_PATH_GRADS, mmap_mode='r+')
        if _bool_act_overlaps_exists:
            _slide_pbar.write(f"Activation overlaps for slide {slide_name} exist, skipping.")
            activations_assembled_wsi_overlaps = np.load(OUT_FILE_PATH_ACT_OVERLAPS, mmap_mode='r+')
    
        if not (_bool_acts_exists and _bool_grads_exists and _bool_act_overlaps_exists):
            logger.debug("Starting assembly of activations and gradients.")
            first_input, first_label, first_m = dataloader.dataset[0]
            acts_ = hooked_model(first_input.unsqueeze(0).to("cuda:0"))
            if not _bool_grads_exists:
                loss = loss_fn(acts_, first_label.unsqueeze(0).to("cuda:0").float())
                loss.backward()

            slide_width = slide_metadata.extent_x  # the width of the slide at the level used for sampling
            slide_height = slide_metadata.extent_y
            tile_width = slide_metadata.tile_extent_x
            tile_height = slide_metadata.tile_extent_y
            tile_stride_x = slide_metadata.stride_x
            tile_stride_y = slide_metadata.stride_y

            if not _bool_acts_exists:
                first_activations = hooked_model.get_activations(target_layer)
                logger.debug(f"First activations shape: {first_activations.shape}")
                _, heatmap_channels, act_heatmap_tile_height, act_heatmap_tile_width = first_activations.shape
                slide_to_heatmap_ratio_x = act_heatmap_tile_width / tile_width
                slide_to_heatmap_ratio_y = act_heatmap_tile_height / tile_height

                act_heatmap_width = int(slide_width * slide_to_heatmap_ratio_x)
                act_heatmap_height = int(slide_height * slide_to_heatmap_ratio_y)
                act_heatmap_tile_stride_x = int(tile_stride_x * slide_to_heatmap_ratio_x)
                act_heatmap_tile_stride_y = int(tile_stride_y * slide_to_heatmap_ratio_y)

                # assert square heatmap tiles and strides
                assert act_heatmap_tile_width == act_heatmap_tile_height
                assert act_heatmap_tile_stride_x == act_heatmap_tile_stride_y
                logger.debug(f"Heatmap width and height: {act_heatmap_width} | {act_heatmap_height}")  

            if not _bool_grads_exists:
                first_gradients = hooked_model.get_gradients(target_layer)
                logger.debug(f"First gradients shape: {first_gradients.shape}")
                _, gradient_channels, gradient_tile_height, gradient_tile_width = first_gradients.shape
                slide_to_heatmap_ratio_x = gradient_tile_width / tile_width
                slide_to_heatmap_ratio_y = gradient_tile_height / tile_height

                grad_heatmap_width = int(slide_width * slide_to_heatmap_ratio_x)
                grad_heatmap_height = int(slide_height * slide_to_heatmap_ratio_y)
                grad_heatmap_tile_stride_x = int(tile_stride_x * slide_to_heatmap_ratio_x)
                grad_heatmap_tile_stride_y = int(tile_stride_y * slide_to_heatmap_ratio_y)
                # assert square heatmap tiles and strides
                assert gradient_tile_width == gradient_tile_height
                assert grad_heatmap_tile_stride_x == grad_heatmap_tile_stride_y
                logger.debug(f"Gradient heatmap width and height: {grad_heatmap_width} | {grad_heatmap_height}")

            logger.debug("Starting assembly of activations and gradients.")
            with safe_file_op_ctxm(OUT_FILE_PATH_ACTS) as act_numpy_file, safe_file_op_ctxm(OUT_FILE_PATH_GRADS) as grad_numpy_file:
                if not _bool_acts_exists:
                    heatmap_assembler = AveragingClippingNumpyMemMapMaskBuilder(
                        clip_top=CUT_EDGE_SUBTILES,
                        clip_bottom=CUT_EDGE_SUBTILES,
                        clip_left=CUT_EDGE_SUBTILES,
                        clip_right=CUT_EDGE_SUBTILES,
                        spatial_dims=(act_heatmap_height, act_heatmap_width),
                        channels=heatmap_channels,
                        filepath=act_numpy_file
                    )
                if not _bool_grads_exists:
                    gradient_assembler = AveragingClippingNumpyMemMapMaskBuilder(
                        clip_top=CUT_EDGE_SUBTILES,
                        clip_bottom=CUT_EDGE_SUBTILES,
                        clip_left=CUT_EDGE_SUBTILES,
                        clip_right=CUT_EDGE_SUBTILES,
                        spatial_dims=(grad_heatmap_height, grad_heatmap_width),
                        channels=gradient_channels,
                        filepath=grad_numpy_file
                    )
                logger.debug("Beginning batch processing.")
                for batch in tqdm(dataloader, desc="Batch"):
                    inputs, labels, metadata = batch
                    logger.debug(f"Processing new batch. SHAPE: {inputs.shape}")
                    # print(metadata)  # slide, x, y
                    X = (metadata['x'] * slide_to_heatmap_ratio_x).to(torch.int64)
                    Y = (metadata['y'] * slide_to_heatmap_ratio_y).to(torch.int64)
                    
                    inputs = inputs.to("cuda:0")
                    outputs_ = hooked_model(inputs)
                    if not _bool_grads_exists:
                        loss = loss_fn(outputs_, labels.to("cuda:0").float())
                        loss.backward()
                        G = hooked_model.get_gradients(target_layer)
                        # gradient_assembler.update_batch_torch(G.cpu().numpy(), X.cpu().numpy(), Y.cpu().numpy())
                        gradient_assembler.update_batch(data_batch=G.cpu(), coords_batch=(Y.cpu().numpy(), X.cpu().numpy()))
                        del G, loss
                    if not _bool_acts_exists:
                        A = hooked_model.get_activations(target_layer)
                        # heatmap_assembler.update_batch_torch(A.cpu().numpy(), X.cpu().numpy(), Y.cpu().numpy())
                        heatmap_assembler.update_batch(data_batch=A.cpu(), coords_batch=(Y.cpu().numpy(), X.cpu().numpy()))
                        del A
                    del inputs, outputs_

                if not _bool_acts_exists:
                    activations_assembled_wsi, activations_assembled_wsi_overlaps = heatmap_assembler.finalize()
                    _slide_pbar.write(f"Saved activations to {OUT_FILE_PATH_ACTS}")
                    np.save(OUT_FILE_PATH_ACT_OVERLAPS, activations_assembled_wsi_overlaps)
                if not _bool_grads_exists:
                    gradients_assembled_wsi, _ = gradient_assembler.finalize()
                    _slide_pbar.write(f"Saved gradients to {OUT_FILE_PATH_GRADS}")
                    # np.save(OUT_FILE_PATH_GRADS_OVERLAPS, gradients_assembled_wsi_overlaps)  # NO NEED, same as activations overlaps
        # OUTPUTS: activations_assembled_wsi, activations_assembled_wsi_overlaps, gradients_assembled_wsi, gradients_assembled_wsi_overlaps


        # =====================================================================
        # Create XAI masks from activations and gradients
        OUT_FILE_PATH_XAI_GRADCAM = TMP_DIR / "xai-gradcam" / f"{slide_name}.npy"
        OUT_FILE_PATH_XAI_GRADCAM_TIFF = TMP_DIR / "xai-gradcam" / f"{slide_name}.tiff"
        if OUT_FILE_PATH_XAI_GRADCAM.exists():
            _slide_pbar.write(f"Grad-CAM for slide {slide_name} exist, skipping.")
            xai_gradcam_assembled_wsi = np.load(OUT_FILE_PATH_XAI_GRADCAM, mmap_mode='r')
            logger.debug(f"Grad-CAM has shape: {xai_gradcam_assembled_wsi.shape}")

        else:
            with safe_file_op_ctxm(OUT_FILE_PATH_XAI_GRADCAM, unlink_on_exception=True) as xai_gradcam_numpy_file:
                logger.debug(f"DEBUG: Shape of activations: {activations_assembled_wsi.shape}, Shape of gradients: {gradients_assembled_wsi.shape}")
                xai_gradcam_assembled_wsi = open_memmap(
                    xai_gradcam_numpy_file,
                    mode='w+',
                    shape=activations_assembled_wsi.shape[1:3]
                )
                tile_operation_to_wsi(
                    operation=grad_cam_numpy,
                    tile_size=2048,
                    inputs={
                        "activations": activations_assembled_wsi,
                        "gradients": gradients_assembled_wsi,
                    },
                    output_array=xai_gradcam_assembled_wsi
                )
                xai_gradcam_assembled_wsi = normalize_highlight_heatmap(xai_gradcam_assembled_wsi)
                xai_gradcam_assembled_wsi.flush()
                _slide_pbar.write(f"Saved Grad-CAM to {OUT_FILE_PATH_XAI_GRADCAM} with shape {xai_gradcam_assembled_wsi.shape}")
            
        if not artifact_exists(
            client=mlflow_client,
            run_id=mlflow_run_id,
            artifact_path=f"xai/gradcam/{slide_name}.tiff",
            file_name=OUT_FILE_PATH_XAI_GRADCAM_TIFF.name
        ):
            # save image tiff mask
            with safe_file_op_ctxm(OUT_FILE_PATH_XAI_GRADCAM_TIFF, unlink_on_exception=True) as xai_gradcam_tiff_file:
                save_image_xopat_compatible(
                    image=xai_gradcam_assembled_wsi,
                    save_path=xai_gradcam_tiff_file,
                    target_extent_x=slide_metadata.extent_x,
                    target_extent_y=slide_metadata.extent_y,
                    microns_per_pixel_x=slide_metadata.mpp_x,
                    microns_per_pixel_y=slide_metadata.mpp_y,
                )
        # upload to mlflow
        upload_image_if_missing(
            client=mlflow_client,
            run_id=mlflow_run_id,
            local_image_path=OUT_FILE_PATH_XAI_GRADCAM_TIFF,
            artifact_subdir="xai/gradcam"
        )
        _slide_pbar.write(f"Ensured Grad-CAM TIFF is uploaded to MLflow.")
    
        
        # OUT_FILE_PATH_XAI_LAYERCAM = PRECOMPUTED_DATA_DIR / "xai-layercam" / f"{slide_name}.npy"
        # OUT_FILE_PATH_XAI_LAYERCAM_TIFF = TMP_DIR / "xai-layercam" / f"{slide_name}.tiff"
        # if OUT_FILE_PATH_XAI_LAYERCAM.exists():
        #     _slide_pbar.write(f"Layer-CAM for slide {slide_name} exist, skipping.")
        #     xai_layercam_assembled_wsi = np.load(OUT_FILE_PATH_XAI_LAYERCAM, mmap_mode='r')
        #     _slide_pbar.write(f"Layer-CAM has shape: {xai_layercam_assembled_wsi.shape}")
        # else:
        #     _slide_pbar.write(f"Computing Layer-CAM for slide {slide_name}...")
        #     with safe_file_op_ctxm(OUT_FILE_PATH_XAI_LAYERCAM, unlink_on_exception=True) as xai_layercam_numpy_file:
        #         xai_layercam_assembled_wsi = open_memmap(
        #             xai_layercam_numpy_file,
        #             mode='w+',
        #             shape=activations_assembled_wsi.shape[1:3]
        #         )
        #         tile_operation_to_wsi(
        #             operation=layer_cam_numpy,
        #             tile_size=2048,
        #             inputs={
        #                 "activations": activations_assembled_wsi,
        #                 "gradients": gradients_assembled_wsi,
        #             },
        #             output_array=xai_layercam_assembled_wsi
        #         )
        #         xai_layercam_assembled_wsi = normalize_highlight_heatmap(xai_layercam_assembled_wsi)
        #         xai_layercam_assembled_wsi.flush()
        #         _slide_pbar.write(f"Saved Layer-CAM to {OUT_FILE_PATH_XAI_LAYERCAM} with shape {xai_layercam_assembled_wsi.shape}")
        # if not artifact_exists(
        #     client=mlflow_client,   
        #     run_id=mlflow_run_id,
        #     artifact_path=f"xai/layercam/{slide_name}.tiff",
        #     file_name=OUT_FILE_PATH_XAI_LAYERCAM_TIFF.name
        # ):
        #     # save image tiff mask
        #     with safe_file_op_ctxm(OUT_FILE_PATH_XAI_LAYERCAM_TIFF, unlink_on_exception=True) as xai_layercam_tiff_file:
        #         save_image_xopat_compatible(
        #             image=xai_layercam_assembled_wsi,
        #             save_path=xai_layercam_tiff_file,
        #             target_extent_x=slide_metadata.extent_x,
        #             target_extent_y=slide_metadata.extent_y,
        #             microns_per_pixel_x=slide_metadata.mpp_x,
        #             microns_per_pixel_y=slide_metadata.mpp_y,
        #         )
        # # upload to mlflow
        # upload_image_if_missing(
        #     client=mlflow_client,
        #     run_id=mlflow_run_id,
        #     local_image_path=OUT_FILE_PATH_XAI_LAYERCAM_TIFF,
        #     artifact_subdir="xai/layercam"
        # )
        # _slide_pbar.write(f"Ensured Layer-CAM TIFF is uploaded to MLflow.")

        # OUT_FILE_PATH_XAI_GRADCAMRAW = PRECOMPUTED_DATA_DIR / "xai-gradcamraw" / f"{slide_name}.npy"
        # OUT_FILE_PATH_XAI_GRADCAMRAW_TIFF = TMP_DIR / "xai-gradcamraw" / f"{slide_name}.tiff"
        # if OUT_FILE_PATH_XAI_GRADCAMRAW.exists():
        #     _slide_pbar.write(f"Raw Grad-CAM for slide {slide_name} exist, skipping.")
        #     xai_gradcamraw_assembled_wsi = np.load(OUT_FILE_PATH_XAI_GRADCAMRAW, mmap_mode='r')
        #     _slide_pbar.write(f"Raw Grad-CAM has shape: {xai_gradcamraw_assembled_wsi.shape}")
        # else:
        #     with safe_file_op_ctxm(OUT_FILE_PATH_XAI_GRADCAMRAW, unlink_on_exception=True) as xai_gradcamraw_numpy_file:
        #         xai_gradcamraw_assembled_wsi = open_memmap(
        #             xai_gradcamraw_numpy_file,
        #             mode='w+',
        #             shape=activations_assembled_wsi.shape[1:3]
        #         )
        #         print("DEBUG: Computing Raw Grad-CAM...", flush=True)
        #         tile_operation_to_wsi(
        #             operation=grad_cam_raw_numpy,
        #             tile_size=2048,
        #             inputs={
        #                 "activations": activations_assembled_wsi,
        #                 "gradients": gradients_assembled_wsi,
        #             },
        #             output_array=xai_gradcamraw_assembled_wsi
        #         )
        #         xai_gradcamraw_assembled_wsi = normalize_highlight_heatmap(xai_gradcamraw_assembled_wsi)
        #         xai_gradcamraw_assembled_wsi.flush()
        #         _slide_pbar.write(f"Saved Raw Grad-CAM to {OUT_FILE_PATH_XAI_GRADCAMRAW} with shape {xai_gradcamraw_assembled_wsi.shape}")
        # if not artifact_exists(
        #     client=mlflow_client,
        #     run_id=mlflow_run_id,
        #     artifact_path=f"xai/gradcamraw/{slide_name}.tiff",
        #     file_name=OUT_FILE_PATH_XAI_GRADCAMRAW_TIFF.name
        # ):
        #     # save image tiff mask
        #     with safe_file_op_ctxm(OUT_FILE_PATH_XAI_GRADCAMRAW_TIFF, unlink_on_exception=True) as xai_gradcamraw_tiff_file:
        #         save_image_xopat_compatible(
        #             image=xai_gradcamraw_assembled_wsi,
        #             save_path=xai_gradcamraw_tiff_file,
        #             target_extent_x=slide_metadata.extent_x,
        #             target_extent_y=slide_metadata.extent_y,
        #             microns_per_pixel_x=slide_metadata.mpp_x,
        #             microns_per_pixel_y=slide_metadata.mpp_y,
        #         )
        # # upload to mlflow
        # upload_image_if_missing(
        #     client=mlflow_client,
        #     run_id=mlflow_run_id,
        #     local_image_path=OUT_FILE_PATH_XAI_GRADCAMRAW_TIFF,
        #     artifact_subdir="xai/gradcamraw"
        # )

        # =====================================================================
        # Gather embeddings for clustering
        OUT_FILE_PATH_EMBEDDINGS = PRECOMPUTED_DATA_DIR / f"embeddings_slide-collected_{i}_{slide_name}.npy"
        if OUT_FILE_PATH_EMBEDDINGS.exists():
            _slide_pbar.write(f"Embeddings for slide {slide_name} exist, skipping.")
            embeddings = np.load(OUT_FILE_PATH_EMBEDDINGS, mmap_mode='r+')
            _slide_pbar.write(f"Embeddings have shape: {embeddings.shape}")
        else:
            overlaps_nzi = activations_assembled_wsi_overlaps > 0
            embeddings = reshape_for_clustering_universal(activations_assembled_wsi[:, overlaps_nzi], channel_dim_index=0)
            _slide_pbar.write(f"Clustering-reshaped embeddings shape: {embeddings.shape}")
            with safe_file_op_ctxm(OUT_FILE_PATH_EMBEDDINGS, unlink_on_exception=True) as emb_numpy_file:
                np.save(emb_numpy_file, embeddings)

        # =====================================================================
        # Perform clustering or load the existing instance
        if clustering_instance_fp is not None:
            _slide_pbar.write(f"Loading precomputed clustering instance from {clustering_instance_fp}")
            clustering_model = ClusteringManager.load_model(
                model_class=clustering_algorithm,
                path=clustering_instance_fp,
                num_clusters=NUM_CLUSTERS,
            )
            _slide_pbar.write(f"Clustering model loaded from precomputed instance.")
        else:
            OUT_FILE_PATH_CLUSTERING_INSTANCE = TMP_DIR / f"clustering-instance_{clustering_algorithm_name}_{i}_{slide_name}.npy"
            if OUT_FILE_PATH_CLUSTERING_INSTANCE.exists():
                _slide_pbar.write(f"Clustering instance for slide {slide_name} exist, skipping.")
                clustering_model = ClusteringManager.load_model(
                    model_class=clustering_algorithm,
                    path=OUT_FILE_PATH_CLUSTERING_INSTANCE,
                    num_clusters=NUM_CLUSTERS,
                )
                _slide_pbar.write(f"Clustering model loaded.")
            else:
                clustering_model = ClusteringManager.create_model(
                    model_class=clustering_algorithm,
                    num_clusters=NUM_CLUSTERS
                )
                _slide_pbar.write(f"Fitting clustering model ({clustering_algorithm_name}) for slide {slide_name} with embeddings shape {embeddings.shape}")

                clustering_model.fit(embeddings)
                _slide_pbar.write(f"Clustering model fitted.")
                with safe_file_op_ctxm(OUT_FILE_PATH_CLUSTERING_INSTANCE, unlink_on_exception=True) as cluster_instance_numpy_file:
                    ClusteringManager.save_model(
                        model_instance=clustering_model,
                        path=cluster_instance_numpy_file
                    )
                _slide_pbar.write(f"Clustering model instance saved to {OUT_FILE_PATH_CLUSTERING_INSTANCE}")

        # =====================================================================
        # Visualise the clusters characteristics to asses quality and centroid distances
        if clustering_instance_fp is None:
            OUT_FILE_PATH_CLUSTERING_VISUALS = TMP_DIR / f"clustering-visuals_{clustering_algorithm_name}_{i}_{slide_name}.svg"
            if OUT_FILE_PATH_CLUSTERING_VISUALS.exists():
                _slide_pbar.write(f"Clustering visuals for slide {slide_name} exist, skipping.")
            else:
                components = ClusteringManager.get_components(
                    model_instance=clustering_model,
                )
                
                distance_matrix = np.linalg.norm(components[:, np.newaxis] - components[np.newaxis, :], axis=2)
                color_lut = ColorPalette(palette=COLOR_PALETTE_ADAM[1:]).get_rgb_lut()
                plot_cluster_distance_matrix(
                    distance_matrix=distance_matrix,
                    cluster_colors=color_lut,
                    figure_fp=OUT_FILE_PATH_CLUSTERING_VISUALS
                )
            

        # =====================================================================
        # Assign cluster indices to each subunit in the WSI 

        OUT_FILE_PATH_CLUSTER_SOFT_ASSIGNMENTS = TMP_DIR / f"cluster-soft-assignments_{clustering_algorithm_name}_{NUM_CLUSTERS}clusters_{i}_{slide_name}.npy"
        if OUT_FILE_PATH_CLUSTER_SOFT_ASSIGNMENTS.exists():
            _slide_pbar.write(f"Soft cluster assignments for slide {slide_name} exist, skipping.")
            cluster_soft_assignments_memmap = np.load(OUT_FILE_PATH_CLUSTER_SOFT_ASSIGNMENTS, mmap_mode='r')
            _slide_pbar.write(f"Soft assignments have shape: {cluster_soft_assignments_memmap.shape}")
        else:
            original_shape = activations_assembled_wsi.shape  # shall be (C, H, W)
            overlaps_nzi = activations_assembled_wsi_overlaps > 0
            with safe_file_op_ctxm(OUT_FILE_PATH_CLUSTER_SOFT_ASSIGNMENTS, unlink_on_exception=True) as soft_assigns_numpy_file:
                # open memmapped array for soft clustering results
                cluster_soft_assignments_memmap = open_memmap(
                    soft_assigns_numpy_file,
                    mode="w+",
                    dtype="float32",
                    shape=(original_shape[1], original_shape[2], NUM_CLUSTERS),
                )
        
                cluster_soft_assignments_memmap[overlaps_nzi, :] = clustering_model.transform(embeddings).astype(np.float32)
                cluster_soft_assignments_memmap.flush()
                _slide_pbar.write(f"Saved soft assignments with shape: {cluster_soft_assignments_memmap.shape}")


        OUT_FILE_PATH_CLUSTER_HARD_ASSIGNMENTS = TMP_DIR / f"cluster-indices_slide-aggregated_{clustering_algorithm_name}_{i}_{slide_name}.npy"
        if OUT_FILE_PATH_CLUSTER_HARD_ASSIGNMENTS.exists():
            _slide_pbar.write(f"Clustering indices for slide {slide_name} exist, skipping.")
            cluster_hard_assignments_memmap = np.load(OUT_FILE_PATH_CLUSTER_HARD_ASSIGNMENTS, mmap_mode='r+')
            _slide_pbar.write(f"Clustering indices have shape: {cluster_hard_assignments_memmap.shape}")
            
        else:
            original_shape = activations_assembled_wsi.shape  # shall be (C, H, W)
            overlaps_nzi = activations_assembled_wsi_overlaps > 0
            _slide_pbar.write(f"Original shape: {original_shape}, (non-zero indices {overlaps_nzi.shape})")
            with safe_file_op_ctxm(OUT_FILE_PATH_CLUSTER_HARD_ASSIGNMENTS, unlink_on_exception=True) as cluster_inds_numpy_file:
                # open memmapped array for argmaxxed clustering results
                cluster_hard_assignments_memmap = open_memmap(
                    cluster_inds_numpy_file,
                    mode="w+",
                    dtype="int8",
                    shape=(original_shape[1], original_shape[2]),
                )

        
                cluster_hard_assignments_memmap[overlaps_nzi] = (
                    np.argmax(cluster_soft_assignments_memmap[overlaps_nzi], axis=1)
                    .astype(np.int8)
                    + 1
                )
                cluster_hard_assignments_memmap.flush()



        # =====================================================================
        # Visualize the clustering results as overlay on the WSI
        OUT_FILE_PATH_INDS_GRAYSCALE = TMP_DIR / f"clustering_gray_{clustering_algorithm_name}"  / f"{slide_name}.tiff"
        # OUT_FILE_PATH_SEGS = CLUSTERING_DIR           / f"clustering_color_{clustering_algorithm_name}" / f"{slide_name}.tiff"
        _slide_pbar.write(f"Preparing segmentation {OUT_FILE_PATH_INDS_GRAYSCALE} for slide {slide_name}")
        if artifact_exists(mlflow_client, mlflow_run_id, "clustering_images", OUT_FILE_PATH_INDS_GRAYSCALE.name):
            _slide_pbar.write(f"Grayscale segmentation for slide {slide_name} exists in MLflow, skipping.")

        elif OUT_FILE_PATH_INDS_GRAYSCALE.exists():
            _slide_pbar.write(f"Segmentation for slide {slide_name} exists, skipping.")
            upload_image_if_missing(
                client=mlflow_client,
                run_id=mlflow_run_id,
                local_image_path=OUT_FILE_PATH_INDS_GRAYSCALE,
                artifact_subdir="clustering_images"
            )
            _slide_pbar.write(f"Ensured upload of grayscale segmentation for {slide_name} to MLflow!")
        
        else:    
            # get WSI full extent at a specific level
            slide_handle = openslide.OpenSlide(slide_path)
            level_extent_x, level_extent_y = slide_handle.level_dimensions[WSI_LEVEL_TO_MATCH_OUTPUTS_TO]
            
            with safe_file_op_ctxm(OUT_FILE_PATH_INDS_GRAYSCALE, unlink_on_exception=True) as inds_gray_numpy_file:
                overlay = (cluster_hard_assignments_memmap.astype(np.float32) / (cluster_hard_assignments_memmap.max())) * 255.0
                overlay = overlay.astype(np.uint8)

                _slide_pbar.write(f"Saving grayscale indices overlay for slide {slide_name} at level {WSI_LEVEL_TO_MATCH_OUTPUTS_TO} dimensions {level_extent_x}x{level_extent_y}")
                save_image_xopat_compatible( 
                    overlay, 
                    inds_gray_numpy_file, 
                    target_extent_x=level_extent_x, 
                    target_extent_y=level_extent_y,
                    microns_per_pixel_x=slide_metadata.mpp_x,
                    microns_per_pixel_y=slide_metadata.mpp_y,
                )
                _slide_pbar.write(f"Done saving grayscale indices for {slide_name}!")
            upload_image_if_missing(
                client=mlflow_client,
                run_id=mlflow_run_id,
                local_image_path=OUT_FILE_PATH_INDS_GRAYSCALE,
                artifact_subdir="clustering_images"
            )
            _slide_pbar.write(f"Ensured upload of grayscale segmentation for {slide_name} to MLflow!")


        SINGLE_OVERLAYS_DIR = TMP_DIR / f"single_cluster_overlays_{clustering_algorithm_name}_{NUM_CLUSTERS}clusters"
        # with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        tasks_futures = []
        for cluster_idx in range(NUM_CLUSTERS):
            OUT_FILE_PATH_SINGLE_CLUSTER_OVERLAY = SINGLE_OVERLAYS_DIR / str(cluster_idx) / f"{slide_name}.tiff"
            if artifact_exists(mlflow_client, mlflow_run_id, "clustering_single_cluster_overlays", OUT_FILE_PATH_SINGLE_CLUSTER_OVERLAY.name):
                _slide_pbar.write(f"Single cluster overlay for cluster {cluster_idx} for slide {slide_name} exists in MLflow, skipping.")
                continue
            elif OUT_FILE_PATH_SINGLE_CLUSTER_OVERLAY.exists():
                _slide_pbar.write(f"Single cluster overlay for cluster {cluster_idx} for slide {slide_name} exists, checking if uploaded to ML Flow.")
                upload_image_if_missing(
                    client=mlflow_client,
                    run_id=mlflow_run_id,
                    local_image_path=OUT_FILE_PATH_SINGLE_CLUSTER_OVERLAY,
                    artifact_subdir=f"clustering_single_cluster_overlays/{cluster_idx}"
                )
                _slide_pbar.write(f"Ensured upload of single cluster overlay for cluster {cluster_idx} for {slide_name} to MLflow!")
                continue
            else:
                # get WSI full extent at a specific level
                slide_handle = openslide.OpenSlide(slide_path)
                level_extent_x, level_extent_y = slide_handle.level_dimensions[WSI_LEVEL_TO_MATCH_OUTPUTS_TO]
                
                parrallel_save_upload_overlay_tiff_single_cluster(
                    WSI_LEVEL_TO_MATCH_OUTPUTS_TO, 
                    mlflow_client, 
                    mlflow_run_id, 
                    _slide_pbar, 
                    slide_metadata, 
                    slide_name,
                    cluster_soft_assignments_memmap, 
                    level_extent_x, 
                    level_extent_y, 
                    cluster_idx, 
                    OUT_FILE_PATH_SINGLE_CLUSTER_OVERLAY)
        _slide_pbar.write(f"Finished processing slide {slide_name} ({i})")

    # finish the mlflow run
    mlflow_client.set_terminated(mlflow_run_id)

def normalize_highlight_heatmap(xai_layercam_assembled_wsi):
    np.power(xai_layercam_assembled_wsi, 1./3., out=xai_layercam_assembled_wsi)   # normalize between 0 and 1
    _max = np.max(np.abs(xai_layercam_assembled_wsi))
    logger.debug(f"Heatmap min/max before normalization: {xai_layercam_assembled_wsi.min()}/{xai_layercam_assembled_wsi.max()}")
    xai_layercam_assembled_wsi /= (_max * 2.0)
    xai_layercam_assembled_wsi += 0.5
    np.clip(xai_layercam_assembled_wsi, 0.0, 1.0, out=xai_layercam_assembled_wsi)
    logger.debug(f"Heatmap min/max after normalization: {xai_layercam_assembled_wsi.min()}/{xai_layercam_assembled_wsi.max()}")
    xai_layercam_assembled_wsi *= 255.0
    return xai_layercam_assembled_wsi

def parrallel_save_upload_overlay_tiff_single_cluster(WSI_LEVEL_TO_MATCH_OUTPUTS_TO, mlflow_client, mlflow_run_id, _slide_pbar, slide_metadata, slide_name, cluster_soft_assignments_memmap, level_extent_x, level_extent_y, cluster_idx, OUT_FILE_PATH_SINGLE_CLUSTER_OVERLAY):
    with safe_file_op_ctxm(OUT_FILE_PATH_SINGLE_CLUSTER_OVERLAY, unlink_on_exception=True) as single_cluster_overlay_numpy_file:
                    # save overlay as tiff
        _slide_pbar.write(f"Saving single cluster overlay for cluster {cluster_idx} for slide {slide_name} at level {WSI_LEVEL_TO_MATCH_OUTPUTS_TO} dimensions {level_extent_x}x{level_extent_y}")
        save_image_xopat_compatible( 
                        np.power(cluster_soft_assignments_memmap[:, :, cluster_idx], 1./3.).clip(0., 1.)*255.0, 
                        single_cluster_overlay_numpy_file, 
                        target_extent_x=level_extent_x, 
                        target_extent_y=level_extent_y,
                        microns_per_pixel_x=slide_metadata.mpp_x,
                        microns_per_pixel_y=slide_metadata.mpp_y,
                    )
        _slide_pbar.write(f"Done saving single cluster overlay for cluster {cluster_idx} for {slide_name}!")
    upload_image_if_missing(
                    client=mlflow_client,
                    run_id=mlflow_run_id,
                    local_image_path=OUT_FILE_PATH_SINGLE_CLUSTER_OVERLAY,
                    artifact_subdir=f"clustering_single_cluster_overlays/{cluster_idx}"
                )
    _slide_pbar.write(f"Ensured upload of single cluster overlay for cluster {cluster_idx} for {slide_name} to MLflow!")

if __name__ == "__main__":
    main()
