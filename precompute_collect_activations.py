from pathlib import Path
import logging
import re

from numpy.lib.format import open_memmap
from numpy import memmap
import numpy as np
from tqdm.auto import tqdm

from explainability.precomputing import append_data_to_a_memmap_npy_file, safe_file_op_ctxm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import click




# OUT_DIR = Path.cwd() / "outputs" / "embeddings"
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # SOURCE_DIR = Path("/mnt") / "data" / "rationai" / "data"/ "XAICNNEmbeddings"/ "PRECOMPUTED" / "VGG16_Prostate"
# SOURCE_DIR = Path("/mnt") / "data" / "rationai" / "data"/ "XAICNNEmbeddings"/ "VGG16_Prostate_NMF_6clusters"
# FILE_REGEX = re.compile(pattern=r"embeddings_slide-collected_(\d+)_([^\.]+).npy")  # embeddings_slide-collected_0_TP-2019_7207-11-0.npy

# with safe_file_op_ctxm(OUT_DIR / "full_dataset_embeddings_nmf_6clusters.npy", unlink_on_exception=False) as full_embeddings_dataset_fp:
#     full_dataset_embedding_collection = open_memmap(
#         filename=full_embeddings_dataset_fp,
#         mode="w+",
#         dtype=np.float32,
#         shape=(0, 512)
#     )
#     full_dataset_embedding_collection.flush()
#     del full_dataset_embedding_collection
    

#     for file_path in SOURCE_DIR.glob("embeddings_slide-collected_*.npy"):
#         logger.info(f"Processing file: {file_path}")
#         match_obj = FILE_REGEX.match(file_path.name)
#         if match_obj is None:
#             logger.warning(f"Filename {file_path.name} does not match the expected pattern. Skipping.")
#             continue
#         slide_idx, slide_name = match_obj.groups()
#         logger.info(f"Extracted slide_idx: {slide_idx}, slide_name: {slide_name}")
#         embeddings = np.load(file_path, mmap_mode='r+')
#         logger.info(f"Loaded embeddings shape: {embeddings.shape}")
#         N, F = embeddings.shape

#         append_data_to_a_memmap_npy_file(
#             npy_file_path=full_embeddings_dataset_fp,
#             data_to_append=embeddings,
#         )
        
#     logger.info("Completed processing all files.")
# full_dataset_embedding_collection = np.load(OUT_DIR / "full_dataset_embeddings_nmf_6clusters.npy", mmap_mode='r+')
# logger.info(f"Full dataset embeddings shape: {full_dataset_embedding_collection.shape}")

@click.command()
@click.option("--source-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, help="Directory containing the source .npy files.")
@click.option("--out-npy-fp", type=click.Path(file_okay=True, dir_okay=False, path_type=Path), required=True, help="File path to save the combined memmap .npy file.")
@click.option("--file-regex-string", type=str, default=None, help="Regex pattern to extract slide index and name from filenames. If None, a default pattern '.*_(\\d+)_([^\\.]+)\\.npy' is used.")
def main(source_dir: Path, out_npy_fp: Path, file_regex_string: str | None):
    """Collect embeddings from multiple .npy files into a single memmap .npy file.
    
    Args:
        source_dir (Path): Directory containing the source .npy files.
        out_dir (Path): Directory to save the combined memmap .npy file.
        file_regex_string (str | None): Regex pattern to extract slide index and name from filenames.
                                        If None, a default pattern ".*_(\\d+)_([^\\.]+)\\.npy" is used.

    Returns:
        None                                    
    """
    out_npy_fp.parent.mkdir(parents=True, exist_ok=True)

    with safe_file_op_ctxm(out_npy_fp, unlink_on_exception=False) as full_embeddings_dataset_fp:
        full_dataset_embedding_collection = open_memmap(
            filename=full_embeddings_dataset_fp,
            mode="w+",
            dtype=np.float32,
            shape=(0, 512)
        )
        full_dataset_embedding_collection.flush()
        del full_dataset_embedding_collection
        if file_regex_string is None:
            # file_regex_string = {anytext}_({number})_({slide name}).npy"
            file_regex_string = r".*_?(\d+)_([^\.]+)\.npy"  # embeddings_slide-collected_0_TP-2019_7207-11-0.npy
        file_regex = re.compile(pattern=file_regex_string)

        for file_path in source_dir.glob("embeddings_slide-collected_*.npy"):
            logger.info(f"Processing file: {file_path}")
            match_obj = file_regex.match(file_path.name)
            if match_obj is None:
                logger.warning(f"Filename {file_path.name} does not match the expected pattern. Skipping.")
                continue
            slide_idx, slide_name = match_obj.groups()
            logger.info(f"Extracted slide_idx: {slide_idx}, slide_name: {slide_name}")
            embeddings = np.load(file_path, mmap_mode='r+')
            logger.info(f"Loaded embeddings shape: {embeddings.shape}")
            N, F = embeddings.shape

            append_data_to_a_memmap_npy_file(
                npy_file_path=full_embeddings_dataset_fp,
                data_to_append=embeddings,
            )

        logger.info("Completed processing all files.")



if __name__ == "__main__":
    main()
    
