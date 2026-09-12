import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


def merge_metadata(metadata_a: pd.DataFrame, metadata_b: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two slide metadata tables.

    Arguments:
        metadata_a (pd.DataFrame): Metadata table of the first data source.
        metadata_b (pd.DataFrame): Metadata table of the second data source.

    Returns:
        pd.DataFrame: Concatenated metadata table.
    """
    merged = pd.concat([metadata_a, metadata_b], ignore_index=True)

    assert not merged["slide_path"].duplicated().any(), (
        "Data sources contain overlapping slides (duplicate `slide_path`)."
    )

    return merged


def merge_tiling(tiling_uri_a: str, tiling_uri_b: str, dataset_name: str) -> None:
    """Merge two tiling datasets (slides.parquet + tiles.parquet) into one.

    Arguments:
        tiling_uri_a (str): MLflow URI of the first tiling dataset directory.
        tiling_uri_b (str): MLflow URI of the second tiling dataset directory.
        dataset_name (str): Dataset name to log the merged tiling dataset under.
    """
    path_a = Path(mlflow.artifacts.download_artifacts(tiling_uri_a))
    path_b = Path(mlflow.artifacts.download_artifacts(tiling_uri_b))

    slides_a = pd.read_parquet(path_a / "slides.parquet")
    slides_b = pd.read_parquet(path_b / "slides.parquet")
    tiles_a = pd.read_parquet(path_a / "tiles.parquet")
    tiles_b = pd.read_parquet(path_b / "tiles.parquet")

    assert not set(slides_a["id"]) & set(slides_b["id"]), (
        f"Data sources contain colliding slide ids in {tiling_uri_a} and {tiling_uri_b}."
    )

    slides = pd.concat([slides_a, slides_b], ignore_index=True)
    tiles = pd.concat([tiles_a, tiles_b], ignore_index=True)

    # A `*_percentage` column missing from one source (e.g. an overlapper that
    # wasn't run there) becomes NaN for its rows after the concat; treat that
    # as no overlap.
    percentage_cols = [col for col in tiles.columns if col.endswith("percentage")]
    tiles[percentage_cols] = tiles[percentage_cols].fillna(0)

    save_mlflow_dataset(slides, tiles, dataset_name)


@with_cli_args(["+preprocessing=merge_data"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata_a_path = mlflow.artifacts.download_artifacts(config.data.metadata_table)
    metadata_b_path = mlflow.artifacts.download_artifacts(
        config.other_data.metadata_table
    )
    metadata_a = pd.read_csv(metadata_a_path)
    metadata_b = pd.read_csv(metadata_b_path)

    merged_metadata = merge_metadata(metadata_a, metadata_b)

    print("Merged slides:", merged_metadata["carcinoma"].value_counts())

    with tempfile.TemporaryDirectory() as tmpdir:
        merged_out = Path(tmpdir) / f"{config.output_name}.csv"
        merged_metadata.to_csv(merged_out, index=False)
        logger.log_artifact(str(merged_out))

    for field, suffix in (
        ("tiles_uri_512", "512"),
        ("tiles_uri_224", "224"),
        ("tiles_filtered_uri_512", "filtered_512"),
        ("tiles_filtered_uri_224", "filtered_224"),
    ):
        uri_a = config.data.get(field)
        uri_b = config.other_data.get(field)

        if uri_a is None or uri_b is None:
            continue

        merge_tiling(uri_a, uri_b, f"{config.output_name}_{suffix}")


if __name__ == "__main__":
    main()
