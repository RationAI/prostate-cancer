"""Script to convert old way of handling embeddings (separate embeddings file per slide) to the new one where they are included in parquet files."""

from pathlib import Path

import hydra
import mlflow
import pandas as pd
import ray
import torch
from omegaconf import DictConfig
from ray.data import SaveMode
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ray.data import Dataset


def attach_embeddings_group(group: pd.DataFrame, embeddings_dir: Path) -> pd.DataFrame:
    assert group["path"].nunique() == 1, "Expected one unique path per group"

    slide_path = group["path"].iloc[0]
    slide_name = Path(slide_path).stem

    embeds = (
        torch.load(
            str(embeddings_dir / f"{slide_name}.pt"),
            map_location="cpu",
        )
        .cpu()
        .numpy()
    )

    if len(group) != len(embeds):
        raise ValueError(
            f"Mismatch: {len(group)} tiles vs {len(embeds)} embeddings for {slide_name}"
        )

    group = group.copy()
    group["embedding"] = embeds.tolist()
    return group


def process_and_shard_tiles(
    slides: pd.DataFrame,
    tiles: pd.DataFrame,
    output_dir: Path,
    embeddings_dir: Path,
) -> None:
    tiles_output = output_dir / "tiles"
    tiles_output.mkdir(parents=True, exist_ok=True)

    tiles_enriched = tiles.join(
        slides.set_index("id")[["path"]],
        on="slide_id",
    )
    ds: Dataset = ray.data.from_pandas(tiles_enriched)

    # batch on the level of slides to avoid opening a single embedding file multiple times
    ds = ds.groupby("slide_id").map_groups(
        attach_embeddings_group,  # type: ignore[arg-type]
        fn_kwargs={"embeddings_dir": embeddings_dir},
        batch_format="pandas",
    )

    ds = ds.drop_columns(["path"])

    ds.write_parquet(
        str(tiles_output),
        max_rows_per_file=10000,
        mode=SaveMode.OVERWRITE
    )


@with_cli_args(["+preprocessing=merge_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_path = Path(mlflow.artifacts.download_artifacts(config.data.tiles_uri_224))
    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")

    embeds_dir = Path(mlflow.artifacts.download_artifacts(config.embeddings_uri))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_output = output_dir / "slides"
    slides_output.mkdir(parents=True, exist_ok=True)
    slides_path = slides_output / "slides.parquet"
    slides.to_parquet(slides_path, index=False)  # slides.parquet is not changed

    with ray.init():  # type: ignore[call-arg]
        process_and_shard_tiles(
            slides,
            tiles,
            output_dir,
            embeds_dir,
        )

    mlflow.log_artifacts(str(output_dir), config.data.data_name + "_sharded")


if __name__ == "__main__":
    main()
