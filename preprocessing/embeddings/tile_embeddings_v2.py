"""Serves the same purpose as tile_embeddings.py, but uses new approach and outputs different format. Was inspired by Adam Kukučka."""

import shutil
from pathlib import Path
from typing import Any

import httpx
import hydra
import mlflow.artifacts
import pandas as pd
import pyarrow as pa
import ray
from omegaconf import DictConfig
from rationai import AsyncClient  # type: ignore[attr-defined]
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling.read_slide_tiles import read_slide_tiles
from ray.data.expressions import col


class EmbedTiles:
    def __init__(self, encoder: str, concurrency: int) -> None:
        self.encoder = encoder
        self.client = AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency, max_keepalive_connections=concurrency
            ),
            timeout=200,
        )

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        embedding = (
            (await self.client.models.embed_image(self.encoder, row["tile"]))
            .reshape(-1)
            .tolist()
        )
        del row["tile"]
        row["embedding"] = embedding
        return row


@with_cli_args(["+preprocessing=tile_embeddings_v2"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    folder = Path(
        mlflow.artifacts.download_artifacts(config.data.tiles_filtered_uri_224)
    )
    slides = pd.read_parquet(folder / "slides.parquet")
    tiles = pd.read_parquet(folder / "tiles.parquet")

    slide_info = slides.set_index("id")[
        ["path", "level", "tile_extent_x", "tile_extent_y"]
    ]
    tiles_enriched = tiles.join(slide_info, on="slide_id")

    ds = ray.data.from_arrow(
        pa.Table.from_pandas(tiles_enriched, preserve_index=False)
    ).repartition(target_num_rows_per_block=config.block_size)
    ds = ds.with_column(
        "tile",
        read_slide_tiles(  # pyright: ignore[reportCallIssue]
            col("path"),
            col("x"),
            col("y"),
            col("tile_extent_x"),
            col("tile_extent_y"),
            col("level"),
        ),
    )

    ds = ds.drop_columns(["path", "level", "tile_extent_x", "tile_extent_y"])
    ds = ds.map(
        EmbedTiles,  # type: ignore[arg-type]
        fn_constructor_args=(config.encoder, config.concurrency),
        compute=ray.data.ActorPoolStrategy(
            max_size=4,
            max_tasks_in_flight_per_actor=max(1, config.concurrency // 4),
        ),
        max_concurrency=config.concurrency,
    )

    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    tiles_parquet_dir = output_path / "tiles"
    if tiles_parquet_dir.exists():
        shutil.rmtree(tiles_parquet_dir)

    slides.to_parquet(output_path / "slides.parquet", index=False)
    ds.write_parquet(str(tiles_parquet_dir), min_rows_per_file=config.rows_per_file)

    logger.log_artifacts(str(output_path), f"{config.data.data_name}")


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):  # type: ignore[call-arg]
        main()
