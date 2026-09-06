"""Merges the shard_* output directories produced by parallel (start/end-sharded) runs of tile_embeddings_v2.py back into a single flat tiles/ + slides/ layout."""

import shutil
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@with_cli_args(["+preprocessing=merge_embeddings_v2"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_path = Path(config.output_path)
    shard_dirs = sorted(output_path.glob("shard_*"))
    if not shard_dirs:
        raise ValueError(f"No shard_* directories found under {output_path}")

    slides = pd.concat(
        [
            pd.read_parquet(shard_dir / "slides" / "slides.parquet")
            for shard_dir in shard_dirs
        ],
        ignore_index=True,
    )

    tiles_dir = output_path / "tiles"
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True)

    for shard_dir in shard_dirs:
        for tile_file in (shard_dir / "tiles").glob("*.parquet"):
            # prefix by shard name so files from different shards (each numbered from 0) don't collide
            shutil.copy2(tile_file, tiles_dir / f"{shard_dir.name}_{tile_file.name}")

    slides_dir = output_path / "slides"
    if slides_dir.exists():
        shutil.rmtree(slides_dir)
    slides_dir.mkdir(parents=True)
    slides.to_parquet(slides_dir / "slides.parquet", index=False)

    for shard_dir in shard_dirs:
        shutil.rmtree(shard_dir)

    logger.log_artifacts(str(output_path), config.data.data_name)


if __name__ == "__main__":
    main()
