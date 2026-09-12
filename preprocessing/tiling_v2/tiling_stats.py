"""Compute slide/tile counts and percentage-column histograms for a data source's tiling datasets."""

import tempfile
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def resolve_roi_threshold(
    thresholds: DictConfig | None, tiles: pd.DataFrame
) -> tuple[str, float] | None:
    """Pick the percentage column and threshold that define tile-level positivity.

    Prefers `carcinoma_roi_t` / `carcinoma_roi_percentage`, falling back to
    `epithelium_roi_t` / `epithelium_roi_percentage` when the carcinoma ROI
    threshold or column is not available for this data source.

    Arguments:
        thresholds (DictConfig | None): `data.thresholds` config, if present.
        tiles (pd.DataFrame): tiles.parquet contents.

    Returns:
        tuple[str, float] | None: (percentage column, threshold) to use, or
            None if neither source is fully available.
    """
    if thresholds is None:
        return None

    for threshold_key, column in (
        ("carcinoma_roi_t", "carcinoma_roi_percentage"),
        ("epithelium_roi_t", "epithelium_roi_percentage"),
    ):
        threshold = thresholds.get(threshold_key)
        if threshold is not None and column in tiles.columns:
            return column, threshold

    return None


def slide_stats(slides: pd.DataFrame) -> dict[str, int]:
    """Count slides overall and by the boolean `carcinoma` flag.

    Arguments:
        slides (pd.DataFrame): slides.parquet contents.

    Returns:
        dict[str, int]: Slide counts.
    """
    positive = slides["carcinoma"]
    return {
        "num_slides": len(slides),
        "num_slides_carcinoma_positive": int(positive.sum()),
        "num_slides_carcinoma_negative": int((~positive).sum()),
    }


def tile_stats(tiles: pd.DataFrame, thresholds: DictConfig | None) -> dict[str, int]:
    """Count tiles overall and, if a ROI threshold/column is available, by positivity.

    Arguments:
        tiles (pd.DataFrame): tiles.parquet contents.
        thresholds (DictConfig | None): `data.thresholds` config, if present.

    Returns:
        dict[str, int]: Tile counts.
    """
    stats = {"num_tiles": len(tiles)}

    resolved = resolve_roi_threshold(thresholds, tiles)
    if resolved is None:
        return stats

    column, threshold = resolved
    positive = tiles[column] > threshold
    stats["num_tiles_positive"] = int(positive.sum())
    stats["num_tiles_negative"] = int((~positive).sum())

    return stats


def plot_percentage_histogram(values: pd.Series, column: str, out_dir: Path) -> Path:
    """Plot a full histogram and a non-zero-only histogram of a percentage column.

    Arguments:
        values (pd.Series): Column values to plot.
        column (str): Column name, used for titles/labels and the output filename.
        out_dir (Path): Directory to save the figure into.

    Returns:
        Path: Path to the saved figure.
    """
    nonzero = values[values > 0]

    fig, (ax_full, ax_nonzero) = plt.subplots(1, 2, figsize=(12, 5))

    ax_full.hist(values, bins=50, range=(0, 1), color="steelblue", edgecolor="black")
    ax_full.set_title(f"All tiles (n={len(values)})")
    ax_full.set_xlabel(column)
    ax_full.set_ylabel("Count")

    ax_nonzero.hist(
        nonzero, bins=50, range=(0, 1), color="darkorange", edgecolor="black"
    )
    ax_nonzero.set_title(f"Non-zero tiles (n={len(nonzero)})")
    ax_nonzero.set_xlabel(column)
    ax_nonzero.set_ylabel("Count")

    fig.suptitle(column)
    fig.tight_layout()

    path = out_dir / f"{column}.png"
    fig.savefig(path)
    plt.close(fig)

    return path


def compute_and_log_stats(
    tiling_uri: str, suffix: str, thresholds: DictConfig | None, logger: MLFlowLogger
) -> None:
    """Compute slide/tile stats and percentage histograms for one tiling dataset.

    Arguments:
        tiling_uri (str): MLflow URI of the tiling dataset directory.
        suffix (str): Namespace suffix for logged metrics/plots (e.g. "512", "filtered_224").
        thresholds (DictConfig | None): `data.thresholds` config, if present.
        logger (MLFlowLogger): Logger used to log metrics and plot artifacts.
    """
    tiling_path = Path(mlflow.artifacts.download_artifacts(tiling_uri))
    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")

    stats = {**slide_stats(slides), **tile_stats(tiles, thresholds)}
    print(f"[{suffix}] stats:", stats)
    logger.log_metrics({f"{suffix}/{name}": value for name, value in stats.items()})

    percentage_cols = [col for col in tiles.columns if col.endswith("percentage")]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        for column in percentage_cols:
            plot_path = plot_percentage_histogram(tiles[column], column, out_dir)
            logger.log_artifact(str(plot_path), artifact_path=f"plots/{suffix}")


@with_cli_args(["+preprocessing=tiling_stats"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    thresholds = config.data.get("thresholds")

    for field, suffix in (
        ("tiles_uri_512", "512"),
        ("tiles_uri_224", "224"),
        ("tiles_filtered_uri_512", "filtered_512"),
        ("tiles_filtered_uri_224", "filtered_224"),
    ):
        tiling_uri = config.data.get(field)
        if tiling_uri is None:
            continue

        compute_and_log_stats(tiling_uri, suffix, thresholds, logger)


if __name__ == "__main__":
    main()
