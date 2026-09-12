import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


def negative_split(
    slides_df: pd.DataFrame,
    n_negative: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split slides into a set of ~N negative slides and the rest, grouped by case.

    Cases are the unit of selection: a case is never left partially in the
    rest split. Cases are greedily picked in order of fewest positive slides
    first (ties broken randomly), accumulating negative slides until at
    least `n_negative` is reached. From each selected case only the negative
    slides are kept for the negative split -- any positive slides belonging
    to a selected case are discarded entirely (they end up in neither
    output). Cases that are not selected stay untouched in rest.

    Arguments:
        slides_df (pd.DataFrame): DataFrame with the slides metadata. Must
            contain "carcinoma" (bool) and "case_id" columns.
        n_negative (int): Target number of negative slides to draw into the
            negative split.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple of (negative, rest) splits.
    """
    case_counts = slides_df.groupby("case_id")["carcinoma"].agg(
        neg_count=lambda labels: int((~labels).sum()),
        pos_count=lambda labels: int(labels.sum()),
    )

    assert case_counts["neg_count"].sum() >= n_negative, (
        f"Requested {n_negative} negative slides, but only "
        f"{case_counts['neg_count'].sum()} negative slides are available in total."
    )

    # shuffle first so ties on pos_count aren't broken by case_id order
    case_counts = case_counts.sample(frac=1, random_state=random_state)
    case_counts = case_counts.sort_values("pos_count", kind="stable")

    selected_case_ids = []
    total_negative = 0
    for case_id, row in case_counts.iterrows():
        if total_negative >= n_negative:
            break
        selected_case_ids.append(case_id)
        total_negative += row["neg_count"]

    is_selected_case = slides_df["case_id"].isin(selected_case_ids)
    negative_slides = slides_df[is_selected_case & ~slides_df["carcinoma"]]
    rest_slides = slides_df[~is_selected_case]

    return negative_slides, rest_slides


def split_tiling(
    tiling_uri: str,
    negative_paths: set[str],
    rest_paths: set[str],
    negative_name: str,
    rest_name: str,
) -> None:
    """Split a tiling dataset (slides.parquet + tiles.parquet) by slide path membership.

    Arguments:
        tiling_uri (str): MLflow URI of the tiling dataset directory.
        negative_paths (set[str]): Slide paths (metadata `slide_path`) that belong
            to the negative split.
        rest_paths (set[str]): Slide paths that belong to the rest split.
        negative_name (str): Dataset name to log the negative split under.
        rest_name (str): Dataset name to log the rest split under.
    """
    tiling_path = Path(mlflow.artifacts.download_artifacts(tiling_uri))
    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")

    negative_slides = slides[slides["path"].isin(negative_paths)]
    rest_slides = slides[slides["path"].isin(rest_paths)]

    negative_tiles = tiles[tiles["slide_id"].isin(negative_slides["id"])]
    rest_tiles = tiles[tiles["slide_id"].isin(rest_slides["id"])]

    save_mlflow_dataset(negative_slides, negative_tiles, negative_name)
    save_mlflow_dataset(rest_slides, rest_tiles, rest_name)


@with_cli_args(["+preprocessing=negative_split"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_df_path = mlflow.artifacts.download_artifacts(config.data.metadata_table)
    slides_df = pd.read_csv(slides_df_path)

    split_names = config.split_names
    assert "negative" in split_names and "rest" in split_names

    negative_slides, rest_slides = negative_split(
        slides_df=slides_df,
        n_negative=config.n_negative,
        random_state=42,
    )

    print("Negative slides:", negative_slides["carcinoma"].value_counts())
    print("Rest slides:", rest_slides["carcinoma"].value_counts())

    with tempfile.TemporaryDirectory() as tmpdir:
        negative_out = (
            Path(tmpdir) / f"{config.data.data_name}_{split_names['negative']}.csv"
        )
        negative_slides.to_csv(negative_out, index=False)
        logger.log_artifact(str(negative_out))

        rest_out = Path(tmpdir) / f"{config.data.data_name}_{split_names['rest']}.csv"
        rest_slides.to_csv(rest_out, index=False)
        logger.log_artifact(str(rest_out))

    negative_paths = set(negative_slides["slide_path"])
    rest_paths = set(rest_slides["slide_path"])

    for field, suffix in (
        ("tiles_uri_512", "512"),
        ("tiles_uri_224", "224"),
        ("tiles_filtered_uri_512", "filtered_512"),
        ("tiles_filtered_uri_224", "filtered_224"),
    ):
        if not hasattr(config.data, field) or config.data[field] is None:
            continue

        split_tiling(
            config.data[field],
            negative_paths,
            rest_paths,
            negative_name=f"{config.data.data_name}_{split_names['negative']}_{suffix}",
            rest_name=f"{config.data.data_name}_{split_names['rest']}_{suffix}",
        )


if __name__ == "__main__":
    main()
