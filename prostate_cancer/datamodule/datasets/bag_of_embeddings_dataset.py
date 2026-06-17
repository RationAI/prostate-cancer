"""These Datasets were taken from Adam Kukučka Ulcerative Colitis project and modified."""

from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

import mlflow
import mlflow.artifacts
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from prostate_cancer.typing import (
    LabeledBagOfTilesSample,
    SlideMetadata,
    UnlabeledBagOfTilesSample,
)


T = TypeVar("T", bound=LabeledBagOfTilesSample | UnlabeledBagOfTilesSample)


class BagOfEmbeddingsDataset(Dataset[T], Generic[T]):
    def __init__(
        self,
        uris: Iterable[str],
        padding: bool = True,
        carcinoma_roi_t: float | None = None,
    ) -> None:
        self.include_labels = carcinoma_roi_t is not None
        self.carcinoma_roi_t = carcinoma_roi_t
        self.slides, self.tiles = self.download_artifacts(uris)

        if self.include_labels:
            self.tiles["carcinoma"] = (
                self.tiles["carcinoma_roi_percentage"] > self.carcinoma_roi_t
            )

        self.padding = padding
        self.max_embeddings = self.tiles["slide_id"].value_counts().max()

    def download_artifacts(
        self, tiling_uris: Iterable[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        slide_dfs = []
        tile_dfs = []

        for tiling_uri in tiling_uris:
            tiling_folder = Path(mlflow.artifacts.download_artifacts(tiling_uri))
            slide_dfs.append(pd.read_parquet(tiling_folder / "slides.parquet"))
            tile_dfs.append(pd.read_parquet(tiling_folder / "tiles.parquet"))

        return (
            pd.concat(slide_dfs, ignore_index=True),
            pd.concat(tile_dfs, ignore_index=True),
        )

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> T:
        slide_metadata = self.slides.iloc[idx]
        slide_name = Path(slide_metadata.path).stem

        slide_tiles = self.tiles[self.tiles["slide_id"] == slide_metadata.id]
        slide_embeddings = torch.tensor(slide_tiles["embedding"])

        pad_amount = self.max_embeddings - slide_embeddings.shape[0]
        assert pad_amount >= 0, "Invalid padding"

        if self.padding:
            slide_embeddings = F.pad(slide_embeddings, (0, 0, 0, pad_amount), value=0.0)

        metadata = SlideMetadata(
            slide_id=slide_metadata["id"],
            slide_name=slide_name,
            slide_path=slide_metadata["path"],
            xs=torch.from_numpy(slide_tiles["x"].to_numpy()),
            ys=torch.from_numpy(slide_tiles["y"].to_numpy()),
        )

        if not self.include_labels:
            return slide_embeddings, metadata  # type: ignore[return-value]

        sl_label = torch.tensor(slide_metadata["carcinoma"]).float()

        tl_labels = torch.zeros(len(slide_embeddings)).float()
        tl_labels[: len(slide_tiles)] = torch.tensor(
            slide_tiles["carcinoma"].to_numpy()
        ).float()

        return slide_embeddings, tl_labels, sl_label, metadata  # type: ignore[return-value]


class LabeledBagOfEmbeddingsDataset(BagOfEmbeddingsDataset[LabeledBagOfTilesSample]):
    def __init__(
        self,
        uris: Iterable[str],
        carcinoma_roi_t: float,
        padding: bool = True,
    ) -> None:
        super().__init__(
            uris=uris,
            padding=padding,
            carcinoma_roi_t=carcinoma_roi_t,
        )


class UnlabeledBagOfEmbeddingsDataset(
    BagOfEmbeddingsDataset[UnlabeledBagOfTilesSample]
):
    def __init__(
        self,
        uris: Iterable[str],
        padding: bool = True,
    ) -> None:
        super().__init__(
            uris=uris,
            padding=padding,
        )
