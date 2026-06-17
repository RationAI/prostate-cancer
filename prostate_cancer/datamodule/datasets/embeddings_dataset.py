from collections.abc import Iterable
from typing import TypeAlias, TypeVar

import pandas as pd
import torch

from prostate_cancer.datamodule.datasets.base import (
    BaseSingleSlideDataset,
    BaseTileDataset,
    get_slide_name,
)
from prostate_cancer.typing import LabeledTileSample, TileMetadata, UnlabeledTileSample


SlideDfMetadata: TypeAlias = pd.Series


T = TypeVar("T", covariant=True)


class EmbeddingsDataset(BaseTileDataset[T]):
    def __init__(
        self,
        uris: Iterable[str],
        carcinoma_roi_t: float | None = None,
        stratified_filter: bool | None = None,
    ) -> None:
        super().__init__(
            uris=uris,
            single_slide_ds_cls=TileEmbeddingsSlide,
            carcinoma_roi_t=carcinoma_roi_t,
            stratified_filter=stratified_filter,
        )


class LabeledEmbeddingsDataset(EmbeddingsDataset[LabeledTileSample]): ...


class UnlabeledEmbeddingsDataset(EmbeddingsDataset[UnlabeledTileSample]): ...


class TileEmbeddingsSlide(BaseSingleSlideDataset):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        include_label: bool,
    ) -> None:
        super().__init__(
            slide_metadata=slide_metadata,
            tiles=tiles,
            include_label=include_label,
        )
        assert "embedding" in tiles.column, (
            "Embeddings Dataset requires embedding column"
        )

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> LabeledTileSample | UnlabeledTileSample:
        tile = self.tiles.iloc[idx]
        vector = torch.tensor(tile["embedding"])
        metadata = TileMetadata(
            slide=get_slide_name(self.slide_metadata), x=tile["x"], y=tile["y"]
        )

        if not self.include_label:
            return vector, metadata

        label = torch.tensor([self.tiles.iloc[idx]["carcinoma"]]).float()
        return vector, label, metadata
