from collections.abc import Iterable
from typing import TypeVar

import torch
from datasets import Dataset as HFDataset

from ml.datamodule.datasets.base import (
    BaseSingleSlideDataset,
    BaseTileDataset,
    get_slide_name,
)
from ml.typing import (
    LabeledTileSample,
    TileMetadata,
    TilingSlideMetadata,
    UnlabeledTileSample,
)


T_co = TypeVar("T_co", covariant=True)


class EmbeddingsDataset(BaseTileDataset[T_co]):
    def __init__(
        self,
        uris: Iterable[str],
        carcinoma_roi_t: float | None = None,
        train_pos_tissue_roi_t: float | None = None,
        stratified_filter: bool | None = None,
        num_slides: int | None = None,
        slide_range: tuple[int | None, int | None] | None = None,
    ) -> None:
        super().__init__(
            uris=uris,
            single_slide_ds_cls=TileEmbeddingsSlide,
            carcinoma_roi_t=carcinoma_roi_t,
            train_pos_tissue_roi_t=train_pos_tissue_roi_t,
            stratified_filter=stratified_filter,
            num_slides=num_slides,
            slide_range=slide_range,
        )


class LabeledEmbeddingsDataset(EmbeddingsDataset[LabeledTileSample]): ...


class UnlabeledEmbeddingsDataset(EmbeddingsDataset[UnlabeledTileSample]): ...


class TileEmbeddingsSlide(BaseSingleSlideDataset):
    def __init__(
        self,
        slide_metadata: TilingSlideMetadata,
        tiles: HFDataset,
        include_label: bool,
    ) -> None:
        super().__init__(
            slide_metadata=slide_metadata,
            tiles=tiles,
            include_label=include_label,
        )
        assert "embedding" in tiles.column_names, (
            "Embeddings Dataset requires embedding column"
        )

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> LabeledTileSample | UnlabeledTileSample:
        tile = self.tiles[idx]

        vector = torch.tensor(tile["embedding"])

        metadata = TileMetadata(
            slide=get_slide_name(self.slide_metadata),
            x=tile["x"],
            y=tile["y"],
        )

        if not self.include_label:
            return vector, metadata

        label = torch.tensor([tile["carcinoma"]]).float()
        return vector, label, metadata
