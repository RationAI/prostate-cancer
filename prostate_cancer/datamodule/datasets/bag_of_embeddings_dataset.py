"""These Datasets were taken from Adam Kukučka Ulcerative Colitis project and modified."""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

import torch
import torch.nn.functional as F
from rationai.mlkit.data.datasets.slides_tiles_loader import SlidesTilesLoader
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

        self._meta = SlidesTilesLoader(uris=uris)
        self.slides = self._meta.slides
        tiles = self._meta.tiles

        if self.include_labels:
            tiles = tiles.map(
                lambda r: {
                    "carcinoma": (r["carcinoma_roi_percentage"] > self.carcinoma_roi_t)
                }
            )

        self.tiles = tiles
        self._meta.tiles = tiles
        # no need to re-build index after .map

        self.padding = padding

        # compute max tiles per slide (HF version)
        slide_ids = self.tiles["slide_id"]

        self.max_embeddings = max(Counter(slide_ids).values())

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> T:
        slide_metadata = self.slides[idx]

        slide_name = Path(slide_metadata["path"]).stem
        slide_tiles = self._meta.filter_tiles_by_slide(slide_metadata["id"])

        slide_embeddings = torch.tensor(slide_tiles["embedding"])

        pad_amount = self.max_embeddings - slide_embeddings.shape[0]
        assert pad_amount >= 0, "Invalid padding"

        if self.padding:
            slide_embeddings = F.pad(
                slide_embeddings,
                (0, 0, 0, pad_amount),
                value=0.0,
            )

        metadata = SlideMetadata(
            slide_id=slide_metadata["id"],
            slide_name=slide_name,
            slide_path=slide_metadata["path"],
            xs=torch.tensor(slide_tiles["x"]),
            ys=torch.tensor(slide_tiles["y"]),
        )

        if not self.include_labels:
            return slide_embeddings, metadata  # type: ignore[return-value]

        sl_label = torch.tensor(slide_metadata["carcinoma"]).float()

        tl_labels = torch.zeros(len(slide_embeddings)).float()
        tl_labels[: len(slide_tiles)] = torch.tensor(slide_tiles["carcinoma"]).float()

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
