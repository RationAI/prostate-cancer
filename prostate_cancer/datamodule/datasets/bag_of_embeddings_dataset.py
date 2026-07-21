"""These Datasets were taken from Adam Kukučka Ulcerative Colitis project and modified."""

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets.slides_tiles_loader import SlidesTilesLoader
from torch.utils.data import Dataset

from prostate_cancer.typing import (
    LabeledBagOfTilesSample,
    SlideMetadata,
    SLLabeledBagOfTilesSample,
    TilingSlideMetadata,
    UnlabeledBagOfTilesSample,
)


T = TypeVar(
    "T",
    bound=LabeledBagOfTilesSample
    | SLLabeledBagOfTilesSample
    | UnlabeledBagOfTilesSample,
)


class BagOfEmbeddingsDataset(Dataset[T], Generic[T], ABC):
    """Base for bag-of-embeddings (MIL) datasets: one item per slide.

    Handles loading slide/tile metadata, assembling the (padded) bag of tile
    embeddings and building the shared slide-level metadata. Subclasses only
    decide which labels (if any) accompany the bag.
    """

    def __init__(
        self,
        uris: Iterable[str],
        padding: bool = True,
    ) -> None:
        self._meta = SlidesTilesLoader(uris=uris)
        self.slides = self._meta.slides
        self.tiles = self._meta.tiles

        self.padding = padding

        # compute max tiles per slide (HF version)
        slide_ids = self.tiles["slide_id"]

        self.max_embeddings = max(Counter(slide_ids).values())

    def __len__(self) -> int:
        return len(self.slides)

    def _load_bag(
        self, idx: int
    ) -> tuple[TilingSlideMetadata, HFDataset, torch.Tensor, SlideMetadata]:
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

        return slide_metadata, slide_tiles, slide_embeddings, metadata

    @abstractmethod
    def __getitem__(self, idx: int) -> T: ...


class UnlabeledBagOfEmbeddingsDataset(
    BagOfEmbeddingsDataset[UnlabeledBagOfTilesSample]
):
    def __init__(
        self,
        uris: Iterable[str],
        padding: bool = True,
    ) -> None:
        super().__init__(uris=uris, padding=padding)

    def __getitem__(self, idx: int) -> UnlabeledBagOfTilesSample:
        _, _, slide_embeddings, metadata = self._load_bag(idx)
        return slide_embeddings, metadata


class SLLabeledBagOfEmbeddingsDataset(
    BagOfEmbeddingsDataset[SLLabeledBagOfTilesSample]
):
    """Bag-of-embeddings dataset carrying only slide-level (SL) labels.

    Unlike `LabeledBagOfEmbeddingsDataset`, this does not require tile-level
    (TL) carcinoma annotations, so it can be used with data that only has
    slide-level ground truth (classic MIL, no TL supervision).
    """

    def __init__(
        self,
        uris: Iterable[str],
        padding: bool = True,
    ) -> None:
        super().__init__(uris=uris, padding=padding)

    def __getitem__(self, idx: int) -> SLLabeledBagOfTilesSample:
        slide_metadata, _, slide_embeddings, metadata = self._load_bag(idx)

        sl_label = torch.tensor(slide_metadata["carcinoma"]).float()

        return slide_embeddings, sl_label, metadata


class LabeledBagOfEmbeddingsDataset(BagOfEmbeddingsDataset[LabeledBagOfTilesSample]):
    """Bag-of-embeddings dataset carrying both SL and TL labels (hybrid MIL)."""

    def __init__(
        self,
        uris: Iterable[str],
        carcinoma_roi_t: float,
        padding: bool = True,
    ) -> None:
        super().__init__(uris=uris, padding=padding)
        self.carcinoma_roi_t = carcinoma_roi_t

        self.tiles = self.tiles.map(
            lambda r: {
                "carcinoma": (r["carcinoma_roi_percentage"] > self.carcinoma_roi_t)
            }
        )
        self._meta.tiles = self.tiles
        # no need to re-build index after .map

    def __getitem__(self, idx: int) -> LabeledBagOfTilesSample:
        slide_metadata, slide_tiles, slide_embeddings, metadata = self._load_bag(idx)

        sl_label = torch.tensor(slide_metadata["carcinoma"]).float()

        tl_labels = torch.zeros(len(slide_embeddings)).float() # pad with zero labels
        tl_labels[: len(slide_tiles)] = torch.tensor(slide_tiles["carcinoma"]).float()

        return slide_embeddings, tl_labels, sl_label, metadata
