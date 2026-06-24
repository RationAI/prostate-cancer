from typing import Any

import torch

from prostate_cancer.base_model import ProstateCancerModel
from prostate_cancer.modeling.backbone.foundation_base import FoundationModel
from prostate_cancer.modeling.decode_head import BinaryCNNClassifier


class FoundationProstateModel(ProstateCancerModel):
    def __init__(
        self,
        backbone: FoundationModel,
        decode_head: BinaryCNNClassifier,
        lr: float,
        tl_threshold: float,
        freeze_backbone: bool,
    ) -> None:
        super().__init__(lr=lr, tl_threshold=tl_threshold)
        self.backbone = backbone
        self.decode_head = decode_head

        self.frozen_backbone = freeze_backbone

        # stay consistent with embedding training
        if freeze_backbone:
            for p in self.backbone.module.parameters():
                p.requires_grad = False
            self.backbone.module.eval()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # no need to save frozen backbone
        state_dict: dict[str, Any] = checkpoint["state_dict"]

        keys_to_remove = [
            k for k in list(state_dict.keys()) if k.startswith("backbone.")
        ]

        for k in keys_to_remove:
            del state_dict[k]

    def on_train_epoch_start(self) -> None:
        # prevent unintentional train mode
        if self.frozen_backbone:
            self.backbone.module.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        logits = self.decode_head(features)
        return logits
