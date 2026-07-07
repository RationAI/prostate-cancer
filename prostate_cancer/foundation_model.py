from typing import Any

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

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
        backbone_lr: float | None = None,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__(lr=lr, tl_threshold=tl_threshold)
        self.backbone = backbone
        self.decode_head = decode_head

        self.frozen_backbone = freeze_backbone

        # full fine-tuning disrupts pretrained weights if updated at the
        # same LR as the freshly initialized head, so it gets its own,
        # lower LR (defaults to the head LR when unset)
        self.backbone_lr = lr if backbone_lr is None else backbone_lr
        self.warmup_steps = warmup_steps

        # stay consistent with embedding training
        if freeze_backbone:
            for p in self.backbone.module.parameters():
                p.requires_grad = False
            self.backbone.module.eval()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.frozen_backbone:
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

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        param_groups = [{"params": self.decode_head.parameters(), "lr": self.lr}]
        if not self.frozen_backbone:
            param_groups.append(
                {"params": self.backbone.parameters(), "lr": self.backbone_lr}
            )

        optimizer = AdamW(param_groups, lr=self.lr)

        if self.warmup_steps == 0:
            return {"optimizer": optimizer}

        warmup_steps = self.warmup_steps

        def warmup(step: int) -> float:
            return min(1.0, (step + 1) / warmup_steps)

        scheduler = LambdaLR(optimizer, lr_lambda=warmup)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
