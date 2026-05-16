from torch import Tensor, nn

from prostate_cancer.base_model import ProstateCancerModel


class CNNProstateModel(ProstateCancerModel):
    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        lr: float,
        tl_threshold: float,
    ) -> None:
        super().__init__(lr=lr, tl_threshold=tl_threshold)
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        logits = self.decode_head(features)
        return logits
