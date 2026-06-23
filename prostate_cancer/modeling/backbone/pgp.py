import timm
import torch

from prostate_cancer.modeling.backbone.foundation_base import FoundationModel


class ProvGigaPath(FoundationModel):
    def __init__(self) -> None:
        super().__init__("PGP", 1536)

    def get_module(self) -> torch.nn.Module:
        # For this, you need to setup HF_TOKEN=<X> env.variable.
        return timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        ).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
