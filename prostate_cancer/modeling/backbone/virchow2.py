import timm
import torch
from timm.layers import SwiGLUPacked  # type: ignore[attr-defined]

from prostate_cancer.modeling.backbone.foundation_base import FoundationModel


class Virchow2(FoundationModel):
    def __init__(self, name: str) -> None:
        super().__init__(name, 2560)

        # For this, you need to setup HF_TOKEN=<X> env.variable.
        self.module = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.module(x)  # size: B x 261 x 1280

        class_token = output[:, 0]  # size: B x 1280
        patch_tokens = output[
            :, 5:
        ]  # size: B x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        # concatenate class token and average pool of patch tokens
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: B x 2560
