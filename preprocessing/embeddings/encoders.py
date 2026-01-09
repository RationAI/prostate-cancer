import timm
import torch

# taken from HuggingFace of Virchow2
from timm.layers import SwiGLUPacked  # type: ignore[attr-defined]


class FoundationModel(torch.nn.Module):
    def __init__(self, name: str, embed_dim: int) -> None:
        """Wrapper for a foundation model - forward and dimension differ depending on the model."""
        super().__init__()
        self.embed_dim = embed_dim


class ProvGigaPath(FoundationModel):
    def __init__(self, name: str) -> None:
        super().__init__(name, 1536)
        # For this, you need to setup HF_TOKEN=<X> env.variable.
        self.module = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        ).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


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
