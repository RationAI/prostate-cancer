from abc import ABC, abstractmethod

import torch


class FoundationModel(torch.nn.Module, ABC):
    def __init__(self, name: str, embed_dim: int) -> None:
        """Wrapper for a foundation model - forward and dimension differ depending on the model."""
        super().__init__()
        self.name = name
        self.embed_dim = embed_dim
        self.module = self.get_module()

    @abstractmethod
    def get_module(self) -> torch.nn.Module: ...
