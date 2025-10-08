from abc import ABC, abstractmethod

import torch
from jaxtyping import Int
from torch import Tensor

from explainability.typing import Embeddings


class Clustering(ABC):
    k: int

    @abstractmethod
    def inference(self, embedding: Embeddings) -> Int[Tensor, "K"]:
        """Given a batch of embeddings, return index of the cluster that the embedding belongs to as a integer tensor of shape (K, )."""

    @abstractmethod
    def find_top_closest(
        self, embeddings: Embeddings, cluster: int, p: int
    ) -> torch.return_types.topk:
        """Given a tensor of embeddings and a cluster index, return top p best representatives from the embeddings of the cluster with the given index."""


class ClusteringMethod(ABC):
    @abstractmethod
    def cluster(self, embeddings: Embeddings, k: int) -> Clustering:
        pass
