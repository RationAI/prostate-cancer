import torch

from explainability.typing import Embeddings


class Clustering:
    k: int

    def inference(self, embedding: Embeddings) -> torch.Tensor:
        """Given a batch of embeddings, return index of the cluster that the embedding belongs to as a integer tensor of shape (batch_size, )."""

    def find_top_closest(
        self, embeddings: Embeddings, cluster: int, p: int
    ) -> torch.return_types.topk:
        """Given a tensor of embeddings, return top p best representatives of the cluster."""


class ClusteringMethod:
    def cluster(self, embeddings: Embeddings, k: int) -> Clustering:
        pass
