import torch
import numpy as np


def reshape_for_clustering_universal(
    embeddings: torch.Tensor | np.ndarray,
    channel_dim_index: int
) -> torch.Tensor | np.ndarray:
    if isinstance(embeddings, np.ndarray):
        n_dims = embeddings.ndim
    elif isinstance(embeddings, torch.Tensor):
        n_dims = embeddings.dim()
    else:
        raise TypeError("embeddings must be a torch.Tensor or np.ndarray")
        
    permute_tuple = tuple(
        [i for i in range(n_dims) if i != channel_dim_index] + [channel_dim_index]
    )
    if isinstance(embeddings, np.ndarray):
        return embeddings.transpose(permute_tuple).reshape(-1, embeddings.shape[channel_dim_index])
    elif isinstance(embeddings, torch.Tensor):
        return embeddings.permute(permute_tuple).reshape(-1, embeddings.shape[channel_dim_index])
