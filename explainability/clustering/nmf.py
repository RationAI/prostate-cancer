from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable

    from explainability.typing import Embeddings

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import NMF

from explainability.clustering.abstract import Clustering, ClusteringMethod


def stack_embeddings(embeddings: Iterable[torch.Tensor]) -> NDArray[np.float64]:
    """Stack a sequence of torch tensors [K_i, C] into a single non-negative 2D numpy array [N, C].

    - Each tensor is converted to CPU, detached, and cast to float32, then clamped to be non-negative.
    - Returns a float64 numpy array to match scikit-learn's defaults.
    """
    arrays: list[np.ndarray] = []
    for tensor in embeddings:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("All embeddings must be torch.Tensor")
        arr = tensor.detach().cpu().float().clamp_min(0.0).numpy()
        if arr.ndim != 2:
            raise ValueError(
                f"Each embedding tensor must be 2D [K, C], got shape {arr.shape}"
            )
        arrays.append(arr)
    if len(arrays) == 0:
        raise ValueError("No embeddings provided to stack.")
    stacked = np.concatenate(arrays, axis=0)
    return stacked.astype(np.float64, copy=False)


def nmf_factorize(
    x: ArrayLike,
    n_components: int,
    init: str = "nndsvda",
    random_state: int | None = 0,
    max_iter: int = 500,
    solver: str = "cd",
    tol: float = 1e-4,
    alpha_weights: float = 0.0,
    alpha_components: float = 0.0,
    l1_ratio: float = 0.0,
    verbose: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NMF]:
    """Run Non-Negative Matrix Factorization X ≈ W @ H.

    Parameters
    - x: 2D array-like of shape [n_samples, n_features], must be non-negative
    - n_components: rank of the factorization (number of latent topics/components)
    - init: initialization method for NMF ('nndsvda', 'nndsvd', 'random', ...)
    - random_state: seed for reproducibility
    - max_iter: maximum number of iterations
    - solver: 'cd' (coordinate descent) or 'mu' (multiplicative updates)
    - tol: tolerance for stopping condition
    - alpha_W, alpha_H: L2/L1 regularization strengths on W and H
    - l1_ratio: the regularization mixing parameter, with 0 <= l1_ratio <= 1
    - verbose: scikit-learn verbosity level

    Returns:
    - W: [n_samples, n_components]
    - H: [n_components, n_features]
    - model: fitted sklearn.decomposition.NMF instance
    """
    x_arr = np.asarray(x)
    if x_arr.ndim != 2:
        raise ValueError(
            f"Input x must be 2D [n_samples, n_features], got shape {x_arr.shape}"
        )
    if np.any(x_arr < 0):
        raise ValueError("Input x must be non-negative for NMF.")

    model = NMF(
        n_components=n_components,
        init=init,
        random_state=random_state,
        max_iter=max_iter,
        solver=solver,
        tol=tol,
        alpha_W=alpha_weights,
        alpha_H=alpha_components,
        l1_ratio=l1_ratio,
        verbose=verbose,
    )
    weights = model.fit_transform(x_arr)
    components = model.components_
    return weights, components, model


def reconstruct(weights: ArrayLike, components: ArrayLike) -> NDArray[np.float64]:
    """Reconstruct X_hat = W @ H."""
    w_arr = np.asarray(weights)
    h_arr = np.asarray(components)
    return (w_arr @ h_arr).astype(np.float64, copy=False)


def explained_frobenius_ratio(
    x: ArrayLike, weights: ArrayLike, components: ArrayLike
) -> float:
    """Compute 1 - ||X - WH||_F / ||X||_F as a simple reconstruction quality metric."""
    x_arr = np.asarray(x)
    x_hat = reconstruct(weights, components)
    num = np.linalg.norm(x_arr - x_hat, ord="fro")
    den = np.linalg.norm(x_arr, ord="fro") + 1e-12
    return float(1.0 - (num / den))


class NMFClustering(Clustering):
    """NMF-based clustering where clusters correspond to latent components.

    - Embeddings are assigned to clusters by argmax over factor weights via model.transform.
    """

    def __init__(self, model: NMF):
        self.model = model
        self.k = int(model.n_components)

    @property
    def weights(self) -> NDArray[np.float64]:
        return self.model.weights_

    @property
    def components(self) -> NDArray[np.float64]:
        return self.model.components_

    def inference(self, embedding: Embeddings) -> torch.Tensor:
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("embedding must be a torch.Tensor")
        if embedding.ndim != 2:
            raise ValueError(
                f"embedding must be 2D [B, C], got shape {tuple(embedding.shape)}"
            )
        # Clamp to non-negative as NMF expects
        x = embedding.detach().cpu().float().clamp_min(0.0).numpy()
        weights = self.model.transform(x)  # [B, k]
        clusters = np.argmax(weights, axis=1).astype(np.int64)
        return torch.from_numpy(clusters)

    def find_top_closest(
        self, embeddings: Embeddings, cluster: int, p: int
    ) -> torch.return_types.topk:
        """Return top-p representatives in the given cluster based on NMF weight for that component."""
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D [N, C], got shape {tuple(embeddings.shape)}"
            )

        if not (0 <= cluster < self.k):
            raise ValueError(f"cluster index {cluster} out of range [0, {self.k})")

        if not (0 <= p <= embeddings.shape[0]):
            raise ValueError(f"p must be in [0, {embeddings.shape[0]}], got {p}")

        x = embeddings.detach().cpu().float().clamp_min(0.0).numpy()
        weights = self.model.transform(x).astype(np.float32)  # [N, k]
        weights_t = torch.from_numpy(weights)

        scores = weights_t[:, cluster]  # [N]
        return torch.topk(scores, k=p)


class NMFClusteringMethod(ClusteringMethod):
    """ClusteringMethod that uses Non-Negative Matrix Factorization."""

    def __init__(
        self,
        init: str = "nndsvda",
        random_state: int | None = 0,
        max_iter: int = 500,
        solver: str = "cd",
        tol: float = 1e-4,
        alpha_weights: float = 0.0,
        alpha_components: float = 0.0,
        l1_ratio: float = 0.0,
        verbose: int = 0,
    ):
        self.init = init
        self.random_state = random_state
        self.max_iter = max_iter
        self.solver = solver
        self.tol = tol
        self.alpha_weights = alpha_weights
        self.alpha_components = alpha_components
        self.l1_ratio = l1_ratio
        self.verbose = verbose

    def cluster(self, embeddings: Embeddings, k: int) -> Clustering:
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("embeddings must be a torch.Tensor")
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D [N, C], got shape {tuple(embeddings.shape)}"
            )
        # Prepare non-negative data for NMF
        x = embeddings.detach().cpu().float().clamp_min(0.0).numpy()
        model = NMF(
            n_components=k,
            init=self.init,
            random_state=self.random_state,
            max_iter=self.max_iter,
            solver=self.solver,
            tol=self.tol,
            alpha_W=self.alpha_weights,
            alpha_H=self.alpha_components,
            l1_ratio=self.l1_ratio,
            verbose=self.verbose,
        )
        model.fit(x)
        return NMFClustering(model=model)
