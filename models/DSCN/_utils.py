import os
from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors





def make_batch_for_sparse_grapsh(batch_x: torch.Tensor) -> torch.Tensor:
    """
    Computes a new batch of data points from the given batch (batch_x)
    in case that the graph-laplacian obtained from the given batch is sparse.
    The new batch is computed based on the nearest neighbors of 0.25
    of the given batch.

    Parameters
    ----------
    batch_x : torch.Tensor
        Batch of data points.

    Returns
    -------
    torch.Tensor
        New batch of data points.
    """

    batch_size = batch_x.shape[0]
    batch_size //= 5
    new_batch_x = batch_x[:batch_size]
    batch_x = new_batch_x
    n_neighbors = 5

    u = AnnoyIndex(batch_x[0].shape[0], "euclidean")
    u.load("ann_index.ann")
    for x in batch_x:
        x = x.detach().cpu().numpy()
        nn_indices = u.get_nns_by_vector(x, n_neighbors)
        nn_tensors = [u.get_item_vector(i) for i in nn_indices[1:]]
        nn_tensors = torch.tensor(nn_tensors)
        new_batch_x = torch.cat((new_batch_x, nn_tensors))

    return new_batch_x


def get_nearest_neighbors(
    X: torch.Tensor, Y: torch.Tensor = None, k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances and the indices of the k nearest neighbors of each data point.

    Parameters
    ----------
    X : torch.Tensor
        Batch of data points.
    Y : torch.Tensor, optional
        Defaults to None.
    k : int, optional
        Number of nearest neighbors to calculate. Defaults to 3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Distances and indices of each data point.
    """
    if Y is None:
        Y = X
    if len(X) < k:
        k = len(X)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids


def compute_scale(
    Dis: np.ndarray, k: int = 2, med: bool = True, is_local: bool = True
) -> np.ndarray:
    """
    Computes the scale for the Gaussian similarity function.

    Parameters
    ----------
    Dis : np.ndarray
        Distances of the k nearest neighbors of each data point.
    k : int, optional
        Number of nearest neighbors for the scale calculation. Relevant for global scale only.
    med : bool, optional
        Scale calculation method. Can be calculated by the median distance from a data point to its neighbors,
        or by the maximum distance. Defaults to True.
    is_local : bool, optional
        Local distance (different for each data point), or global distance. Defaults to True.

    Returns
    -------
    np.ndarray
        Scale (global or local).
    """

    if is_local:
        if not med:
            scale = np.max(Dis, axis=1)
        else:
            scale = np.median(Dis, axis=1)
    else:
        if not med:
            scale = np.max(Dis[:, k - 1])
        else:
            scale = np.median(Dis[:, k - 1])
    return scale


def get_gaussian_kernel(
    D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True
) -> torch.Tensor:
    """
    Computes the Gaussian similarity function according to a given distance matrix D and a given scale.

    Parameters
    ----------
    D : torch.Tensor
        Distance matrix.
    scale :
        Scale.
    Ids : np.ndarray
        Indices of the k nearest neighbors of each sample.
    device : torch.device
        Defaults to torch.device("cpu").
    is_local : bool, optional
        Determines whether the given scale is global or local. Defaults to True.

    Returns
    -------
    torch.Tensor
        Matrix W with Gaussian similarities.
    """

    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale**2))
    else:
        # local scales
        W = torch.exp(
            -torch.pow(D, 2).to(device)
            / (torch.tensor(scale).float().to(device).clamp_min(1e-7) ** 2)
        )
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.0
    return sym_W








