"""
Deep Spectral Clustering Network for Incomplete Multi-view Clustering (DSCN-IMC)
"""

from typing import List
import torch
import torch.nn as nn

from .autoencoder import ManifoldRegLoss


class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(
        self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False
    ) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W,
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network
            is_normalized (bool, optional): Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
        """
        m = Y.size(0)
        if is_normalized:
            D = torch.sum(W, dim=1)
            Y = Y / D[:, None]

        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)

        return loss



class IMvSpectralNetLoss(nn.Module):

    def forward(
        self, W: List[torch.Tensor], Y: torch.Tensor, M: torch.Tensor, P: torch.Tensor,
        H: torch.Tensor, lamda: float,
        is_normalized: bool = True  # will have smaller losses.
    ) -> torch.Tensor:
        spectral_loss = SpectralNetLoss()
        manifold_loss = ManifoldRegLoss()
        loss = sum(spectral_loss(w, Y[m], is_normalized) for w, m in zip(W, M.T)) / len(W)
        loss_2 = manifold_loss(dict(
            P_view=P, H_common=H, M=M, viewNum=M.shape[1],
        ))
        return loss + loss_2 * lamda
