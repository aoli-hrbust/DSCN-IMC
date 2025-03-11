"""
Deep Spectral Clustering Network for Incomplete Multi-view Clustering (DSCN-IMC)
"""

import torch
import numpy as np
import torch.nn as nn
from typing import List

from .autoencoder import MultiviewEncoder
from utils.clustering_layer import get_q_cluster


class SpectralNetModel(nn.Module):

    def __init__(self, architecture, input_dim: List[int], clusterNum: int, use_gcn: bool):
        super(SpectralNetModel, self).__init__()
        assert clusterNum == architecture[-1]
        self.architecture = architecture
        self.clusterNum = clusterNum
        self.view_layers = nn.ModuleList()
        self.viewNum = len(input_dim)
        self.use_gcn = use_gcn
        self.hidden_dims = 128
        self.use_fc = True

        self.encoder = MultiviewEncoder(hidden_dims=self.hidden_dims, in_channels=input_dim, use_gcn=use_gcn)
        self.fc = nn.Linear(self.hidden_dims, clusterNum) if self.use_fc else nn.Identity()

        self.centers = nn.Parameter(torch.empty(clusterNum, clusterNum))
        nn.init.normal_(self.centers.data)
    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the output of the network using the Cholesky decomposition.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the network.

        Returns
        -------
        torch.Tensor
            The orthonormalized output.

        Notes
        -----
        This function applies QR decomposition to orthonormalize the output (`Y`) of the network.
        The inverse of the R matrix is returned as the orthonormalization weights.
        """

        m = Y.shape[0]
        _, R = torch.linalg.qr(Y)
        orthonorm_weights = np.sqrt(m) * torch.inverse(R)
        return orthonorm_weights

    def forward(
        self, X: List[torch.Tensor],
        M: torch.Tensor,
        S: torch.Tensor = None,
        should_update_orth_weights: bool = True,
    ):
        X_view = [X[v][M[:, v]] for v in range(self.viewNum)]
        inputs = dict(M=M, X_view=X_view, S_view=S)
        inputs = self.encoder(inputs)
        H_common = inputs['H_common']
        Y_tilde = self.fc(H_common)

        if should_update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)

        Y = Y_tilde @ self.orthonorm_weights
        Q_cluster = get_q_cluster(Y, self.centers)
        return Y, H_common, Q_cluster  # spectral embedding & common rep.
