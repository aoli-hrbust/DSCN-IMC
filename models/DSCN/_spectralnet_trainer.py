"""
Deep Spectral Clustering Network for Incomplete Multi-view Clustering (DSCN-IMC)
"""

import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import math
from torch.utils.data import (
    DataLoader,
    TensorDataset
)


from ._spectralnet_loss import IMvSpectralNetLoss
from ._spectralnet_model import SpectralNetModel
from .ptsne_training import (
    calculate_optimized_p_cond,
    make_joint,
)
from utils.clustering_layer import (
    kl_clustering_loss,
    target_distribution
)
from ._utils import (
    make_batch_for_sparse_grapsh,
    get_nearest_neighbors,
    compute_scale,
    get_gaussian_kernel
)
from utils.metrics import *

class SpectralTrainer:
    def __init__(
        self,
        input_dim: List[int],
        clusterNum: int,
        config: dict,
        device: torch.device,
        is_sparse: bool = False,
    ):
        self.device = device
        self.is_sparse = is_sparse
        self.spectral_config = config
        self.lr = self.spectral_config["lr"]
        self.n_nbg = self.spectral_config["n_nbg"]
        self.epochs = self.spectral_config["epochs"]
        self.scale_k = self.spectral_config["scale_k"]
        self.batch_size = self.spectral_config["batch_size"]
        self.is_local_scale = self.spectral_config["is_local_scale"]
        self.use_gcn = self.spectral_config["use_gcn"]
        self.architecture = self.spectral_config["architecture"]
        self.ppl = self.spectral_config["ppl"]
        self.lamda = self.spectral_config["lamda"]
        self.aug_size = self.spectral_config.get("aug_size", 32)
        self.aug_p = self.spectral_config.get('aug_p', 0.5)
        self.spectral_net = SpectralNetModel(
            self.architecture,
            input_dim=input_dim,
            clusterNum=clusterNum,
            use_gcn=self.use_gcn,
        ).to(self.device)

    def train(
        self,
        X: List[torch.Tensor],
        y: torch.Tensor,
        M: torch.Tensor,
        data,
    ):
        clusterNum: int = len(np.unique(y))
        orthnorm_weights = None
        self.X = convert_tensor(X)
        self.y = convert_tensor(y, dtype=torch.long)
        self.M = convert_tensor(M, dtype=torch.bool)
        self.counter = 0
        self.criterion = IMvSpectralNetLoss()

        self.mm = MaxMetrics()
        best_outputs = None

        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)
        self.history = []

        train_loader, ortho_loader, test_loader = self._get_data_loader()

        print("Training SpectralNet:")
        t = trange(self.epochs, leave=True)
        for epoch in t:
            train_loss = 0.0
            for (*X_grad, M_grad), (*X_orth, M_orth) in zip(train_loader, ortho_loader):
                X_grad = [x.to(self.device) for x in X_grad]
                X_orth = [x.to(self.device) for x in X_orth]
                M_grad = M_grad.to(self.device)
                M_orth = M_orth.to(self.device)

                S_grad, P_grad = self._get_graph_distribution(X_grad, M_grad)
                S_orth, P_orth = self._get_graph_distribution(X_orth, M_orth)

                if self.is_sparse:
                    X_grad = [
                        make_batch_for_sparse_grapsh(x) for x, m in zip(X_grad, M.T)
                    ]
                    X_orth = [
                        make_batch_for_sparse_grapsh(x) for x, m in zip(X_orth, M.T)
                    ]

                # Orthogonalization step
                self.spectral_net.eval()
                self.spectral_net(
                    X_orth, M_orth, S=S_orth, should_update_orth_weights=True
                )

                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()

                Y, H, Q_cluster = self.spectral_net(
                    X_grad, M_grad, S=S_grad, should_update_orth_weights=False
                )
                W = [self._get_affinity_matrix(x[m]) for x, m in zip(X_grad, M_grad.T)]

                loss = self.criterion(W, Y, M_grad, P_grad, H, self.lamda)
                if epoch + 1 > 5:
                    loss_3 = kl_clustering_loss(
                        Q_cluster, target_distribution(Q_cluster)
                    )
                    loss += loss_3

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # End of batch. Begin epoch's evaluation.
            train_loss /= len(train_loader)
            Y, H, Q_cluster = self.predict(test_loader)
            # metrics, _, centers = KMeans_Evaluate(Y, data, return_centroid=True)

            centers, ypred = KMeans_Torch(Y, n_clusters=clusterNum)
            ypred = convert_numpy(ypred)
            metrics = get_all_metrics(y, ypred)
            if 1 + epoch == 5:
                print("Initialize centers")
                self.spectral_net.centers.data = centers
            if 1 + epoch > 5 and (1 + epoch) % 2 == 0:
                metrics = get_all_metrics(y, convert_numpy(Q_cluster.argmax(1)))

            # 看看orthnorm-weights是不是收敛的。
            current_orthnorm_weights = self.spectral_net.orthonorm_weights.detach()
            if orthnorm_weights is not None:
                orthnorm_diff = F.mse_loss(
                    current_orthnorm_weights, orthnorm_weights
                ).item()
                metrics.update(orthnorm_diff=orthnorm_diff)
            orthnorm_weights = current_orthnorm_weights

            metrics.update(loss=train_loss)
            if self.mm.update(**metrics)["ACC"]:
                best_outputs = convert_numpy(
                    dict(Y=Y, H=H, Q_cluster=Q_cluster, mu=self.spectral_net.centers)
                )
            self.history.append(metrics)

            t.set_description(
                "Train Loss: {:.7f}, ACC: {:.2f} NMI: {:.2f}".format(
                    train_loss, metrics["ACC"] * 100, metrics["NMI"] * 100
                )
            )
            t.refresh()

        return self.mm.report(current=False), best_outputs

    def _get_graph_distribution(self, X: List[Tensor], M: Tensor):
        S_view = [
            calculate_optimized_p_cond(x[m], math.log2(self.ppl), dev=self.device)
            for x, m in zip(X, M.T)
        ]
        P_view = [make_joint(s) for s in S_view]
        return S_view, P_view

    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function computes the affinity matrix W using the Gaussian kernel.

        Args:
            X (torch.Tensor):   The input data

        Returns:
            torch.Tensor: The affinity matrix W
        """

        is_local = self.is_local_scale
        n_neighbors = self.n_nbg
        scale_k = self.scale_k
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(
            Dx, scale, indices, device=self.device, is_local=is_local
        )
        return W

    def _get_data_loader(self) -> tuple:
        """
        This function returns the data loaders for training, validation and testing.

        Returns:
            tuple:  The data loaders
        """
        n = self.X[0].shape[0]
        if self.y is None:
            self.y = torch.zeros(n)

        dataset = TensorDataset(*self.X, self.M)
        train_dataset = dataset
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        ortho_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, ortho_loader, test_loader

    @torch.no_grad()
    def evaluate(self, X: List[torch.Tensor], y: torch.Tensor, M: torch.Tensor, **kwargs):
        self.spectral_net.eval()
        clusterNum = len(np.unique(y))
        X = convert_tensor(X, dtype=torch.float)
        # y = convert_tensor(y, dtype=torch.long)
        M = convert_tensor(M, dtype=torch.bool)
        dataset = TensorDataset(*X, M)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        Y, H, Y_hat = self.predict(test_loader)
        _, ypred = KMeans_Torch(X=Y, n_clusters=clusterNum)
        ypred = convert_numpy(ypred)
        metrics = get_all_metrics(y, ypred)
        sort_idx = np.argsort(y)
        best_outputs = convert_numpy(
            dict(Y=Y[sort_idx], H=H[sort_idx], Y_hat=Y_hat[sort_idx])
        )
        return metrics, best_outputs

    @torch.no_grad()
    def predict(self, test_loader):
        """Predicts the cluster assignments for the given data.

        Parameters
        ----------
        X : torch.Tensor
            Data to be clustered.

        Returns
        -------
        np.ndarray
            The cluster assignments for the given data.
        """
        Y_list = []
        H_list = []
        Y_hat_list = []
        Q_cluster_list = []
        for *X, M in test_loader:
            X = [x.to(self.device) for x in X]
            M = M.to(self.device)
            S, P = self._get_graph_distribution(X, M)
            Y, H, Q_cluster = self.spectral_net(
                X, M, S=S, should_update_orth_weights=False
            )
            Y_list.append(Y.detach().cpu())
            H_list.append(H.detach().cpu())
            Q_cluster_list.append(Q_cluster.detach().cpu())

        Y = torch.cat(Y_list, 0)
        Y = F.normalize(Y)
        H = torch.cat(H_list, 0)
        H = F.normalize(H)
        Q_cluster = torch.cat(Q_cluster_list, 0)
        return Y, H, Q_cluster
