"""
Deep Spectral Clustering Network for Incomplete Multi-view Clustering (DSCN-IMC)
"""

from utils.torch_utils import get_device
from ._spectralnet_trainer import SpectralTrainer
from data.dataset import (
    PartialMultiviewDataset,
    train_test_split
)
from utils.io_utils import (
    train_begin,
    save_variables,
    save_var,
    train_end
)
from utils.metrics import *



def train_main(
    datapath=None,
    eta=0.5,
    views=None,
    lr: float = 0.0001,
    epochs: int = 30,
    batch_size: int = 1024,
    eval_epochs: int = 5,
    k: int = 15,
    use_gcn: bool = True,
    is_sparse: bool = False,
    device=get_device(),
    lamda: float = 0.1,
    savedir: P = P("output/debug/"),
    save_vars: bool = False,
    save_history: bool = False,
    unseen_samples: bool = False,
    train_size: float = 0.9,
    aug_size=10,
    **kwargs,
):
    if train_size < 0.5:
        raise ValueError(f'train_size too small: {train_size}')

    if unseen_samples and train_size == 1.0:
        unseen_samples = False  # Train with all samples.

    method = "DSCN-IMC"
    ppl = k
    config = dict(
        datapath=datapath,
        eta=eta,
        views=views,
        method=method,
        batch_size=batch_size,
        lr=lr,
        k=k,
        eval_epochs=eval_epochs,
        epochs=epochs,
        device=device,
        use_gcn=use_gcn,
        is_sparse=is_sparse,
        train_size=train_size,
        lamda=lamda,
        unseen_samples=unseen_samples,
        aug_size=aug_size,
    )
    train_begin(savedir, config, f"Begin train {method}")

    data = PartialMultiviewDataset(
        datapath=datapath,
        paired_rate=1 - eta,
        view_ids=views,
        normalize="center",
    )

    spectral_epochs: int = epochs  # 30
    spectral_lr: float = 1e-3
    spectral_batch_size: int = batch_size  # 1024
    spectral_n_nbg: int = 30
    spectral_scale_k: int = k  # 15
    spectral_is_local_scale: bool = True
    spectral_hiddens: list = [1024, 1024, 512, data.clusterNum]


    spectral_config = {
        "epochs": spectral_epochs,
        "lr": spectral_lr,
        "n_nbg": spectral_n_nbg,
        "scale_k": spectral_scale_k,
        "is_local_scale": spectral_is_local_scale,
        "batch_size": spectral_batch_size,
        "architecture": spectral_hiddens,
        "use_gcn": use_gcn,
        "ppl": ppl,
        "lamda": lamda,
        'aug_size': aug_size,
    }

    trainer = SpectralTrainer(
        input_dim=data.view_dims,
        clusterNum=data.clusterNum,
        config=spectral_config, device=device, is_sparse=is_sparse
    )

    if unseen_samples:
        train_idx, test_idx = train_test_split(
            range(data.sampleNum), train_size=train_size, shuffle=True
        )
        train_X = [data.X[v][train_idx] for v in range(data.viewNum)]
        test_X = [data.X[v][test_idx] for v in range(data.viewNum)]
        train_y = data.Y[train_idx]
        test_y = data.Y[test_idx]
        train_M = data.mask[train_idx, :]
        test_M = data.mask[test_idx, :]

        begin = time.time()
        trainer.train(X=train_X, y=train_y, M=train_M, data=data)
        metrics, outputs = trainer.evaluate(
            X=test_X,
            y=test_y,
            M=test_M,
        )

        T = time.time() - begin
        metrics["T"] = T
    else:
        begin = time.time()
        metrics, outputs = trainer.train(X=data.X, y=data.Y, M=data.mask, data=data)
        T = time.time() - begin
        metrics["T"] = T

    history = trainer.history

    if save_vars:
        save_variables(savedir, outputs)

    if save_history:
        save_var(savedir, history, "history")

    train_end(savedir, metrics, method)
