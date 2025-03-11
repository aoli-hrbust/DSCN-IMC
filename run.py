"""
Deep Spectral Clustering Network for Incomplete Multi-view Clustering (DSCN-IMC)
"""

import traceback

from models.DSCN import train_main

from utils.io_utils import *
from joblib import Parallel, delayed

import warnings


warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    root = P("./json/")


    def func(kwargs: dict):
        savedir = root.joinpath(encode_path(**kwargs))
        if (
            savedir.exists() and savedir.joinpath("metrics.json").exists()
        ):
            return

        dataname = kwargs["dataname"]
        eta = kwargs.pop("eta")
        datapath = P("data/dataset").joinpath(
            dataname
        )

        try:
            train_main(datapath=datapath, eta=eta / 100, lr=0.0001, epochs=50, batch_size=1024, k=15, savedir=savedir,
                       save_vars=False, save_history=False, **kwargs)
        except:
            traceback.print_exc()


    Parallel(n_jobs=1, verbose=999)(
        delayed(func)(kwargs)
        for kwargs in kv_product(
            idx=range(5),
            eta=[10],
            dataname=[
                 "USPS-MNIST.mat",
            ],
        )
    )
