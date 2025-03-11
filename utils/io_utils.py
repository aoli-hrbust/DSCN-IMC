# MIT License

# Copyright (c) 2025 Ao Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import itertools
import logging
import pickle
from pathlib import Path as P
from pprint import pformat
import jsons


def encode_path(**kwargs):
    items = sorted(kwargs.items(), key=lambda x: x[0])
    return "-".join([f"{k}={v}" for k, v in items])


def kv_product(**kwargs):
    """
    >>> for kwargs in kv_product(a='abc', b='xyz'): print(kwargs)
    """
    for val in itertools.product(*kwargs.values()):
        yield dict(zip(kwargs.keys(), val))


def save_var(savedir: P, var, name: str):
    """
    Save a single variable to savedir.
    """
    savedir.mkdir(exist_ok=1, parents=1)
    f = savedir.joinpath(name).with_suffix(".pkl")
    pickle.dump(var, f.open("wb"))
    logging.info(f"Save Var to {f}")


def save_variables(savedir: P, variables: dict):
    for key, val in variables.items():
        save_var(savedir, val, key)


def save_json(savedir: P, var, name: str):
    savedir.mkdir(exist_ok=1, parents=1)
    f = savedir.joinpath(name).with_suffix(".json")
    f.write_text(jsons.dumps(var, jdkwargs=dict(indent=4)))
    logging.info(f"Save Var to {f}")


def train_begin(savedir: P, config: dict, message: str = None):
    message = message or "Train begins\n"
    logging.info(f"{message} {pformat(config)}")
    save_json(savedir, config, "config")


def train_end(savedir: P, metrics: dict, message: str = None):
    message = message or "Train ends"
    logging.info(f"{message} {metrics}")
    save_json(savedir, metrics, "metrics")


