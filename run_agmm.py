"""Need to install https://github.com/microsoft/AdversarialGMM"""

import argparse
import itertools
import os
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mliv.dgps import fn_dict, get_data, get_tau_fn
from mliv.neuralnet import AGMM
from mliv.neuralnet.deepiv_fit import deep_iv_fit
from mliv.neuralnet.rbflayer import gaussian, inverse_multiquadric
from mliv.neuralnet.utilities import log_metrics, plot_results
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from dgp.mc1 import MC1
from dgp.mc2 import MC2
from dgp.mc2a import MC2a
from dgp.mc3 import MC3

warnings.simplefilter("ignore")


def train_network(Y_train, T_train, Z_train, lr=1e-3, epochs=500, n_hidden=30, g_features=30):
    n_t = T_train.shape[1]
    n_z = Z_train.shape[1]

    p = 0.1  # dropout prob of dropout layers throughout notebook

    learner = nn.Sequential(
        # nn.Dropout(p=p),
        nn.Linear(n_t, n_hidden),
        nn.ReLU(),
        # nn.Dropout(p=p),
        nn.Linear(n_hidden, 1),
    )

    # For any method that use a projection of z into features g(z)
    # adversary_g = nn.Sequential(
    #     nn.Dropout(p=p),
    #     nn.Linear(n_z, g_features),
    #     nn.ReLU(),
    #     nn.Dropout(p=p),
    #     nn.Linear(g_features, g_features),
    #     nn.ReLU(),
    # )
    # The kernel function
    # kernel_fn = gaussian

    # For any method that uses an unstructured adversary test function f(z)
    adversary_fn = nn.Sequential(
        nn.Dropout(p=p),
        nn.Linear(n_z, g_features),
        nn.ReLU(),
        nn.Dropout(p=p),
        nn.Linear(g_features, 1),
    )

    # Looks like it works for n = 1000
    learner_lr = lr
    adversary_lr = lr
    learner_l2 = 0  # 1e-3
    adversary_l2 = 1e-4
    # adversary_norm_reg = 1e-3
    bs = 100
    # sigma = 2.0 / g_features
    # n_centers = 100
    device = torch.cuda.current_device() if torch.cuda.is_available() else None

    np.random.seed(12356)
    agmm = AGMM(learner, adversary_fn).fit(
        Z_train,
        T_train,
        Y_train,
        learner_lr=learner_lr,
        adversary_lr=adversary_lr,
        learner_l2=learner_l2,
        adversary_l2=adversary_l2,
        n_epochs=epochs,
        bs=bs,
        logger=None,
        model_dir="agmm_model",
        device=device,
    )

    trained_network = deepcopy(agmm.learner).eval()
    diff = torch.zeros_like(T_train)
    diff_grad = torch.zeros((len(T_train), 1), requires_grad=True)
    diff[:, [0]] += diff_grad
    trained_network(T_train + diff).sum().backward()
    wad = diff_grad.grad.mean().item()
    return wad


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dgp", type=str, default="mc2")
    parser.add_argument("--h-width", type=int, default=30)
    parser.add_argument("--inst-width", type=int, default=30)
    args = parser.parse_args()
    seed = args.seed
    if seed == "":
        seed = os.environ.get("LSB_JOBINDEX")
    seed = int(seed)
    lr = {1000: 1e-3, 5000: 1e-4}

    dgp_dict = {"mc2": MC2, "mc2a": MC2a, "mc3": MC3}
    dgp_constructor = dgp_dict[args.dgp]

    result = []
    for n in [1000, 5000]:
        for dim in [0, 5, 10]:
            for corr in [0, 0.5, 0.8]:
                if dim == 0 and corr > 0:
                    continue

                dgp = dgp_constructor(n=n, batch_size=n, dimension=dim, corr=corr, device="cpu")
                torchvec = dgp.data(seed)[1]
                Y_train = torchvec["response"]
                T_train = torchvec["endogenous"]
                Z_train = torchvec["instrument"]
                wad_tab2 = train_network(
                    Y_train=Y_train,
                    T_train=T_train,
                    Z_train=Z_train,
                    lr=lr[n],
                    epochs=args.epochs,
                    n_hidden=args.h_width,
                    g_features=args.inst_width,
                )

                result.append(
                    pd.Series(
                        [n, dim, corr, wad_tab2, seed],
                        index=["n", "dim", "corr", "wad_t2", "seed"],
                    )
                )
    result = pd.DataFrame(result)
    result.to_csv(
        f"checkpts/agmm_{seed}_{args.dgp}_{args.h_width}_{args.inst_width}.csv",
        index=False,
    )
