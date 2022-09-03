import datetime
import json
import os
from pathlib import Path

import numpy as np
import torch
from architecture.architectures import (
    Nonparametric,
    PartiallyAdditive,
    PartiallyAdditiveWithSpline,
    PartiallyLinear,
)
from dgp.gasdemand import GasDemand

from dgp.mc1 import MC1
from dgp.mc2 import MC2
from dgp.mc2a import MC2a
from dgp.mc3 import MC3
from dgp.mc4 import MC4
from dgp.calibrated_gasdemand import CalibratedGasDemand

from dgp.compiani import Strawberry
from dgp.compiani_org import Strawberry_org
from pipeline.callbacks import callback, log_callback, tensorboard_callback
from pipeline.pipeline import (
    l2_regularizer,
    l1_regularizer,
    null_regularizer,
    stopping_criterion,
)
from torch import nn


def get_regularizer(cfg):
    if cfg == "none":
        return null_regularizer
    elif cfg[0] == "l1":
        _, lmbda = cfg
        return lambda m: l1_regularizer(m, lmbda=lmbda)
    elif cfg[0] == "l2":
        _, lmbda = cfg
        return lambda m: l2_regularizer(m, lmbda=lmbda)


# Dictionaries for options
dataset_dict = {
    "mc1": MC1,
    "mc2": MC2,
    "mc3": MC3,
    "mc2a": MC2a,
    "mc4": MC4,
    "gasdemand": GasDemand,
    "strawberry": Strawberry,
    "strawberry_org": Strawberry_org,
    "calibrated_gasdemand": CalibratedGasDemand,
}

activation_dict = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}
callback_dict = {
    "callback": callback,
    "tensorboard_callback": tensorboard_callback,
    "log_callback": log_callback,
    "none": None,
}

optimizer_dict = {"gd": torch.optim.SGD, "adam": torch.optim.Adam}
stopping_criterion_dict = {"stopping_criterion": stopping_criterion}
model_dict = {
    "partiallylinear": PartiallyLinear,
    "nonparametric": Nonparametric,
    "partiallyadditive": PartiallyAdditive,
    "partiallyadditivewithspline": PartiallyAdditiveWithSpline,
}

CONFIG_PATH = Path("configs")


def preprocess(parser, configpath=CONFIG_PATH, provided_args=None):
    """Parsing arguments and add in device as an argument"""
    args = (
        parser.parse_args(provided_args) if provided_args is not None else parser.parse_args()
    )

    args.device = None

    # Add device if not specified
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    config = read_config(configpath / f"{args.config}.json", args)
    dgp_args = parse_dgp_args(config, args)
    dgp, data, corr_mat = generate_data(config, dgp_args)

    model_config = parse_model_args(input_dim=data[0]["endogenous"].shape[1], config=config)
    if "spline" in config["model_name"]:
        model_config["knot_locs"] = dgp.endogenous_knots

    train_step_kwargs = parse_train_step_args(model_config, config)
    optimizer_config, optimizer_constructor, model, optimizer = initialize_model(
        args, config, model_config
    )

    config_lst = (
        args,
        config,
        dgp_args,
        model_config,
        train_step_kwargs,
        optimizer_config,
    )
    dgp_objects = (dgp, data, corr_mat)
    model_objects = (model, optimizer_constructor, optimizer)

    if "moment_function" in model_config:
        del model_config["moment_function"]

    return config_lst, dgp_objects, model_objects


def read_config(path, args):
    if args.no_tqdm:
        os.environ["TQDM"] = "False"
    else:
        os.environ["TQDM"] = "True"

    with open(path, "r") as f:
        config = json.load(f)
        config = {k: v.lower() if type(v) is str else v for k, v in config.items()}
    config["n"] = args.n

    # Get seed
    if args.seed == -1:
        config["seed"] = os.environ.get("LSB_JOBINDEX")
        if config["seed"] is None:
            raise ValueError("Must pass in seed")
        config["seed"] = int(config["seed"])
    else:
        config["seed"] = args.seed

    # Want bootstrap but didn't pass in seed, assume seed come from JOBINDEX
    if args.bootstrap_seed == -1 and args.bootstrap:
        assert args.seed != -1
        config["bootstrap_seed"] = int(os.environ.get("LSB_JOBINDEX"))
    elif args.bootstrap:
        config["bootstrap_seed"] = args.bootstrap_seed
    else:
        config["bootstrap_seed"] = None

    if args.exp_bootstrap:
        config["exp_bootstrap"] = True
    else:
        config["exp_bootstrap"] = False

    config["batch_size"] = args.batch_size if args.batch_size > 0 else config["n"]
    config["timestamp"] = (
        str(datetime.datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")
    )
    return config


def parse_dgp_args(config, args):
    if config["dataset"] == "gasdemand":
        return {"covariate": config["covariates"]}

    # Arguments to the DGP
    dgp_args = dict(n=config["n"], batch_size=config["batch_size"], device=args.device)
    if "data_nuisance_dimension" in config:
        dgp_args["dimension"] = config["data_nuisance_dimension"]
    if "data_nuisance_relevant" in config:
        dgp_args["high_dim_relevant"] = config["data_nuisance_relevant"]
    if "data_corr" in config:
        dgp_args["corr"] = config["data_corr"]
    if "timeseries_kwargs" in config:
        dgp_args.update(config["timeseries_kwargs"])

    return dgp_args


def parse_model_args(input_dim, config):
    if "partially" in config["model_name"]:
        input_dim -= 1

    model_config = dict(
        input_dim=input_dim,
        depth=config["arch_depth"],
        width=config["arch_width"],
        hidden_activation=activation_dict[config["arch_hidden_activation"]],
    )

    return model_config


def generate_data(config, dgp_args):
    data_constructor = dataset_dict[config["dataset"]]
    dgp = data_constructor(**dgp_args)

    if (
        (config["dataset"] == "gasdemand")
        or (config["dataset"] == "strawberry")
        or (config["dataset"] == "strawberry_org")
    ):
        config["n"] = dgp.n
    assert config["n"] == dgp.n or config["n"] == dgp.n - dgp.n_lags

    npvec, torchvec, dataset, loader = dgp.data(config["seed"])
    corr_mat = np.corrcoef(np.c_[npvec["instrument"]], rowvar=False)
    return dgp, (npvec, torchvec, dataset, loader), corr_mat


def parse_train_step_args(model_config, config):
    """Modifies model_config and returns train_config"""
    train_step_kwargs = dict(regularizer=get_regularizer(config["regularizer"]))
    if config["bootstrap_seed"] is not None:
        rng = np.random.RandomState(config["bootstrap_seed"])

        # 2-pt weights
        if not config["exp_bootstrap"]:
            np_weights = np.where(rng.rand(config["n"]) < 4 / 5, 1 / 2, 3)
            bootstrap_weights = torch.tensor(np_weights).unsqueeze(1).float()
        else:
            np_weights = -np.log(rng.rand(config["n"]))
            bootstrap_weights = torch.tensor(np_weights).unsqueeze(1).float()

        train_step_kwargs["bootstrap_weights"] = bootstrap_weights
        model_config["bootstrap_weights"] = bootstrap_weights

    return train_step_kwargs


def initialize_model(args, config, model_config):
    torch.manual_seed(config["seed"])
    optimizer_config = dict(
        lr=config["opt_learning_rate"], weight_decay=config["opt_weight_decay"]
    )
    model = model_dict[config["model_name"]](**model_config).to(args.device)

    optimizer_constructor = (
        optimizer_dict[config["optimizer"]] if "optimizer" in config else "adam"
    )
    optimizer = optimizer_constructor(model.parameters(), **optimizer_config)

    return optimizer_config, optimizer_constructor, model, optimizer
