import json
import pickle

import numpy as np
import pandas as pd


def to_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def to_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def save_results(
    outdir, filename, config_lst, results, logger, no_config_save=False, **kwargs
):
    (_, config, _, model_config, _, optimizer_config) = config_lst

    # don't save all the bootstrap weights, seed is enough
    if "bootstrap_weights" in model_config:
        del model_config["bootstrap_weights"]

    configs = {
        "config": config,
        "model_config": model_config,
        "optimizer_config": optimizer_config,
    }

    timestamp = config["timestamp"]
    results["filename"] = filename
    results["dataset"] = config["dataset"]
    results["bootstrap_seed"] = config["bootstrap_seed"]
    results["depth"] = config["arch_depth"]
    results["width"] = config["arch_width"]
    results["activation"] = config["arch_hidden_activation"]
    results["dimension"] = (
        config["data_nuisance_dimension"] if "data_nuisance_dimension" in config else None
    )
    results["corr"] = config["data_corr"] if "data_corr" in config else None

    random_str = str(int(np.random.rand() * 1_000_000_000_000))

    to_json(results, outdir / f"{timestamp}_{random_str}.results.json")

    if not no_config_save:
        to_pickle(configs, outdir / f"{timestamp}_{random_str}.configs.pickle")

    if logger is not None:
        pd.DataFrame(logger).to_csv(outdir / f"{timestamp}_{random_str}.csv", index=False)

    if len(kwargs) > 0:
        to_pickle(kwargs, outdir / f"{timestamp}_{random_str}.misc.json")

    print(f"Saving to {outdir}/{timestamp}_{random_str}*")


def get_filename(config, *args):
    model_name = config["model_name"]
    dataset = config["dataset"]
    timestamp = config["timestamp"]
    seed = config["seed"]
    bootstrap_seed = (
        "bootstrap_" + str(config["bootstrap_seed"])
        if config["bootstrap_seed"] is not None
        else ""
    )
    extra = "_" + "_".join(args) if len(args) > 0 else ""
    extra = "" if len(extra) <= 1 else extra

    filename = f"{model_name}_{dataset}_{seed}_{bootstrap_seed}{extra}_{timestamp}"
    return filename
