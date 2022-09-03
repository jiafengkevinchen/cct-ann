import pandas as pd
import argparse
import itertools
import json
import os
import datetime
import warnings
from glob import glob

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# from xfit import *
from pipeline.xfit import *

from run_model import generate_parser, main


from pipeline.callbacks import logger
from pipeline.pipeline import train_loop, weight_fn, compute_se
from utils.parseconfigs import callback_dict, preprocess, stopping_criterion_dict
from utils.postprocessing import get_filename, save_results
from pathlib import Path

from submit_experiments import fit, fit_boot

warnings.simplefilter("ignore", FutureWarning)

CONFIG_PATH = Path("configs")
OUTPUT_PATH = Path("checkpts")


##############################
##############################
##############################
###### Helper Functions ######
##############################
##############################
##############################


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def generate_config(
    config_options, master, overwrite=False, change_width=False, **kwargs
):
    names = []
    for options in product_dict(**config_options):

        name = master + "_" + str(hash(str(options)))
        name = name.replace(" ", "")

        with open(f"configs/{master}.json", "r") as f:
            configs = json.load(f)

        configs.update(options)

        with open(f"configs/{name}.json", "w") as f:
            json.dump(configs, f)
        names.append(name)
    return names


def table(
    seed, config_options, master, boot=False, B=100, change_width=False, **kwargs
):

    config_names = generate_config(config_options, master, change_width=change_width)
    sample = []
    bootstrap = []
    bootstrap_exp = []

    # raises Value error if during bootstrap input config has multiple models
    if (boot == True) & (len(config_names) > 1):
        raise ValueError("For Bootstrap use 1 model")

    else:

        for config in tqdm(config_names):
            with open(f"configs/{config}.json", "r") as f:
                configs = json.load(f)
                print("PARAMETERS:")
                print(configs)

            if boot == True:
                print(f"Fit Boot for config:{config}")
                for b in tqdm(range(B)):

                    boot_result, _ = fit_boot(b, seed=seed, config=config, exp=False)

                    bootstrap.append(pd.Series(boot_result))

            if boot == False:
                print(f"Fit model for config:{config}")

                result = fit(seed, config, **kwargs)
                result.update(seed=seed, config=config, **configs)
                is_beta, is_se = whole_sample(seed, config, **kwargs)
                result["IS"], result["IS_se"] = is_beta, is_se

                sample.append(pd.Series(result))

        return (
            pd.DataFrame(sample),
            pd.DataFrame(bootstrap),
            pd.DataFrame(bootstrap_exp),
        )


##############################
##############################
##############################
##### Testing many models ####
##############################
##############################
##############################

# processed config examples

STR_CONFIGS = {
    "arch_depth": [1, 3],
    "arch_width": [15, 30],
    "arch_hidden_activation": ["sigmoid", "relu"],
    "opt_learning_rate": [0.01],
    "regularizer": ["none", "yes"],
    "train_max_epoch": [5000, 10000],
    "callback": ["callback"],
}

SIG1_CONFIGS = {
    "arch_depth": [1],
    "arch_width": [15],
    "arch_hidden_activation": ["sigmoid"],
    "opt_learning_rate": [0.05],
    "regularizer": ["none"],
    "train_max_epoch": [2500],
    "callback": ["none"],
}

REL3_CONFIGS = {
    "arch_depth": [3],
    "arch_width": [30],
    "arch_hidden_activation": ["relu"],
    "opt_learning_rate": [0.05],
    "regularizer": ["none"],
    "train_max_epoch": [2500],
    "callback": ["none"],
}


# runs multiple models generated from STR_CONFIGS file on non organic
# strawberry data and saves their performance
def experiments_strawberry(
    seed, change_width=False, model_params=STR_CONFIGS, **kwargs
):
    sample, _, _ = table(
        seed=seed,
        config_options=dict(model_name=["nonparametric"], **model_params, **kwargs),
        master="Compiani",
        change_width=change_width,
        boot=False,
    )
    date = (
        datetime.datetime.now().minute,
        datetime.datetime.now().hour,
        datetime.datetime.now().day,
        datetime.datetime.now().month,
    )
    sample.to_csv(f"checkpts/table_strawberry_{date}.csv", index=False)

    return sample


# runs multiple models generated from STR_CONFIGS file on organic
# strawberry data and saves their performance
def experiments_org_strawberry(
    seed, change_width=False, model_params=STR_CONFIGS, **kwargs
):
    sample, _, _ = table(
        seed=seed,
        config_options=dict(model_name=["nonparametric"], **model_params, **kwargs),
        master="Compiani_org",
        change_width=change_width,
        boot=False,
    )
    date = (
        datetime.datetime.now().minute,
        datetime.datetime.now().hour,
        datetime.datetime.now().day,
        datetime.datetime.now().month,
    )

    sample.to_csv(f"checkpts/table_strawberry_org_{date}.csv", index=False)

    return sample


##############################
##############################
##############################
########## Bootstrap #########
##############################
##############################
##############################


# saves and outputs bootstrap results for non-organic strawberry
# requires a model config file, I recommend using a single model
# since bootstrap is computationally heavy


def bootstrap_strawberry(seed, model_params=SIG1_CONFIGS, organic=False, **kwargs):
    config_names = generate_config(
        config_options=dict(model_name=["nonparametric"], **model_params, **kwargs),
        master="Compiani" if not organic else "Compiani_org",
        change_width=False,
    )
    bootstrap = []

    for config in config_names:
        with open(f"configs/{config}.json", "r") as f:
            configs = json.load(f)

        boot_result, _ = fit_boot(seed, seed=1, config=config, exp=False)
        boot_result.update(configs)
        bootstrap.append(pd.Series(boot_result))

    date = (
        datetime.datetime.now().minute,
        datetime.datetime.now().hour,
        datetime.datetime.now().day,
        datetime.datetime.now().month,
    )

    pd.DataFrame(bootstrap).to_csv(
        f"checkpts/boot_table_strawberry_{date}_{'organic' if organic else ''}_{int(np.random.rand()*10000)}.csv",
        index=False,
    )

    return bootstrap


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="")
    args = parser.parse_args()

    seed = args.seed
    if seed == "":
        seed = os.environ.get("LSB_JOBINDEX")
    seed = int(seed)

    # dictionary which stores model parameters for experiments with
    # different types of architectures
    EXP_CONFIGS = {
        "arch_depth": [1, 3],
        "arch_width": [15, 30],
        "arch_hidden_activation": ["sigmoid", "relu"],
        "opt_learning_rate": [0.01],
        "regularizer": ["none", "yes"],
        "train_max_epoch": [5000, 10000],
        "callback": ["none"],
    }

    # dictionary which stores model parameters for bootstrap
    # it should strictly contain a single architecture
    # otherwise bootstrap wont run

    BOOT_CONFIG_1 = {
        "arch_depth": [1],
        "arch_width": [15],
        "arch_hidden_activation": ["relu"],
        "opt_learning_rate": [0.01],
        "regularizer": ["none"],
        "train_max_epoch": [5000],
        "callback": ["none"],
    }

    BOOT_CONFIG_2 = {
        "arch_depth": [3],
        "arch_width": [15],
        "arch_hidden_activation": ["relu"],
        "opt_learning_rate": [0.05],
        "regularizer": ["none"],
        "train_max_epoch": [10000],
        "callback": ["none"],
    }

    BOOT_CONFIG_3 = {
        "arch_depth": [1],
        "arch_width": [15],
        "arch_hidden_activation": ["sigmoid"],
        "opt_learning_rate": [0.01],
        "regularizer": ["none"],
        "train_max_epoch": [5000],
        "callback": ["none"],
    }

    BOOT_CONFIG_4 = {
        "arch_depth": [3],
        "arch_width": [30],
        "arch_hidden_activation": ["sigmoid"],
        "opt_learning_rate": [0.05],
        "regularizer": ["none"],
        "train_max_epoch": [10000],
        "callback": ["none"],
    }

    ##############################
    ##############################
    ##############################
    ########## Run Models ########
    ##############################
    ##############################
    ##############################

    # runs multiple ANNs with different architectures
    #     experiments_strawberry(seed=1, change_width=False, model_params = EXP_CONFIGS)

    #     experiments_org_strawberry(seed=1, change_width=False, model_params = EXP_CONFIGS)

    # Runs bootstrapped models for 4 models in XC table
    bootstrap_strawberry(seed=seed, model_params=BOOT_CONFIG_1)
    bootstrap_strawberry(seed=seed, organic=True, model_params=BOOT_CONFIG_1)

    bootstrap_strawberry(seed=seed, model_params=BOOT_CONFIG_2)
    bootstrap_strawberry(seed=seed, organic=True, model_params=BOOT_CONFIG_2)

    bootstrap_strawberry(seed=seed, model_params=BOOT_CONFIG_3)
    bootstrap_strawberry(seed=seed, organic=True, model_params=BOOT_CONFIG_3)

    bootstrap_strawberry(seed=seed, model_params=BOOT_CONFIG_4)
    bootstrap_strawberry(seed=seed, organic=True, model_params=BOOT_CONFIG_4)

