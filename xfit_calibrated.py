import argparse
import json
import os

import pandas as pd

from submit_experiments import generate_config
from xfit import score_experiment

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=str, default="")
    args = ap.parse_args()

    seed = args.seed
    if seed == "":
        seed = os.environ.get("LSB_JOBINDEX")
    seed = int(seed)

    mc_configs = {
        "arch_depth": [1, 3],
        "arch_width": [10, 20],
        "arch_hidden_activation": ["sigmoid", "relu"],
        "opt_learning_rate": [0.005],
    }

    configs = generate_config(
        config_options=dict(model_name=["nonparametric"], **mc_configs),
        master="calibrated_gasdemand",
        overwrite=False,
        change_width=False,
    )

    results = []
    for cf in configs:
        for n in [5000, 10000]:
            res = score_experiment(seed, cf, n=n, neighbors=lambda x: 150 if n == 5000 else 300)
            with open(f"configs/{cf}.json", "r") as f:
                conf = json.load(f)
            res.update(conf)
            res["n"] = n
            res["seed"] = seed

            print(res)
            results.append(pd.Series(res))

    pd.DataFrame(results).to_csv(
        f"checkpts/calibrated_gasdemand_score_experiments_seed_{seed}.csv", index=False
    )
