import sys
import argparse
import json
import os

import pandas as pd
from submit_experiments import generate_config
from pipeline.xfit import (
    whole_sample_optimal,
    half_sample,
    _half_sample,
    half_sample_osmd,
    _half_sample_optimal_vstar,
    whole_sample_optimal_v_star,
    nfold_osmd,
)


def score_experiment(seed, config, n, neighbors=lambda x: 50 if x == 1000 else 100):
    # Contains ES, ID, IDSPL without sample splitting
    result_obj = whole_sample_optimal(seed, config, n=n)  # IS and ES 5NN
    neighbors = neighbors()
    es_high_neighbor = whole_sample_optimal_v_star(seed, config, n=n, n_neighbors=neighbors)
    result_obj.update(es_high_neighbor)

    # ID with xfit
    result_obj["id_score_xfit"], result_obj["id_score_xfit_se"] = half_sample(
        seed, config, n=n, fit_func=_half_sample
    )

    # ES with xfit
    def fit_func(a, b):
        return _half_sample_optimal_vstar(a, b, n_neighbors=neighbors)

    es_x = half_sample_osmd(seed, config, n=n, fit_func=fit_func)
    result_obj.update(es_x)
    result_obj["n_neighbors"] = neighbors

    return result_obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=str, default="")
    # ap.add_argument("--change-width", action="store_true")
    # ap.add_argument("--mc-configs", type=str, default="mc_configs")
    ap.add_argument("--two-fold-small-neighbor", action="store_true")
    ap.add_argument("--five-fold", action="store_true")
    args = ap.parse_args()

    seed = args.seed
    if seed == "":
        seed = os.environ.get("LSB_JOBINDEX")
    seed = int(seed)

    with open("configs/mc_configs_3l.json", "r") as f:
        m3l = json.load(f)

    configs = generate_config(
        config_options=dict(model_name=["nonparametric"], **m3l),
        master="mc2",
        overwrite=False,
        change_width=False,
    )

    with open("configs/mc_configs_simple.json", "r") as f:
        m1lvw = json.load(f)

    configs.extend(
        generate_config(
            config_options=dict(model_name=["nonparametric"], **m1lvw),
            master="mc2",
            overwrite=False,
            change_width=True,
        )
    )

    if args.five_fold:
        results = []
        for cf in configs:
            for n in [1000, 5000]:
                for n_neighbors in [5, 50 if n == 1000 else 100]:
                    res = nfold_osmd(
                        5,
                        seed,
                        cf,
                        n=n,
                        fit_func=lambda a, b: _half_sample_optimal_vstar(
                            a, b, n_neighbors=n_neighbors
                        ),
                    )
                    res["n_neighbors"] = n_neighbors

                    with open(f"configs/{cf}.json", "r") as f:
                        conf = json.load(f)
                    res.update(conf)
                    res["n"] = n
                    res["seed"] = seed
                    results.append(pd.Series(res))
        pd.DataFrame(results).to_csv(
            f"checkpts/five_fold_score_experiments_seed_{seed}.csv", index=False
        )
        sys.exit(0)

    if args.two_fold_small_neighbor:
        results = []
        for cf in configs:
            for n in [1000, 5000]:
                for n_neighbors in [5, 50 if n == 1000 else 100]:

                    # Half sample OSMD
                    res = half_sample_osmd(
                        seed,
                        cf,
                        n=n,
                        fit_func=lambda a, b: _half_sample_optimal_vstar(
                            a, b, n_neighbors=n_neighbors
                        ),
                    )

                    # Full sample OSMD
                    res_full = whole_sample_optimal_v_star(seed, cf, n=n, n_neighbors=n_neighbors)

                    res.update(res_full)
                    res["n_neighbors"] = n_neighbors

                    with open(f"configs/{cf}.json", "r") as f:
                        conf = json.load(f)
                    res.update(conf)
                    res["n"] = n
                    res["seed"] = seed
                    results.append(pd.Series(res))
        pd.DataFrame(results).to_csv(f"checkpts/score_experiments_seed_{seed}.csv", index=False)
        sys.exit(0)

    results = []
    for cf in configs:
        for n in [1000, 5000]:
            res = score_experiment(seed, cf, n=n)
            with open(f"configs/{cf}.json", "r") as f:
                conf = json.load(f)
            res.update(conf)
            res["n"] = n
            res["seed"] = seed
    pd.DataFrame(results).to_csv(f"checkpts/score_experiments_seed_{seed}.csv", index=False)
