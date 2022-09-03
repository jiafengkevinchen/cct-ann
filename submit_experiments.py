"""
Main experiments with SMD estimators. Computation is designed to be
distributed on a LSF cluster

Usage: python submit_experiments --seed 1 --tab1 --tab2 --tab2-mc2a \
                                 --tab2-mc3 --tab3 --tab4 --empirical
"""

import argparse
import itertools
import json
import os

import pandas as pd

from pipeline.pipeline import compute_se
from run_model import generate_parser, main


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def generate_config(config_options, master, overwrite=False, change_width=False, **kwargs):
    names = []
    for options in product_dict(**config_options):
        if (
            "data_corr" in options
            and "data_nuisance_dimension" in options
            and options["data_nuisance_dimension"] == 0
            and options["data_corr"] != 0
        ):
            continue
        name = master + "_" + str(hash(str(options)))
        name = name.replace(" ", "")

        if os.path.exists(f"configs/{name}.json") and not overwrite:
            names.append(name)
            continue

        with open(f"configs/{master}.json", "r") as f:
            configs = json.load(f)

        configs.update(options)

        if change_width:
            configs["arch_width"] = options["data_nuisance_dimension"] + 3 + 1

        if "callback" not in config_options:
            configs["callback"] = "none"
        with open(f"configs/{name}.json", "w") as f:
            json.dump(configs, f)
        names.append(name)
    return names


def fit(seed, config, weighting=True, n=5000):
    """For seed and config, estimate and return results"""

    # SMD results
    parser = generate_parser()
    main_locals = main(
        parser,
        provided_args=(
            f"--n {n} --seed {seed} --no-save "
            f"--no-tqdm --no-logger --no-config-save --config {config}"
        ).split(),
        return_locals=True,
        identity_only=not weighting,
    )
    results = main_locals["results"]

    # Arguments passed to splines
    spl_info = dict(
        npvec=main_locals["npvec"],
        deg=3,
        knots_inst=main_locals["dgp"].instrument_knots,
        knots_endo=main_locals["dgp"].endogenous_knots,
    )

    # Estimate analytic standard errors
    for p in [1, 2, 3, spl_info]:
        p_str = p if type(p) is not dict else "spline"
        if weighting:
            (results[f"se_poly_{p_str}_opt"], results[f"se_nonpar_poly_{p_str}_opt"],) = compute_se(
                main_locals["torchvec"],
                main_locals["model"],
                weights=main_locals.get("weights"),
                inefficient_derivative=main_locals.get("inefficient_derivative"),
                inefficient_prediction=main_locals.get("inefficient_prediction"),
                order=p,
                weighting=weighting,
            )

        id_weight_model = main_locals["model"]
        id_weight_model.load_state_dict(main_locals["inefficient_model"])

        (results[f"se_poly_{p_str}_id"], results[f"se_nonpar_poly_{p_str}_id"],) = compute_se(
            main_locals["torchvec"],
            id_weight_model,
            weights=main_locals.get("weights"),
            inefficient_derivative=main_locals.get("inefficient_derivative"),
            inefficient_prediction=main_locals.get("inefficient_prediction"),
            order=p,
            weighting=False,
        )
    return results


def fit_boot(bs, config, exp=True, seed=1, n=5000):
    """For boostrap_seed and config, estimate and return results"""

    parser = generate_parser()
    results = main(
        parser,
        provided_args=(
            f"--n {n} --seed {seed} "
            f"--bootstrap --bootstrap-seed {bs} --no-save "
            f"--no-tqdm --no-logger --no-config-save --config {config}"
        ).split(),
    )
    results["bootstrap_seed"] = bs

    results_exp = None
    if exp:
        results_exp = main(
            parser,
            provided_args=(
                f"--n {n} --seed {seed} "
                f"--bootstrap --bootstrap-seed {bs} --exp-bootstrap --no-save "
                f"--no-tqdm --no-logger --no-config-save --config {config}"
            ).split(),
        )

        results_exp["bootstrap_seed"] = bs
    return results, results_exp


def coverage(seed, config, n_boot):
    return_dict = {}
    for bs in range(1, n_boot + 1):
        results, _ = fit_boot(bs, config, exp=False, seed=seed)
        return_dict[bs] = results
    return return_dict


def table(seed, config_options, master, n=5000, exp=False, change_width=False, **kwargs):
    config_names = generate_config(config_options, master, change_width=change_width)

    sample = []
    bootstrap = []
    bootstrap_exp = []
    for config in config_names:
        with open(f"configs/{config}.json", "r") as f:
            configs = json.load(f)

        result = fit(seed, config, n=n, **kwargs)
        boot_result, boot_exp_result = fit_boot(seed, config, n=n, exp=exp)

        result.update(seed=seed, config=config, n=n, **configs)
        boot_result.update(seed=seed, config=config, n=n, **configs)

        sample.append(pd.Series(result))
        bootstrap.append(pd.Series(boot_result))

        if boot_exp_result is not None:
            boot_exp_result.update(seed=seed, config=config, n=n, **configs)
            bootstrap_exp.append(pd.Series(boot_exp_result))

    return pd.DataFrame(sample), pd.DataFrame(bootstrap), pd.DataFrame(bootstrap_exp)


def empirical(seed):
    result_dict = {}
    for config in ["gasoline_master", "gasoline_master_short"]:
        result_boot = fit_boot(bs=seed, config=config)
        result_dict[config] = {"bootstrap": result_boot}
        if str(seed) == "1":
            result_dict[config]["sample"] = fit(seed, config)
    return result_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="")
    # parser.add_argument("--name", type=str, default="")
    parser.add_argument("--mc-configs", type=str, default="mc_configs")
    parser.add_argument("--tab1", action="store_true")
    parser.add_argument("--tab2", action="store_true")
    parser.add_argument("--tab2-mc2a", action="store_true")
    parser.add_argument("--tab2-mc3", action="store_true")
    parser.add_argument("--tab3", action="store_true")
    parser.add_argument("--tab4", action="store_true")
    parser.add_argument("--empirical", action="store_true")
    parser.add_argument("--r1", action="store_true")
    parser.add_argument("--calibrated", action="store_true")
    # parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--change-width", action="store_true")
    parser.add_argument("--mc3-relu", action="store_true")
    args = parser.parse_args()

    with open(f"configs/{args.mc_configs}.json", "r") as f:
        MC_CONFIGS = json.load(f)

    def tab1(seed, n, change_width=False, **kwargs):
        sample, boot, boot_e = table(
            seed=seed,
            change_width=change_width,
            config_options=dict(
                model_name=["nonparametric"],
                **MC_CONFIGS,
                **kwargs,
            ),
            master="mc1",
            n=n,
        )

        sample.to_csv(f"checkpts/sample_tab1_{seed}_n_{n}.csv", index=False)
        boot.to_csv(f"checkpts/boot_tab1_{seed}_n_{n}.csv", index=False)

        if len(boot_e) > 0:
            boot_e.to_csv(f"checkpts/bootexp_tab1_{seed}_n_{n}.csv", index=False)

    def tab2(seed, n, change_width=False, master="mc2", **kwargs):
        sample, boot, boot_e = table(
            seed=seed,
            config_options=dict(
                model_name=["nonparametric"],
                **MC_CONFIGS,
                **kwargs,
            ),
            master=master,
            change_width=change_width,
            n=n,
        )

        sample.to_csv(f"checkpts/sample_tab2_{seed}_n_{n}.csv", index=False)
        boot.to_csv(f"checkpts/boot_tab2_{seed}_n_{n}.csv", index=False)
        if len(boot_e) > 0:
            boot_e.to_csv(f"checkpts/bootexp_tab2_{seed}_n_{n}.csv", index=False)

    def tab3(seed, n, change_width=False, **kwargs):
        sample, boot, boot_e = table(
            seed=seed,
            config_options=dict(
                model_name=["partiallylinear"],
                **MC_CONFIGS,
                **kwargs,
            ),
            master="mc2",
            change_width=change_width,
            n=n,
        )

        sample.to_csv(f"checkpts/sample_tab3_{seed}_n_{n}.csv", index=False)
        boot.to_csv(f"checkpts/boot_tab3_{seed}_n_{n}.csv", index=False)

        if len(boot_e) > 0:
            boot_e.to_csv(f"checkpts/bootexp_tab3_{seed}_n_{n}.csv", index=False)

    def tab4(seed, n, change_width=False, **kwargs):
        sample, boot, boot_e = table(
            seed=seed,
            config_options=dict(
                model_name=["partiallyadditive"],
                **MC_CONFIGS,
                **kwargs,
            ),
            master="mc2",
            change_width=change_width,
            n=n,
        )

        sample.to_csv(f"checkpts/sample_tab4a_{seed}_n_{n}.csv", index=False)
        boot.to_csv(f"checkpts/boot_tab4a_{seed}_n_{n}.csv", index=False)

        if len(boot_e) > 0:
            boot_e.to_csv(f"checkpts/bootexp_tab4a_{seed}_n_{n}.csv", index=False)

        new_config = {k: v for k, v in MC_CONFIGS.items()}
        new_config["data_nuisance_dimension"] = [5, 10]
        sample, boot, boot_e = table(
            seed=seed,
            config_options=dict(
                model_name=["partiallyadditivewithspline"],
                **new_config,
                **kwargs,
            ),
            master="mc2",
            n=n,
        )

        sample.to_csv(f"checkpts/sample_tab4b_{seed}_n_{n}.csv", index=False)
        boot.to_csv(f"checkpts/boot_tab4b_{seed}_n_{n}.csv", index=False)

        if len(boot_e) > 0:
            boot_e.to_csv(f"checkpts/bootexp_tab4b_{seed}_n_{n}.csv", index=False)

    def r1_design(seed, n, change_width=False, **kwargs):
        config_names = generate_config(
            dict(
                model_name=["nonparametric"],
                **MC_CONFIGS,
                **kwargs,
            ),
            master="mc4",
            change_width=change_width,
        )

        sample = []

        for config in config_names:
            with open(f"configs/{config}.json", "r") as f:
                configs = json.load(f)

            parser = generate_parser()
            main_locals = main(
                parser,
                provided_args=(
                    f"--n {n} --seed {seed} --no-save --no-tqdm "
                    f"--no-logger --no-config-save --config {config}"
                ).split(),
                return_locals=True,
                identity_only=True,
            )
            result = main_locals["results"]
            result.update(seed=seed, config=config, n=n, **configs)
            sample.append(pd.Series(result))

        sample = pd.DataFrame(sample)
        sample.to_csv(f"checkpts/sample_r1design_{seed}_n_{n}.csv", index=False)

    def relu_mc3(seed, n, change_width=False, **kwargs):
        config_names = generate_config(
            dict(
                model_name=["nonparametric"],
                **MC_CONFIGS,
                **kwargs,
            ),
            master="mc3",
            change_width=change_width,
        )

        sample = []

        for config in config_names:
            with open(f"configs/{config}.json", "r") as f:
                configs = json.load(f)

            parser = generate_parser()
            main_locals = main(
                parser,
                provided_args=(
                    f"--n {n} --seed {seed} --no-save --no-tqdm "
                    f"--no-logger --no-config-save --config {config}"
                ).split(),
                return_locals=True,
            )
            result = main_locals["results"]
            result.update(seed=seed, config=config, n=n, **configs)
            sample.append(pd.Series(result))

        sample = pd.DataFrame(sample)
        sample.to_csv(f"checkpts/mc3_relu_{seed}_n_{n}.csv", index=False)

    def calibrated_empirical(seed, n, **kwargs):
        mc_configs = {
            "arch_depth": [1, 3],
            "arch_width": [10, 20],
            "arch_hidden_activation": ["sigmoid", "relu"],
            "opt_learning_rate": [0.005],
        }

        config_names = generate_config(
            dict(
                model_name=["nonparametric"],
                **mc_configs,
                **kwargs,
            ),
            master="calibrated_gasdemand",
        )

        sample = []

        for config in config_names:
            with open(f"configs/{config}.json", "r") as f:
                configs = json.load(f)

            # print(json.dumps(configs, indent=2))

            parser = generate_parser()
            main_locals = main(
                parser,
                provided_args=(
                    f"--n {n} --seed {seed} --no-save --no-tqdm "
                    f"--no-logger --no-config-save --config {config}"
                ).split(),
                return_locals=True,
                identity_only=False,
            )

            result = main_locals["results"]
            # npvec = main_locals["npvec"]
            # dgp = main_locals["dgp"]

            result.update(seed=seed, config=config, n=n, **configs)
            sample.append(pd.Series(result))

        sample = pd.DataFrame(sample)
        sample.to_csv(f"checkpts/calibrated_{seed}_n_{n}.csv", index=False)

    def empirical_ex(boot_seed, config):

        with open(f"configs/{config}.json", "r") as f:
            cf = json.load(f)
        parser = generate_parser()

        if boot_seed == 1:
            main_locals = main(
                parser,
                provided_args=f"--seed 0 --no-save --no-tqdm --no-logger \
                    --no-config-save --config {config}".split(),
                return_locals=True,
                identity_only=False,
            )
            results = main_locals["results"]
            results.update(cf)

            with open(f"checkpts/empirical_{config}.json", "w") as f:
                json.dump(results, f)

        results_boot = main(
            parser,
            provided_args=(
                f"--seed 0 --bootstrap --bootstrap-seed {boot_seed} --no-save "
                f"--no-tqdm --no-logger --no-config-save --config {config}"
            ).split(),
        )
        results_boot["bootstrap_seed"] = boot_seed

        results_boot.update(cf)
        results_boot["seed"] = boot_seed

        with open(f"checkpts/empirical_boot_{seed}_{config}.json", "w") as f:
            json.dump(results_boot, f)

    seed = args.seed
    if seed == "":
        seed = os.environ.get("LSB_JOBINDEX")
    seed = int(seed)

    # Table 1
    if args.tab1:
        tab1(seed, 5000, change_width=args.change_width)
        tab1(seed, 1000, change_width=args.change_width)

    # Table 2
    if args.tab2:
        tab2(seed, 5000, change_width=args.change_width)
        tab2(seed, 1000, change_width=args.change_width)

    # Table 3
    if args.tab3:
        tab3(seed, 5000, change_width=args.change_width)
        tab3(seed, 1000, change_width=args.change_width)

    # Table 4
    if args.tab4:
        tab4(seed, 5000, change_width=args.change_width)
        tab4(seed, 1000, change_width=args.change_width)

    if args.tab2_mc2a:
        tab2(seed, 5000, change_width=args.change_width, master="mc2a")
        tab2(seed, 1000, change_width=args.change_width, master="mc2a")

    if args.tab2_mc3:
        tab2(seed, 5000, change_width=args.change_width, master="mc3")
        tab2(seed, 1000, change_width=args.change_width, master="mc3")

    if args.mc3_relu:
        relu_mc3(seed, 5000, change_width=args.change_width)
        relu_mc3(seed, 1000, change_width=args.change_width)

    if args.r1:
        r1_design(seed, 5000, change_width=args.change_width)
        r1_design(seed, 1000, change_width=args.change_width)

    if args.calibrated:
        calibrated_empirical(seed, 5000)
        calibrated_empirical(seed, 10000)

    if args.empirical:
        for gm in ["gasoline_master", "gasoline_master_short"]:
            with open(f"configs/{gm}.json", "r") as f:
                gasoline_master = json.load(f)
            options = [
                dict(arch_depth=1, arch_hidden_activation="sigmoid"),
                dict(arch_depth=3, arch_hidden_activation="sigmoid"),
                dict(arch_depth=3, arch_hidden_activation="relu"),
            ]
            for option in options:
                gasoline_master.update(option)
                name = gm + "_" + str(hash(str(option)))
                name = name.replace(" ", "")
                with open(f"configs/{name}.json", "w") as f:
                    json.dump(gasoline_master, f)
                empirical_ex(seed, name)

    with open("checkpts/mc_configs.json", "w") as f:
        json.dump(MC_CONFIGS, f)
