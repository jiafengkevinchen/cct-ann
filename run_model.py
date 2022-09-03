import argparse
import warnings
from copy import deepcopy
from pathlib import Path

import torch

from pipeline.callbacks import logger
from pipeline.pipeline import train_loop, weight_fn
from utils.parseconfigs import callback_dict, preprocess, stopping_criterion_dict
from utils.postprocessing import get_filename, save_results

warnings.simplefilter("ignore", FutureWarning)

CONFIG_PATH = Path("configs")
OUTPUT_PATH = Path("checkpts")


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--config", type=str, default="mc1")
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=-1)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--no-logger", action="store_true")
    parser.add_argument("--se", action="store_true")
    parser.add_argument("--exp-bootstrap", action="store_true")
    parser.add_argument("--no-config-save", action="store_true")
    return parser


def main(parser, provided_args=None, return_locals=False, identity_only=False):
    # --------------------------- Preprocess ---------------------------
    config_lst, dgp_objects, model_objects = preprocess(
        parser, configpath=CONFIG_PATH, provided_args=provided_args
    )
    (
        args,
        config,
        dgp_args,
        model_config,
        train_step_kwargs,
        optimizer_config,
    ) = config_lst

    dgp, data, corr_mat = dgp_objects
    npvec, torchvec, dataset, loader = data
    model, optimizer_constructor, optimizer = model_objects
    torch.manual_seed(config["seed"])

    # --------------------------- Inefficient estimate ---------------------------
    cache_df = train_loop(
        model,
        optimizer,
        loader,
        inverse_design_instrument=torchvec["inverse_design_instrument"],
        max_epoch=config["train_max_epoch"],
        min_epochs=config["train_min_epoch"],
        stopping_kwargs=dict(
            param_tol=config["train_stopping_param_tol"],
            grad_tol=config["train_stopping_grad_tol"],
        ),
        history=config["train_stopping_history_length"],
        print_freq=config["train_callback_freq"],
        callback=callback_dict[config["callback"]],
        stopping_criterion=stopping_criterion_dict[config["stopping_criterion"]],
        train_step_kwargs=train_step_kwargs,
        name=f"{config['model_name']}_inefficient",
    )

    inefficient_parameter_estimate = model.get_parameter_of_interest(
        torchvec["endogenous"]
    )
    inefficient_model = deepcopy(model.state_dict())

    if not args.no_tqdm:
        print(f"initial estimate = {inefficient_parameter_estimate}")

    inefficient_derivative = None
    inefficient_prediction = None
    corrected_identity_weighting = None

    if hasattr(model, "_forward_filter_residuals"):
        corrected_identity_weighting = model._forward_filter_residuals(
            torchvec["endogenous"],
            torchvec["response"],
            torchvec["inverse_design_instrument"],
            torchvec["transformed_instrument"],
        )

    if hasattr(model, "get_parameter_of_interest_with_correction"):
        inefficient_prediction = model(torchvec["endogenous"]).detach()
        inefficient_derivative = model.get_derivatives(torchvec["endogenous"]).detach()

    if identity_only and return_locals:
        results = {
            "identity_weighting": inefficient_parameter_estimate,
            "initial_loss": cache_df["loss"].iloc[-1],
            "corrected_identity_weighting": corrected_identity_weighting,
        }
        return locals()

    # --------------------------- Efficiency weighting ---------------------------
    weights = weight_fn(
        prediction=model(torchvec["endogenous"]),
        truth=torchvec["response"],
        basis=torchvec["instrument"],
        n_neighbors=5,
    )
    torchvec["weights"] = weights
    dataset_with_weights, loader_with_weights = dgp.package_dataset(torchvec)

    # Refresh the optimizer
    optimizer = optimizer_constructor(model.parameters(), **optimizer_config)

    # Train again
    cache_df_eff = train_loop(
        model,
        optimizer,
        loader_with_weights,
        inverse_design_instrument=torchvec["inverse_design_instrument"],
        max_epoch=config["train_max_epoch"],
        min_epochs=config["train_min_epoch"],
        stopping_kwargs=dict(
            param_tol=config["train_stopping_param_tol"],
            grad_tol=config["train_stopping_grad_tol"],
        ),
        history=config["train_stopping_history_length"],
        print_freq=config["train_callback_freq"],
        callback=callback_dict[config["callback"]],
        stopping_criterion=stopping_criterion_dict[config["stopping_criterion"]],
        has_weights=True,
        train_step_kwargs=train_step_kwargs,
        name=f"{config['model_name']}_efficient",
    )
    efficient_parameter_estimate = model.get_parameter_of_interest(
        torchvec["endogenous"]
    )
    efficient_model = deepcopy(model.state_dict())

    param_with_correction = None
    param_se = None
    if hasattr(model, "get_parameter_of_interest_with_correction"):
        param_with_correction = model.get_parameter_of_interest_with_correction(
            endogenous=torchvec["endogenous"],
            response=torchvec["response"],
            inefficient_derivative=inefficient_derivative,
            inefficient_prediction=inefficient_prediction,
            weights=weights,
            basis=torchvec["transformed_instrument"],
            inverse_design=torchvec["inverse_design_instrument"],
            return_standard_error=args.se,
        )
        if not args.no_tqdm:
            print(f"final estimate = {param_with_correction}")
        if args.se:
            param_with_correction, param_se = param_with_correction

    else:
        if not args.no_tqdm:
            print(f"final estimate = {efficient_parameter_estimate}")

    # --------------------------- Post-processing ---------------------------
    results = {
        "optimal_weighting_uncorrected": efficient_parameter_estimate,
        "identity_weighting": inefficient_parameter_estimate,
        "initial_loss": cache_df["loss"].iloc[-1],
        "final_loss": cache_df_eff["loss"].iloc[-1],
        "corrected_identity_weighting": corrected_identity_weighting,
    }

    if hasattr(model, "get_parameter_of_interest_with_correction"):
        results["optimal_weighting"] = param_with_correction
        # if args.se:
        #     results["efficient_parameter_se"] = param_se
    else:
        results["optimal_weighting"] = results["optimal_weighting_uncorrected"]

    if not args.no_save:
        save_results_kwargs = {}
        if args.save_weights:
            save_results_kwargs["efficient_model"] = efficient_model
            save_results_kwargs["inefficient_model"] = inefficient_model

        save_results(
            OUTPUT_PATH,
            get_filename(config, args.name),
            config_lst,
            results,
            logger if not args.no_logger else None,
            no_config_save=args.no_config_save,
            **save_results_kwargs,
        )

    if return_locals:
        return locals()

    return results


if __name__ == "__main__":
    parser = generate_parser()
    main(parser)
