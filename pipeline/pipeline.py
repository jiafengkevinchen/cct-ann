import os

import numpy as np
import pandas as pd
import torch
from statsmodels.tools import add_constant
from tqdm.auto import tqdm
from sklearn.neighbors import KNeighborsRegressor
from .splines import spl_experiment
from numpy.linalg import pinv


def project(inverse_design, transformed_instrument, to_project):
    return (
        transformed_instrument
        @ inverse_design
        @ (transformed_instrument.T @ to_project / len(to_project))
    )


def compute_inverse_design(basis, rcond=1e-5):
    """Compute (Z'Z)^-1, package into torch"""
    return torch.as_tensor(
        pinv((basis.T @ basis / len(basis)).detach().numpy(), rcond=rcond, hermitian=True)
    ).float()


def loss_fn(prediction, truth, basis, inverse_design=None, weights=None, bootstrap_weights=None):
    """Implements loss function for mean regression"""
    if inverse_design is None:
        inverse_design = compute_inverse_design(basis)

    residual = (
        truth - prediction
        if bootstrap_weights is None
        else (truth - prediction) * bootstrap_weights
    )

    projected_residual = basis @ inverse_design @ (basis.T @ residual) / len(basis)
    if weights is None:
        loss = (projected_residual**2).mean()
    else:
        loss = (projected_residual**2 * weights).mean()
    return loss


def loss_fn_qiv(
    prediction,
    truth,
    basis,
    inverse_design=None,
    weights=None,
    target_quantile=0.5,
    smoothing=20,
):
    if inverse_design is None:
        inverse_design = compute_inverse_design(basis)
    residual = truth - prediction
    thresholded_residual = torch.sigmoid(residual * smoothing) - target_quantile
    projected_residual = basis @ inverse_design @ (basis.T @ thresholded_residual) / len(basis)
    loss = (projected_residual**2).mean()
    return loss


def _nearest_neighbor_variance_estimation(residual2, instrument, **kwargs):
    model = KNeighborsRegressor(**kwargs)
    model.fit(instrument, residual2)
    predicted = model.predict(instrument)
    return model, predicted


def weight_fn(
    prediction,
    truth,
    basis,
    inverse_design=None,
    min_var=0.01,
    normalize=False,
    **kwargs,
):
    residual_sq = ((truth - prediction) ** 2).detach().numpy()
    basis = basis.numpy()
    m, predicted = _nearest_neighbor_variance_estimation(residual_sq, basis, **kwargs)
    weights = torch.tensor((1 / predicted)).float()

    if normalize:
        return weights / weights.sum() * len(basis)
    else:
        return weights


def null_regularizer(model):
    device = next(model.parameters()).device
    return torch.Tensor([0]).to(device)


def l1_regularizer(model, lmbda=0.0001):
    device = next(model.parameters()).device
    return lmbda * sum([p.abs().sum() for p in model.parameters()]).to(device)


def l2_regularizer(model, lmbda=0.0001):
    device = next(model.parameters()).device
    return lmbda * sum([(p**2).sum() for p in model.parameters()]).to(device)


def stopping_criterion(cached_history, param_tol=1e-3, grad_tol=1e-3):
    k = len(cached_history)
    return (
        abs(
            cached_history["param"].iloc[: k // 2].mean()
            - cached_history["param"].iloc[k // 2 :].mean()
        )
        < param_tol
        and cached_history["grad_norm"].mean() < grad_tol
    )


def train_step(
    model,
    optimizer,
    response,
    endogenous,
    transformed_instrument,
    inverse_design_instrument,
    regularizer=null_regularizer,
    loss_fn=loss_fn,
    **kwargs,
):
    optimizer.zero_grad()
    prediction = model(endogenous)
    loss = loss_fn(
        prediction=prediction,
        truth=response,
        basis=transformed_instrument,
        inverse_design=inverse_design_instrument,
        weights=kwargs["weights"] if "weights" in kwargs else None,
        bootstrap_weights=kwargs["bootstrap_weights"] if "bootstrap_weights" in kwargs else None,
    )
    penalty = regularizer(model)
    (loss + penalty).backward()
    grad_norm = max([x.grad.abs().max() for x in model.parameters()]).item()
    optimizer.step()

    report = {
        "loss": loss.item(),
        "penalty": penalty.item(),
        "param": model.get_parameter_of_interest(endogenous),
        "grad_norm": grad_norm,
    }
    return report


def train_loop(
    model,
    optimizer,
    loader,
    inverse_design_instrument=None,
    max_epoch=500,
    history=20,
    print_freq=10,
    min_epochs=1,
    stopping_criterion=stopping_criterion,
    callback=None,
    has_weights=False,
    stopping_kwargs=dict(),
    train_step_kwargs=dict(),
    name="",
):
    cache = []
    cache_df = None

    for epoch in tqdm(range(max_epoch), disable=os.environ.get("TQDM") == "False"):
        for obj in loader:
            if has_weights:
                (
                    response,
                    endogenous,
                    instrument,
                    transformed_instrument,
                    weights,
                ) = obj
            else:
                response, endogenous, instrument, transformed_instrument = obj
                weights = None

            result = train_step(
                model,
                optimizer,
                response=response,
                endogenous=endogenous,
                transformed_instrument=transformed_instrument,
                inverse_design_instrument=inverse_design_instrument,
                weights=weights,
                **train_step_kwargs,
            )

            cache.append(result)
            cache = cache[-history:]
            cache_df = pd.DataFrame(cache)
            if (
                stopping_criterion(cache_df, **stopping_kwargs)
                and epoch > min_epochs
                and len(cache_df) >= history
            ):
                return cache_df
        if callback is not None and epoch > 0 and epoch % print_freq == 0:
            callback(epoch, cache_df, name)
    return cache_df


def transform_endogenous(endo, order=1, interact_x=False):
    t = torch.cat([endo**p / p for p in range(1, order + 1)], dim=1)
    mask = torch.ones_like(endo)
    mask[:, 1:] = 0
    g = torch.cat([endo ** (p - 1) for p in range(1, order + 1)], dim=1)
    mask = torch.cat([mask for p in range(1, order + 1)], dim=1)
    g = mask * g

    if interact_x:
        interacts = []
        interacts_grad = []
        k = endo.shape[1]
        for i in range(k):
            for j in range(i + 1, k):
                interacts.append(endo[:, [i]] * endo[:, [j]])
                if i == 0:
                    interacts_grad.append(endo[:, [j]])
                else:
                    interacts_grad.append(torch.zeros((len(endo), 1)))
        t = torch.cat([t] + interacts, dim=1)
        g = torch.cat([g] + interacts_grad, dim=1)

    transformed_endogenous = torch.tensor(add_constant(t)).float()
    transformed_endogenous_gradient = torch.cat([torch.zeros(len(g), 1), g], dim=1)
    return transformed_endogenous, transformed_endogenous_gradient


def interact(tilde_x):
    tf = []
    k = tilde_x.shape[1]
    for i in range(k):
        for j in range(i + 1, k):
            tf.append(tilde_x[:, [i]] * tilde_x[:, [j]])
    return torch.cat(tf, dim=1)


def transform_endogenous_wrapper(
    endo, inst, full_inst, nonpar, dim_x_tilde, order, interact_x=False
):
    if type(order) is not dict:
        to_transform = endo if nonpar else endo[:, 1:]
        tf_endogenous, tf_endogenous_gradient = transform_endogenous(
            to_transform, order=order, interact_x=interact_x
        )

        tf_inst, _ = transform_endogenous(inst, order=order + 1, interact_x=interact_x)
        transformed_instrument = torch.cat([full_inst, tf_inst], dim=1)

        if dim_x_tilde > 0:
            tilde_x = endo[:, -dim_x_tilde:]
            tilde_x_interacts = interact(tilde_x)
            transformed_instrument = torch.cat([transformed_instrument, tilde_x_interacts], dim=1)
            tf_endogenous = torch.cat([tf_endogenous, tilde_x_interacts], dim=1)
            tf_endogenous_gradient = torch.cat(
                [tf_endogenous_gradient, torch.zeros_like(tilde_x_interacts)], dim=1
            )

        inverse_design = compute_inverse_design(transformed_instrument)
        return (
            tf_endogenous,
            tf_endogenous_gradient,
            transformed_instrument,
            inverse_design,
        )
    else:
        (tf_endogenous, tf_endogenous_gradient, tf_inst, _) = spl_experiment(
            order["npvec"],
            order["deg"],
            order["knots_inst"],
            order["knots_endo"],
            pl=not nonpar,
            full_return=True,
        )

        if not nonpar:
            tf_endogenous = add_constant(tf_endogenous[:, 2:])  # remove constant and first
            tf_endogenous_gradient = np.c_[
                np.zeros((len(tf_endogenous), 1)), tf_endogenous_gradient[:, 2:]
            ]

        tf_endogenous = torch.tensor(tf_endogenous).float()
        tf_endogenous_gradient = torch.tensor(tf_endogenous_gradient).float()
        tf_inst = torch.tensor(tf_inst).float()
        inverse_design = compute_inverse_design(tf_inst)

        return (tf_endogenous, tf_endogenous_gradient, tf_inst, inverse_design)


def compute_se(
    torchvec,
    model,
    weights=None,
    inefficient_derivative=None,
    inefficient_prediction=None,
    order=1,
    weighting=True,
):
    try:
        se_nonpar = None
        dim_x_tilde = torchvec["endogenous"].shape[1] - 3
        if hasattr(model, "get_standard_error_nonparametric"):
            (
                tf_endogenous,
                tf_endogenous_gradient,
                transformed_instrument,
                inverse_design,
            ) = transform_endogenous_wrapper(
                torchvec["endogenous"],
                torchvec["instrument"],
                torchvec["transformed_instrument"],
                True,
                dim_x_tilde,
                order,
                interact_x=True,
            )

            inv_variance = weight_fn(
                prediction=model(torchvec["endogenous"]),
                truth=torchvec["response"],
                basis=transformed_instrument,
                inverse_design=inverse_design,
            )

            if weighting:
                filtered, Gamma = model.forward_filter_residuals(
                    endogenous=torchvec["endogenous"],
                    response=torchvec["response"],
                    inefficient_derivative=inefficient_derivative,
                    inefficient_prediction=inefficient_prediction,
                    weights=weights,
                    basis=transformed_instrument,
                    inverse_design=inverse_design,
                )

                se_nonpar = model.get_standard_error_nonparametric(
                    filtered,
                    Gamma,
                    tf_endogenous,
                    tf_endogenous_gradient,
                    transformed_instrument,
                    inv_variance,
                    inverse_design,
                )
            else:
                se_nonpar = model.get_standard_error_nonparametric(
                    model.get_derivatives(torchvec["endogenous"]),
                    0,
                    tf_endogenous,
                    tf_endogenous_gradient,
                    transformed_instrument,
                    1,
                    inverse_design,
                    weighting=False,
                    residuals=torchvec["response"] - model(torchvec["endogenous"]),
                )

        (
            tf_endogenous,
            tf_endogenous_gradient,
            transformed_instrument,
            inverse_design,
        ) = transform_endogenous_wrapper(
            torchvec["endogenous"],
            torchvec["instrument"],
            torchvec["transformed_instrument"],
            False,
            dim_x_tilde,
            order,
        )

        inverse_variance = weight_fn(
            prediction=model(torchvec["endogenous"]),
            truth=torchvec["response"],
            basis=transformed_instrument,
            inverse_design=inverse_design,
        )
        se = model.get_standard_error(
            endogenous_of_interest=torchvec["endogenous"][:, [0]],
            transformed_endogenous=tf_endogenous,
            transformed_instrument=transformed_instrument,
            inverse_variance=inverse_variance,
            inverse_design_instrument=inverse_design,
        )

        return se, se_nonpar
    except RuntimeError:
        return np.nan, np.nan
