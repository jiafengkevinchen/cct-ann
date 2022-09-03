from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from run_model import generate_parser, main
from utils.parseconfigs import callback_dict, preprocess, stopping_criterion_dict

from pipeline.pipeline import (
    _nearest_neighbor_variance_estimation,
    train_loop,
    weight_fn,
)
from pipeline.splines import spl_experiment
from numpy.linalg import pinv


def get_provided_args(seed, config, n=2500):
    return (
        f"--n {n} --seed {seed} --no-save "
        f"--no-tqdm --no-logger --no-config-save --config {config}"
    ).split()


def compute_identity_score_correction_term(
    transformed_instrument,
    inverse_design_instrument,
    transformed_endogenous,
    transformed_endogenous_gradient,
    regularization=0,
    rcond=1e-15,
):
    """Compute v^* and project v^* onto the instruments.
    This is the identity score correction term. Estimated v^* is the
    transformed_endogenous @ beta and estimated E[v^* | instrument] is
    the transformed_instrument @ rho"""
    n = len(transformed_endogenous)
    inv_inst_product = inverse_design_instrument / n
    term1 = (
        (transformed_endogenous.T @ transformed_instrument)
        @ inv_inst_product
        @ (transformed_instrument.T @ transformed_endogenous)
    ) / n
    grad_mean = transformed_endogenous_gradient.mean(0)
    term2 = grad_mean[:, None] @ grad_mean[None, :]

    beta = -pinv(term1 + term2 + regularization * np.eye(len(term2)), rcond=rcond) @ grad_mean
    vhat = transformed_endogenous @ beta / (1 + grad_mean @ beta)
    rho = inv_inst_product @ (transformed_instrument.T @ vhat)
    return rho, beta


def _compute_beta_wstar(endo_basis, endo_grad_basis, inst_basis, var0, cond_variance, gamma):
    """Calculate closed form for w*"""
    xx_inv_x = np.linalg.pinv(inst_basis.T @ inst_basis) @ inst_basis.T
    K = endo_grad_basis + gamma[:, None] * endo_basis
    meanK = K.mean(0)
    second_term = np.outer(meanK, meanK)
    first_term_outer = inst_basis @ (xx_inv_x @ endo_basis)
    first_term = (
        first_term_outer.T @ ((var0 / cond_variance)[:, None] * first_term_outer) / len(inst_basis)
    )
    beta = -np.linalg.pinv(first_term + second_term) @ meanK
    w_star = endo_basis @ beta
    w_star_grad = endo_grad_basis @ beta
    v_star = w_star / (1 + w_star_grad + gamma * w_star).mean()
    alpha_coef = xx_inv_x @ v_star
    return alpha_coef


def get_grad_and_residual(h, fold):
    gradh = h.get_derivatives(fold["torchvec"]["endogenous"]).detach().numpy().flatten()
    residual = (
        (fold["torchvec"]["response"] - h(fold["torchvec"]["endogenous"]))
        .detach()
        .numpy()
        .flatten()
    )
    return gradh, residual


def whole_sample(seed, config, n=5000, return_estimation=False, deg=3, regularization=0):
    """Identity score estimator that doesn't use split sample"""
    parser = generate_parser()
    provided_args = get_provided_args(seed, config, n=n)
    estimation = main(
        parser,
        provided_args=provided_args,
        return_locals=True,
        identity_only=True,
    )
    h = estimation["model"]

    endo_basis, endo_grad_basis, inst_basis, _ = spl_experiment(
        npvec=estimation["npvec"],
        deg=deg,
        knots_inst=estimation["dgp"].instrument_knots,
        knots_endo=estimation["dgp"].endogenous_knots,
        pl=False,
        full_return=True,
    )

    answers = []
    if isinstance(regularization, (np.floating, float)) or type(regularization) is int:
        regularization = [regularization]
    for reg in regularization:
        rho, beta = compute_identity_score_correction_term(
            inst_basis,
            np.linalg.pinv(inst_basis.T @ inst_basis / len(inst_basis)),
            endo_basis,
            endo_grad_basis,
            regularization=reg,
        )
        gradh, residual = get_grad_and_residual(h, estimation)
        score_values = gradh - (inst_basis @ rho) * residual

        theta = score_values.mean()
        estimated_se = score_values.std() / (len(score_values) ** 0.5)

        ans = (theta, estimated_se, estimation) if return_estimation else (theta, estimated_se)
        answers.append((reg, ans))
    if len(answers) == 1:
        return answers[0][1]
    else:
        return answers


def whole_sample_optimal(seed, config, n=5000, n_neighbors=5):
    """IS and ES estimators that doesn't use split sample"""
    # Get IS estimate from preliminary
    theta_prelim, estimated_se_prelim, estimation = whole_sample(
        seed, config, n=n, return_estimation=True
    )

    h = estimation["model"]

    endo_basis, endo_grad_basis, inst_basis, spline_coef = spl_experiment(
        npvec=estimation["npvec"],
        deg=3,
        knots_inst=estimation["dgp"].instrument_knots,
        knots_endo=estimation["dgp"].endogenous_knots,
        pl=False,
        full_return=True,
    )

    # What the P-ISMD spline got
    spl_coef = (endo_grad_basis @ spline_coef).mean()
    gradh, residual = get_grad_and_residual(h, estimation)

    # Estimating the conditional variance
    knn_estimator, cond_variance = _nearest_neighbor_variance_estimation(
        (residual - residual.mean()) ** 2,
        estimation["npvec"]["instrument"],
        n_neighbors=n_neighbors,
    )

    # Calculate Gamma
    xx_inv_x = np.linalg.pinv(inst_basis.T @ inst_basis) @ inst_basis.T
    grad_diff = gradh - theta_prelim

    # Reproject Gamma
    gamma = inst_basis @ (xx_inv_x @ (((residual - residual.mean()) * grad_diff) / cond_variance))

    return_obj = {
        "id_score": theta_prelim,
        "id_score_se": estimated_se_prelim,
        "id_spline": spl_coef,
    }

    var0 = np.std(gradh - gamma * residual) ** 2

    # compute beta and w*
    alpha_coef = _compute_beta_wstar(
        endo_basis, endo_grad_basis, inst_basis, var0, cond_variance, gamma
    )
    alpha = inst_basis @ alpha_coef
    adjustment = gamma + alpha * var0 / cond_variance
    score_values = gradh - adjustment * residual
    theta = score_values.mean()
    estimated_se = score_values.std() / (len(score_values) ** 0.5)

    return_obj["es"] = theta
    return_obj["es_se"] = estimated_se

    return return_obj


def _half_sample(fold1, fold2, regularization=0):
    h = fold1["model"]

    endo_basis, endo_grad_basis, inst_basis, _ = spl_experiment(
        npvec=fold1["npvec"],
        deg=3,
        knots_inst=fold1["dgp"].instrument_knots,
        knots_endo=fold1["dgp"].endogenous_knots,
        pl=False,
        full_return=True,
    )

    # fold 2
    gradh, residual = get_grad_and_residual(h, fold2)
    _, _, inst_basis_2, _ = spl_experiment(
        npvec=fold2["npvec"],
        deg=3,
        knots_inst=fold2["dgp"].instrument_knots,
        knots_endo=fold2["dgp"].endogenous_knots,
        pl=False,
        full_return=True,
    )

    if isinstance(regularization, (np.floating, float)) or type(regularization) is int:
        regularization = [regularization]

    answers = []
    for reg in regularization:
        rho, beta = compute_identity_score_correction_term(
            inst_basis,
            np.linalg.pinv(inst_basis.T @ inst_basis / len(inst_basis)),
            endo_basis,
            endo_grad_basis,
            regularization=reg,
        )

        scores = gradh - (inst_basis_2 @ rho) * residual
        ans = (scores.mean(), scores.std() / len(scores) ** 0.5)
        answers.append((reg, ans))
    if len(answers) == 1:
        return answers[0][1]
    else:
        return answers


def half_sample(seed, config, n=5000, fit_func=_half_sample):
    n_fold = n // 2
    parser = generate_parser()
    estimated = []
    for a in [0, 1]:
        provided_args = get_provided_args(2 * seed + a, config, n=n_fold)
        estimation = main(
            parser,
            provided_args=provided_args,
            return_locals=True,
            identity_only=True,
        )
        estimated.append(estimation)

    sol1 = fit_func(estimated[0], estimated[1])
    sol2 = fit_func(estimated[1], estimated[0])

    if isinstance(sol1[0], (np.floating, float)):
        a1, se1 = sol1
        a2, se2 = sol2
        return (a1 + a2) / 2, ((se1**2 + se2**2) / 4) ** 0.5

    else:
        ests = {}
        ses = {}
        for i in range(len(sol1)):
            _, (a1, se1) = sol1[i]
            reg, (a2, se2) = sol2[i]
            ans = (a1 + a2) / 2
            ans_se = ((se1**2 + se2**2) / 4) ** 0.5
            ests[f"isx_{reg}"] = ans
            ests[f"isx_se_{reg}"] = ans_se
        return ests, ses


# Splines
def _spline_identity_score(npvec, knots_inst, knots_endo, deg=3, return_coefs=False, rcond=1e-15):
    endo_basis, endo_grad_basis, inst_basis, coef = spl_experiment(
        npvec=npvec,
        deg=deg,
        knots_inst=knots_inst,
        knots_endo=knots_endo,
        pl=False,
        full_return=True,
        rcond=rcond,
    )

    h_func = endo_basis @ coef
    gradh = endo_grad_basis @ coef
    residual = npvec["response"].flatten() - h_func

    _, _, inst_basis_large, _ = spl_experiment(
        npvec=npvec,
        deg=deg,
        knots_inst=knots_inst,
        knots_endo=knots_endo,
        pl=False,
        full_return=True,
        rcond=rcond,
    )

    rho, beta = compute_identity_score_correction_term(
        inst_basis_large,
        np.linalg.pinv(inst_basis_large.T @ inst_basis_large / len(inst_basis_large)),
        endo_basis,
        endo_grad_basis,
        rcond=rcond,
    )

    if return_coefs:
        return coef, rho

    score_values = gradh - (inst_basis_large @ rho) * residual
    theta = score_values.mean()
    estimated_se = score_values.std() / (len(score_values) ** 0.5)
    return theta, estimated_se


def spline_score(npvec, knots_inst, knots_endo, deg=3):
    theta_prelim, se_prelim = _spline_identity_score(npvec, knots_inst, knots_endo, deg=deg)

    endo_basis, endo_grad_basis, inst_basis, spline_coef = spl_experiment(
        npvec=npvec,
        deg=deg,
        knots_inst=knots_inst,
        knots_endo=knots_endo,
        pl=False,
        full_return=True,
    )

    gradh = endo_grad_basis @ spline_coef
    residual = npvec["response"].flatten() - endo_basis @ spline_coef

    knn_estimator, cond_variance = _nearest_neighbor_variance_estimation(
        (residual - residual.mean()) ** 2,
        npvec["instrument"],
        n_neighbors=100,
    )

    xx_inv_x = np.linalg.pinv(inst_basis.T @ inst_basis) @ inst_basis.T
    grad_diff = gradh - theta_prelim
    gamma = inst_basis @ (xx_inv_x @ (((residual - residual.mean()) * grad_diff) / cond_variance))

    return_obj = {
        "id_score_spline": theta_prelim,
        "id_score_spline_se": se_prelim,
    }

    var0 = np.std(gradh - gamma * residual) ** 2

    alpha_coef = _compute_beta_wstar(
        endo_basis, endo_grad_basis, inst_basis, var0, cond_variance, gamma
    )
    alpha = inst_basis @ alpha_coef

    adjustment = gamma + alpha * var0 / cond_variance

    score_values = gradh - adjustment * residual
    theta = score_values.mean()
    estimated_se = score_values.std() / (len(score_values) ** 0.5)

    return_obj["es_spline"] = theta
    return_obj["es_se_spline"] = estimated_se

    return return_obj


def whole_sample_optimal_osmd(seed, config, n=5000):
    parser = generate_parser()
    provided_args = get_provided_args(seed, config, n=n)
    estimation = main(
        parser,
        provided_args=provided_args,
        return_locals=True,
        identity_only=False,
    )
    h = estimation["model"]  # OP-OSMD

    endo_basis, endo_grad_basis, inst_basis, spline_coef = spl_experiment(
        npvec=estimation["npvec"],
        deg=3,
        knots_inst=estimation["dgp"].instrument_knots,
        knots_endo=estimation["dgp"].endogenous_knots,
        pl=False,
        full_return=True,
    )

    # Conditional variance
    cond_variance = 1 / estimation["weights"]

    gradh, residual = get_grad_and_residual(h, estimation)
    filtered, gamma = h.forward_filter_residuals(
        endogenous=estimation["torchvec"]["endogenous"],
        response=estimation["torchvec"]["response"],
        inefficient_derivative=estimation["inefficient_derivative"],
        inefficient_prediction=estimation["inefficient_prediction"],
        weights=estimation["weights"],
        basis=estimation["torchvec"]["transformed_instrument"],
        inverse_design=estimation["torchvec"]["inverse_design_instrument"],
    )
    gamma = gamma.numpy().flatten()
    cond_variance = cond_variance.numpy().flatten()

    var0 = np.std(gradh - gamma * residual) ** 2
    alpha_coef = _compute_beta_wstar(
        endo_basis, endo_grad_basis, inst_basis, var0, cond_variance, gamma
    )
    alpha = inst_basis @ alpha_coef

    adjustment = gamma + alpha * var0 / cond_variance
    score_values = gradh - adjustment * residual

    theta = score_values.mean()
    estimated_se = score_values.std() / (len(score_values) ** 0.5)

    return_obj = dict()
    return_obj["es_osmd"] = theta
    return_obj["es_osmd_se"] = estimated_se

    return return_obj


def _half_sample_optimal_osmd(fold1, fold2):
    theta_prelim, theta_se_prelim = _half_sample(fold1, fold2)

    # On fold 1:
    h = fold1["model"]
    endo_basis, endo_grad_basis, inst_basis, spline_coef = spl_experiment(
        npvec=fold1["npvec"],
        deg=3,
        knots_inst=fold1["dgp"].instrument_knots,
        knots_endo=fold1["dgp"].endogenous_knots,
        pl=False,
        full_return=True,
    )
    gradh, residual = get_grad_and_residual(h, fold1)

    cond_variance = 1 / fold1["weights"]
    gradh, residual = get_grad_and_residual(h, fold1)
    filtered, gamma = h.forward_filter_residuals(
        endogenous=fold1["torchvec"]["endogenous"],
        response=fold1["torchvec"]["response"],
        inefficient_derivative=fold1["inefficient_derivative"],
        inefficient_prediction=fold1["inefficient_prediction"],
        weights=fold1["weights"],
        basis=fold1["torchvec"]["transformed_instrument"],
        inverse_design=fold1["torchvec"]["inverse_design_instrument"],
    )

    gamma = gamma.numpy().flatten()
    cond_variance = cond_variance.numpy().flatten()

    xx_inv_x = np.linalg.pinv(inst_basis.T @ inst_basis) @ inst_basis.T
    gamma_coef = xx_inv_x @ gamma
    var0 = np.std(gradh - gamma * residual) ** 2

    alpha_coef = _compute_beta_wstar(
        endo_basis, endo_grad_basis, inst_basis, var0, cond_variance, gamma
    )

    alpha_before_projection = (inst_basis @ alpha_coef) * var0 / cond_variance
    alpha_coef = np.linalg.pinv(inst_basis.T @ inst_basis) @ (
        inst_basis.T @ alpha_before_projection
    )

    # On fold 2:
    gradh, residual = get_grad_and_residual(h, fold2)
    _, _, inst_basis, _ = spl_experiment(
        npvec=fold2["npvec"],
        deg=3,
        knots_inst=fold2["dgp"].instrument_knots,
        knots_endo=fold2["dgp"].endogenous_knots,
        pl=False,
        full_return=True,
    )

    adjustment = inst_basis @ gamma_coef + inst_basis @ alpha_coef
    score_values = gradh - adjustment * residual
    theta = score_values.mean()
    estimated_se = score_values.std() / (len(score_values) ** 0.5)

    return theta, estimated_se


def half_sample_osmd(seed, config, n=5000, fit_func=_half_sample_optimal_osmd):
    n_fold = n // 2
    parser = generate_parser()
    estimated = []
    for a in [0, 1]:
        provided_args = get_provided_args(2 * seed + a, config, n=n_fold)
        estimation = main(
            parser,
            provided_args=provided_args,
            return_locals=True,
            identity_only=False,
        )
        estimated.append(estimation)

    sol1 = fit_func(estimated[0], estimated[1])
    sol2 = fit_func(estimated[1], estimated[0])

    if isinstance(sol1[0], (np.floating, float)):
        a1, se1 = sol1
        a2, se2 = sol2
        return {
            "es_x_osmd": (a1 + a2) / 2,
            "es_x_osmd_se": ((se1**2 + se2**2) / 4) ** 0.5,
        }
    else:
        return_dict = {}
        for i in range(len(sol1)):
            _, (a1, se1) = sol1[i]
            reg, (a2, se2) = sol2[i]
            ans = (a1 + a2) / 2
            ans_se = ((se1**2 + se2**2) / 4) ** 0.5
            return_dict[f"es_x_osmd_{reg}"] = ans
            return_dict[f"es_x_osmd_se_{reg}"] = ans_se
        return return_dict


def _half_sample_optimal_vstar(estimation, fold2, n_neighbors=100, regularization=0):
    h = estimation["model"]  # OSMD

    endo_basis, endo_grad_basis, inst_basis, spline_coef = spl_experiment(
        npvec=estimation["npvec"],
        deg=3,
        knots_inst=estimation["dgp"].instrument_knots,
        knots_endo=estimation["dgp"].endogenous_knots,
        pl=False,
        full_return=True,
    )

    # Conditional variance
    cond_variance = 1 / estimation["weights"]

    gradh, residual = get_grad_and_residual(h, estimation)
    filtered, gamma = h.forward_filter_residuals(
        endogenous=estimation["torchvec"]["endogenous"],
        response=estimation["torchvec"]["response"],
        inefficient_derivative=estimation["inefficient_derivative"],
        inefficient_prediction=estimation["inefficient_prediction"],
        weights=estimation["weights"],
        basis=estimation["torchvec"]["transformed_instrument"],
        inverse_design=estimation["torchvec"]["inverse_design_instrument"],
    )
    gamma = gamma.numpy().flatten()
    # cond_variance = cond_variance.numpy().flatten()
    knn_estimator, cond_variance = _nearest_neighbor_variance_estimation(
        (residual - residual.mean()) ** 2,
        estimation["npvec"]["instrument"],
        n_neighbors=n_neighbors,
    )

    gradh_f2, residual_f2 = get_grad_and_residual(h, fold2)
    _, _, inst_basis_f2, _ = spl_experiment(
        npvec=fold2["npvec"],
        deg=3,
        knots_inst=fold2["dgp"].instrument_knots,
        knots_endo=fold2["dgp"].endogenous_knots,
        pl=False,
        full_return=True,
    )

    # Compute v* directly
    F = (endo_grad_basis + gamma[:, None] * endo_basis).mean(0)

    projected_basis = inst_basis @ (
        np.linalg.pinv(inst_basis.T @ inst_basis) @ (inst_basis.T @ endo_basis)
    )

    R = projected_basis.T @ ((1 / cond_variance)[:, None] * projected_basis) / len(projected_basis)

    if isinstance(regularization, (np.floating, float)) or type(regularization) is int:
        regularization = [regularization]

    answers = []
    for reg in regularization:
        v_coef = np.linalg.pinv(R + reg * np.eye(len(R)), rcond=1e-4) @ F
        v = endo_basis @ v_coef
        alpha_before_projection = gamma - v / cond_variance
        alpha_coef = np.linalg.pinv(inst_basis.T @ inst_basis) @ (
            inst_basis.T @ alpha_before_projection
        )

        adjustment = inst_basis_f2 @ alpha_coef
        score_values = gradh_f2 - adjustment * residual_f2

        theta = score_values.mean()
        estimated_se = score_values.std() / (len(score_values) ** 0.5)

        answers.append((reg, (theta, estimated_se)))

    if len(answers) == 1:
        return answers[0][1]
    else:
        return answers


def whole_sample_optimal_v_star(
    seed, config, n=5000, n_neighbors=100, regularization=0, return_estimation=False, deg=3
):
    parser = generate_parser()
    provided_args = get_provided_args(seed, config, n=n)
    estimation = main(
        parser,
        provided_args=provided_args,
        return_locals=True,
        identity_only=False,
    )
    h = estimation["model"]  # OSMD

    endo_basis, endo_grad_basis, inst_basis, spline_coef = spl_experiment(
        npvec=estimation["npvec"],
        deg=deg,
        knots_inst=estimation["dgp"].instrument_knots,
        knots_endo=estimation["dgp"].endogenous_knots,
        pl=False,
        full_return=True,
    )

    # Conditional variance
    cond_variance = 1 / estimation["weights"]

    gradh, residual = get_grad_and_residual(h, estimation)
    filtered, gamma = h.forward_filter_residuals(
        endogenous=estimation["torchvec"]["endogenous"],
        response=estimation["torchvec"]["response"],
        inefficient_derivative=estimation["inefficient_derivative"],
        inefficient_prediction=estimation["inefficient_prediction"],
        weights=estimation["weights"],
        basis=estimation["torchvec"]["transformed_instrument"],
        inverse_design=estimation["torchvec"]["inverse_design_instrument"],
    )
    gamma = gamma.numpy().flatten()

    knn_estimator, cond_variance = _nearest_neighbor_variance_estimation(
        (residual - residual.mean()) ** 2,
        estimation["npvec"]["instrument"],
        n_neighbors=n_neighbors,
    )

    # Compute v* directly
    F = (endo_grad_basis + gamma[:, None] * endo_basis).mean(0)

    projected_basis = inst_basis @ (
        np.linalg.pinv(inst_basis.T @ inst_basis) @ (inst_basis.T @ endo_basis)
    )

    R = projected_basis.T @ ((1 / cond_variance)[:, None] * projected_basis) / len(projected_basis)

    return_obj = dict()
    if isinstance(regularization, (np.floating, float)) or type(regularization) is int:
        regularization = [regularization]
    for reg in regularization:
        R_regged = R + reg * np.eye(len(R))

        v = endo_basis @ (np.linalg.pinv(R_regged) @ F)
        projected_normalized_v = (
            inst_basis
            @ (np.linalg.pinv(inst_basis.T @ inst_basis) @ (inst_basis.T @ v))
            / cond_variance
        )

        adjustment = gamma - projected_normalized_v
        score_values = gradh - adjustment * residual

        theta = score_values.mean()
        estimated_se = score_values.std() / (len(score_values) ** 0.5)

        reg_str = f"_{reg}" if reg != 0 else ""
        return_obj[f"es_vstar{reg_str}"] = theta
        return_obj[f"es_vstar_se{reg_str}"] = estimated_se

        projected_normalized_v_divide_first = inst_basis @ (
            np.linalg.pinv(inst_basis.T @ inst_basis) @ (inst_basis.T @ (v / cond_variance))
        )

        score_values_divide_first = gradh - (gamma - projected_normalized_v_divide_first) * residual
        return_obj[f"es_vstar_div_first{reg_str}"] = score_values_divide_first.mean()
        return_obj[f"es_vstar_se_div_first{reg_str}"] = (
            score_values_divide_first.std() / len(score_values_divide_first) ** 0.5
        )
    if return_estimation:
        return return_obj, estimation
    else:
        return return_obj


def _half_sample_spline_is(npvec, evalvec, knots_inst, knots_endo, deg=3, rcond=1e-15):
    spline_coef, adjustment_coef = _spline_identity_score(
        npvec, knots_inst, knots_endo, deg=deg, return_coefs=True, rcond=rcond
    )

    endo_basis, endo_grad_basis, inst_basis, spline_coef = spl_experiment(
        npvec=evalvec,
        deg=deg,
        knots_inst=knots_inst,
        knots_endo=knots_endo,
        pl=False,
        full_return=True,
        rcond=rcond,
    )
    residuals = evalvec["response"].flatten() - (endo_basis @ spline_coef)

    score_values = endo_grad_basis @ spline_coef - (inst_basis @ adjustment_coef) * residuals
    return score_values.mean(), score_values.std() / (len(score_values) ** 0.5)


def _half_sample_spline_es(
    npvec, evalvec, knots_inst, knots_endo, deg=3, n_neighbors=100, rcond=1e-15
):
    spline_coef, adjustment_coef = spline_score_vstar(
        npvec,
        knots_inst,
        knots_endo,
        deg=deg,
        n_neighbors=n_neighbors,
        return_coefs=True,
        rcond=rcond,
    )

    endo_basis, endo_grad_basis, inst_basis, spline_coef = spl_experiment(
        npvec=evalvec,
        deg=deg,
        knots_inst=knots_inst,
        knots_endo=knots_endo,
        pl=False,
        full_return=True,
        rcond=rcond,
    )
    residuals = evalvec["response"].flatten() - (endo_basis @ spline_coef)

    score_values = endo_grad_basis @ spline_coef - (inst_basis @ adjustment_coef) * residuals
    return score_values.mean(), score_values.std() / (len(score_values) ** 0.5)


def spline_score_vstar(
    npvec, knots_inst, knots_endo, deg=3, n_neighbors=100, return_coefs=False, rcond=1e-15
):
    theta_prelim, se_prelim = _spline_identity_score(
        npvec, knots_inst, knots_endo, deg=deg, rcond=rcond
    )

    endo_basis, endo_grad_basis, inst_basis, spline_coef = spl_experiment(
        npvec=npvec,
        deg=deg,
        knots_inst=knots_inst,
        knots_endo=knots_endo,
        pl=False,
        full_return=True,
        rcond=rcond,
    )

    gradh = endo_grad_basis @ spline_coef
    residual = npvec["response"].flatten() - endo_basis @ spline_coef
    invzz = np.linalg.pinv(inst_basis.T @ inst_basis, rcond=rcond)

    if n_neighbors == "projection":
        cond_variance = (
            inst_basis @ (invzz @ (inst_basis.T @ (residual - residual.mean()) ** 2))
        ).clip(min=0.1)
    else:
        knn_estimator, cond_variance = _nearest_neighbor_variance_estimation(
            (residual - residual.mean()) ** 2,
            npvec["instrument"],
            n_neighbors=n_neighbors,
        )

    xx_inv_x = invzz @ inst_basis.T
    grad_diff = gradh - theta_prelim
    gamma = inst_basis @ (xx_inv_x @ (((residual - residual.mean()) * grad_diff) / cond_variance))

    return_obj = {
        "id_score_spline": theta_prelim,
        "id_score_spline_se": se_prelim,
    }

    F = (endo_grad_basis + gamma[:, None] * endo_basis).mean(0)

    projected_basis = inst_basis @ (invzz @ (inst_basis.T @ endo_basis))

    R = projected_basis.T @ ((1 / cond_variance)[:, None] * projected_basis) / len(projected_basis)

    v = endo_basis @ (np.linalg.pinv(R, rcond=1e-4) @ F)
    projected_normalized_v = inst_basis @ (invzz @ (inst_basis.T @ v)) / cond_variance

    adjustment = gamma + projected_normalized_v

    if return_coefs:
        coef_for_adjustment = invzz @ (inst_basis.T @ (gamma - projected_normalized_v))
        return spline_coef, coef_for_adjustment

    adjustment = gamma - projected_normalized_v  # in theory it's the correct one
    score_values = gradh - adjustment * residual

    theta = score_values.mean()
    estimated_se = score_values.std() / (len(score_values) ** 0.5)
    return_obj["es_vstar"] = theta
    return_obj["es_vstar_se"] = estimated_se

    return return_obj


def break_into_folds(torchvec, nfolds):
    """Returns a list of tuples of torchvec objects of the type (train, test)"""
    n = len(torchvec["response"])
    fold_size = n // nfolds

    return_values = []
    for i in range(nfolds):
        test_indices = (
            np.arange(i * fold_size, (i + 1) * fold_size)
            if i < nfolds - 1
            else np.arange(i * fold_size, n)
        )
        train_indices = np.setdiff1d(np.arange(n), test_indices, assume_unique=True)

        test_indices = torch.tensor(test_indices).long()
        train_indices = torch.tensor(train_indices).long()

        train_dict = {
            k: v[train_indices] if k != "inverse_design_instrument" else v
            for k, v in torchvec.items()
        }
        test_dict = {
            k: v[test_indices] if k != "inverse_design_instrument" else v
            for k, v in torchvec.items()
        }

        train_loader = [
            [train_dict[k] for k in train_dict.keys() if k != "inverse_design_instrument"]
        ]

        return_values.append((train_dict, test_dict, train_loader))

    return return_values


def fit_model_on_folds(
    parser,
    nfolds,
    provided_args=None,
):
    config_lst, dgp_objects, model_objects = preprocess(
        parser, configpath=Path("configs"), provided_args=provided_args
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
    model_initial_weights = deepcopy(model.state_dict())

    torch.manual_seed(config["seed"])

    def fit_nn(model, torchvec, loader):
        optimizer = optimizer_constructor(model.parameters(), **optimizer_config)

        _ = train_loop(
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

        inefficient_prediction = model(torchvec["endogenous"]).detach()
        inefficient_derivative = model.get_derivatives(torchvec["endogenous"]).detach()

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
        _ = train_loop(
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

        efficient_model = deepcopy(model)
        return {
            "npvec": {k: v.numpy().astype("float") for k, v in torchvec.items()},
            "dgp": dgp,
            "weights": weights,
            "inefficient_prediction": inefficient_prediction,
            "inefficient_derivative": inefficient_derivative,
            "torchvec": torchvec,
            "model": efficient_model,
        }

    folds = break_into_folds(torchvec, nfolds)
    return_values = []
    for (train_dict, test_dict, train_loader) in folds:
        model.load_state_dict(model_initial_weights)
        estimated_model = fit_nn(model, train_dict, train_loader)
        test_fold = {
            "torchvec": test_dict,
            "npvec": {k: v.numpy().astype("float") for k, v in test_dict.items()},
            "dgp": estimated_model["dgp"],
        }
        return_values.append((estimated_model, test_fold))
    return return_values


def nfold_osmd(nfolds, seed, config, n=5000, fit_func=_half_sample_optimal_osmd):
    parser = generate_parser()
    provided_args = get_provided_args(seed, config, n=n)
    estimated_objects = fit_model_on_folds(parser, nfolds, provided_args)

    stats = np.array([fit_func(train, test) for train, test in estimated_objects])
    estimate = stats[:, 0].mean()
    estimated_se = ((stats[:, 1] ** 2).sum() ** 0.5) / len(stats)

    return {"es_x_osmd_nf": estimate, "es_x_osmd_nf_se": estimated_se}
