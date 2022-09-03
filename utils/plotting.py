import pickle
import os
import numpy as np

from copy import deepcopy


# def get_fitted_models(
#     path="checkpts/checkpts-2020-09-14-all/",
#     crit="chen07nonparametric_chen07_",
#     n=10,
#     seed=2138,
# ):
#     result = []
#     np.random.seed(seed)
#     for _ in range(n):
#         for x in np.random.choice(
#             [x for x in os.listdir(path) if x.startswith(crit)],
#             size=600,
#             replace=False,
#         ):
#             with open(f"{path}{x}", "rb") as f:
#                 obj = pickle.load(f)
#                 if (
#                     obj["config"]["arch_depth"] == 3
#                 ):  # and obj["config"]["data_nuisance_dimension"] == 5:
#                     break

#         model = Chen07Nonparametric(**obj["model_config"])
#         model.load_state_dict(obj["efficient_model"])
#         result.append(model)
#     return result


# def calculate_model_value(model, dgp, grid, ind, seed=2001):
#     _, dat, _, _ = dgp.data(seed)
#     vals = []
#     for y in grid:
#         inp = dat["endogenous"].clone()
#         inp[:, ind] = y
#         val = model(inp).mean().item()
#         vals.append(val)
#     return np.array(vals)


# def splines(seed, grid_h1, grid_h2, npvec):
#     spl_dgp = Chen07(5000, 5000, "cpu").data(seed)[0]
#     pen = (0.02, 0.02, 0.02, 0.02)
#     b, coef = spline_experiment(spl_dgp, interact=False, pen=pen)
#     b_in, coef_in = spline_experiment(spl_dgp, interact=True, pen=pen)
#     normalizer = spl_dgp["endogenous"][:, 1].max()

#     interact1, interact2 = evaluate_spline(
#         npvec, grid_h1, grid_h2, coef_in, b_in, interact=True, normalizer=normalizer
#     )

#     no_interact1, no_interact2 = evaluate_spline(
#         npvec, grid_h1, grid_h2, coef, b, interact=False, normalizer=normalizer
#     )
#     return interact1, interact2, no_interact1, no_interact2


# def evaluate_spline(npvec, grid_h1, grid_h2, coef, b, **spl_basis_args):
#     ysp1 = []
#     for y in grid_h1:
#         splb = deepcopy(npvec)
#         splb["endogenous"][:, 1] = y
#         splb, _ = get_spl_basis_and_penalty(splb, **spl_basis_args)

#         ysp1.append((splb @ coef + b * npvec["endogenous"][:, 0]).mean())

#     ysp2 = []
#     for y in grid_h2:
#         splb = deepcopy(npvec)
#         splb["endogenous"][:, 2] = y
#         splb, _ = get_spl_basis_and_penalty(splb, **spl_basis_args)
#         ysp2.append((splb @ coef + b * npvec["endogenous"][:, 0]).mean())
#     return np.array(ysp1), np.array(ysp2)


def plot_mean_std(
    panels, axs, starting_point=0, vertical_offset=0, **plot_kwargs,
):
    for i, p in enumerate(panels):
        vert_length = len(p)
        axs[i].errorbar(
            y=np.arange(starting_point, starting_point + vert_length)[::-1]
            + vertical_offset,
            x=p["mean"],
            xerr=p["std"],
            **plot_kwargs,
        )

