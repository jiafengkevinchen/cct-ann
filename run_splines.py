import warnings
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dgp.mc2 import MC2
from dgp.mc3 import MC3
from dgp.mc2a import MC2a
from dgp.mc4 import MC4
from pipeline.splines import optimally_weighted_spline_experiment, spl_experiment
from pipeline.xfit import spline_score

warnings.simplefilter("ignore", RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--main", action="store_true")
parser.add_argument("--boot", action="store_true")
parser.add_argument("--mc3a", action="store_true")
parser.add_argument("--mc3b", action="store_true")
parser.add_argument("--temp", action="store_true")
parser.add_argument("--temp-esx", action="store_true")
parser.add_argument("--r1", action="store_true")


args = parser.parse_args()

if args.main:
    nsim = 1000
    spline_score_results = []
    total = 20 * nsim

    pbar = tqdm(total=total)

    for n in [1000, 5000]:
        for dim in [0, 5, 10]:
            for corr in [0, 0.5]:
                if dim == 0 and corr == 0.5:
                    continue
                for deg in [3, 4]:
                    dgp = MC2(n=n, batch_size=n, dimension=dim, corr=corr, device="cpu")
                    for s in range(1, nsim + 1):
                        pbar.update(1)
                        npvec = dgp.data(s)[0]
                        arg = dict(
                            npvec=npvec,
                            knots_inst=dgp.instrument_knots,
                            knots_endo=dgp.endogenous_knots,
                        )
                        res = spline_score(**arg, deg=deg)
                        res["pismd_spline"], res["pismd_spline_se"] = spl_experiment(
                            **arg, deg=deg, pl=False, se=True,
                        )
                        res["oposmd_spline"] = optimally_weighted_spline_experiment(
                            **arg, deg=deg, pl=False
                        )
                        res["n"] = n
                        res["dim"] = dim
                        res["corr"] = corr
                        res["deg"] = deg
                        spline_score_results.append(res)

    spline_score_df = pd.DataFrame(spline_score_results)
    spline_score_df.to_csv("checkpts/spline_smd_score_results.csv", index=False)

if args.boot:
    spline_boot = []

    total = 2 * 5 * 1000
    pbar = tqdm(total=total)

    for n in [1000, 5000]:
        for dim in [0, 5, 10]:
            for corr in [0, 0.5]:
                if dim == 0 and corr == 0.5:
                    continue
                dgp = MC2(n=n, batch_size=n, dimension=dim, corr=corr, device="cpu")
                data_seed = 0

                for boot in range(1000):
                    pbar.update(1)
                    rng = np.random.RandomState(boot)
                    np_weights = np.where(rng.rand(n) < 4 / 5, 1 / 2, 3)
                    npvec = dgp.data(data_seed)[0]

                    osmd, psmd = optimally_weighted_spline_experiment(
                        npvec,
                        deg=3,
                        knots_inst=dgp.instrument_knots,
                        knots_endo=dgp.endogenous_knots,
                        return_initial=True,
                        bootstrap_weights=np_weights,
                    )

                    res = {
                        "oposmd_spline": osmd,
                        "pismd_spline": psmd,
                    }
                    res["n"] = n
                    res["dim"] = dim
                    res["corr"] = corr
                    spline_boot.append(res)
    spline_boot = pd.DataFrame(spline_boot)
    spline_boot.to_csv("checkpts/boot_splines_smd.csv", index=False)


if args.mc3a:
    pbar = tqdm(total=10)
    res = []
    for n in [1000, 5000]:
        for dim in [0, 5, 10]:
            for corr in [0, 0.5]:
                if dim == 0 and corr == 0.5:
                    continue
                pbar.update(1)
                dgp = MC2a(n, n, "cpu", dim, corr=corr)
                ans = np.array(
                    [
                        spl_experiment(
                            dgp.data(s)[0], 4, dgp.instrument_knots, dgp.endogenous_knots,
                        )
                        for s in range(1, 1001)
                    ]
                )
                res.append(
                    {"n": n, "dim": dim, "corr": corr, "mean": ans.mean(), "std": ans.std(),}
                )

    spline_r2_res = pd.DataFrame(res)

    pbar = tqdm(total=10)
    opt_res = []
    for n in [1000, 5000]:
        for dim in [0, 5, 10]:
            for corr in [0, 0.5]:
                if dim == 0 and corr == 0.5:
                    continue
                pbar.update(1)
                dgp = MC2a(n, n, "cpu", dim, corr=corr)
                ans = np.array(
                    [
                        optimally_weighted_spline_experiment(
                            dgp.data(s)[0], 4, dgp.instrument_knots, dgp.endogenous_knots,
                        )
                        for s in range(1, 1001)
                    ]
                )
                opt_res.append(
                    {"n": n, "dim": dim, "corr": corr, "mean": ans.mean(), "std": ans.std(),}
                )
    spline_r2_opt_res = pd.DataFrame(opt_res)

    spline_r2_res.to_csv("checkpts/pismd_splines_mc3a_estimator.csv", index=False)
    spline_r2_opt_res.to_csv("checkpts/oposmd_splines_mc3a_estimator.csv", index=False)

if args.mc3b:
    pbar = tqdm(total=10)
    res_mc3 = []
    for n in [1000, 5000]:
        for dim in [0, 5, 10]:
            for corr in [0, 0.5]:
                if dim == 0 and corr == 0.5:
                    continue
                pbar.update(1)
                dgp = MC3(n, n, "cpu", dim, corr=corr)
                ans = np.array(
                    [
                        spl_experiment(
                            dgp.data(s)[0],
                            4,
                            dgp.instrument_knots,
                            dgp.endogenous_knots,
                            rcond=1e-6,
                        )
                        for s in range(1, 1001)
                    ]
                )
                res_mc3.append(
                    {"n": n, "dim": dim, "corr": corr, "mean": ans.mean(), "std": ans.std(),}
                )

    spline_r2_res_mc3 = pd.DataFrame(res_mc3)

    pbar = tqdm(total=10)
    opt_res_mc3 = []
    for n in [1000, 5000]:
        for dim in [0, 5, 10]:
            for corr in [0, 0.5]:
                if dim == 0 and corr == 0.5:
                    continue
                pbar.update(1)
                dgp = MC3(n, n, "cpu", dim, corr=corr)
                ans = np.array(
                    [
                        optimally_weighted_spline_experiment(
                            dgp.data(s)[0],
                            4,
                            dgp.instrument_knots,
                            dgp.endogenous_knots,
                            rcond=1e-6,
                        )
                        for s in range(1, 1001)
                    ]
                )
                opt_res_mc3.append(
                    {"n": n, "dim": dim, "corr": corr, "mean": ans.mean(), "std": ans.std(),}
                )

    spline_r2_opt_res_mc3 = pd.DataFrame(opt_res_mc3)

    spline_r2_res_mc3.to_csv("checkpts/pismd_splines_mc3b_estimator.csv", index=False)
    spline_r2_opt_res_mc3.to_csv("checkpts/oposmd_splines_mc3b_estimator.csv", index=False)

if args.temp:
    from pipeline.xfit import (
        spline_score_vstar,
        _half_sample_spline_es,
        _half_sample_spline_is,
    )

    nsim = 1000
    spline_score_results = []
    total = 10 * nsim * 2

    pbar = tqdm(total=total)
    for n in [1000, 5000]:
        for dim in [0, 5, 10]:
            for corr in [0, 0.5]:
                if dim == 0 and corr == 0.5:
                    continue
                for deg in [3, 4]:
                    dgp = MC2(n=n, batch_size=n, dimension=dim, corr=corr, device="cpu")
                    dgp_half = MC2(
                        n=n // 2, batch_size=n // 2, dimension=dim, corr=corr, device="cpu",
                    )

                    for s in range(1, nsim + 1):
                        pbar.update(1)
                        npvec = dgp.data(s)[0]
                        arg = dict(
                            npvec=npvec,
                            knots_inst=dgp.instrument_knots,
                            knots_endo=dgp.endogenous_knots,
                        )

                        n1 = dgp_half.data(2 * s)[0]
                        n2 = dgp_half.data(2 * s + 1)[0]

                        arg_esx = dict(
                            knots_inst=dgp_half.instrument_knots,
                            knots_endo=dgp_half.endogenous_knots,
                            deg=deg,
                            n_neighbors=100 if n == 5000 else 50,
                        )

                        res = spline_score_vstar(
                            **arg, deg=deg, n_neighbors=100 if n == 5000 else 50
                        )

                        # ES-X
                        est1, se1 = _half_sample_spline_es(n1, n2, **arg_esx)
                        est2, se2 = _half_sample_spline_es(n2, n1, **arg_esx)
                        est = (est1 + est2) / 2
                        se = ((se1 ** 2 + se2 ** 2) / 4) ** 0.5
                        res["esx_vstar"] = est
                        res["esx_vstar_se"] = se

                        # IS-X
                        del arg_esx["n_neighbors"]
                        est1, se1 = _half_sample_spline_is(n1, n2, **arg_esx)
                        est2, se2 = _half_sample_spline_is(n2, n1, **arg_esx)
                        est = (est1 + est2) / 2
                        se = ((se1 ** 2 + se2 ** 2) / 4) ** 0.5
                        res["isx_vstar"] = est
                        res["isx_vstar_se"] = se

                        res["n"] = n
                        res["dim"] = dim
                        res["corr"] = corr
                        res["deg"] = deg
                        spline_score_results.append(res)

    spline_score_df = pd.DataFrame(spline_score_results)
    spline_score_df.to_csv("checkpts/spline_smd_score_results_vstar_full.csv", index=False)

if args.temp_esx:
    from pipeline.xfit import (
        spline_score_vstar,
        _half_sample_spline_es,
        _half_sample_spline_is,
    )

    nsim = 1000
    spline_score_results = []
    total = 10 * nsim

    pbar = tqdm(total=total)
    for n in [1000, 5000]:
        for dim in [0, 5, 10]:
            for corr in [0, 0.5]:
                if dim == 0 and corr == 0.5:
                    continue
                for deg in [3, 4]:
                    dgp = MC2(n=n, batch_size=n, dimension=dim, corr=corr, device="cpu")
                    dgp_half = MC2(
                        n=n // 2, batch_size=n // 2, dimension=dim, corr=corr, device="cpu",
                    )

                    for s in range(1, nsim + 1):
                        pbar.update(1)
                        npvec = dgp.data(s)[0]
                        arg = dict(
                            npvec=npvec,
                            knots_inst=dgp.instrument_knots,
                            knots_endo=dgp.endogenous_knots,
                        )

                        n1 = dgp_half.data(2 * s)[0]
                        n2 = dgp_half.data(2 * s + 1)[0]

                        arg_esx = dict(
                            knots_inst=dgp_half.instrument_knots,
                            knots_endo=dgp_half.endogenous_knots,
                            deg=deg,
                            n_neighbors=5,
                        )

                        res = dict()
                        est1, se1 = _half_sample_spline_es(n1, n2, **arg_esx)
                        est2, se2 = _half_sample_spline_es(n2, n1, **arg_esx)
                        est = (est1 + est2) / 2
                        se = ((se1 ** 2 + se2 ** 2) / 4) ** 0.5
                        res["esx_vstar"] = est
                        res["esx_vstar_se"] = se
                        res["n"] = n
                        res["dim"] = dim
                        res["corr"] = corr
                        res["deg"] = deg
                        spline_score_results.append(res)
    spline_score_df = pd.DataFrame(spline_score_results)
    spline_score_df.to_csv(
        "checkpts/spline_smd_score_results_vstar_esx_small.csv", index=False
    )

if args.r1:
    spline_score_results = []
    nsim = 1000
    total = nsim * 6
    pbar = tqdm(total=total)
    for n in [1000, 5000]:
        for dim in [0, 5, 10]:
            dgp = MC4(n=n, batch_size=n, dimension=dim, device="cpu")
            for s in range(1, nsim + 1):
                pbar.update(1)
                deg = 4
                res = {
                    "spl_estimate": spl_experiment(
                        dgp.data(s)[0],
                        deg,
                        knots_inst=dgp.instrument_knots,
                        knots_endo=dgp.endogenous_knots,
                    )
                }
                res["n"] = n
                res["dim"] = dim
                res["deg"] = deg
                spline_score_results.append(res)
    spline_df = pd.DataFrame(spline_score_results)
    spline_df.to_csv("checkpts/spline_smd_r1.csv", index=False)

