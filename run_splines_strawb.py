import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dgp.compiani import Strawberry
from pipeline.splines import optimally_weighted_spline_experiment, spl_experiment
from xfit import spline_score

warnings.simplefilter("ignore", RuntimeWarning)

spline_score_results = []
dgp = Strawberry()

for deg in [3, 4]:
    npvec, _, _, _ = dgp.data()
    arg = dict(
        npvec=npvec, knots_inst=dgp.instrument_knots, knots_endo=dgp.endogenous_knots,
    )
    res = spline_score(**arg, deg=deg)
    res["pismd_spline"], res["pismd_spline_se"] = spl_experiment(
        **arg, deg=deg, pl=False, se=True,
    )
    res["oposmd_spline"] = optimally_weighted_spline_experiment(
        **arg, deg=deg, pl=False
    )
    res["deg"] = deg
    spline_score_results.append(res)


spline_score_df = pd.DataFrame(spline_score_results)
spline_score_df.to_csv("checkpts/spline_strawb_smd_score_results.csv", index=False)
