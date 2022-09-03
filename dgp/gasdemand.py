import numpy as np
import pandas as pd
from pipeline.splines import spl_experiment
from dgp.dgp import DGP


class GasDemand(DGP):
    def __init__(self, *args, covariate=None, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            print(f"All arguments ignored by Gas Demand args={args} kwargs={kwargs}")
        df = pd.read_stata("data/chen_christensen/gasoline_demand_BHP2012.dta")
        self.covariates = covariate
        self.clean_data(df)
        self.batch_size = self.n = len(self.df)
        self.device = "cpu"

    def clean_data(self, df):
        df["log_y"] /= 11.51
        if self.covariates is None:
            self.covariates = [
                "log_y",
                "log_hhsize",
                "log_driver",
                "log_hhr_age",
                "total_wrkr",
                "publictransit_d",
            ]
        self.outcome = "log_q"
        self.treatment = "log_p"
        self.instruments = ["distance_oil1000"] + self.covariates

        self.std_lhs, self.mean_lhs = df["log_q"].std(), df["log_q"].mean()
        # df["outcome_normalized"] = (df["log_q"] - self.mean_lhs) / self.std_lhs

        self.df = df[[self.outcome] + [self.treatment] + self.instruments].dropna()
        df = self.df

        self.instrument_knots = np.zeros((len(self.instruments), 2))
        for i in range(len(self.instruments)):
            self.instrument_knots[i] = np.quantile(
                df[self.instruments[i]].values, [1 / 3, 2 / 3]
            )

        self.endogenous_knots = np.zeros((len(self.covariates) + 1, 2))

        for i, c in enumerate([self.treatment] + self.covariates):
            self.endogenous_knots[i] = np.quantile(df[c], [1 / 3, 2 / 3])

    def transform_instrument(self):
        _, _, inst_basis, _ = spl_experiment(
            dict(
                response=self.df[self.outcome].values,
                endogenous=self.df[[self.treatment] + self.covariates].values,
                instrument=self.df[self.instruments].values,
            ),
            deg=4,
            knots_inst=self.instrument_knots,
            knots_endo=self.endogenous_knots,
            interact="full",
            full_return=True,
            se=False,
        )
        return inst_basis
        # instruments = self.df[self.instruments].values
        # n, p_instr = instruments.shape
        # pairwise_interactions = np.array(
        #     [
        #         instruments[:, i] * instruments[:, j]
        #         for i in range(p_instr)
        #         for j in range(p_instr)
        #         if i >= j
        #     ]
        # ).T
        # return np.c_[
        #     np.ones((n, 1)),
        #     instruments,
        #     pairwise_interactions,
        #     instruments ** 2,
        #     instruments ** 3,
        # ]

    def data(self, *args, **kwargs):
        npvec, torchvec = self.process_data(
            response=self.df[self.outcome].values,
            endogenous=self.df[[self.treatment] + self.covariates].values.T,
            instrument=self.df[self.instruments].T,
            transformed_instrument=self.transform_instrument(),
        )

        # print([print(k, v.shape) for k, v in npvec.items()])
        dataset, loader = self.package_dataset(torchvec)
        return npvec, torchvec, dataset, loader

