import numpy as np
import pandas as pd
from pipeline.splines import spl_experiment
from dgp.dgp import DGP


class Strawberry(DGP):
    def __init__(self, *args, covariate=None, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            print(f"All arguments ignored by Strawberry args={args} kwargs={kwargs}")
        df = pd.read_csv("data/compiani/data_1.csv")
        # drop redundant columns and columns with 0 variation
        string_cols =  list(df.columns[:5])+list(df.columns[-9:])
        redund_strawb = string_cols + ['usda'] + ['x_usda_lettuce_2'] + ['x_outf_2']
        df = df.drop(columns = redund_strawb)
        df['q_own'] = np.log(df['q_own'])
        df['q_other'] = np.log(df['q_other'])
        df['p_own'] = np.log(df['p_own'])
        df['p_other'] = np.log(df['p_other'])
        self.covariates = covariate
        self.clean_data(df)
        self.batch_size = self.n = len(self.df)
        self.device = "cpu"

    def clean_data(self, df):
        
        if self.covariates is None:
            self.covariates = ["p_own", "p_other", 'x_usda_lettuce','x_outf', "income"]
            
        self.outcome = "q_own"
        self.treatment = "p_own"
        self.instruments = ['x_usda_lettuce', 'x_outf', "income", "spot_own", "spot_other", "z_own", "z_other", "z_out"]
        self.df = df[[self.outcome] + [self.treatment] + self.instruments + ['p_other']].dropna()

        df = self.df

        self.instrument_knots = np.zeros((len(self.instruments), 2))
        for i in range(len(self.instruments)):
            self.instrument_knots[i] = np.quantile(
                df[self.instruments[i]].values, [1 / 3, 2 / 3]
            )

        self.endogenous_knots = np.zeros((len(self.covariates), 2))

        for i, c in enumerate(self.covariates):
            self.endogenous_knots[i] = np.quantile(df[c], [1 / 3, 2 / 3])

    def transform_instrument(self):
        _, _, inst_basis, _ = spl_experiment(
            dict(
                response=self.df[self.outcome].values,
                endogenous=self.df[self.covariates].values,
                instrument=self.df[self.instruments].values,
            ),
            deg=3,
            knots_inst=self.instrument_knots,
            knots_endo=self.endogenous_knots,
            interact="full",
            full_return=True,
            se=False,
        )
        return inst_basis

    def data(self, *args, **kwargs):
        
        npvec, torchvec = self.process_data(
            response=self.df[self.outcome].values,
            endogenous=self.df[self.covariates].values.T,
            instrument=self.df[self.instruments].T,
            transformed_instrument=self.transform_instrument(),
        )

        # print([print(k, v.shape) for k, v in npvec.items()])
        dataset, loader = self.package_dataset(torchvec)
        return npvec, torchvec, dataset, loader


