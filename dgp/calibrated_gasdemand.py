import os
import pandas as pd
import numpy as np
from pipeline.splines import _instrument_basis
import torch
from dgp.dgp import DGP

from scipy.special import expit as sigmoid

ESTIMATED_ELASTICITY = 1.43


def load_data():
    df = pd.read_stata("data/chen_christensen/gasoline_demand_BHP2012.dta")
    df["log_y"] /= 11.51
    covariates = [
        "log_y",
        "log_hhsize",
        "log_driver",
        "log_hhr_age",
        "total_wrkr",
        "publictransit_d",
    ]
    outcome = "log_q"
    treatment = "log_p"
    instruments = ["distance_oil1000"] + covariates

    data = df[[outcome] + [treatment] + instruments].dropna()
    return data, covariates, outcome, treatment, instruments


def generated_calibration_weights():
    data, covariates, outcome, treatment, instruments = load_data()
    torch.manual_seed(123)

    # two layer neural network predicting log q from covariates in pytorch
    net = torch.nn.Sequential(
        torch.nn.Linear(len(instruments), 15), torch.nn.Sigmoid(), torch.nn.Linear(15, 1)
    )
    x = torch.tensor(data[instruments].values).float()
    y = torch.tensor(data[treatment].values).float().unsqueeze(-1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

    for _ in range(15000):
        optimizer.zero_grad()
        loss = ((y - net(x)) ** 2).mean()
        loss.backward()
        optimizer.step()
        if _ % 3000 == 0:
            print(_, loss.item())

    # predict log_p from covariates
    net_c = torch.nn.Sequential(
        torch.nn.Linear(len(covariates), 15), torch.nn.Sigmoid(), torch.nn.Linear(15, 1)
    )
    xx = torch.tensor(data[covariates].values).float()
    yy = (
        torch.tensor((data[outcome] + ESTIMATED_ELASTICITY * data[treatment]).values)
        .float()
        .unsqueeze(-1)
    )  # 1.43 is estimated elasticity

    optimizer = torch.optim.Adam(net_c.parameters(), lr=0.005)

    for _ in range(15000):
        optimizer.zero_grad()
        loss = ((yy - net_c(xx)) ** 2).mean()
        loss.backward()
        optimizer.step()
        if _ % 3000 == 0:
            print(_, loss.item())

    # save state dict
    torch.save(net_c.state_dict(), "data/chen_christensen/covariate_function.pt")
    torch.save(net.state_dict(), "data/chen_christensen/instrument_function.pt")


class CalibratedGasDemand(DGP):
    def __init__(self, n, batch_size=None, device=None):
        self.n = n
        self.device = "cpu"
        self.batch_size = self.n
        df, covariates, outcome, treatment, instruments = self.generate_data(
            n=100000, seed=2138
        )

        self.instrument_knots = np.zeros((len(instruments), 2))
        for i, c in enumerate(instruments):
            self.instrument_knots[i] = np.quantile(df[c].values, [1 / 3, 2 / 3])

        self.endogenous_knots = np.zeros((len(covariates) + 1, 2))
        for i, c in enumerate([treatment] + covariates):
            self.endogenous_knots[i] = np.quantile(df[c], [1 / 3, 2 / 3])

    def generate_data(self, n, seed=None):
        if seed is not None:
            rng = self.create_rng(seed)
        else:
            rng = np.random
        data, covariates, outcome, treatment, instruments = load_data()
        if not os.path.exists("data/chen_christensen/covariate_function.pt"):
            generated_calibration_weights()

        # Initialize the torch nns
        cov_weights = torch.load("data/chen_christensen/covariate_function.pt")
        ins_weights = torch.load("data/chen_christensen/instrument_function.pt")
        cov_func = torch.nn.Sequential(
            torch.nn.Linear(len(covariates), 15), torch.nn.Sigmoid(), torch.nn.Linear(15, 1)
        ).eval()
        ins_func = torch.nn.Sequential(
            torch.nn.Linear(len(instruments), 15), torch.nn.Sigmoid(), torch.nn.Linear(15, 1)
        ).eval()
        cov_func.load_state_dict(cov_weights)
        ins_func.load_state_dict(ins_weights)

        sampled_data = data.sample(n=n, replace=True, random_state=rng).reset_index(drop=True)

        # fmt:off
        noised_up_instruments = sampled_data[instruments] + \
            sampled_data[instruments].std() * 0.1 * pd.DataFrame(rng.randn(n, len(instruments)), columns=instruments)
        treatment_resid = sampled_data[treatment].values - \
            ins_func(torch.tensor(sampled_data[instruments].values).float()).detach().numpy().flatten()
        outcome_resid = sampled_data[outcome].values + ESTIMATED_ELASTICITY * sampled_data[treatment].values \
            - cov_func(torch.tensor(sampled_data[covariates].values).float()).detach().numpy().flatten()
        predicted_treatment = ins_func(torch.tensor(noised_up_instruments.values).float()).detach().numpy().flatten()
        predicted_outcome_covariate = cov_func(torch.tensor(noised_up_instruments[covariates].values).float()).detach().numpy().flatten()
        # fmt:on

        first_stage_resid = 1.3 * treatment_resid * rng.randn(n)
        simulated_logp = predicted_treatment + first_stage_resid

        ds = sigmoid(simulated_logp) * (1 - sigmoid(simulated_logp))
        avg_ds = ds.mean()

        elasticity = -ESTIMATED_ELASTICITY * (
            1
            - 5 * avg_ds
            + 5 * ds
            + (predicted_outcome_covariate - predicted_outcome_covariate.mean())
            / predicted_outcome_covariate.std()
            * 0.2
        )

        structural_function = (
            predicted_outcome_covariate
            + (elasticity + ESTIMATED_ELASTICITY * 5 * ds) * simulated_logp
            - ESTIMATED_ELASTICITY * 5 * sigmoid(simulated_logp)
        )

        structural_residual = outcome_resid * rng.randn(n) + 1.2 * first_stage_resid
        simulated_logq = structural_function + structural_residual

        simulated_data = noised_up_instruments.copy()
        simulated_data["log_q"] = simulated_logq
        simulated_data["log_p"] = simulated_logp
        simulated_data["elasticity"] = elasticity
        return simulated_data, covariates, outcome, treatment, instruments

    def data(self, seed=None):
        data, covariates, outcome, treatment, instruments = self.generate_data(self.n, seed)
        npvec, torchvec = self.process_data(
            response=data[outcome].values,
            endogenous=data[[treatment] + covariates].values.T,
            instrument=data[instruments].values.T,
            transformed_instrument=self.transform_instrument(data[instruments]),
        )

        dataset, loader = self.package_dataset(torchvec)
        return npvec, torchvec, dataset, loader

    def transform_instrument(self, instruments):
        return _instrument_basis(instruments.values, 4, self.instrument_knots, interact="full")
