from .dgp import DGP
import numpy as np
from statsmodels.tools import add_constant
from scipy.stats import norm
from scipy.stats import t as t_distribution


class SimpleTimeSeries(DGP):
    def __init__(
        self,
        n,
        batch_size,
        device,
        x_ar_param=0.7,
        lag_coef_ratio=0.2,
        endogeneity=0.5,
        n_lags=3,
        announce_truth=False,
        **kwargs,
    ):
        super().__init__(n + n_lags, batch_size + n_lags, device)
        self.x_ar_param = x_ar_param
        self.lag_coef_ratio = lag_coef_ratio
        self.endogeneity = endogeneity
        self.g = np.sin
        self.n_lags = n_lags

        self.x_long_run_variance = 1 / (1 - x_ar_param ** 2)  #
        self.true_parameter = np.exp(
            -self.x_long_run_variance / 2
        )  # https://math.stackexchange.com/questions/2134801/compute-e-sin-x-if-x-is-normally-distributed

        if announce_truth:
            print(f"True parameter: {self.true_parameter}")

    def data(self, seed=None):
        if seed is not None:
            rng = self.create_rng(seed)
        else:
            rng = np.random

        # Scalar AR for x
        u_t = rng.randn(self.n)
        x = np.empty(self.n)
        x[0] = u_t[0]
        for t in range(1, self.n):
            x[t] = self.x_ar_param * x[t - 1] + u_t[t]

        e_t = self.endogeneity * u_t + np.sqrt(1 - self.endogeneity ** 2) * rng.randn(
            self.n
        )
        y = np.empty(self.n)

        y[0] = e_t[0]
        for t in range(1, self.n):
            y[t] = (
                self.g(x[t])
                + sum(
                    [
                        ylag * (self.lag_coef_ratio ** (i + 1))
                        for i, ylag in enumerate(y[max(t - self.n_lags, 0) : t][::-1])
                    ]
                )
                + e_t[t]
            )

        covariates = [x[self.n_lags :]] + [
            y[self.n_lags - i : -i] for i in range(1, self.n_lags + 1)
        ]
        instruments = [x[self.n_lags - 1 : -1]] + [
            y[self.n_lags - i : -i] for i in range(1, self.n_lags + 1)
        ]

        self.x = x
        self.y = y

        transformed_instrument = self.transform_instrument(instruments)
        npvec, torchvec = self.process_data(
            y[self.n_lags :], covariates, instruments, transformed_instrument
        )
        dataset, loader = self.package_dataset(torchvec)
        return npvec, torchvec, dataset, loader

    def transform_instrument(self, instruments):
        return add_constant(
            np.vstack(
                [
                    np.array(instruments),
                    np.array(instruments) ** 2,
                    np.array(instruments) ** 3,
                ]
            ).T
        )


class ClaytonTimeSeries(DGP):
    def __init__(
        self,
        n,
        batch_size,
        device,
        x_ar_param=0.7,
        lag_coef_ratio=0.6,
        endogeneity=0.5,
        n_lags=3,
        announce_truth=False,
        **kwargs,
    ):
        super().__init__(n + n_lags, batch_size + n_lags, device)
        self.x_ar_param = x_ar_param
        self.lag_coef_ratio = lag_coef_ratio
        self.endogeneity = endogeneity
        self.g = np.sin
        self.n_lags = n_lags

        self.announce_truth = announce_truth

    def data(self, seed=None):
        if seed is not None:
            rng = self.create_rng(seed)
        else:
            rng = np.random

        v_t = rng.rand(self.n)
        e_t = norm.ppf(v_t)
        u_t = np.empty(self.n)
        u_t[0] = v_t[0]
        for t in range(1, self.n):
            u_t[t] = (
                u_t[t - 1]
                * np.sqrt(v_t[t])
                / (1 + u_t[t - 1] * np.sqrt(v_t[t]) - np.sqrt(v_t[t]))
            )
        t_dist = t_distribution(5)
        x = t_dist.ppf(u_t)
        y = np.empty(self.n)

        y[0] = e_t[0]
        for t in range(1, self.n):
            y[t] = (
                self.g(x[t])
                + sum(
                    [
                        ylag * (self.lag_coef_ratio ** (i + 1))
                        for i, ylag in enumerate(y[max(t - self.n_lags, 0) : t][::-1])
                    ]
                )
                + e_t[t]
            )

        covariates = [x[self.n_lags :]] + [
            y[self.n_lags - i : -i] for i in range(1, self.n_lags + 1)
        ]
        instruments = [x[self.n_lags - 1 : -1]] + [
            y[self.n_lags - i : -i] for i in range(1, self.n_lags + 1)
        ]

        self.x = x
        self.y = y

        transformed_instrument = self.transform_instrument(instruments)
        npvec, torchvec = self.process_data(
            y[self.n_lags :], covariates, instruments, transformed_instrument
        )
        dataset, loader = self.package_dataset(torchvec)

        self.sample_parameter_value = np.cos(self.x).mean()

        if self.announce_truth:
            print(f"True parameter (sample): {self.sample_parameter_value}")

        return npvec, torchvec, dataset, loader

    def transform_instrument(self, instruments):
        interactions = [
            instruments[i] * instruments[j]
            for i in range(len(instruments))
            for j in range(len(instruments))
            if i > j
        ]
        return add_constant(
            np.vstack(
                [
                    np.array(instruments),
                    # np.abs(np.array(instruments)),
                    # np.abs(np.array(instruments)) ** 1.3,
                    # np.array(instruments) ** 3,
                    np.array(interactions),
                ]
            ).T
        )

