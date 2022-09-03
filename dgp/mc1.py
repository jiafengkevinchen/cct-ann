import numpy as np
from .dgp import DGP
from scipy import stats


class MC1(DGP):
    def __init__(
        self, n, batch_size, device, dimension=5, high_dim_relevant=True, corr=0
    ):
        super().__init__(n, batch_size, device)
        self.h01 = lambda x: 1 / (1 + np.exp(-x))
        self.h02 = lambda x: np.log(1 + x)

        self.dimension = dimension
        if dimension > 0:
            rng_cov = self.create_rng(9999)
            covariance_matrix = rng_cov.randn(dimension, dimension)
            covariance_matrix = (
                np.eye(dimension) + covariance_matrix.T @ covariance_matrix
            )
            covariance_matrix = (
                covariance_matrix
                / np.diag(covariance_matrix)[:, None] ** 0.5
                / np.diag(covariance_matrix)[None, :] ** 0.5
            )
            self.covariance_matrix = covariance_matrix
            self.covariance_matrix_root = np.linalg.cholesky(self.covariance_matrix)
            self.complex_func = lambda x_high_dim: (
                x_high_dim[:, 0] ** 3 * 5
                + x_high_dim[:, min(1, self.dimension - 1)]
                * (np.clip(x_high_dim, a_min=0.5, a_max=None)).max(axis=1)
                + np.exp(-x_high_dim[:, -1]) / 2
            )
            self.high_dim_relevant = high_dim_relevant
            self.corr = corr
            self.corr_mat = self.corr * np.ones((self.dimension, 3))

    def compute_knots(self):
        npvec, _, _, _ = self.data(seed=21480099, transform_instrument=False)

        x = npvec["endogenous"]
        self.endogenous_knots = np.zeros((x.shape[1], 2))
        for i in range(x.shape[1]):
            self.endogenous_knots[i] = np.quantile(x[:, i], [1 / 3, 2 / 3])

        x = npvec["instrument"]
        self.instrument_knots = np.zeros((x.shape[1], 2))
        for i in range(x.shape[1]):
            self.instrument_knots[i] = np.quantile(x[:, i], [1 / 3, 2 / 3])

    def data(self, seed=None, transform_instrument=True):
        if seed is not None:
            rng = self.create_rng(seed)
        else:
            rng = np.random
        h01 = self.h01
        h02 = self.h02
        n = self.n

        x1 = rng.rand(n)
        x2 = rng.rand(n)
        x3 = rng.rand(n)

        u = rng.randn(n) * np.sqrt((x1 ** 2 + x2 ** 2 + x3 ** 2) / 3)
        e = rng.randn(n) * np.sqrt(0.1)

        y2 = x1 + x2 + x3 + 0.9 * u + e
        y1 = x1 + h01(y2) + h02(x2) + u

        endogenous = np.c_[x1, y2, x2]
        instrument = np.c_[x1, x2, x3]

        if self.dimension > 0:
            x_high_dim_untransformed = (1 - self.corr ** 2) ** 0.5 * (
                self.covariance_matrix_root @ rng.randn(self.dimension, n)
            ).T + (self.corr_mat @ np.c_[x1, x2, x3].T).T
            x_high_dim = stats.norm.cdf(x_high_dim_untransformed)

            if self.high_dim_relevant:
                y1 = x1 + h01(y2) + h02(x2) + u + self.complex_func(x_high_dim)
            else:
                y1 = x1 + h01(y2) + h02(x2) + u
            endogenous = np.c_[endogenous, x_high_dim]
            instrument = np.c_[instrument, x_high_dim]

        transformed_instrument = (
            self.transform_instrument(instrument) if transform_instrument else None
        )

        npvec, torchvec = self.process_data(
            response=y1,
            endogenous=endogenous.T,
            instrument=instrument.T,
            transformed_instrument=transformed_instrument,
        )

        dataset, loader = self.package_dataset(torchvec)
        return npvec, torchvec, dataset, loader

    def simple_transform_instrument(self, x):
        """Chen 2007 instrument transform"""
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        return np.array(
            [np.ones(len(x1))]
            + [x1 ** i for i in range(1, 5)]
            + [np.maximum(x1 - 0.5, 0) ** 4]
            + [x2 ** i for i in range(1, 5)]
            + [np.maximum(x2 - 0.5, 0) ** 4]
            + [x3 ** i for i in range(1, 5)]
            + [np.maximum(x3 - k, 0) ** 4 for k in [0.1, 0.25, 0.5, 0.75, 0.9]]
            + [
                x1 * x3,
                x2 * x3,
                x1 * np.maximum(x3 - 0.25, 0) ** 4,
                x2 * np.maximum(x3 - 0.25, 0) ** 4,
                x1 * np.maximum(x3 - 0.75, 0) ** 4,
                x2 * np.maximum(x3 - 0.75, 0) ** 4,
            ]
        ).T

    def transform_instrument(self, x):
        self.compute_knots()
        if x.shape[1] > 3:
            extra = x[:, 3:]
            first_three = x[:, :3]

            # The interactions don't seem to change stuff qualitatively,
            # maybe reduces bias a little

            interactions = np.hstack(
                [
                    (first_three[:, [j]] * extra[:, [i]])
                    for i in range(extra.shape[1])
                    for j in range(first_three.shape[1])
                ]
            )
            return np.c_[
                self.simple_transform_instrument(x),
                x[:, 3:],
                interactions,
                x[:, 3:] ** 2,  # This matters a lot
                # x[:, 3:] ** 3,  # what if we make it even larger?
                # TODO: refactor this into an option
            ]
        else:
            return self.simple_transform_instrument(x)
