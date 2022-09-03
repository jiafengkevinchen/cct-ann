import numpy as np
from scipy import stats

from .mc1 import MC1
from scipy.special import expit
from pipeline.splines import _instrument_basis


class MC4(MC1):
    def __init__(
        self, n, batch_size, device, dimension=5, corr=0,
    ):
        super().__init__(
            n, batch_size, device, dimension=dimension, corr=corr,
        )
        self.dimension = dimension

        rng = self.create_rng(214802189)

        k = 4
        self.fs_w = 0.1 + 0.9 * rng.rand(k, self.dimension + 2)
        self.middle_w = 0.1 + 0.9 * rng.rand(k)
        self.second = 0.1 + 0.9 * rng.rand(self.dimension)

    def data(self, seed=None, transform_instrument=True):
        if seed is not None:
            rng = self.create_rng(seed)
        else:
            rng = np.random

        n = self.n

        eps1 = rng.randn(n)
        eps2 = 0.9 * eps1 + (1 - 0.9 ** 2) ** 0.5 * rng.randn(n)
        # eps2 = rng.randn(n)

        x = rng.randn(n, self.dimension + 2)

        y1 = np.tanh(x @ self.fs_w.T / 5) @ self.middle_w + eps1
        covs = x[:, 2:].copy()
        y = y1 + covs @ self.second + eps2

        endogenous = np.c_[y1[:, None], covs].copy()
        instrument = x.copy()

        transformed_instrument = (
            self.transform_instrument(instrument) if transform_instrument else None
        )

        npvec, torchvec = self.process_data(
            response=y,
            endogenous=endogenous.T,
            instrument=instrument.T,
            transformed_instrument=transformed_instrument,
        )

        dataset, loader = self.package_dataset(torchvec)
        return npvec, torchvec, dataset, loader

    def transform_instrument(self, x):
        self.compute_knots()
        return _instrument_basis(x, 4, knots_inst=self.instrument_knots, interact="full")
