import numpy as np
from scipy import stats

from .mc1 import MC1


class MC2a(MC1):
    """MC2, but with square in the endogenous variable of interest"""

    def __init__(
        self,
        n,
        batch_size,
        device,
        dimension=5,
        high_dim_relevant=True,
        corr=0,
        heteroskedastic=True,
    ):
        super().__init__(
            n,
            batch_size,
            device,
            dimension=dimension,
            high_dim_relevant=high_dim_relevant,
            corr=corr,
        )
        self.dimension = dimension
        self.heteroskedastic = heteroskedastic

    def data(self, seed=None, transform_instrument=True):
        if seed is not None:
            rng = self.create_rng(seed)
        else:
            rng = np.random
        h01 = self.h01
        h02 = self.h02
        n = self.n

        x2 = rng.rand(n)
        u2 = rng.randn(n)
        u3 = rng.randn(n)

        x1 = stats.norm.cdf(u2)
        x3 = stats.norm.cdf(u3)

        eps1, eps2, eps3 = rng.randn(3, n)
        eps = (eps1 + eps2 + eps3) / 3
        if self.heteroskedastic:
            eps *= np.sqrt((x1 ** 2 + x2 ** 2 + x3 ** 2) / 3)

        y2 = x1 + 0.5 * eps2 + eps
        y3 = stats.norm.cdf(u3 + 0.5 * eps3)
        x_high_dim = None

        endogenous = np.c_[y2, y3, x2]
        instrument = np.c_[x1, x2, x3]

        if self.dimension > 0:
            x_high_dim_untransformed = (1 - self.corr ** 2) ** 0.5 * (
                self.covariance_matrix_root @ rng.randn(self.dimension, n)
            ).T + (self.corr_mat @ np.c_[x1, x2, x3].T).T
            x_high_dim = stats.norm.cdf(x_high_dim_untransformed)

            endogenous = np.c_[endogenous, x_high_dim]
            instrument = np.c_[instrument, x_high_dim]

            if self.high_dim_relevant:
                y1 = y2 ** 2 + h01(y3) + h02(x2) + eps + self.complex_func(x_high_dim)
            else:
                y1 = y2 ** 2 + h01(y3) + h02(x2) + eps
        else:
            y1 = y2 ** 2 + h01(y3) + h02(x2) + eps

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
