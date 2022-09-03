import numpy as np
from torch import as_tensor

from torch.utils.data import TensorDataset, DataLoader


class DGP:
    def __init__(self, n, batch_size, device):
        self.n = n
        self.batch_size = batch_size
        self.device = device

    def create_rng(self, seed):
        rng = np.random.RandomState(seed)
        return rng

    def process_data(
        self, response, endogenous, instrument, transformed_instrument=None
    ):
        np_vec = {
            "response": response[:, None] if len(response.shape) == 1 else response,
            "endogenous": np.array(endogenous).T,
            "instrument": np.array(instrument).T,
            "transformed_instrument": transformed_instrument
            if transformed_instrument is not None
            else None,
            "inverse_design_instrument": np.linalg.pinv(
                transformed_instrument.T @ transformed_instrument / self.n,
                rcond=1e-5,
                hermitian=True,
            )
            if transformed_instrument is not None
            else None,
        }
        if np_vec["transformed_instrument"] is None:
            del np_vec["transformed_instrument"]
            del np_vec["inverse_design_instrument"]

        torch_vec = {k: as_tensor(v).float().to(self.device) for k, v in np_vec.items()}

        return np_vec, torch_vec

    def package_dataset(self, torchvec):
        if self.n > self.batch_size:
            keys = list(torchvec.keys())
            dataset = TensorDataset(
                *[torchvec[k] for k in keys if k != "inverse_design_instrument"]
            )
            loader = DataLoader(dataset, batch_size=self.batch_size)
            return dataset, loader
        else:
            keys = list(torchvec.keys())
            dataset = [[torchvec[k] for k in keys if k != "inverse_design_instrument"]]
            return dataset, dataset
