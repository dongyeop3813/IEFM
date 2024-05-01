from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
import torch
from fab.utils.plotting import plot_contours, plot_marginal_pair
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.logging_utils import fig_to_image


class Normal(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        sigma=1.0,
    ):
        self.name = "std.normal"
        self.sigma = sigma

        super().__init__(
            dimensionality=dimensionality
        )

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return -(samples.norm(p=2, dim=-1) ** 2) / (2 * self.sigma)

    @property
    def dimensionality(self):
        return 2
