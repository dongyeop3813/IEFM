from functools import partial
from typing import Callable
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from matplotlib.animation import FuncAnimation

from dem.models.components.noise_schedules import OTcnfNoiseSchedule
from dem.models.components.vf_estimator import estimate_VF_for_fixed_time
from dem.energies.gmm_energy import GMM, plot_vecfield, plot_vecfield_error
from dem.energies.normal_dist_energy import Normal

from fab.utils.plotting import plot_contours


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--energy_function', default='GMM')
    parser.add_argument('-s', '--normal_sigma', default=30.0, type=float)
    parser.add_argument('-g', '--grid_step', default=30, type=int)
    parser.add_argument('-t', '--time', default=0.5, type=float)
    args = parser.parse_args()

    ENERGY_FUNCTION = args.energy_function

    grid_step = args.grid_step
    num_points = grid_step ** 2

    normal_dist_sigma = args.normal_sigma

    noise_schedule = OTcnfNoiseSchedule(0.001)
    time = torch.tensor(args.time, dtype=torch.float)

    if ENERGY_FUNCTION == "GMM":
        energy_function = GMM()
        plotting_bounds = (-1.4 * 40.0, 1.4 * 40.0)

    elif ENERGY_FUNCTION == "Normal":
        energy_function = Normal(sigma=normal_dist_sigma)
        plotting_bounds = (
            -normal_dist_sigma * 2.5, normal_dist_sigma * 2.5
        )
        ENERGY_FUNCTION += f"-sgm={normal_dist_sigma}"

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    plot_contours(
        energy_function,
        bounds=plotting_bounds,
        ax=ax,
        n_contour_levels=50,
        grid_width_n_points=200,
    )

    vecfield = partial(
        estimate_VF_for_fixed_time, 
        time,
        energy_function=energy_function, 
        noise_schedule=noise_schedule,
        num_mc_samples=1000,
    )

    plot_vecfield_error(fig, ax, vecfield, vecfield, grid_step=40)

    fig.savefig(f"./figure/{ENERGY_FUNCTION}-time={time}.png")
