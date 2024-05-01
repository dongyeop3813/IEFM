from functools import partial
from typing import Callable
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from matplotlib.animation import FuncAnimation

from dem.models.components.noise_schedules import (
    OTcnfNoiseSchedule, VEcnfNoiseSchedule
)
from dem.models.components.vf_estimator import estimate_VF_for_fixed_time
from dem.energies.gmm_energy import GMM, plot_vecfield
from dem.energies.normal_dist_energy import Normal

from fab.utils.plotting import plot_contours


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--energy_function', default='GMM')
    parser.add_argument('-n', '--normal_sigma', default=30.0, type=float)
    parser.add_argument('-M', '--sigma_max', default=1.0, type=float)
    parser.add_argument('-m', '--sigma_min', default=0.001, type=float)
    parser.add_argument('-g', '--grid_step', default=30, type=int)
    parser.add_argument('-t', '--time', default=0.5, type=float)
    parser.add_argument('-p', '--prob_path', default='OT', type=str)
    args = parser.parse_args()

    ENERGY_FUNCTION = args.energy_function

    grid_step = args.grid_step
    num_points = grid_step ** 2

    normal_dist_sigma = args.normal_sigma
    sigma_max = args.sigma_max
    sigma_min = args.sigma_min

    prob_path = args.prob_path

    if prob_path == "OT":
        noise_schedule = OTcnfNoiseSchedule(sigma_min, sigma_max)
    elif prob_path == "VE":
        noise_schedule = VEcnfNoiseSchedule(sigma_min, sigma_max)
    else:
        raise Exception("Invalid prob path")

    time = torch.tensor(args.time, dtype=torch.float)

    print(f"[INFO] Energy function: {ENERGY_FUNCTION}")
    print(f"[INFO] Time: {args.time}")
    print(f"[INFO] sigma max: {sigma_max}")
    print(f"[INFO] sigma min: {sigma_min}")

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
        option="VE",
    )

    plot_vecfield(ax, vecfield, grid_step, plotting_bounds, debug=True)

    fig.savefig(
        f"./figure/VFplot/{ENERGY_FUNCTION}-VE-time={args.time:0.2f}" \
        f"-sgmM={sigma_max:0.2f}-sgmm={sigma_min:0.2f}.png"
    )
