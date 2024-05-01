from functools import partial
import argparse

import torch

from dem.models.components.noise_schedules import OTcnfNoiseSchedule
from dem.models.components.vf_estimator import _VF_estimator
from dem.energies.gmm_energy import GMM
from dem.energies.normal_dist_energy import Normal


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

    noise_schedule = OTcnfNoiseSchedule(0.001, 1.0)
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

    x = torch.tensor([20.0, 20.0])
    vector = _VF_estimator(
        time,
        x,
        energy_function,
        noise_schedule,
        1000,
    )

    print(f"U_K({x.tolist()}, {time.tolist()}) = {vector.tolist()}")