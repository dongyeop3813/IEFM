import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.clipper import Clipper
from dem.models.components.noise_schedules import BaseNoiseSchedule


def OT_conditional_VF(
    perturbed: torch.Tensor, 
    origin: torch.Tensor,
    time: torch.Tensor,
    noise_schedule: BaseNoiseSchedule,
):
    """
    Optimal transport conditioanl vector field.
    For given time points and origin, 
    evaluate CVF at given points (perturbed).

    Here K = sample size, D = dimension of data.
    
    :param perturbed: Points where CVF will be evaluated. (D dimension)
    :param origin: Points where move starts. (K x D dimension)
    :param time: time step t. (scalar)

    :return: Evaluted (conditional) vector field (K x D dimension)
    """

    coeff = 1 - noise_schedule.sigma_min
    return (origin - coeff * perturbed) / (1 - coeff * time + 1e-5)


def VF_estimator(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    option: str = "OT CVF",
) -> torch.Tensor:
    """
    Here D = dimension of data.
    
    :param t: timestep (scalar)
    :param x: data (D dimension)

    :return: Estimated vector field (D dimension)
    """

    # energy_function = -E(x)

    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)

    h_t = noise_schedule.h(repeated_t).unsqueeze(1)

    # K x D dimension
    samples = (repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())) / (repeated_t.unsqueeze(1) + 1e-5)

    # log (unnormalized) prob of each sample (dimension K)
    log_prob = energy_function(samples)

    # choose probabilith path
    if option == "OT CVF":
        conditional_VF = OT_conditional_VF
    else:
        raise Exception("Invalid probability path")

    # weights (dimension K)
    weights = torch.softmax(log_prob, dim=0)

    # marginal vector field estimator
    return torch.matmul(weights, conditional_VF(x, samples, t, noise_schedule))


def estimate_VF(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    option: str = "OT CVF",
):

    # vectorizing map for batch processing
    vmmaped_fxn = torch.vmap(VF_estimator, in_dims=(0, 0, None, None, None, None), randomness="different")

    return vmmaped_fxn(t, x, energy_function, noise_schedule, num_mc_samples, option)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from dem.models.components.noise_schedules import OTcnfNoiseSchedule
    from dem.energies.gmm_energy import GMM
    from fab.utils.plotting import plot_contours
    from matplotlib.animation import FuncAnimation

    noise_schedule = OTcnfNoiseSchedule(0.001)
    energy_function = GMM()
    grid_steps = 20
    batch_size = grid_steps ** 2
    plotting_bounds = (-1.4 * 40, 1.4 * 40)

    x_coord, y_coord = np.meshgrid(
        np.linspace(*plotting_bounds, grid_steps), 
        np.linspace(*plotting_bounds, grid_steps)
    )
    x = torch.tensor([x_coord, y_coord], dtype=torch.float).T.flatten(start_dim=0, end_dim=1)

    for time in np.linspace(0, 1, 10):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_contours(
            energy_function,
            bounds=plotting_bounds,
            ax = ax,
            n_contour_levels=50,
            grid_width_n_points=200,
        )

        t = torch.tensor([time * 0.1] * batch_size, dtype=torch.float)
        estimated_VF = estimate_VF(t, x, energy_function, noise_schedule, 1000).detach()

        ax.quiver(x_coord, y_coord, estimated_VF[:,0], estimated_VF[:,1])
        fig.savefig(f"./gmm_vf_time={time}.png")

    # ani = FuncAnimation(fig, animate, 10, repeat=False)
    # ani.save("./gmm_vf.mp4")
