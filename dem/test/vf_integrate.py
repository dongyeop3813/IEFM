from functools import partial
import argparse

import torch
import matplotlib.pyplot as plt

from dem.models.components.noise_schedules import (
    OTcnfNoiseSchedule, VEcnfNoiseSchedule
)
from dem.models.components.vf_estimator import estimate_VF
from dem.models.components.optimal_transport import wasserstein
from dem.energies.gmm_energy import GMM
from dem.energies.multi_double_well_energy import MultiDoubleWellEnergy

from fab.utils.plotting import plot_contours, plot_marginal_pair

import ot as pot

from torchdiffeq import odeint


SAMPLE_SIZE = 1000
INTEGRATION_METHOD = 'dopri5'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-E', '--energy_function', default='GMM', type=str)

    parser.add_argument('-M', '--sigma_max', default=1.0, type=float)
    parser.add_argument('-m', '--sigma_min', default=1e-5, type=float)

    parser.add_argument('-s', '--start_time', default=0.01, type=float)
    parser.add_argument('-K', '--num_mc_samples', default=1000, type=int)
    parser.add_argument('-p', '--prob_path', default='OT', type=str)
    args = parser.parse_args()

    return args


def get_filename(sigma_max, sigma_min, num_mc_samples, start_time, end_time, prob_path):
    filename = f'./figure/VF_integration/{ENERGY_FUNCTION}-{prob_path}/sample'

    if sigma_max != 1.0:
        filename += f'-sgM={sigma_max:0.2f}'

    if sigma_min != 1e-5:
        filename += f'-sgm={sigma_min:0.5f}'

    if num_mc_samples != 1000:
        filename += f'-K={num_mc_samples}'

    if start_time != 0.01:
        filename += f'-s={start_time:0.3f}'

    if end_time != 1.0:
        filename += f'-e={end_time:0.2f}'

    filename += '.png'

    return filename


def wasserstein2(energy_function, sample):
    test_set = energy_function.sample_test_set(SAMPLE_SIZE)

    # sample_energies = energy_function(energy_function.normalize(sample))
    # test_energies = energy_function(energy_function.normalize(test_set))

    # energy_w2 = pot.emd2_1d(test_energies.cpu().numpy(), sample_energies.cpu().numpy())

    return wasserstein(test_set, sample, power=2)


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def sample_from_prior(std_dev, dim, device='cpu'):
    return torch.randn((SAMPLE_SIZE, dim), device=device) * std_dev


def make_vector_field(
    energy_function,
    noise_schedule,
    num_mc_samples,
    device,
    option,
):
    def fxn(t, x):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return estimate_VF(
            t.repeat(len(x)), 
            x,
            energy_function,
            noise_schedule,
            num_mc_samples,
            option,
            device,
        )

    return fxn


def sample_trajectory_by_ode_integration(vf, prior_sample, time):
    trajectory = odeint(
        vf,
        prior_sample,
        t=time,
        method=INTEGRATION_METHOD,
        atol=1e-5,
        rtol=1e-5,
    ).detach()

    return trajectory


def draw_figure(energy_function, sample_trajectory, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    plotting_bounds=(-1.4 * 40, 1.4 * 40)

    sample_trajectory = sample_trajectory.to('cpu')

    sample = sample_trajectory[-1]

    energy_function.gmm.to('cpu')

    log_prob = energy_function.gmm.log_prob

    draw_sample_figure(
        axs[0], log_prob,
        sample, plotting_bounds,
    )
    axs[0].set_title(title)

    energy_function.gmm.to('cpu')

    draw_trajectory_figure(
        axs[1], log_prob,
        sample_trajectory, plotting_bounds
    )

    return fig, axs


def draw_sample_figure(ax, log_prob, sample, plotting_bounds):
    plot_contours(
        log_prob,
        bounds=plotting_bounds,
        ax=ax,
        n_contour_levels=50,
        grid_width_n_points=200,
    )

    plot_marginal_pair(sample, ax=ax, bounds=plotting_bounds)


def draw_trajectory_figure(ax, log_prob, trajectory, plotting_bounds):
    plot_contours(
        log_prob,
        bounds=plotting_bounds,
        ax=ax,
        n_contour_levels=50,
        grid_width_n_points=200,
    )

    sample_idx = 0

    ax.plot(trajectory[:,sample_idx,0], trajectory[:,sample_idx,1])


if __name__ == "__main__":
    args = parse_args()

    sigma_max = args.sigma_max
    sigma_min = args.sigma_min
    num_mc_samples = args.num_mc_samples
    start_time = args.start_time
    prob_path = args.prob_path
    ENERGY_FUNCTION = args.energy_function

    num_steps = 1000
    end_time = 1.0

    device = get_device()

    if prob_path == 'OT':
        noise_schedule = OTcnfNoiseSchedule(sigma_min, sigma_max)
    elif prob_path == 'VE':
        noise_schedule = VEcnfNoiseSchedule(sigma_min, sigma_max)
    else:
        raise Exception("Invalid prob path")

    if ENERGY_FUNCTION == "GMM":
        energy_function = GMM(device=device)
    elif ENERGY_FUNCTION == "DW4":
        energy_function = MultiDoubleWellEnergy(
            8,
            4,
            "data/test_split_DW4.npy",
            "data/train_split_DW4.npy",
            "data/val_split_DW4.npy",
            device=device,
            plot_samples_epoch_period=1,
            data_normalization_factor=1.0,
            is_molecule=True,
        )
    else:
        raise Exception("Invalid energy function")

    print('[+] Information:')
    print(f'\t Energy_function: {ENERGY_FUNCTION}')
    print(f'\t prob_path: {prob_path}')
    print(f'\t sigma_max: {sigma_max}')
    print(f'\t sigma_min: {sigma_min}')
    print(f'\t num_mc_samples: {num_mc_samples}')
    print(f'\t start_time: {start_time}')

    x = sample_from_prior(sigma_max, dim=energy_function.dimensionality, device=device)
    time = torch.linspace(start_time, end_time, num_steps + 1, device=device)

    vecfield = make_vector_field(
        energy_function=energy_function,
        noise_schedule=noise_schedule,
        num_mc_samples=num_mc_samples,
        device=device,
        option=prob_path,
    )

    trajectory = sample_trajectory_by_ode_integration(vecfield, x, time)
    sample = trajectory[-1]

    w2 = wasserstein2(energy_function, sample)
    print(f'\tw2 distance: {w2}')

    if ENERGY_FUNCTION == "GMM":
        fig_title = 'Sampling by U_K integration\n' + f'W2={w2:0.3f}'
        fig, axs = draw_figure(energy_function, trajectory, fig_title)

        filename = get_filename(
            sigma_max, sigma_min, num_mc_samples, start_time, end_time, prob_path
        )

        fig.savefig(filename)
