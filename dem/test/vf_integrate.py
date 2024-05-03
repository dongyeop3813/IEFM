import argparse
import wandb

import torch
import matplotlib.pyplot as plt

from dem.models.components.noise_schedules import (
    OTcnfNoiseSchedule, VEcnfNoiseSchedule
)
from dem.models.components.vf_estimator import estimate_VF
from dem.models.components.optimal_transport import wasserstein
from dem.energies.gmm_energy import GMM
from dem.energies.multi_double_well_energy import MultiDoubleWellEnergy
from dem.energies.lennardjones_energy import LennardJonesEnergy

from fab.utils.plotting import plot_contours, plot_marginal_pair

from torchdiffeq import odeint


SAMPLE_SIZE = 1000
INTEGRATION_METHOD = 'dopri5'

FILTER_OUTLIER = True
THRESHOLD = -10


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


def get_filename(cfg):
    filename = './figure/VF_integration/' \
        f'{cfg.energy_function}-{cfg.prob_path}/sample'

    if cfg.sigma_max != 1.0:
        filename += f'-sgM={cfg.sigma_max:0.2f}'

    if cfg.sigma_min != 1e-5:
        filename += f'-sgm={cfg.sigma_min:0.5f}'

    if cfg.num_mc_samples != 1000:
        filename += f'-K={cfg.num_mc_samples}'

    if cfg.start_time != 0.01:
        filename += f'-s={cfg.start_time:0.3f}'

    if cfg.end_time != 1.0:
        filename += f'-e={cfg.end_time:0.2f}'

    filename += '.png'

    return filename


def wasserstein2(energy_function, sample):
    test_set = energy_function.sample_test_set(SAMPLE_SIZE)
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


def get_energy_function(cfg):
    if cfg.energy_function == "GMM":
        energy_function = GMM(device=cfg.device)
    elif cfg.energy_function == "DW4":
        energy_function = MultiDoubleWellEnergy(
            8,
            4,
            "data/test_split_DW4.npy",
            "data/train_split_DW4.npy",
            "data/val_split_DW4.npy",
            device=cfg.device,
            plot_samples_epoch_period=1,
            data_normalization_factor=1.0,
            is_molecule=True,
        )
    elif cfg.energy_function == "LJ13":
        energy_function = LennardJonesEnergy(
            39,
            13,
            "data/test_split_LJ13-1000.npy",
            "data/train_split_LJ13-1000.npy",
            "data/test_split_LJ13-1000.npy",
            device=cfg.device,
            plot_samples_epoch_period=1,
            data_normalization_factor=1.0,
            is_molecule=True,
        )
    else:
        raise Exception("Invalid energy function")

    return energy_function


def get_noise_schedule(cfg):
    if cfg.prob_path == 'OT':
        noise_schedule = OTcnfNoiseSchedule(cfg.sigma_min, cfg.sigma_max)
    elif cfg.prob_path == 'VE':
        noise_schedule = VEcnfNoiseSchedule(cfg.sigma_min, cfg.sigma_max)
    else:
        raise Exception("Invalid prob path")
    return noise_schedule


def filter_outlier(sample):
    # E(x) = -energy_function(x)
    sample_energy = -energy_function(sample)
    indices = sample_energy < THRESHOLD
    SAMPLE_SIZE = indices.sum()
    return sample[indices]


def draw_figure_and_save(cfg, energy_function, trajectory, w2=None):
    if cfg.energy_function == "GMM":
        fig_title = 'Sampling by U_K integration\n' + f'W2={w2:0.3f}'
        fig, _ = draw_figure(energy_function, trajectory, fig_title)
        fig.savefig(cfg.fig_filename)

    elif cfg.energy_function == "DW4":
        img = energy_function.get_dataset_fig(sample)
        img.save(cfg.fig_filename)

    elif cfg.energy_function == "LJ13":
        img = energy_function.get_dataset_fig(sample)
        img.save(cfg.fig_filename)


def sample_trajectory(cfg, energy_function, noise_schedule):
    device = cfg.device

    x = sample_from_prior(cfg.sigma_max, dim=energy_function.dimensionality, device=device)
    time = torch.linspace(cfg.start_time, cfg.end_time, cfg.num_steps + 1, device=device)

    vecfield = make_vector_field(
        energy_function=energy_function,
        noise_schedule=noise_schedule,
        num_mc_samples=cfg.num_mc_samples,
        device=device,
        option=cfg.prob_path,
    )

    return sample_trajectory_by_ode_integration(vecfield, x, time)


if __name__ == "__main__":
    cfg = parse_args()

    cfg.num_steps = 1000
    cfg.end_time = 1.0
    cfg.device = get_device()
    cfg.fig_filename = get_filename(cfg)

    wandb.init(
        project="efm_vf_integrate",
        config={
            'energy_function': cfg.energy_function,
            'num_mc_samples': cfg.num_mc_samples,
            'ODE_integration_start_time': cfg.start_time,
            'ODE_integration_steps': cfg.num_steps,
            'probability_path': cfg.prob_path,
            'sigma_max': cfg.sigma_max,
            'sigma_min': cfg.sigma_min,
            'sample_size': SAMPLE_SIZE,
            'fig_file': cfg.fig_filename,
        },
        tags=[cfg.energy_function, cfg.prob_path]
    )

    noise_schedule = get_noise_schedule(cfg)
    energy_function = get_energy_function(cfg)

    trajectory = sample_trajectory(
        cfg, 
        energy_function, 
        noise_schedule
    )
    sample = trajectory[-1]

    if FILTER_OUTLIER:
        sample = filter_outlier(sample)

    w2 = wasserstein2(energy_function, sample)
    wandb.log({'test/2-Wasserstein': w2})

    draw_figure_and_save(cfg, energy_function, trajectory, w2)

    wandb.finish()