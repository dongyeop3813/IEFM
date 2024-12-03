import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.clipper import Clipper
from dem.models.components.noise_schedules import BaseNoiseSchedule
from dem.models.components.score_estimator import estimate_grad_Rt


DEBUG = False
EPSILON = 1e-23
USE_SOFTMAX_TRICK = False


def VE_conditional_VF(
    perturbed: torch.Tensor,
    endpoint: torch.Tensor,
    time: torch.Tensor,
    noise_schedule: BaseNoiseSchedule,
):
    """
    VE conditioanl vector field.
    For given time points and origin, 
    evaluate CVF at given points (perturbed).

    Here K = sample size, D = dimension of data.
    
    :param perturbed: Points where CVF will be evaluated. (D dimension)
    :param endpoint: Points where transport ends. (K x D dimension)
    :param time: time step t. (scalar)

    :return: Evaluted (conditional) vector field (K x D dimension)
    """
    return noise_schedule.c * (perturbed - endpoint)


def OT_conditional_VF(
    perturbed: torch.Tensor, 
    endpoint: torch.Tensor,
    time: torch.Tensor,
    noise_schedule: BaseNoiseSchedule,
):
    """
    Optimal transport conditioanl vector field.
    For given time points and origin, 
    evaluate CVF at given points (perturbed).

    Here K = sample size, D = dimension of data.
    
    :param perturbed: Points where CVF will be evaluated. (D dimension)
    :param endpoint: Points where transport ends. (K x D dimension)
    :param time: time step t. (scalar)

    :return: Evaluted (conditional) vector field (K x D dimension)
    """

    coeff = 1 - noise_schedule.sigma_diff
    return (endpoint - coeff * perturbed) / (1 - coeff * time + EPSILON)


def sampling_from_OT_importance(t, x, sigma_t, device):
    return (
        (x + (torch.randn_like(x, device=device) * sigma_t)) 
        /
        (t.unsqueeze(1) + EPSILON)
    )


def sampling_from_VE_importance(t, x, sigma_t, device):
    return (
        x + torch.randn_like(x, device=device) * sigma_t
    )


def _VF_estimator(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    option: str = "OT",
    device="cpu",
) -> torch.Tensor:
    """
    Estimate marginal vector field with signle time, x input.
    Here D = dimension of data.

    :param t: timestep (scalar)
    :param x: data (D dimension)

    :return: Estimated vector field (D dimension)
    """

    # energy_function = -E(x)

    # choose probabilith path
    if option == "OT":
        conditional_VF = OT_conditional_VF
        sample_from_importance_dist = sampling_from_OT_importance

    elif option == "VE":
        conditional_VF = VE_conditional_VF
        sample_from_importance_dist = sampling_from_VE_importance

    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)

    sigma_t = noise_schedule.sigma(repeated_t).unsqueeze(1)

    # sample from importance distribution q(-;x) (K x D dimension)
    endpoint_candidates = sample_from_importance_dist(
        repeated_t,
        repeated_x,
        sigma_t,
        device=device,
    )

    # log (unnormalized) prob of each sample (dimension K)
    log_prob = energy_function(endpoint_candidates)

    if USE_SOFTMAX_TRICK:
        log_prob = log_prob - log_prob.max()

    # weights (dimension K)
    weights = torch.softmax(log_prob, dim=0)

    cvf = conditional_VF(x, endpoint_candidates, t, noise_schedule)

    if DEBUG:
        print((weights != 0))

        print(weights)

        print(f"endpoint has nan {endpoint_candidates.isnan().max()}")
        print(f"weights has nan {weights.isnan().max()}")
        print(f"cvf has nan {cvf.isnan().max()}")
        print(cvf)

    # marginal vector field estimator
    return torch.matmul(
        weights, 
        cvf
    )


def estimate_VF(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    option: str = "OT",
    device="cpu",
):
    """
        Estimate marginal vector field with batched time, x input.
    """
    if option == 'OT' or option == 'VE':
        # vectorizing map for batch processing
        vmmaped_fxn = torch.vmap(
            _VF_estimator, 
            in_dims=(0, 0, None, None, None, None), 
            randomness="different"
        )

        return vmmaped_fxn(
            t, x, energy_function, noise_schedule, num_mc_samples, 
            option, device=device
        )

    elif option == 'PFODE':
        sigma_t = noise_schedule.sigma(t).unsqueeze(1)
        sigma_t_prime = noise_schedule.sigma_prime(t).unsqueeze(1)

        weight = -sigma_t * sigma_t_prime
        scores = estimate_grad_Rt(
            t, x, energy_function, noise_schedule, num_mc_samples
        )

        return weight * scores


def estimate_VF_for_fixed_time(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    option: str = "OT",
    device="cpu",
):
    """
        Estimate marginal vector field with batched x and single time input.
    """

    if option == 'VE' or option == 'OT':
        vmmaped_fxn = torch.vmap(
            _VF_estimator,
            in_dims=(None, 0, None, None, None, None),
            randomness="different"
        )

        return vmmaped_fxn(
            t, x, energy_function, noise_schedule, num_mc_samples,
            option, device=device
        )

    elif option == 'PFODE':
        t = t.repeat_interleave(x.size(0))

        sigma_t = noise_schedule.sigma(t).unsqueeze(1)
        sigma_t_prime = noise_schedule.sigma_prime(t).unsqueeze(1)

        weight = -sigma_t * sigma_t_prime
        scores = estimate_grad_Rt(
            t, x, energy_function, noise_schedule, num_mc_samples
        )

        return weight * scores
