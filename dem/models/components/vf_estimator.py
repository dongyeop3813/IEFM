import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.clipper import Clipper
from dem.models.components.noise_schedules import BaseNoiseSchedule


DEBUG = False


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
    c = torch.log(torch.tensor(noise_schedule.sigma_diff).to(perturbed))
    return c * (perturbed - endpoint)


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
    return (endpoint - coeff * perturbed) / (1 - coeff * time + 1e-23)


def sampling_from_OT_importance(t, x, sigma_t, device):
    return (
        (x + (torch.randn_like(x, device=device) * sigma_t.sqrt())) 
        /
        (t.unsqueeze(1) + 1e-23)
    )


def sampling_from_VE_importance(t, x, sigma_t, device):
    return (
        x + torch.randn_like(x, device=device) * sigma_t.sqrt()
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

    h_t = noise_schedule.h(repeated_t).unsqueeze(1)

    # sample from importance distribution q(-;x) (K x D dimension)
    endpoint_candidates = sample_from_importance_dist(
        repeated_t,
        repeated_x,
        h_t,
        device=device,
    )

    # log (unnormalized) prob of each sample (dimension K)
    log_prob = energy_function(endpoint_candidates)

    # weights (dimension K)
    weights = torch.softmax(log_prob, dim=0)

    cvf = conditional_VF(x, endpoint_candidates, t, noise_schedule)

    if DEBUG:
        print(endpoint_candidates)
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

    vmmaped_fxn = torch.vmap(
        _VF_estimator,
        in_dims=(None, 0, None, None, None, None),
        randomness="different"
    )

    return vmmaped_fxn(
        t, x, energy_function, noise_schedule, num_mc_samples,
        option, device=device
    )
