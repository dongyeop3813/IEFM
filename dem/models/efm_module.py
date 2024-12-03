import time
from functools import partial
from typing import Any, Dict, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from hydra.utils import get_original_cwd
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

from torchmetrics import MeanMetric

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.data_utils import remove_mean
from dem.energies.gmm_energy import plot_vecfield_error

from .components.clipper import Clipper
from .components.cnf import CNF
from .components.distribution_distances import compute_distribution_distances
from .components.ema import EMAWrapper
from .components.lambda_weighter import BaseLambdaWeighter
from .components.noise_schedules import BaseNoiseSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.scaling_wrapper import ScalingWrapper
from .components.vf_estimator import estimate_VF
from .components.score_scaler import BaseScoreScaler


def t_stratified_loss(batch_t, batch_loss, num_bins=5, loss_name=None):
    """Stratify loss by binning t."""
    flat_losses = batch_loss.flatten().detach().cpu().numpy()
    flat_t = batch_t.flatten().detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


def get_wandb_logger(loggers):
    """Gets the wandb logger if it is the list of loggers otherwise returns None."""
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    return wandb_logger


class EFMLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        lambda_weighter: BaseLambdaWeighter,
        buffer: PrioritisedReplayBuffer,
        num_init_samples: int,
        num_estimator_mc_samples: int,
        num_samples_to_generate_per_epoch: int,
        num_samples_to_sample_from_buffer: int,
        num_samples_to_save: int,
        eval_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        nll_with_cfm: bool,
        nll_with_dem: bool,
        nll_on_buffer: bool,
        logz_with_cfm: bool,
        cfm_sigma: float,
        cfm_prior_std: float,
        use_otcfm: bool,
        nll_integration_method: str,
        compile: bool,
        prioritize_cfm_training_samples: bool = False,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
        clipper: Optional[Clipper] = None,
        score_scaler: Optional[BaseScoreScaler] = None,
        partial_prior=None,
        diffusion_scale=1.0,
        cfm_loss_weight=1.0,
        use_ema=False,
        use_exact_likelihood=False,
        debug_use_train_data=False,
        init_from_prior=False,
        compute_nll_on_train_data=False,
        use_buffer=True,
        tol=1e-5,
        version=1,
        prob_path="OT",
        ode_start_time=0.00,
    ) -> None:
        """Initialize a `EFMLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param buffer: Buffer of sampled objects
        """
        super().__init__()
        # Seems to slow things down
        # torch.set_float32_matmul_precision('high')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net(energy_function=energy_function)
        self.cfm_net = net(energy_function=energy_function)

        self.efm_cnf = CNF(
            self.net,
            is_diffusion=False,
            use_exact_likelihood=use_exact_likelihood,
            method=nll_integration_method,
            num_steps=num_integration_steps,
            atol=tol,
            rtol=tol,
            time_at_prior=ode_start_time,
            time_at_data=1.0,
        )

        self.cfm_cnf = CNF(
            self.cfm_net,
            is_diffusion=False,
            use_exact_likelihood=use_exact_likelihood,
            method=nll_integration_method,
            num_steps=num_integration_steps,
            atol=tol,
            rtol=tol,
        )

        self.prob_path = prob_path

        self.energy_function = energy_function
        self.noise_schedule = noise_schedule
        self.buffer = buffer
        self.dim = self.energy_function.dimensionality
        self.ode_start_time = ode_start_time

        self.clipper = clipper

        self.num_init_samples = num_init_samples
        self.num_estimator_mc_samples = num_estimator_mc_samples
        self.num_samples_to_generate_per_epoch = num_samples_to_generate_per_epoch
        self.num_samples_to_sample_from_buffer = num_samples_to_sample_from_buffer
        self.num_integration_steps = num_integration_steps
        self.num_samples_to_save = num_samples_to_save

        self.lambda_weighter = self.hparams.lambda_weighter(self.noise_schedule)

        self.partial_prior = partial_prior
        self.init_from_prior = init_from_prior

        # Evaluation option
        self.eval_batch_size = eval_batch_size

        self.should_eval_nll_with_cfm = nll_with_cfm
        self.should_eval_logz_with_cfm = logz_with_cfm

        self.should_eval_nll_with_efm = nll_with_dem

        self.should_eval_nll_on_buffer = nll_on_buffer
        self.should_compute_nll_on_train_data = compute_nll_on_train_data

        # If we should train cfm, set hyperparameters of cfm.
        self.should_train_cfm = (
            self.should_eval_nll_with_cfm or self.hparams.debug_use_train_data
        )
        if self.should_train_cfm:
            self.cfm_prior_std = cfm_prior_std
            flow_matcher = ConditionalFlowMatcher
            if use_otcfm:
                flow_matcher = ExactOptimalTransportConditionalFlowMatcher

            self.cfm_sigma = cfm_sigma
            self.conditional_flow_matcher = flow_matcher(sigma=cfm_sigma)

            self.prioritize_cfm_training_samples = prioritize_cfm_training_samples

        # Loss metric
        self.dem_train_loss = MeanMetric()
        self.cfm_train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Validation metric with cfm_net
        self.val_nll_logdetjac = MeanMetric()
        self.val_nll_log_p_1 = MeanMetric()
        self.val_nll = MeanMetric()
        self.val_nfe = MeanMetric()
        self.val_energy_w2 = MeanMetric()
        self.val_dist_w2 = MeanMetric()
        self.val_dist_total_var = MeanMetric()

        # Validation metric with efm_net
        self.val_dem_nll_logdetjac = MeanMetric()
        self.val_dem_nll_log_p_1 = MeanMetric()
        self.val_dem_nll = MeanMetric()
        self.val_dem_nfe = MeanMetric()
        self.val_dem_logz = MeanMetric()

        self.val_logz = MeanMetric()

        # Metric on buffer
        self.val_buffer_nll_logdetjac = MeanMetric()
        self.val_buffer_nll_log_p_1 = MeanMetric()
        self.val_buffer_nll = MeanMetric()
        self.val_buffer_nfe = MeanMetric()
        self.val_buffer_logz = MeanMetric()

        self.val_train_nll_logdetjac = MeanMetric()
        self.val_train_nll_log_p_1 = MeanMetric()
        self.val_train_nll = MeanMetric()
        self.val_train_nfe = MeanMetric()
        self.val_train_logz = MeanMetric()

        # Test metric with cfm_net
        self.test_nll_logdetjac = MeanMetric()
        self.test_nll_log_p_1 = MeanMetric()
        self.test_nll = MeanMetric()
        self.test_nfe = MeanMetric()

        # Test metric with efm_net
        self.test_dem_nll_logdetjac = MeanMetric()
        self.test_dem_nll_log_p_1 = MeanMetric()
        self.test_dem_nll = MeanMetric()
        self.test_dem_nfe = MeanMetric()
        self.test_dem_logz = MeanMetric()
        self.test_logz = MeanMetric()

        # Metric on buffer
        self.test_buffer_nll_logdetjac = MeanMetric()
        self.test_buffer_nll_log_p_1 = MeanMetric()
        self.test_buffer_nll = MeanMetric()
        self.test_buffer_nfe = MeanMetric()
        self.test_buffer_logz = MeanMetric()

        self.test_train_nll_logdetjac = MeanMetric()
        self.test_train_nll_log_p_1 = MeanMetric()
        self.test_train_nll = MeanMetric()
        self.test_train_nfe = MeanMetric()
        self.test_train_logz = MeanMetric()

        # Initialize temp variables
        self.last_samples = None
        self.last_energies = None
        self.eval_step_outputs = []

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x)

    def get_cfm_loss(self, samples: torch.Tensor) -> torch.Tensor:
        x0 = self.cfm_prior.sample(self.num_samples_to_sample_from_buffer)
        x1 = samples
        x1 = self.energy_function.unnormalize(x1)

        t, xt, ut = self.conditional_flow_matcher.sample_location_and_conditional_flow(
            x0, x1
        )

        if self.energy_function.is_molecule and self.cfm_sigma != 0:
            xt = remove_mean(
                xt, self.energy_function.n_particles, self.energy_function.n_spatial_dim
            )

        vt = self.cfm_net(t, xt)
        loss = (vt - ut).pow(2).mean(dim=-1)

        return loss

    def get_pointwise_loss(
        self, times: torch.Tensor, samples: torch.Tensor
    ) -> torch.Tensor:
        estimated_VF = estimate_VF(
            times,
            samples,
            self.energy_function,
            self.noise_schedule,
            num_mc_samples=self.num_estimator_mc_samples,
            device=self.device,
            option=self.prob_path,
        )

        if self.clipper is not None and self.clipper.should_clip_scores:
            if self.energy_function.is_molecule:
                estimated_VF = estimated_VF.reshape(
                    -1,
                    self.energy_function.n_particles,
                    self.energy_function.n_spatial_dim,
                )

            estimated_VF = self.clipper.clip_scores(estimated_VF)

            if self.energy_function.is_molecule:
                estimated_VF = estimated_VF.reshape(
                    -1, self.energy_function.dimensionality
                )

        predicted_VF = self.forward(times, samples)

        error_norms = (predicted_VF - estimated_VF).pow(2).mean(-1)

        return self.lambda_weighter(times) * error_norms

    def sample_time(self, num_samples):
        """
        Sample time points from uniform distribution on [s, e].
        """
        s = self.ode_start_time
        t = (1 - s) * torch.rand((num_samples,), device=self.device) + s
        return t

    def add_noise_to(self, origins, times):
        """
        Perturb given data points with conditional t|1.
        Here, times and origins are batch.
        """

        sigma_t = self.noise_schedule.h(times).sqrt().unsqueeze(-1)

        if self.prob_path == "OT":
            noisy_data = times.unsqueeze(1) * origins + (
                torch.randn_like(origins) * sigma_t
            )
        elif self.prob_path == "VE":
            noisy_data = origins + (torch.randn_like(origins) * sigma_t)
        elif self.prob_path == "PFODE":
            noisy_data = origins + (torch.randn_like(origins) * sigma_t)
        else:
            raise Exception("Invalid probability path")

        if self.energy_function.is_molecule:
            noisy_data = remove_mean(
                noisy_data,
                self.energy_function.n_particles,
                self.energy_function.n_spatial_dim,
            )

        return noisy_data

    def get_EFM_loss(
        self,
        sample_from_target: torch.Tensor,
    ) -> torch.Tensor:

        num_samples = sample_from_target.size(0)

        times = self.sample_time(num_samples=num_samples)

        perturbed_samples = self.add_noise_to(
            sample_from_target,
            times=times,
        )

        loss = self.get_pointwise_loss(times, perturbed_samples)

        return loss, times, perturbed_samples

    def training_step(self, batch, batch_idx):
        loss = 0.0
        if not self.hparams.debug_use_train_data:
            if self.hparams.use_buffer:
                iter_samples, _, _ = self.buffer.sample(
                    self.num_samples_to_sample_from_buffer
                )
            else:
                iter_samples = self.prior.sample(self.num_samples_to_sample_from_buffer)

            efm_loss, times, _ = self.get_EFM_loss(iter_samples)

            self.log_dict(
                t_stratified_loss(
                    times, efm_loss, loss_name="train/stratified/dem_loss"
                )
            )

            efm_loss = efm_loss.mean()
            loss = loss + efm_loss

            # update and log metrics
            self.dem_train_loss(efm_loss)
            self.log(
                "train/dem_loss",
                self.dem_train_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.should_train_cfm:
            if self.hparams.debug_use_train_data:
                cfm_samples = self.energy_function.sample_train_set(
                    self.num_samples_to_sample_from_buffer
                )
                times = torch.rand(
                    (self.num_samples_to_sample_from_buffer,), device=cfm_samples.device
                )
            else:
                cfm_samples, _, _ = self.buffer.sample(
                    self.num_samples_to_sample_from_buffer,
                    prioritize=self.prioritize_cfm_training_samples,
                )

            cfm_loss = self.get_cfm_loss(cfm_samples)
            self.log_dict(
                t_stratified_loss(
                    times, cfm_loss, loss_name="train/stratified/cfm_loss"
                )
            )
            cfm_loss = cfm_loss.mean()
            self.cfm_train_loss(cfm_loss)
            self.log(
                "train/cfm_loss",
                self.cfm_train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

            loss = loss + self.hparams.cfm_loss_weight * cfm_loss
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)

    def generate_samples(
        self,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_samples_to_generate_per_epoch

        noise = self.prior.sample(num_samples)

        return self.integrate(
            samples=noise,
            return_full_trajectory=return_full_trajectory,
        )

    @torch.no_grad()
    def integrate(
        self,
        samples: torch.Tensor = None,
        return_full_trajectory=False,
    ) -> torch.Tensor:

        trajectory = self.efm_cnf.to(self.device).generate(samples).detach()

        if return_full_trajectory:
            return trajectory

        return trajectory[-1]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        self.last_samples = self.generate_samples()
        self.last_energies = self.energy_function(self.last_samples)

        # Insert sample into the buffer
        self.buffer.add(self.last_samples, self.last_energies)

        # Compute and log energy_w2, dist_w2 and total_var.
        self.compute_and_log_energy_w2(stage="val")

        if self.energy_function.is_molecule:
            self.compute_and_log_dist_w2(stage="val")
            self.compute_and_log_dist_total_var(stage="val")

    def compute_nll(
        self,
        cnf,
        prior,
        samples: torch.Tensor,
    ):
        aug_samples = torch.cat(
            [samples, torch.zeros(samples.shape[0], 1, device=samples.device)], dim=-1
        )
        aug_output = cnf.integrate(aug_samples).detach()[-1]
        x_0, logdetjac = aug_output[..., :-1], aug_output[..., -1]
        log_p_0 = prior.log_prob(x_0)
        log_p_1 = log_p_0 + logdetjac
        nll = -log_p_1
        return nll, x_0, logdetjac, log_p_0

    def compute_and_log_energy_w2(self, stage):
        ground_truth_sample = self.generate_ground_truth_sample(
            stage, self.eval_batch_size
        )
        true_energies = self.energy_function(
            self.energy_function.normalize(ground_truth_sample)
        )

        # Generate sample and get energy
        if stage == "test":
            generated_samples = self.generate_samples(num_samples=self.eval_batch_size)
            generated_energies = self.energy_function(generated_samples)
        elif stage == "val":
            if len(self.buffer) < self.eval_batch_size:
                return
            _, generated_energies = self.buffer.get_last_n_inserted(
                self.eval_batch_size
            )
        else:
            raise Exception("Invalid stage")

        # Calculate w2 on energy distribution
        energy_w2 = pot.emd2_1d(
            true_energies.cpu().numpy(), generated_energies.cpu().numpy()
        )

        self.log(
            f"{stage}/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def compute_and_log_dist_w2(self, stage="val"):
        data_set = self.generate_ground_truth_sample(stage, self.eval_batch_size)

        if stage == "test":
            generated_samples = self.generate_samples(num_samples=self.eval_batch_size)
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

        dist_w2 = pot.emd2_1d(
            self.energy_function.interatomic_dist(generated_samples)
            .cpu()
            .numpy()
            .reshape(-1),
            self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1),
        )

        self.log(
            f"{stage}/dist_w2",
            self.val_dist_w2(dist_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def compute_and_log_dist_total_var(self, stage="val"):
        data_set = self.generate_ground_truth_sample(stage, self.eval_batch_size)

        if stage == "test":
            generated_samples = self.generate_samples(num_samples=self.eval_batch_size)
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

        generated_samples_dists = (
            self.energy_function.interatomic_dist(generated_samples)
            .cpu()
            .numpy()
            .reshape(-1),
        )
        data_set_dists = (
            self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1)
        )

        H_data_set, x_data_set = np.histogram(data_set_dists, bins=200)
        H_generated_samples, _ = np.histogram(
            generated_samples_dists, bins=(x_data_set)
        )
        total_var = (
            0.5
            * np.abs(
                H_data_set / H_data_set.sum()
                - H_generated_samples / H_generated_samples.sum()
            ).sum()
        )

        self.log(
            f"{stage}/dist_total_var",
            self.val_dist_total_var(total_var),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def compute_log_z(self, cnf, prior, samples, prefix, name):
        nll, _, _, _ = self.compute_nll(cnf, prior, samples)
        # energy function will unnormalize the samples itself
        logz = self.energy_function(self.energy_function.normalize(samples)) + nll
        logz_metric = getattr(self, f"{prefix}_{name}logz")
        logz_metric.update(logz)
        self.log(
            f"{prefix}/{name}logz",
            logz_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def compute_and_log_nll(self, cnf, prior, samples, stage, name):
        # Choose proper metric functions (for given stage and model)
        nfe_metric = getattr(self, f"{stage}_{name}nfe")
        nll_metric = getattr(self, f"{stage}_{name}nll")
        logdetjac_metric = getattr(self, f"{stage}_{name}nll_logdetjac")
        log_p_1_metric = getattr(self, f"{stage}_{name}nll_log_p_1")

        cnf.nfe = 0.0

        # Compute NLL
        nll, forwards_samples, logdetjac, log_p_1 = self.compute_nll(
            cnf, prior, samples
        )

        # Log NLL
        nfe_metric.update(cnf.nfe)
        nll_metric.update(nll)
        logdetjac_metric.update(logdetjac)
        log_p_1_metric.update(log_p_1)

        self.log_dict(
            {
                f"{stage}/{name}_nfe": nfe_metric,
                f"{stage}/{name}nll_logdetjac": logdetjac_metric,
                f"{stage}/{name}nll_log_p_1": log_p_1_metric,
            },
            on_epoch=True,
        )

        self.log(
            f"{stage}/{name}nll",
            nll_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def generate_ground_truth_sample(self, stage: str, size: int) -> torch.Tensor:
        if stage == "val":
            ground_truth_sample = self.energy_function.sample_val_set(size)
        elif stage == "test":
            ground_truth_sample = self.energy_function.sample_test_set(size)
        else:
            raise Exception("Invalid stage")

        return ground_truth_sample

    def get_sample_for_evaluation(self, size: int) -> torch.Tensor:
        # Evaluation on lastly generated sample
        sample = self.last_samples

        # Generate samples if needed
        if sample is None or self.eval_batch_size > len(sample):
            sample = self.generate_samples(self.eval_batch_size)

        # Discard some sample to match the size if too much
        elif len(sample) > self.eval_batch_size:
            indices = torch.randperm(len(sample))[: self.eval_batch_size]
            sample = sample[indices]

        return sample

    def eval_nll_with_efm(self, stage, ground_truth_sample):
        ground_truth_sample = self.energy_function.normalize(ground_truth_sample)

        self.compute_and_log_nll(
            self.efm_cnf, self.prior, ground_truth_sample, stage, "dem_"
        )

        self.compute_log_z(self.efm_cnf, self.prior, ground_truth_sample, stage, "dem_")

    def eval_nll_with_cfm(self, stage, sample):
        self.compute_and_log_nll(self.cfm_cnf, self.cfm_prior, sample, stage, "")

    def eval_logz_with_cfm(self, stage):
        backwards_samples = self.cfm_cnf.generate(
            self.cfm_prior.sample(self.eval_batch_size),
        )[-1]

        self.compute_log_z(self.cfm_cnf, self.cfm_prior, backwards_samples, stage, "")

    def eval_step(self, stage: str) -> None:
        """
        Perform a single eval step.
        Compare ground_truth_sample and generated sample
        and log sample quality metric
        :param stage: test or val
        """
        loss_fn = getattr(self, f"{stage}_loss")

        ground_truth_sample = self.generate_ground_truth_sample(
            stage, self.eval_batch_size
        )

        generated_sample = self.get_sample_for_evaluation(self.eval_batch_size)

        # Calculate and log loss
        loss, _, _ = self.get_EFM_loss(ground_truth_sample)
        loss_fn(loss.mean(-1))
        self.log(f"{stage}/loss", loss_fn, on_step=True, on_epoch=True, prog_bar=True)

        if self.should_eval_nll_with_efm:
            self.eval_nll_with_efm(stage, ground_truth_sample)

        if self.should_eval_nll_with_cfm:
            self.eval_nll_with_cfm(stage, ground_truth_sample)

            # compute nll on buffer if not training cfm only
            if not self.hparams.debug_use_train_data and self.should_eval_nll_on_buffer:
                sample_from_buffer, _, _ = self.buffer.sample(self.eval_batch_size)
                self.eval_nll_with_cfm(stage, sample_from_buffer)

            if self.should_compute_nll_on_train_data:
                sample_from_train_set = self.energy_function.sample_train_set(
                    self.eval_batch_size
                )
                self.eval_nll_with_cfm(stage, sample_from_train_set)

        if self.should_eval_logz_with_cfm:
            self.eval_logz_with_cfm(stage)

        self.eval_step_outputs.append(
            {
                "data": ground_truth_sample,
                "gen": generated_sample,
            }
        )

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("test")

    def log_with_energy_function(self, stage: str):
        wandb_logger = get_wandb_logger(self.loggers)

        if self.should_eval_nll_with_cfm:
            buffer_samples, _, _ = self.buffer.sample(
                self.eval_batch_size,
            )

            cfm_samples = self.cfm_cnf.generate(
                self.cfm_prior.sample(self.eval_batch_size),
            )[-1]
        else:
            buffer_samples, cfm_samples = None, None

        self.energy_function.log_on_epoch_end(
            self.last_samples,
            self.last_energies,
            wandb_logger,
            unprioritized_buffer_samples=buffer_samples,
            cfm_samples=cfm_samples,
            replay_buffer=self.buffer,
        )

    def compute_and_log_distribution_distances(
        self,
        stage,
        ground_truth_sample,
        generated_sample,
    ) -> None:

        generated_sample = self.energy_function.unnormalize(generated_sample)[:, None]
        ground_truth_sample = ground_truth_sample[:, None]

        names, dists = compute_distribution_distances(
            generated_sample,
            ground_truth_sample,
            self.energy_function,
        )

        names = [f"{stage}/{name}" for name in names]
        d = dict(zip(names, dists))
        self.log_dict(d, sync_dist=True)

    def eval_epoch_end(self, stage: str):
        self.log_with_energy_function(stage)

        # convert to dict of tensors assumes [batch, ...]
        outputs = {
            name: torch.cat([output[name] for output in self.eval_step_outputs], dim=0)
            for name in ["data", "gen"]
        }

        if "data" in outputs:
            self.compute_and_log_distribution_distances(
                stage,
                outputs["data"],
                outputs["gen"],
            )

        self.eval_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)

        self.eval_epoch_end("test")

        batch_size = 1000
        final_samples = []
        n_batches = self.num_samples_to_save // batch_size
        print("Generating samples")
        for i in range(n_batches):
            start = time.time()
            samples = self.generate_samples(num_samples=batch_size)
            final_samples.append(samples)
            end = time.time()
            print(f"batch {i} took {end - start:0.2f}s")

            if i == 0:
                self.energy_function.log_on_epoch_end(
                    samples,
                    self.energy_function(samples),
                    wandb_logger,
                )

        final_samples = torch.cat(final_samples, dim=0)
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        path = f"{output_dir}/samples_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path)
        print(f"Saving samples to {path}")
        import os

        os.makedirs(self.energy_function.name, exist_ok=True)
        path2 = f"{self.energy_function.name}/samples_{self.hparams.version}_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path2)
        print(f"Saving samples to {path2}")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        self.prior = self.partial_prior(
            device=self.device, scale=self.noise_schedule.h(0) ** 0.5
        )

        # Buffer initialization
        if self.init_from_prior:
            init_states = self.prior.sample(self.num_init_samples)
        else:
            init_states = self.generate_samples(self.num_init_samples)
        init_energies = self.energy_function(init_states)

        self.buffer.add(init_states, init_energies)

        if self.should_eval_nll_with_cfm:
            self.cfm_prior = self.partial_prior(
                device=self.device, scale=self.cfm_prior_std
            )

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.cfm_net = torch.compile(self.cfm_net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.hparams.lr_scheduler_update_frequency,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = EFMLitModule(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
