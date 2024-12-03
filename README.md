<div align="center">

# Iterated Energy-based Flow Matching for Sampling from Boltzmann Densities

[![Preprint](http://img.shields.io/badge/paper-arxiv.2402.06121-B31B1B.svg)](https://arxiv.org/abs/2408.16249)
</div>

Our code is inspired by [iterated denoising energy matching](https://github.com/jarridrb/DEM).

## Installation

For installation, we recommend the use of Micromamba. Please refer [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for an installation guide for Micromamba.
First, we install dependencies

```bash
# create micromamba environment
micromamba create -f environment.yaml
micromamba activate dem

# install requirements
pip install -r requirements.txt

```

Note that the hydra configs interpolate using some environment variables set in the file `.env`. We provide
an example `.env.example` file for convenience. Note that to use wandb we require that you set WANDB_ENTITY in your
`.env` file.

To run an experiment, e.g., GMM with iEFM, you can run on the command line

```bash
python dem/train.py experiment=gmm_iefm_unnormalized
```
