# On the Redundancy of Timestep Embeddings in Diffusion Models
## Abstract

Diffusion models rely heavily on explicit timestep embeddings to modulate the denoising process across various noise scales. In this work, we challenge the necessity of these temporal signals by analyzing their impact on U-Net and Diffusion Transformer architectures. Beyond empirical evidence, we provide a theoretical framework demonstrating that, under certain conditions, the global minimizer of the diffusion training objective can be achieved without explicit timestep conditioning. Our findings reveal a surprising robustness when timestep embeddings are completely removed. Extensive ablation studies on the CelebA and CIFAR-10 datasets show that these time-agnostic models can maintain high structural fidelity and even surpass their conditioned counterparts in competitive metrics, including FID, precision, and recall. Our analysis suggests these architectures can implicitly infer noise scales from the corrupted input under specific assumptions, rendering explicit temporal conditioning redundant. This study challenges long-standing temporal conditioning paradigms and paves the way for more efficient and structurally focused generative architectures.

## Overview

Implementation of [On the Redundancy of Timestep Embeddings in Diffusion Models](https://arxiv.org/abs/2606.20416) in PyTorch. The repository provides a modular framework for training and evaluating diffusion models with and without timestep embeddings, supporting both U-Net and DiT architectures.

The codebase implements:

- Diffusion training and sampling logic in [ddpm.py](ddpm.py)
- A DiT-style Transformer backbone in [dit.py](dit.py)
- U-Net-based diffusion blocks in [ddpm.py](ddpm.py)
- Dataset loaders and utility helpers in [utils.py](utils.py)
- Training scripts for CIFAR-10, CelebA, and Tiny ImageNet in [train_cifar10.py](train_cifar10.py) and [train_celeba.py](train_celeba.py)
- Sampling utilities in [sample.py](sample.py) and [sample_cifar10.py](sample_cifar10.py)

The repository also includes evaluation and metric-related scripts for FID, Inception Score, precision/recall, and FLOPs analysis.

## Repository Structure

- [ddpm.py](ddpm.py): DDPM / DDIM diffusion training and sampling implementation
- [dit.py](dit.py): DiT architecture, timestep embedding modules, and model configurations
- [modules.py](modules.py): reusable model components
- [train_cifar10.py](train_cifar10.py): training entrypoint for CIFAR-10 experiments
- [train_celeba.py](train_celeba.py): training entrypoint for CelebA experiments
- [sample.py](sample.py): sampling script for the main models
- [sample_cifar10.py](sample_cifar10.py): sampling script for CIFAR-10
- [utils.py](utils.py): image saving, data helpers, and utilities
- [fid_score.py](fid_score.py), [inception_score.py](inception_score.py), [precision_recall/](precision_recall/): evaluation metrics

## Installation

This project requires Python 3.9+ and PyTorch with CUDA support (recommended). Install the main dependencies with:

```bash
pip install torch torchvision timm diffusers torchsummary calflops tqdm numpy pillow
```

## Quick Start

### Train CIFAR-10

```bash
python train_cifar10.py --state_num 1 --batch_size 128 --epochs 1000
```

### Train CelebA

```bash
python train_celeba.py --state_num 1 --batch_size 128 --epochs 1000
```

### Sample images from CIFAR-10

```bash
python sample_cifar10.py
```

### Sample images from CelebA

```bash
python sample.py
```

## Experiment Notes

The repository includes multiple experimental settings, including:

- DiT default configuration
- DiT with modified timestep conditioning
- U-Net default configuration
- U-Net with modified timestep conditioning

These experiments are organized through the `state_num` argument and the corresponding result directories under ```results/```.

## Notes

- Training logs are written to `train.log` by default.
- Checkpoints are saved under the experiment directory inside ```results/```.
- The implementation is intended for research experiments and is not yet packaged as a production-ready library.

## Citation

If you use this repository in your work, please cite the paper:

```bibtex
@misc{chávez2026redundancytimestepembeddingsdiffusion,
      title={On the Redundancy of Timestep Embeddings in Diffusion Models}, 
      author={José A. Chávez},
      year={2026},
      eprint={2606.20416},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.20416}, 
}
```
