"""
Models module for Lightning Diffusers.

This module exports PyTorch Lightning modules that implement various diffusion models.
Currently includes the MNIST DDPM (Denoising Diffusion Probabilistic Model) implementation.
"""

from .mnist_ddpm_module import MnistDDPMModule

__all__ = [
    "MnistDDPMModule",
]
