"""
Callbacks module for Lightning Diffusers.

This module exports PyTorch Lightning callbacks that provide additional functionality
during the training of diffusion models. Currently includes the MNIST DDPM callback
for generating and logging sample images during training.
"""

from .mnist_ddpm_callback import MnistDDPMCallback

__all__ = [
    "MnistDDPMCallback",
]
