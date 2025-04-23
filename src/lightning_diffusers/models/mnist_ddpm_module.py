"""
MNIST DDPM (Denoising Diffusion Probabilistic Model) implementation using PyTorch Lightning.

This module defines a Lightning Module that implements the DDPM algorithm for the MNIST dataset.
It uses the diffusers library for the UNet model and noise scheduler components.
"""

from typing import Callable, Iterable, Tuple, cast

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from diffusers.models import UNet2DModel
from diffusers.schedulers import DDPMScheduler

# Type definition for optimizer factory functions
OptimizerCallable = Callable[[Iterable], torch.optim.Optimizer]


class MnistDDPMModule(pl.LightningModule):
    """
    PyTorch Lightning module implementing DDPM for MNIST dataset.

    This module handles the training process for a Denoising Diffusion Probabilistic Model
    on the MNIST dataset, including forward pass, loss calculation, and optimizer configuration.
    """

    def __init__(
        self,
        unet: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        optimizer: OptimizerCallable,
    ) -> None:
        """
        Initialize the MNIST DDPM module.

        Args:
            unet: The U-Net model used for noise prediction
            noise_scheduler: The DDPM noise scheduler
            optimizer: A callable that returns an optimizer when given model parameters
        """
        super().__init__()
        self.save_hyperparameters()

        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net model.

        Args:
            x_noisy: Noisy input images
            t: Timestep for the diffusion process

        Returns:
            Predicted noise
        """
        return self.unet(x_noisy, t).sample

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step.

        This method:
        1. Extracts images from the batch
        2. Samples random timesteps
        3. Adds noise to the images
        4. Predicts the noise using the model
        5. Calculates the MSE loss between predicted and actual noise

        Args:
            batch: A tuple of (images, labels) where only images are used

        Returns:
            The calculated loss value
        """
        x, _ = batch  # Extract images, ignore labels
        bsz = x.shape[0]  # Batch size

        # Get the number of timesteps from the scheduler configuration
        num_timesteps = self.noise_scheduler.config.get("num_train_timesteps", None)
        assert num_timesteps is not None

        # Sample random timesteps for each image in the batch
        t = cast(
            torch.IntTensor,
            torch.randint(
                low=0,
                high=num_timesteps,
                size=(bsz,),
                device=x.device,
                dtype=torch.long,
            ),
        )

        # Sample random noise
        z = torch.randn_like(x)
        # Add noise to the input images according to the timesteps
        x_noisy = self.noise_scheduler.add_noise(x, z, t)
        # Predict the noise
        z_pred = self(x_noisy, t)
        # Calculate MSE loss between predicted and actual noise
        loss = F.mse_loss(z_pred, z)

        # Log the training loss
        self.log(name="train-loss", value=loss)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Returns:
            The configured optimizer
        """
        optimizer = self.optimizer(self.parameters())
        return optimizer
