"""
MNIST DDPM Callback module for image generation during training.

This module provides a PyTorch Lightning callback that generates sample images
at the end of each training epoch using the current state of the DDPM model
and logs them to Weights & Biases (wandb).
"""

from diffusers.pipelines import DDPMPipeline
from diffusers.utils import make_image_grid
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

from lightning_diffusers.models import MnistDDPMModule


class MnistDDPMCallback(Callback):
    """
    PyTorch Lightning callback for generating and logging MNIST images during DDPM training.

    This callback generates sample images at the end of each training epoch using
    the current state of the DDPM model and logs them to Weights & Biases (wandb).
    The images are arranged in a grid for better visualization.
    """

    def __init__(
        self,
        num_generate_images: int = 16,
        num_grid_rows: int = 4,
        num_grid_cols: int = 4,
    ) -> None:
        """
        Initialize the MNIST DDPM callback.

        Args:
            num_generate_images: Number of images to generate (default: 16)
            num_grid_rows: Number of rows in the image grid (default: 4)
            num_grid_cols: Number of columns in the image grid (default: 4)
        """
        super().__init__()
        self.num_generate_images = num_generate_images
        self.num_grid_rows = num_grid_rows
        self.num_grid_cols = num_grid_cols
        # Ensure the total number of images matches the grid dimensions
        assert self.num_generate_images == self.num_grid_rows * self.num_grid_cols

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Generate and log images at the end of each training epoch.

        This method:
        1. Creates a DDPM pipeline using the current model
        2. Generates sample images
        3. Arranges them in a grid
        4. Logs the grid to Weights & Biases

        Args:
            trainer: The PyTorch Lightning trainer instance
            pl_module: The Lightning module being trained (must be MnistDDPMModule)
        """
        # Ensure the module is of the expected type
        assert isinstance(pl_module, MnistDDPMModule)

        # Create a DDPM pipeline using the current model components
        pipe = DDPMPipeline(unet=pl_module.unet, scheduler=pl_module.noise_scheduler)
        pipe.set_progress_bar_config(leave=False, desc="Generating images")

        # Use the same number of inference steps as training timesteps
        num_inference_steps = pl_module.noise_scheduler.config.num_train_timesteps

        # Generate images using the pipeline
        output = pipe(
            num_inference_steps=num_inference_steps,
            batch_size=self.num_generate_images,
        )
        # Create a grid from the generated images
        image = make_image_grid(
            images=output.images,
            rows=self.num_grid_rows,
            cols=self.num_grid_cols,
        )

        # Log the image grid to Weights & Biases
        logger = trainer.logger
        assert isinstance(logger, WandbLogger)
        logger.log_image(key="generated", images=[image])
