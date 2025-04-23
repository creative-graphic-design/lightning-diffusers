import os

from diffusers.pipelines import DDPMPipeline
from diffusers.utils import make_image_grid
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

from lightning_diffusers.models import MnistDDPMModule


class MnistDDPMCallback(Callback):
    def __init__(
        self,
        num_generate_images: int = 16,
        num_grid_rows: int = 4,
        num_grid_cols: int = 4,
    ) -> None:
        super().__init__()
        self.num_generate_images = num_generate_images
        self.num_grid_rows = num_grid_rows
        self.num_grid_cols = num_grid_cols
        assert self.num_generate_images == self.num_grid_rows * self.num_grid_cols

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert isinstance(pl_module, MnistDDPMModule)

        pipe = DDPMPipeline(unet=pl_module.unet, scheduler=pl_module.noise_scheduler)
        pipe.set_progress_bar_config(leave=False, desc="Generating images")

        num_inference_steps = pl_module.noise_scheduler.config.num_train_timesteps

        output = pipe(
            num_inference_steps=num_inference_steps,
            batch_size=self.num_generate_images,
        )
        image = make_image_grid(
            images=output.images,
            rows=self.num_grid_rows,
            cols=self.num_grid_cols,
        )

        logger = trainer.logger
        assert isinstance(logger, WandbLogger)
        logger.log_image(key="generated", images=[image])
