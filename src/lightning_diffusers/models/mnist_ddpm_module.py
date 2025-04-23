from typing import Callable, Iterable, Tuple, cast

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from diffusers.models import UNet2DModel
from diffusers.schedulers import DDPMScheduler

OptimizerCallable = Callable[[Iterable], torch.optim.Optimizer]


class MnistDDPMModule(pl.LightningModule):
    def __init__(
        self,
        unet: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        optimizer: OptimizerCallable,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(x_noisy, t).sample

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, _ = batch
        bsz = x.shape[0]

        num_timesteps = self.noise_scheduler.config.get("num_train_timesteps", None)
        assert num_timesteps is not None

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

        z = torch.randn_like(x)
        x_noisy = self.noise_scheduler.add_noise(x, z, t)
        z_pred = self(x_noisy, t)
        loss = F.mse_loss(z_pred, z)

        self.log(name="train-loss", value=loss)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.optimizer(self.parameters())
        return optimizer
