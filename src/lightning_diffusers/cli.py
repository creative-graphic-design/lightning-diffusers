from lightning.pytorch.cli import LightningCLI

from lightning_diffusers.data import MnistDataModule
from lightning_diffusers.models import MnistDDPMModule


def main() -> None:
    LightningCLI(
        model_class=MnistDDPMModule,
        datamodule_class=MnistDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_configure_optimizers=True,
        save_config_kwargs={"overwrite": True},
    )
