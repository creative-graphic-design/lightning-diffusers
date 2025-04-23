from lightning.pytorch.cli import LightningCLI


def main() -> None:
    LightningCLI(
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )
