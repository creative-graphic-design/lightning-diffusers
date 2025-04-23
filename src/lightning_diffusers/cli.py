"""
Command Line Interface (CLI) module for Lightning Diffusers.

This module provides an interface for training and evaluating models from the command line
using PyTorch Lightning's `LightningCLI`.
"""

from lightning.pytorch.cli import LightningCLI


def main() -> None:
    """
    Main entry point function.

    Initializes LightningCLI to parse command line arguments and run the model.

    Configuration:
        - subclass_mode_model=True: Automatically detect model subclasses
        - subclass_mode_data=True: Automatically detect data module subclasses
        - save_config_kwargs={"overwrite": True}: Overwrite configuration files
    """
    LightningCLI(
        subclass_mode_model=True,  # Automatically detect model subclasses
        subclass_mode_data=True,  # Automatically detect data module subclasses
        save_config_kwargs={"overwrite": True},  # Overwrite configuration files
    )
