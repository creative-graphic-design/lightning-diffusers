# Lightning Diffusers

🤗 Diffusers meets ⚡ PyTorch-Lightning: A simple and flexible training template for diffusion models.

## Overview

Lightning Diffusers combines the power of [Hugging Face Diffusers](https://github.com/huggingface/diffusers) with the organization and scalability of [PyTorch Lightning](https://lightning.ai/). This framework provides a clean, modular approach to training diffusion models with minimal boilerplate code.

## Features

- **Modular Configuration**: Separate configuration files for models, data, and training
- **Experiment Tracking**: Built-in integration with Weights & Biases
- **Extensible Architecture**: Easy to add new models, datasets, and callbacks
- **Reproducible Experiments**: Consistent training setup with configuration files

## Installation

```bash
pip install git+https://github.com/creative-graphic-design/lightning-diffusers
```

## Usage

### Training with a single configuration file

```shell
WANDB_API_KEY=xxxxxxx uv run lightning-diffusers fit \
    --config wandb/ddpm/config.yaml
```

### Training with separate configuration files

```shell
WANDB_API_KEY=xxxxxxx uv run lightning-diffusers fit \
    --data configs/data/mnist.yaml \
    --model configs/models/ddpm.yaml \
    --trainer configs/trainers/mnist_ddpm.yaml
```

## Project Structure

- `configs/`: Configuration files for models, data, and trainers
- `src/lightning_diffusers/`: Main package
  - `models/`: PyTorch Lightning modules for diffusion models
  - `callbacks/`: Custom callbacks for training and visualization
  - `cli.py`: Command-line interface

## Examples

Currently implemented examples:
- MNIST DDPM (Denoising Diffusion Probabilistic Model)

## License

This project is open source and available under the [Apache License 2.0](https://github.com/creative-graphic-design/lightning-diffusers/blob/main/LICENSE).
