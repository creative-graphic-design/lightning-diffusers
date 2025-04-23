# lightning-diffusers: 🤗 Diffusers meets ⚡ Lightning

```shell
WANDB_API_KEY=xxxxxxx uv run lightning-diffusers fit \
    --config wandb/ddpm/config.yaml
````

```shell
WANDB_API_KEY=xxxxxxx uv run lightning-diffusers fit \
    --data configs/data/mnist.yaml \
    --model configs/models/ddpm.yaml \
    --optimizer configs/optimizers/adamw.yaml \
    --trainer configs/trainers/mnist_ddpm.yaml
```
