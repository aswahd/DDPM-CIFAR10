import os
import random
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torchvision import transforms
import torch
from diffusers import DDPMScheduler
from custom_pipeline import CustomPipeline, CustomUNet2DConditionModel


@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    seed = 0


config = TrainingConfig()

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CustomUNet2DConditionModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    ),
    num_class_embeds=10
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Load model
ckpt_path = 'cond-ddpm-cifar10-128/model_59.pt'
model.load_state_dict(torch.load(ckpt_path))
model = model.to(device)

save_dir = "samples"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

vis_every = 100
y = random.randint(0, 10)
save_as = f"vis_sampling_{y}.svg"
# Generate steps for the selected class
y = torch.tensor([y], device=device)
pipeline = CustomPipeline(unet=model, scheduler=noise_scheduler, class_label=y)


@torch.no_grad()
def forward(
        batch_size: int = 1,
        generator: torch.Generator = None,
        num_inference_steps: int = 1000,
):
    r"""
    Args:
        batch_size (`int`, *optional*, defaults to 1):
            The number of images to generate.
        generator (`torch.Generator`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        num_inference_steps (`int`, *optional*, defaults to 1000):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

    Returns:
        [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
        `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
        generated images.
    """
    # Sample gaussian noise to begin loop
    if isinstance(model.sample_size, int):
        image_shape = (batch_size, model.in_channels, model.sample_size, model.sample_size)
    else:
        image_shape = (batch_size, model.in_channels, *model.sample_size)

    image = torch.randn(image_shape, generator=generator, device=device)

    out = (image / 2 + 0.5).clamp(0, 1)
    out = out.cpu().permute(0, 2, 3, 1).numpy()[0]
    out = (out * 255).round().astype("uint8")
    out = Image.fromarray(out)

    samples_all_steps = [out]
    # set step values
    noise_scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(noise_scheduler.timesteps):
        # 1. predict noise model_output
        model_output = model(image, t).sample

        # 2. compute previous image: x_t -> x_t-1
        image = noise_scheduler.step(model_output, t, image, generator=generator).prev_sample

        if (t + 1) % vis_every == 0:
            out = (image / 2 + 0.5).clamp(0, 1)
            out = out.cpu().permute(0, 2, 3, 1).numpy()[0]
            out = (out * 255).round().astype("uint8")
            out = Image.fromarray(out)
            samples_all_steps.append(out)
    return samples_all_steps


images = forward(batch_size=1, num_inference_steps=1000)
fig, axes = plt.subplots(1, 10, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.axis("off")
    ax.set_xlabel(f"Step {i}")
    ax.imshow(images[i])
plt.savefig(os.path.join(save_dir, save_as), format='svg')
