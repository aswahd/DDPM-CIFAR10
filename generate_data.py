import os
from dataclasses import dataclass
from torchvision import transforms
import torch
from diffusers import DDPMScheduler
from custom_pipeline import CustomPipeline, CustomUNet2DConditionModel


@dataclass
class EvalConfig:
    image_size = 32  # the generated image resolution
    batch_size = 64  # how many images to sample during generation
    mixed_precision = 'fp32'  # `no` for float32, `fp16` for automatic mixed precision
    seed = 0


config = EvalConfig()
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
ckpt_path = 'cond-ddpm-cifar10-128/model_99.pt'
model.load_state_dict(torch.load(ckpt_path))
model = model.to(device)
model.eval()


save_dir = "outlier_data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


num_imgs_per_class = 1000
batch_size = config.batch_size

# Generate for each class
for i in range(10):
    save_to = os.path.join(save_dir, f'{i}')
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    y = torch.tensor([i] * config.batch_size, device=device)
    pipeline = CustomPipeline(unet=model, scheduler=noise_scheduler, class_label=y)

    counter = 0
    for _ in range(num_imgs_per_class // batch_size):
        images = pipeline(batch_size=config.batch_size).images

        for img in images:
            img.save(os.path.join(save_to, f'{counter}.png'))
            counter += 1
