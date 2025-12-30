from dataclasses import dataclass
# from datasets import load_dataset
# import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler, UNet1DModel
import torch
from PIL import Image
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
from accelerate import Accelerator
# from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch.nn.functional as F
# from torchvision.datasets import MNIST
# from torch_ema import ExponentialMovingAverage
import numpy as np
# from data.dataset import RescaledLogRDmapsDataset
from datasets.diffusion_dataset import get_train_test_with_config
import argparse
from utils.config import load_config

def evaluate(config, epoch, pipeline, test_dataloader, accelerator, tracker, global_step, device='cuda'):
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device)
    model = pipeline.unet
    model.eval()
    num_sampling_steps = 100
    train_steps = len(alphas_cumprod)
    skip = train_steps // num_sampling_steps
    seq = range(0, train_steps-1, skip)
    seq_next = [-1] + list(seq[:-1])
    # test_iter = iter(test_dataloader)
    total_mse = 0.0
    progress_bar = tqdm(total=len(test_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in enumerate(test_dataloader):
        if step >= 2:
            break
        mix = batch['mix'].to('cuda')  # [B, 2, 3072]
        clean1 = batch['clean1'].to('cuda')
        clean2 = batch['clean2'].to('cuda')
        clean_samples = torch.cat([clean1, clean2], dim=1)  # [B, 4, 3072]
        bs = clean_samples.shape[0]
        xt = torch.randn((bs, 4, config.data.signal_len), device=device)
        with torch.no_grad():
            for i, j in (zip(reversed(seq), reversed(seq_next))):
                t = torch.full((xt.shape[0],), i, dtype=torch.long).to(device)
                input_noisy = torch.cat([xt, mix], dim=1)
                noise = torch.randn_like(xt) if i > 0 else torch.tensor(0.).to(device)
                at = alphas_cumprod[i]
                at_next = alphas_cumprod[j] if j >= 0 else torch.tensor(1.0).to(device)
                model_output = model(input_noisy, t, return_dict=False)[0]
                if config.model.predict_target == 'epsilon':
                    et = model_output
                    x_0 = (xt - torch.sqrt(1 - at) * et) / torch.sqrt(at)
                elif config.model.predict_target == 'x0':
                    x_0 = model_output
                    et = (xt - torch.sqrt(at) * x_0) / torch.sqrt(1 - at)
                xt_next = torch.sqrt(at_next) * x_0 + torch.sqrt(1 - at_next) * et
                xt = xt_next
                # print(x_0[0])
        
        predict = xt
        # 计算指标
        mse = F.mse_loss(predict, clean_samples)
        # print("Eval Step:{}; MSE:{}".format(step, mse.item()))
        total_mse += mse.item()
        progress_bar.update(1)
        logs = {"mse": total_mse/(step+1), "num_test": step+1}
        progress_bar.set_postfix(**logs)
    avg_mse = total_mse / (step)
    print("Epoch:{}; Eval MSE:{}".format(epoch, avg_mse))
    tracker.log({"eval_mse": avg_mse}, step=global_step)
    model.train()
    return avg_mse
    

def train(config, trainset, testset):
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=config.training.train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=config.training.test_batch_size, shuffle=False)
    model = UNet1DModel(
        sample_size=config.data.signal_len,  # the target signal length
        in_channels=4,  # the number of input channels, 2 for real and imaginary parts
        out_channels=4,  # the number of output channels
        extra_in_channels=2, # condition
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 256, 256, 256, 256),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock1D",  # a regular ResNet downsampling block
            "DownBlock1D", 
            "DownBlock1D",
            "DownBlock1D",
            "AttnDownBlock1D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock1D",
        ),
        norm_num_groups=4, 
        up_block_types=(
            "UpBlock1D",  # a regular ResNet upsampling block
            "AttnUpBlock1D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock1D",
            "UpBlock1D",
            "UpBlock1D",
            "UpBlock1D",
        ),
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.model.num_diffusion_timesteps)
    if config.training.pretrained is not None:
        model = UNet1DModel.from_pretrained(config.training.pretrained, use_safetensors=True, local_files_only=True)
    # print(config.training.learning_rate)
    print(config.training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.training.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.training.num_epochs),
    )
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, test_dataloader)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, test_dataloader):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.training.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.training.output_dir is not None:
            os.makedirs(config.training.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    global_step = 0

    # Now you train the model
    for epoch in range(config.training.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        loss_sum = 0
        
        for step, batch in enumerate(train_dataloader):
            # clean_images = batch["input"]
            mix = batch['mix'].to('cuda')  # [B, 2, 3072]
            clean1 = batch['clean1'].to('cuda')
            clean2 = batch['clean2'].to('cuda')
            # print(clean1)
            clean_samples = torch.cat([clean1, clean2], dim=1)  # [B, 4, 3072]
            # Sample noise to add to the images
            noise = torch.randn(clean_samples.shape, device=clean_samples.device)
            bs = clean_samples.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_samples.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_samples, noise, timesteps)
            # 拼接输入
            input_noisy = torch.cat([noisy_images, mix], dim=1)  # [B, 4+2, 3072]
            target = noise if config.model.predict_target == 'epsilon' else clean_samples
            with accelerator.accumulate(model):
                noise_pred = model(input_noisy, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, target)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            loss_sum += loss.detach().item()
            progress_bar.update(1)
            logs = {"loss": loss_sum/(step+1), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % config.training.save_data_epochs == 0 or epoch == config.training.num_epochs - 1:
                evaluate(config, epoch, pipeline, test_dataloader, accelerator, accelerator.trackers[0], global_step, device='cuda')
            if (epoch + 1) % config.training.save_model_epochs == 0 or epoch == config.training.num_epochs - 1:
                if config.training.push_to_hub:
                    pass
                else:
                    pipeline.save_pretrained(config.training.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config.yaml', help='Output directory name')
    # parser.add_argument('--mode', type=str, default='8PSK', help='modulation mode')
    args = parser.parse_args()
    config = load_config(os.sep.join(['configs', args.config]))
    trainset, testset = get_train_test_with_config(config)
    train(config, trainset, testset)


if __name__ == '__main__':
    main()