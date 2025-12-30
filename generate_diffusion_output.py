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
import numpy as np
import argparse
from utils.config import load_config
from datasets.diffusion_dataset import get_test_with_config



def evaluate_ddnm(config, epoch, pipeline, dataset, test_dataloader, accelerator, tracker, global_step, device='cuda', args=None):
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device)
    model = pipeline.unet
    model.eval()
    num_sampling_steps = config.sampling.num_inference_steps
    train_steps = len(alphas_cumprod)
    skip = train_steps // num_sampling_steps
    seq = range(0, train_steps-1, skip)
    seq_next = [-1] + list(seq[:-1])
    # test_iter = iter(test_dataloader)
    total_mse = 0.0
    progress_bar = tqdm(total=len(test_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    idx = 0
    path = config.sampling.output_dir
    for step, batch in enumerate(test_dataloader):
        # if step >= 2:
        #     break
        mix = batch['mix'].to('cuda')  # [B, 2, 3072]
        clean1 = batch['clean1'].to('cuda')
        clean2 = batch['clean2'].to('cuda')
        clean_samples = torch.cat([clean1, clean2], dim=1)  # [B, 4, 3072]
        bs = clean_samples.shape[0]
        xt = torch.randn((bs, 4, config.data.signal_len), device=device)
        sigma_n = (mix - clean1 - clean2).std()
        eta = config.sampling.eta # DDNM
        y_0 = mix
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
                if at_next.sqrt() * 0.5 * sigma_n < eta * (1-at_next).sqrt():
                    diff = y_0 - x_0[:,0:2] - x_0[:, 2:4]
                    x0_t_hat = x_0
                    x0_t_hat[:, 0:2] += 0.5 * diff
                    x0_t_hat[:, 2:4] += 0.5 * diff
                    new_noise_var = eta**2 * (1-at_next) - 0.25 * sigma_n**2 * at_next
                    xt_next = torch.sqrt(at_next) * x0_t_hat + torch.sqrt(1-at_next) * (1-eta**2)**0.5 * et + new_noise_var**0.5 * noise
                else:
                    diff = y_0 - x_0[:,0:2] - x_0[:, 2:4]
                    diff *= eta * (1-at_next)**0.5 / (0.5 * sigma_n * at_next.sqrt())
                    x0_t_hat = x_0
                    x0_t_hat[:, 0:2] += 0.5 * diff
                    x0_t_hat[:, 2:4] += 0.5 * diff
                    xt_next = torch.sqrt(at_next) * x0_t_hat + torch.sqrt(1-at_next) * (1-eta**2)**0.5 * et

                xt = xt_next
                # print(x_0[0])
        
        predict = xt
        # 计算指标
        mse = F.mse_loss(predict, clean_samples)
        total_mse += mse.item()
        progress_bar.update(1)
        logs = {"mse": total_mse/(step+1), "num_test": step+1}
        progress_bar.set_postfix(**logs)
        for n in range(predict.shape[0]):
            one_sample = predict[n]
            dataset.update(one_sample, idx)
            idx += 1
    avg_mse = total_mse / (step+1)
    print("Epoch:{}; Eval MSE:{}".format(epoch, avg_mse))
    tracker.log({"eval_mse": avg_mse}, step=global_step)
    model.train()
    os.makedirs(path, exist_ok=True)
    print('Saving to ', path)
    dataset.save_shards(path)
    return avg_mse
    

def generate(config, dataset, args):
    # train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.training.test_batch_size, shuffle=False)
    print('loading pretrained model from ' + config.training.output_dir + '/unet')
    model = UNet1DModel.from_pretrained(config.training.output_dir + '/unet', use_safetensors=True, local_files_only=True)
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.model.num_diffusion_timesteps)
    test_loop(config, model, noise_scheduler, dataset, test_dataloader, args)


def test_loop(config, model, noise_scheduler, dataset, test_dataloader, args):
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
    
    model = accelerator.prepare(
        model
    )
    global_step = 0

    # Now you train the model
    for epoch in range(1):
        progress_bar = tqdm(total=1, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        loss_sum = 0
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            evaluate_ddnm(config, epoch, pipeline, dataset, test_dataloader, accelerator, accelerator.trackers[0], global_step, device='cuda', args=args)
            if (epoch + 1) % config.training.save_model_epochs == 0 or epoch == config.training.num_epochs - 1:
                if config.training.push_to_hub:
                    pass
                else:
                    pipeline.save_pretrained(config.training.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training', help='Output directory name')
    args = parser.parse_args()
    config = load_config(os.sep.join(['configs', args.config]))
    # config.output_dir = f"results/DDPM-PCMA-8PSK-real-{args.train_ratio}"
    dataset = get_test_with_config(config)
    generate(config, dataset, args)

if __name__ == '__main__':
    main()