import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--gpu', type=str, required=True)
parser.add_argument('--obj_type', type=str, required=True, choices=['a_chair', 'an_office_chair'])
args = parser.parse_args()
# python lora_sliders/eval_shape_lora_sliders_visual.py --name 04_09_2024_19_37_32_armsslider_with_prompts_masking_percentile_100_beta_0_75 --gpu 6 --obj_type an_office_chair

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
from lora_sliders.lora import LoRANetwork
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

n = 38
rank = 4
size = 160
alpha = 1.0
sigma_max = 160
render_mode = "nerf"
cond_drop_prob = 0.5
guidance_scale = 7.5
name = args.name
t_start = 0
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
base_dir = "/home/noamatia/repos/shape_lora_sliders/lora_sliders/outputs"
lora_weight = os.path.join(base_dir, name, "model_best.pt")
results_path = os.path.join(base_dir, name, "results", args.obj_type)
os.makedirs(results_path, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
model.wrapped.cond_drop_prob = cond_drop_prob
model.freeze_all_parameters()
network = LoRANetwork(model.wrapped, rank, alpha).to(device)
network.load_state_dict(torch.load(lora_weight))
diffusion = diffusion_from_config(load_config('diffusion'))
cameras = create_pan_cameras(size, device)

obj_type = args.obj_type.replace("_", " ")
if "armsslider" in name:
    test_model_kwargs_high = dict(texts=[f'{obj_type} with armrests'])
    test_model_kwargs_low = dict(texts=[f'{obj_type} without armrests'])
elif "backslider" in name:
    test_model_kwargs_high = dict(texts=[f'{obj_type} with a solid backrest'])
    test_model_kwargs_low = dict(texts=[f'{obj_type} with a spindle backrest'])
else:
    raise ValueError(f"Unknown name: {name}")

for i in range(25):
    x_T = torch.load(f'lora_sliders/outputs/x_Ts/x_T_{i}.pt')
    eps = torch.randn_like(x_T)
    latents_high = sample_latents(model=model, 
                            diffusion=diffusion, 
                            model_kwargs=test_model_kwargs_high, 
                            guidance_scale=guidance_scale, 
                            device=device, 
                            progress=True, 
                            sigma_max=sigma_max, 
                            x_T=x_T,
                            network=network,
                            scale=1,
                            t_start=t_start,
                            eps=eps)
    images = decode_latent_images(xm, latents_high[0], cameras, rendering_mode=render_mode)
    images[7].save(os.path.join(results_path, f'example_{i}_percentile_0.png'))
    latents_low = sample_latents(model=model, 
                            diffusion=diffusion, 
                            model_kwargs=test_model_kwargs_low, 
                            guidance_scale=guidance_scale, 
                            device=device, 
                            progress=True, 
                            sigma_max=sigma_max, 
                            x_T=x_T,
                            network=network,
                            scale=-1,
                            t_start=t_start,
                            eps=eps)
    images = decode_latent_images(xm, latents_low[0], cameras, rendering_mode=render_mode)
    images[7].save(os.path.join(results_path, f'example_{i}_percentile_100.png'))
    for percentile in percentiles:
        if os.path.exists(os.path.join(results_path, f'example_{i}_percentile_{percentile}.png')):
            continue
        latents_diff_abs = torch.abs(latents_high - latents_low)
        threshold = torch.quantile(latents_diff_abs, percentile / 100)
        mask = torch.where(latents_diff_abs < threshold, torch.zeros_like(latents_diff_abs), torch.ones_like(latents_diff_abs))
        masked_latent = latents_low + mask * (latents_high - latents_low)
        images = decode_latent_images(xm, masked_latent[0], cameras, rendering_mode=render_mode)
        images[7].save(os.path.join(results_path, f'example_{i}_percentile_{percentile}.png'))