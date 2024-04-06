import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
import json
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from lora_sliders.lora import LoRANetwork
from diffusers.utils import export_to_gif
from shap_e.util.notebooks import decode_latent_images
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config

    
def parse_arg(arg: str, is_int: bool = False) -> np.ndarray:
    arg = arg.split(',')
    arg = [f.strip() for f in arg]
    if is_int:
        arg = [int(s) for s in arg]
    return np.array(arg)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name of the run.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha of LoRA.")
    parser.add_argument("--rank", type=int, default=4, help="Rank of LoRA.")
    parser.add_argument("--slider_name", type=str, default="armsslider", help="Name of the slider.")
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='/home/noamatia/repos/shape_lora_sliders/lora_sliders/datasets/arms', help='data directory')
    parser.add_argument('--prompts', type=str, default=' , ', help='prompts for generation')
    parser.add_argument( "--folders", type=str, default='withoutarms/latents, witharms/latents', help="folders with different attribute-scaled images")
    parser.add_argument( "--scales", type=str, default = '-1, 1', help="scales for different attribute-scaled images")
    parser.add_argument( "--batch_size", type=int, default=6, help="batch size")
    parser.add_argument( "--grad_acc_steps", type=int, default=11, help="max timesteps")
    parser.add_argument('--cond_drop_prob', type=float, default=0.5, help='chance for model to ignore text condition during training')
    parser.add_argument('--num_timesteps', type=int, default=1024, help='number of timesteps (1024 in paper)')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument( "--wandb_project", type=str, default="ShapeLoraSliders", help="wandb project name")
    parser.add_argument( "--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument( "--render_mode", type=str, default="nerf", help="render mode")
    parser.add_argument('--size', type=int, default=160, help='images size')
    parser.add_argument('--test_steps', type=int, default=10, help='test steps')
    parser.add_argument("--subset", type=str, help="json subset") # only_arms_latents
    parser.add_argument('--masking_diff_percentile', type=int, default=0, help='masking diff percentile')
    parser.add_argument( "--beta", type=float, default=1.0, help="beta")
    return parser.parse_args()

def build_name_and_paths(args: argparse.Namespace) -> tuple:
    paths = {}
    name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}_{args.slider_name}_{args.name}"
    output_dir = os.path.join('lora_sliders', 'outputs', name)
    paths['samples'] = os.path.join(output_dir, 'samples')
    os.makedirs(paths['samples'], exist_ok=True)
    paths['x_T_test'] = os.path.join(output_dir, 'x_T_test.pt')
    paths['model_best'] = os.path.join(output_dir, 'model_best.pt')
    paths['model_final'] = os.path.join(output_dir, 'model_final.pt')
    return name, paths

def build_models(args: argparse.Namespace, device: torch.device) -> tuple:
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model.wrapped.cond_drop_prob = args.cond_drop_prob
    model.freeze_all_parameters()
    network = LoRANetwork(model.wrapped, args.rank, args.alpha).to(device)
    model.print_parameter_status()
    network.print_parameter_status()
    diffusion = diffusion_from_config(load_config('diffusion'))
    optimizer = torch.optim.Adam(network.prepare_optimizer_params(), lr=args.lr)
    return xm, model, network, diffusion, optimizer

def build_folders_prompts_and_scales(args: argparse.Namespace) -> tuple:
    folders = parse_arg(args.folders)
    prompts = parse_arg(args.prompts)
    scales = parse_arg(args.scales, is_int=True)
    assert folders.shape[0] == scales.shape[0], "The number of folders and scales must be the same."
    assert prompts.shape[0] == scales.shape[0], "The number of prompts and scales must be the same."
    assert scales.shape[0] == 2, "Currently the number of scales must be 2."
    scale_high = abs(random.choice(list(scales)))
    scale_low = -scale_high
    folder_high = folders[scales==scale_high][0]
    folder_low = folders[scales==scale_low][0]
    prompt_high = prompts[scales==scale_high][0]
    prompt_low = prompts[scales==scale_low][0]
    return folder_high, folder_low, prompt_high, prompt_low, scale_high, scale_low

def build_latents(args: argparse.Namespace, folder:str ) -> list:
    latents = os.listdir(os.path.join(args.data_dir, folder))
    latents = [latent for latent in latents if '.pt' in latent]
    if args.subset:
        with open(os.path.join(args.data_dir, f'{args.subset}.json'), 'r') as f:
            subset = json.load(f)
        latents = [latent for latent in latents if latent in subset]
    return latents

def load_latents(args: argparse.Namespace, folder: str, device: torch.device, latents: list) -> list:
    lats = [torch.load(os.path.join(args.data_dir, folder, latent)).to(device) for latent in latents]
    lats += [lats[i] for i in range(args.batch_size - (len(lats) % args.batch_size))]
    lats = [lats[i:i + args.batch_size] for i in range(0, len(lats), args.batch_size)]
    lats = [torch.cat(lats).detach().unsqueeze(1) for lats in lats]
    return lats

def build_model_kwargs(prompt: str, batch_size: int) -> tuple:
    model_kwargs = dict(texts=[prompt for _ in range(batch_size)])
    test_model_kwargs = dict(texts=[prompt])
    return model_kwargs, test_model_kwargs

def build_masks(masking_diff_percentile: int, latents_high: list, latents_low: list) -> list:
    masks = []
    for batch_high, batch_low in zip(latents_high, latents_low):
        batch_diff_abs = torch.abs(batch_high - batch_low)
        threshold = torch.quantile(batch_diff_abs, masking_diff_percentile / 100, dim=2).unsqueeze(2)
        mask = torch.where(batch_diff_abs < threshold, torch.ones_like(batch_diff_abs), torch.zeros_like(batch_diff_abs))
        masks.append(mask)
    return masks

def sanity_check(latents_high, latents_low, masks, cameras, xm):
    output_dir = 'sanity_check'
    os.makedirs(output_dir, exist_ok=True)

    def decode_and_export(latent, name):
        images = decode_latent_images(xm, latent, cameras, rendering_mode='nerf')
        export_to_gif(images, os.path.join(output_dir, name))
        
    i = 0
    for batch_high, batch_low, mask in zip(latents_high, latents_low, masks):
        masked_latent = batch_low + mask * (batch_high - batch_low)
        for j in range(len(batch_high)):
            decode_and_export(batch_high[j], f'{i}_{j}_high.gif')
            decode_and_export(batch_low[j], f'{i}_{j}_low.gif')
            decode_and_export(masked_latent[j], f'{i}_{j}_masked.gif')
