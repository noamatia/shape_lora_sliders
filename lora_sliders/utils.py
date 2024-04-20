import os
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='spice_with_prompts_masking_percentile_90_beta_0_75', help="Name of the run.")
parser.add_argument("--alpha", type=float, default=1.0, help="Alpha of LoRA.")
parser.add_argument("--rank", type=int, default=4, help="Rank of LoRA.")
parser.add_argument("--slider_name", type=str, default="backslider", help="Name of the slider.")
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--data_dir', type=str, default='/home/noamatia/repos/shape_lora_sliders/lora_sliders/datasets/back', help='data directory')
parser.add_argument('--prompts', type=str, default='with a spindle backrest, with a solid backrest', help='prompts for generation')
parser.add_argument( "--folders", type=str, default='spindle/latents, solid/latents', help="folders with different attribute-scaled images")
parser.add_argument( "--scales", type=str, default = '-1, 1', help="scales for different attribute-scaled images")
parser.add_argument( "--batch_size", type=int, default=4, help="batch size")
parser.add_argument( "--grad_acc_steps", type=int, default=16, help="max timesteps")
parser.add_argument('--cond_drop_prob', type=float, default=0.5, help='chance for model to ignore text condition during training')
parser.add_argument('--num_timesteps', type=int, default=1024, help='number of timesteps (1024 in paper)')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument( "--wandb_project", type=str, default="ShapeLoraSliders", help="wandb project name")
parser.add_argument( "--guidance_scale", type=float, default=7.5, help="guidance scale")
parser.add_argument( "--render_mode", type=str, default="nerf", help="render mode")
parser.add_argument('--size', type=int, default=160, help='images size')
parser.add_argument('--test_steps', type=int, default=10, help='test steps')
parser.add_argument("--subset", type=str, help="json subset")
parser.add_argument('--masking_diff_percentile', type=int, default=90, help='masking diff percentile')
parser.add_argument( "--beta", type=float, default=0.75, help="beta")
parser.add_argument("--uid_to_object", type=str, default='uid_to_object', help="uid to object")
parser.add_argument("--test_object", type=str, default='a chair', help="uid to object")
parser.add_argument("--spice_model_path", type=str, default="/scratch/noam/spice_models/semantic_editing_chair/best_model.pt", help="path to spice model") #  /scratch/noam/spice_models/semantic_editing_chair/best_model.pt
parser.add_argument("--gpu", type=str, default="4", help="gpu id")
args = parser.parse_args()
if args.spice_model_path:
    sys.path.insert(0, os.path.join(os.path.abspath(os.getcwd()), "spic_e"))
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"

import json
import wandb
import torch
import random
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

def build_name_and_paths() -> tuple:
    paths = {}
    name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}_{args.slider_name}_{args.name}"
    output_dir = os.path.join('lora_sliders', 'outputs', name)
    paths['samples'] = os.path.join(output_dir, 'samples')
    os.makedirs(paths['samples'], exist_ok=True)
    paths['x_T_test'] = os.path.join(output_dir, 'x_T_test.pt')
    paths['eps_test'] = os.path.join(output_dir, 'eps_test.pt')
    paths['model_best'] = os.path.join(output_dir, 'model_best.pt')
    paths['model_final'] = os.path.join(output_dir, 'model_final.pt')
    return name, paths

def build_models(device: torch.device) -> tuple:
    if args.spice_model_path:
        cache_dir = os.path.join(os.path.abspath(os.getcwd()), "spic_e_model_cache")
    else:
        cache_dir = os.path.join(os.path.abspath(os.getcwd()), "shap_e_model_cache")
    xm = load_model('transmitter', device=device, cache_dir=cache_dir)
    model = load_model('text300M', device=device, cache_dir=cache_dir)
    if args.spice_model_path:
        model.wrapped.backbone.make_ctrl_layers()
        model.wrapped.set_up_controlnet_cond()
        model.load_state_dict(torch.load(args.spice_model_path))
    model.wrapped.cond_drop_prob = args.cond_drop_prob
    model.freeze_all_parameters()
    network = LoRANetwork(model.wrapped, args.rank, args.alpha).to(device)
    model.print_parameter_status()
    network.print_parameter_status()
    diffusion = diffusion_from_config(load_config('diffusion'))
    optimizer = torch.optim.Adam(network.prepare_optimizer_params(), lr=args.lr)
    return xm, model, network, diffusion, optimizer

def build_folders_prompts_and_scales() -> tuple:
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

def build_latents(folder:str ) -> list:
    latents = os.listdir(os.path.join(args.data_dir, folder))
    latents = [latent for latent in latents if '.pt' in latent]
    if args.subset:
        with open(os.path.join(args.data_dir, f'{args.subset}.json'), 'r') as f:
            subset = json.load(f)
        latents = [latent for latent in latents if latent in subset]
    return latents

def load_latents(folder: str, device: torch.device, latents: list) -> list:
    lats = [torch.load(os.path.join(args.data_dir, folder, latent)).to(device) for latent in latents]
    lats += [lats[i] for i in range(args.batch_size - (len(lats) % args.batch_size))]
    lats = [lats[i:i + args.batch_size] for i in range(0, len(lats), args.batch_size)]
    lats = [torch.cat(lats).detach().unsqueeze(1) for lats in lats]
    return lats

def build_model_kwargs_train(prompt: str, latents: list,  loaded_latents: list = None) -> list:
    if args.uid_to_object:
        assert prompt != '', "Prompt must be provided if uid_to_object is provided."
        with open(os.path.join(args.data_dir, f'{args.uid_to_object}.json'), 'r') as f:
            uid_to_object = json.load(f)
        prompts = [uid_to_object[latent.split('.')[0]] + ' ' + prompt for latent in latents]
    else:
        assert prompt == '', "Prompt must be empty if uid_to_object is not provided."
        prompts = ['' for _ in latents]
    prompts += [prompts[i] for i in range(args.batch_size - (len(prompts) % args.batch_size))]
    prompts = [prompts[i:i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    if loaded_latents is None:
        model_kwargs = [dict(texts=prompts[i]) for i in range(len(prompts))]
    else:
        assert len(prompts) == len(loaded_latents), "The number of prompts and latents must be the same."
        assert len(prompts[-1]) == len(loaded_latents[-1]), "The number of prompts and latents must be the same."
        model_kwargs = [dict(texts=prompts[i], cond=loaded_latents[i]) for i in range(len(prompts))]
    return model_kwargs

def build_model_kwargs_test(prompt: str, loaded_latents: list = None) -> dict:
    if args.test_object:
        assert prompt != '', "Prompt must be provided if test_object is provided."
        prompt = args.test_object + ' ' + prompt
    else:
        assert prompt == '', "Prompt must be empty if test_object is not provided."
    if loaded_latents is None:
        return dict(texts=[''])
    else:
        return dict(texts=[prompt], cond=loaded_latents[0][0])
        
def build_model_kwargs(prompt: str, latents: list, type: str, loaded_latents: list = None) -> tuple:
    model_kwargs_list = build_model_kwargs_train(prompt, latents, loaded_latents)
    wandb.config[f"model_kwargs_{type}_list"] = model_kwargs_list
    test_model_kwargs = build_model_kwargs_test(prompt, loaded_latents)
    wandb.config[f"test_model_kwargs_{type}"] = test_model_kwargs
    return model_kwargs_list, test_model_kwargs

def build_masks(latents_high: list, latents_low: list) -> list:
    masks = []
    for batch_high, batch_low in zip(latents_high, latents_low):
        batch_diff_abs = torch.abs(batch_high - batch_low)
        threshold = torch.quantile(batch_diff_abs, args.masking_diff_percentile / 100, dim=2).unsqueeze(2)
        mask = torch.where(batch_diff_abs < threshold, torch.ones_like(batch_diff_abs), torch.zeros_like(batch_diff_abs))
        masks.append(mask)
    return masks

def decode_and_export(latent, output_dir, name, cameras, xm) -> str:
    images = decode_latent_images(xm, latent, cameras, rendering_mode=args.render_mode)
    return export_to_gif(images, os.path.join(output_dir, name))

def sanity_check(latents_high, latents_low, masks, cameras, xm):
    output_dir = 'sanity_check'
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    for batch_high, batch_low, mask in zip(latents_high, latents_low, masks):
        masked_latent = batch_low + mask * (batch_high - batch_low)
        for j in range(len(batch_high)):
            decode_and_export(batch_high[j], output_dir, f'{i}_{j}_high.gif', cameras, xm)
            decode_and_export(batch_low[j], output_dir, f'{i}_{j}_low.gif', cameras, xm)
            decode_and_export(masked_latent[j], output_dir, f'{i}_{j}_masked.gif', cameras, xm)
