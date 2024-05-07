import os
import sys
spice = True
if spice:
    sys.path.insert(0, os.path.join(os.path.abspath(os.getcwd()), "spic_e", "shap_e"))
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
import json
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from lora import LoRANetwork
from diffusers.utils import export_to_gif
from shap_e.util.notebooks import decode_latent_images
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='spice_with_prompts_masking_percentile_90_beta_0_75', help="Name of the run.")
parser.add_argument("--alpha", type=float, default=1.0, help="Alpha of LoRA.")
parser.add_argument("--rank", type=int, default=4, help="Rank of LoRA.")
parser.add_argument("--obj", type=str, default="chair", help="Name of the slider.")
parser.add_argument("--prompt_index", type=int, default=0, help="Rank of LoRA.")
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument( "--scales", type=str, default = '-1, 1', help  ="scales for different attribute-scaled images")
parser.add_argument( "--batch_size", type=int, default=4, help="batch size")
parser.add_argument( "--grad_acc_steps", type=int, default=16, help="max timesteps")
parser.add_argument('--cond_drop_prob', type=float, default=0.5, help='chance for model to ignore text condition during training')
parser.add_argument('--num_timesteps', type=int, default=1024, help='number of timesteps (1024 in paper)')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument( "--wandb_project", type=str, default="ShapeLoraSliders", help="wandb project name")
parser.add_argument( "--guidance_scale", type=float, default=7.5, help="guidance scale")
parser.add_argument( "--render_mode", type=str, default="nerf", help="render mode")
parser.add_argument('--size', type=int, default=160, help='images size')
parser.add_argument('--test_steps', type=int, default=50, help='test steps')
parser.add_argument('--masking_diff_percentile', type=int, default=90, help='masking diff percentile')
parser.add_argument("--beta", type=float, default=0.75, help="beta")
parser.add_argument('--test', action='store_true', help='test and use wandb')
args = parser.parse_args()

with open(os.path.join('/storage/noamatia/repos/shape_lora_sliders/lora_sliders/llama3', f'{args.obj}.json'), 'r') as f:
    pairs = json.load(f)
pair = pairs[args.prompt_index]
low_prompt, high_prompt = pair[0], pair[1]
slider_name = f'{low_prompt.replace(" ", "_")}_{high_prompt.replace(" ", "_")}'
data_dir = f'/storage/noamatia/shape_lora_sliders_datasets/{args.obj}/{slider_name}'

def parse_arg(arg: str, is_int: bool = False) -> np.ndarray:
    arg = arg.split(',')
    arg = [f.strip() for f in arg]
    if is_int:
        arg = [int(s) for s in arg]
    return np.array(arg)

def build_name_and_paths() -> tuple:
    paths = {}
    name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}_{slider_name}_{args.name}"
    output_dir = os.path.join('/storage/noamatia/shape_lora_sliders', name)
    paths['samples'] = os.path.join(output_dir, 'samples')
    paths['x_T_test'] = os.path.join(output_dir, 'x_T_test.pt')
    paths['eps_test'] = os.path.join(output_dir, 'eps_test.pt')
    paths['models'] = os.path.join(output_dir, 'models')
    return name, paths

def build_models(device: torch.device) -> tuple:
    if spice:
        cache_dir = os.path.join(os.path.abspath(os.getcwd()), "spic_e_model_cache")
    else:
        cache_dir = os.path.join(os.path.abspath(os.getcwd()), "shap_e_model_cache")
    xm = load_model('transmitter', device=device, cache_dir=cache_dir)
    model = load_model('text300M', device=device, cache_dir=cache_dir)
    if spice:
        model.wrapped.backbone.make_ctrl_layers()
        model.wrapped.set_up_controlnet_cond()
        model.load_state_dict(torch.load(os.path.join('/storage/noamatia/repos/shape_lora_sliders/lora_sliders/spice_models', f'{args.obj}.pt')))
    model.wrapped.cond_drop_prob = args.cond_drop_prob
    model.freeze_all_parameters()
    network = LoRANetwork(model.wrapped, args.rank, args.alpha).to(device)
    model.print_parameter_status()
    network.print_parameter_status()
    diffusion = diffusion_from_config(load_config('diffusion'))
    optimizer = torch.optim.Adam(network.prepare_optimizer_params(), lr=args.lr)
    return xm, model, network, diffusion, optimizer

def build_folders_prompts_and_scales() -> tuple:
    # folders = parse_arg(args.folders)
    # prompts = parse_arg(args.prompts)
    scales = parse_arg(args.scales, is_int=True)
    # assert folders.shape[0] == scales.shape[0], "The number of folders and scales must be the same."
    # assert prompts.shape[0] == scales.shape[0], "The number of prompts and scales must be the same."
    assert scales.shape[0] == 2, "Currently the number of scales must be 2."
    scale_high = abs(random.choice(list(scales)))
    scale_low = -scale_high
    # folder_high = folders[scales==scale_high][0]
    # folder_low = folders[scales==scale_low][0]
    # prompt_high = prompts[scales==scale_high][0]
    # prompt_low = prompts[scales==scale_low][0]
    folder_high = f'{high_prompt.replace(" ", "_")}/latents'
    folder_low = f'{low_prompt.replace(" ", "_")}/latents'
    return folder_high, folder_low, high_prompt, low_prompt, scale_high, scale_low

def build_latents(test_or_train: str, folder:str ) -> list:
    latents = os.listdir(os.path.join(data_dir, test_or_train, folder))
    latents = [latent for latent in latents if '.pt' in latent]
    if args.test:
        wandb.config[f"{test_or_train}_dataset_size"] =  len(latents)
        wandb.config[f"{test_or_train}_latents_ids"] =  latents
    return latents

def load_latents(test_or_train: str, folder: str, device: torch.device, latents: list) -> list:
    lats = [torch.load(os.path.join(data_dir, test_or_train, folder, latent)).to(device) for latent in latents]
    if len(lats) % args.batch_size != 0:
        lats += [lats[i] for i in range(args.batch_size - (len(lats) % args.batch_size))]
    lats = [lats[i:i + args.batch_size] for i in range(0, len(lats), args.batch_size)]
    lats = [torch.cat(lats).detach().unsqueeze(1) for lats in lats]
    return lats

def build_model_kwargs_list(prompt: str, latents: list, test_or_train: str, low_or_high: str, loaded_latents: list = None) -> tuple:
    uid_to_object = False
    if uid_to_object:
        assert prompt != '', "Prompt must be provided if uid_to_object is provided."
        with open(os.path.join(data_dir, test_or_train, f'{args.uid_to_object}.json'), 'r') as f:
            uid_to_object = json.load(f)
        prompts = [uid_to_object[latent.split('.')[0]] + ' ' + prompt for latent in latents]
    else:
        # assert prompt == '', "Prompt must be empty if uid_to_object is not provided."
        prompts = [prompt for _ in latents]
    uids = [latent.split('.')[0] for latent in latents]
    if len(prompts) % args.batch_size != 0:
        prompts += [prompts[i] for i in range(args.batch_size - (len(prompts) % args.batch_size))]
        uids += [uids[i] for i in range(args.batch_size - (len(uids) % args.batch_size))]
    prompts = [prompts[i:i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    uids = [uids[i:i + args.batch_size] for i in range(0, len(uids), args.batch_size)]
    if loaded_latents is None:
        model_kwargs = [dict(texts=prompts[i]) for i in range(len(prompts))]
    else:
        try:
            assert len(prompts) == len(loaded_latents), "The number of prompts and latents must be the same."
        except:
            import pdb; pdb.set_trace()
        assert len(prompts[-1]) == len(loaded_latents[-1]), "The number of prompts and latents must be the same."
        model_kwargs = [dict(texts=prompts[i], cond=loaded_latents[i], uid=uids[i]) for i in range(len(prompts))]
    if args.test:
        wandb.config[f"model_kwargs_{test_or_train}_{low_or_high}"] = model_kwargs
    return model_kwargs

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

def sanity_check(latents_high, latents_low, masks, cameras, xm, test_or_train):
    output_dir = 'sanity_check'
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    for batch_high, batch_low, mask in zip(latents_high, latents_low, masks):
        masked_latent = batch_low + mask * (batch_high - batch_low)
        for j in range(len(batch_high)):
            decode_and_export(batch_high[j], output_dir, f'{test_or_train}_{i}_{j}_high.gif', cameras, xm)
            decode_and_export(batch_low[j], output_dir, f'{test_or_train}_{i}_{j}_low.gif', cameras, xm)
            decode_and_export(masked_latent[j], output_dir, f'{test_or_train}_{i}_{j}_masked.gif', cameras, xm)

def decode_and_export_test_conditions(model_kwargs_high_test_list, model_kwargs_low_test_list, paths, cameras, xm):
    log_data = {}
    for model_kwargs_high_test, model_kwargs_low_test in tqdm(zip(model_kwargs_high_test_list, model_kwargs_low_test_list), 
                                                              total=len(model_kwargs_high_test_list), 
                                                              desc='Decoding and exporting test conditions'):
        for i in range(args.batch_size):
            uid_test_high = model_kwargs_high_test['uid'][i]
            uid_test_low = model_kwargs_low_test['uid'][i]
            assert uid_test_high == uid_test_low, "The high and low uid must be the same."
            uid_output_dir_high = os.path.join(paths['samples'], uid_test_high, 'high')
            uid_output_dir_low = os.path.join(paths['samples'], uid_test_low, 'low')
            os.makedirs(uid_output_dir_high, exist_ok=True)
            os.makedirs(uid_output_dir_low, exist_ok=True)
            log_data[f"high_condition_{uid_test_high}"] = wandb.Video(decode_and_export(model_kwargs_high_test['cond'][i], uid_output_dir_high, f'condition.gif', cameras, xm))
            log_data[f"low_condition_{uid_test_low}"] = wandb.Video(decode_and_export(model_kwargs_low_test['cond'][i], uid_output_dir_low, f'condition.gif', cameras, xm))
    wandb.log(log_data)