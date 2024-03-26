import gc
import torch
import argparse
import numpy as np

def flush(*args):
    for arg in args:
        del arg
    torch.cuda.empty_cache()
    gc.collect()
    
def parse_arg(arg: str, is_int: bool = False) -> np.ndarray:
    arg = arg.split(',')
    arg = [f.strip() for f in arg]
    if is_int:
        arg = [int(s) for s in arg]
    return np.array(arg)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha of LoRA.")
    parser.add_argument("--rank", type=int, default=4, help="Rank of LoRA.")
    parser.add_argument("--name", type=str, default="armsslider", help="Name of the slider.")
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='/home/noamatia/repos/shape_sliders/datasets/arms', help='data directory')
    parser.add_argument( "--folders", type=str, default='withoutarms/latents, witharms/latents', help="folders with different attribute-scaled images")
    parser.add_argument( "--scales", type=str, default = '-1, 1', help="scales for different attribute-scaled images")
    parser.add_argument('--prompt', type=str, default='', help='prompt for generation')
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
    parser.add_argument('--subset', type=int, help='subset of data to use')
    parser.add_argument('--eta', type=int, default=4, help='sliders guidance scale')
    parser.add_argument('--netural_prompt', type=str, default='a chair', help='neutral prompt')
    parser.add_argument('--positive_prompt', type=str, default='a chair with arms', help='positive prompt')
    parser.add_argument('--negative_prompt', type=str, default='a chair without arms', help='negative prompt')
    return parser.parse_args()
