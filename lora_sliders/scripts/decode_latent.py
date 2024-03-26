import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
import wandb
import random
from lora_sliders.trainning.utils import *
from tqdm import tqdm
from datetime import datetime
from diffusers.utils import export_to_gif
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
test_latents = torch.load('/home/noamatia/repos/shape_lora_sliders/datasets/color/red/8bd6af15b49f4abc8f539c1ca4efeb2d.pt').to(device).detach()
cameras = create_pan_cameras(256, device)
images = decode_latent_images(xm, test_latents[0], cameras, rendering_mode='nerf')
export_to_gif(images, 'test.gif')