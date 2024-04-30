import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import random
from tqdm import tqdm
from datetime import datetime
from lora_sliders.lora import LoRANetwork
from diffusers.utils import export_to_gif
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

n = 20
rank = 4
size = 160
prompt = ""
alpha = 1.0
sigma_max = 160
scales = [-1, 1]
render_mode = "nerf"
cond_drop_prob = 0.5
guidance_scale = 7.5
name = "armsslider_masking_percentile_90_beta_0_75_02_04_2024_00_03_58"
base_dir = "/home/noamatia/repos/shape_lora_sliders/lora_sliders/outputs"
lora_weight = os.path.join(base_dir, name, "model_best.pt")
output_dir = os.path.join(base_dir, name, 'test', str(datetime.now()))
os.makedirs(output_dir, exist_ok=True)

def flush(*args):
    for arg in args:
        del arg
    torch.cuda.empty_cache()
    gc.collect()
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
model.wrapped.cond_drop_prob = cond_drop_prob
model.freeze_all_parameters()
network = LoRANetwork(model.wrapped, rank, alpha).to(device)
network.load_state_dict(torch.load(lora_weight))
diffusion = diffusion_from_config(load_config('diffusion'))
test_model_kwargs = dict(texts=[prompt])
cameras = create_pan_cameras(size, device)

for i in tqdm(range(n), total=n):
    seed = random.randint(0, 5000)
    x_T = torch.randn((1, model.d_latent), device=device).expand(1, -1) * sigma_max
    for scale in scales:
        network.set_lora_slider(scale)
        with network:
            with torch.no_grad():
                test_latents = sample_latents(
                    device=device,
                    batch_size=1,
                    model=model,
                    diffusion=diffusion,
                    guidance_scale=guidance_scale,
                    model_kwargs=test_model_kwargs,
                    clip_denoised=True,
                    use_fp16=True,
                    use_karras=True,
                    karras_steps=64,
                    sigma_min=1e-3,
                    sigma_max=sigma_max,
                    s_churn=0,
                    progress=True,
                    x_T=x_T,
                    # network=network,
                    # scale=scale,
                )
        images = decode_latent_images(xm, test_latents[0], cameras, rendering_mode=render_mode)
        result_path = os.path.join(output_dir, f'example_{i}_scale_{scale}.gif')
        export_to_gif(images, result_path)
        flush(test_latents)