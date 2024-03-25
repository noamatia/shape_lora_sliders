import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
import wandb
import random
from utils import *
from tqdm import tqdm
from lora import LoRANetwork
from datetime import datetime
from diffusers.utils import export_to_gif
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

args = parse_args()      
name = f"{args.name}_visual_{str(datetime.now())}"
output_dir = os.path.join('outputs', name)
samples_dir = os.path.join(output_dir, 'samples')
models_dir = os.path.join(output_dir, 'models')
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
wandb.init(project=args.wandb_project, name=name, config=args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
model.wrapped.cond_drop_prob = args.cond_drop_prob
model.freeze_all_parameters()
network = LoRANetwork(model.wrapped, args.rank, args.alpha).to(device)
model.print_parameter_status()
network.print_parameter_status()
diffusion = diffusion_from_config(load_config('diffusion'))
optimizer = torch.optim.Adam(network.prepare_optimizer_params(), lr=args.lr)

folders = parse_arg(args.folders)
scales = parse_arg(args.scales, is_int=True)
assert folders.shape[0] == scales.shape[0], "The number of folders and scales must be the same."
assert scales.shape[0] == 2, "Currently the number of scales must be 2."
scale_high = abs(random.choice(list(scales)))
scale_low = -scale_high
folder_high = folders[scales==scale_high][0]
folder_low = folders[scales==scale_low][0]
latents = os.listdir(os.path.join(args.data_dir, folder_high))
latents = [latent for latent in latents if '.pt' in latent]

def load_latents(folder: str) -> list:
    lats = [torch.load(os.path.join(args.data_dir, folder, latent)).to(device) for latent in latents]
    lats = random.sample(lats, 18)
    lats = [lats[i:i + args.batch_size] for i in range(0, len(lats) - args.batch_size + 1, args.batch_size)]
    lats = [torch.cat(lats).detach().unsqueeze(1) for lats in lats]
    return lats

latents_high = load_latents(folder_high)
latents_low = load_latents(folder_low)
assert len(latents_high) == len(latents_low), "The number of high and low latents must be the same."
print(f"Loaded {len(latents_high)} latent files")
prompts = [args.prompt for _ in range(args.batch_size)]
model_kwargs = dict(texts=prompts)
test_model_kwargs = dict(texts=[args.prompt])
cameras = create_pan_cameras(args.size, device)

def train_step(scale: int, batch: torch.Tensor) -> int:
    network.set_lora_slider(scale)
    timesteps = torch.tensor(random.sample(range(args.num_timesteps), args.batch_size)).to(device).detach()
    with network:
        losses = diffusion.training_losses(model=model, x_start=batch, t=timesteps, model_kwargs=model_kwargs)
    loss = losses['loss'] / args.grad_acc_steps
    flush(timesteps)
    return loss

def backpropagate(loss: torch.Tensor, grad_acc_step: int):
    loss.backward()
    if grad_acc_step % args.grad_acc_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        
def test_step(scale: int, i: int, log_data: dict, sacle_type: str):
    network.set_lora_slider(scale)
    with network:
        test_latents = sample_latents(
            device=device,
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=args.guidance_scale,
            model_kwargs=test_model_kwargs,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
            progress=True
        )
    images = decode_latent_images(xm, test_latents[0], cameras, rendering_mode=args.render_mode)
    result_path = os.path.join(samples_dir, f'{i}_{sacle_type}.gif')
    log_data[f"output_{sacle_type}"] = wandb.Video(export_to_gif(images, result_path))
    flush(test_latents)

grad_acc_step, best_loss = 0, 1e9
pbar = tqdm(range(args.epochs))
for i in pbar:  
    loss_for_epoch_high, loss_for_epoch_low = 0, 0
    for batch_high, batch_low in tqdm(zip(latents_high, latents_low)):
        loss_high = train_step(scale_high, batch_high)
        loss_for_epoch_high += loss_high.item()
        grad_acc_step += 1
        backpropagate(loss_high, grad_acc_step)
        loss_low = train_step(scale_low, batch_low)
        loss_for_epoch_low += loss_low.item()
        grad_acc_step += 1
        backpropagate(loss_low, grad_acc_step)
    flush(latents_high, latents_low, prompts)
    log_data = {"loss_high": loss_for_epoch_high, "loss_low": loss_for_epoch_low}
    avg_loss_for_epoch = (loss_for_epoch_high + loss_for_epoch_low) / 2
    pbar.set_description(f"loss: {avg_loss_for_epoch:.4f}")
    if avg_loss_for_epoch < best_loss:
        best_loss = avg_loss_for_epoch
        network.save_weights(os.path.join(models_dir, f'model_best.pt'))
    if i % args.test_steps == 0:
        test_step(scale_high, i, log_data, "high")
        test_step(scale_low, i, log_data, "low")
    wandb.log(log_data)
network.save_weights(os.path.join(models_dir, f'model_final.pt'))
    