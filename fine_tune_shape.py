import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
import wandb
import random
from utils import *
from tqdm import tqdm
from datetime import datetime
from diffusers.utils import export_to_gif
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

args = parse_args()
name = f"fine_tune_shape_{str(datetime.now())}"
output_dir = os.path.join('outputs', name)
os.makedirs(output_dir, exist_ok=True)
wandb.init(project=args.wandb_project, name=name, config=args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
model.wrapped.cond_drop_prob = args.cond_drop_prob
model.freeze_all_parameters()
model.unfreeze_transformer_backbone()
diffusion = diffusion_from_config(load_config('diffusion'))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

latents = [lat for lat in os.listdir(args.data_dir) if '.pt' in lat]
assert len(latents) == 1, "There must be only one latent file in the folder"
latents = torch.cat([torch.load(os.path.join(args.data_dir, latents[0])) for _ in range(args.batch_size)]).to(device).detach()
latents = latents.unsqueeze(1)
prompts = [args.prompt for _ in range(args.batch_size)]
model_kwargs = dict(texts=prompts)
test_model_kwargs = dict(texts=[args.prompt])
cameras = create_pan_cameras(args.size, device)

pbar = tqdm(range(args.epochs))
for i in pbar:  
    loss_for_epoch = 0
    for _ in tqdm(range(args.grad_acc_steps)):
        timesteps = torch.tensor(random.sample(range(args.num_timesteps), args.batch_size)).to(device).detach()
        losses = diffusion.training_losses(model=model, x_start=latents, t=timesteps, model_kwargs=model_kwargs)
        loss = losses['loss'] / args.grad_acc_steps
        loss.backward()
        loss_for_epoch += loss.item()
        flush(latents, prompts, timesteps)
    optimizer.step()
    model.zero_grad()
    log_data = {"loss": loss_for_epoch}
    pbar.set_description(f"loss: {loss_for_epoch:.4f}")
    if i % args.test_steps == 0:
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
        result_path = os.path.join(output_dir, f'{i}.gif')
        log_data["output"] = wandb.Video(export_to_gif(images, result_path))
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{i}.pt'))
    wandb.log(log_data)
torch.save(model.state_dict(), os.path.join(output_dir, f'model_final.pt'))
    