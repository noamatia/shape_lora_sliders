import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
import wandb
import random
from lora_sliders.trainning.utils import *
from tqdm import tqdm
from lora_sliders.lora import LoRANetwork
from datetime import datetime
from diffusers.utils import export_to_gif
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images


args = parse_args() 
name = f"fine_tune_shape_lora_{str(datetime.now())}"
output_dir = os.path.join('outputs', name)
os.makedirs(output_dir, exist_ok=True)
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

latents = []
for root, dirs, files in os.walk(args.data_dir):
    for file in files:
        if file.endswith(".pt"):
            latents.append(torch.load(os.path.join(root, file)).to(device))
print(f"Loaded {len(latents)} latent files")
latents = [latents[i:i + args.batch_size] for i in range(0, len(latents) - args.batch_size + 1, args.batch_size)]
latents = [torch.cat(latent).detach().unsqueeze(1) for latent in latents]
prompts = [args.prompt for _ in range(args.batch_size)]
model_kwargs = dict(texts=prompts)
test_model_kwargs = dict(texts=[args.prompt])
cameras = create_pan_cameras(args.size, device)

pbar = tqdm(range(args.epochs))
for i in pbar:  
    loss_for_epoch, grad_acc_step = 0, 0
    grad_acc_step = 0
    while grad_acc_step < args.grad_acc_steps:
        for batch in tqdm(latents):
            timesteps = torch.tensor(random.sample(range(args.num_timesteps), args.batch_size)).to(device).detach()
            with network:
                losses = diffusion.training_losses(model=model, x_start=batch, t=timesteps, model_kwargs=model_kwargs)
            loss = losses['loss'] / args.grad_acc_steps
            loss.backward()
            loss_for_epoch += loss.item()
            grad_acc_step += 1
            flush(timesteps)
            if grad_acc_step % args.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    flush(latents, prompts)
    log_data = {"loss": loss_for_epoch}
    pbar.set_description(f"loss: {loss_for_epoch:.4f}")
    if i % args.test_steps == 0:
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
        result_path = os.path.join(output_dir, f'{i}.gif')
        log_data["output"] = wandb.Video(export_to_gif(images, result_path))
        network.save_weights(os.path.join(output_dir, f'model_epoch_{i}.pt'))
    wandb.log(log_data)
network.save_weights(os.path.join(output_dir, f'model_final.pt'))
    