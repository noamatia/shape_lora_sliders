import wandb
from utils import *
from tqdm import tqdm
from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import create_pan_cameras
from shap_e.diffusion.gaussian_diffusion import mean_flat

    
name, paths = build_name_and_paths()
wandb.init(project=args.wandb_project, name=name, config=vars(args))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
xm, model, network, diffusion, optimizer = build_models(device)
cameras = create_pan_cameras(args.size, device)

folder_high, folder_low, prompt_high, prompt_low, scale_high, scale_low = build_folders_prompts_and_scales()
print(f"Folder high: {folder_high}, Folder low: {folder_low}")
print(f"Prompt high: {prompt_high}, Prompt low: {prompt_low}")
print(f"Scale high: {scale_high}, Scale low: {scale_low}")

latents = build_latents(folder_high)
wandb.config["dataset_size"] =  len(latents)
wandb.config["latents_ids"] =  latents
latents_high, latents_low = load_latents(folder_high, device, latents), load_latents(folder_low, device, latents)
assert len(latents_high) == len(latents_low), "The number of high and low latents must be the same."
print(f"Loaded {len(latents)} latent files in {len(latents_high)} batches.")
model_kwargs_high_list, test_model_kwargs_high =  build_model_kwargs(prompt_high, latents, "high")
model_kwargs_low_list, test_model_kwargs_low =  build_model_kwargs(prompt_low, latents, "low")
masks = build_masks(latents_high, latents_low)
# sanity_check(latents_high, latents_low, masks, cameras, xm)

sigma_max = 160
x_T_test = torch.randn((1, model.d_latent), device=device).expand(1, -1) * sigma_max
torch.save(x_T_test, paths['x_T_test'])
eps_test = torch.randn_like(x_T_test)
torch.save(eps_test, paths['eps_test'])

def train_step(scale: int, batch: torch.Tensor, timesteps: torch.Tensor, model_kwargs: dict, noise: torch.Tensor) -> int:
    network.set_lora_slider(scale)
    with network:
        losses = diffusion.training_losses(model, batch, timesteps, model_kwargs, noise)
    loss = losses['loss'] / args.grad_acc_steps
    return loss, losses['output']
        
def test_step(scale: int, i: int, log_data: dict, sacle_type: str, test_model_kwargs: dict):
    test_latents = sample_latents(model=model, 
                                    diffusion=diffusion, 
                                    model_kwargs=test_model_kwargs, 
                                    guidance_scale=args.guidance_scale, 
                                    device=device, 
                                    progress=True, 
                                    sigma_max=sigma_max, 
                                    x_T=x_T_test,
                                    eps=eps_test,
                                    network=network,
                                    scale=scale,
                                    t_start=0)
    log_data[f"output_{sacle_type}"] = wandb.Video(decode_and_export(test_latents[0], paths['samples'], f'{i}_{sacle_type}.gif', cameras, xm))

grad_acc_step, best_loss = 0, 1e9
pbar = tqdm(range(args.epochs))
for i in pbar:  
    loss_for_epoch_high, loss_for_epoch_low, loss_for_epoch_masked_diff = 0, 0, 0
    for batch_high, batch_low, mask, model_kwargs_high, model_kwargs_low in tqdm(zip(latents_high, latents_low, masks, model_kwargs_high_list, model_kwargs_low_list)):
        timesteps = torch.tensor(random.sample(range(args.num_timesteps), args.batch_size)).to(device).detach()
        noise = torch.randn_like(batch_high)
        
        loss_high, output_high = train_step(scale_high, batch_high, timesteps, model_kwargs_high, noise)
        loss_for_epoch_high += loss_high.item()
        loss_low, output_low = train_step(scale_low, batch_low, timesteps, model_kwargs_low, noise)
        loss_for_epoch_low += loss_low.item()
        total_loss = args.beta * (loss_high + loss_low)
        
        masked_diff_loss = mean_flat((mask * (output_high - output_low)) ** 2).mean() / args.grad_acc_steps
        loss_for_epoch_masked_diff += masked_diff_loss.item()
        total_loss += (1 - args.beta) * masked_diff_loss
        
        grad_acc_step += 1
        total_loss.backward()
        if grad_acc_step % args.grad_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
    log_data = {"loss_high": loss_for_epoch_high, "loss_low": loss_for_epoch_low, "loss_masked_diff": loss_for_epoch_masked_diff}
    avg_loss_for_epoch = (loss_for_epoch_high + loss_for_epoch_low + loss_for_epoch_masked_diff) / 3
    pbar.set_description(f"loss: {avg_loss_for_epoch:.4f}")
    if avg_loss_for_epoch < best_loss:
        best_loss = avg_loss_for_epoch
        network.save_weights(paths['model_best'])
    if i % args.test_steps == 0:
        test_step(scale_high, i, log_data, "high", test_model_kwargs_high)
        test_step(scale_low, i, log_data, "low", test_model_kwargs_low)
    wandb.log(log_data)
network.save_weights(paths['model_final'])
    