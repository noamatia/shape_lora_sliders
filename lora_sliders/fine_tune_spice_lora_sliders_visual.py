from utils import *
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

latents_train = build_latents('train', folder_high)
latents_test = build_latents('test', folder_high)
latents_high_train, latents_low_train = load_latents('train', folder_high, device, latents_train), load_latents('train', folder_low, device, latents_train)
latents_high_test, latents_low_test = load_latents('test', folder_high, device, latents_test), load_latents('test', folder_low, device, latents_test)
assert len(latents_high_train) == len(latents_low_train), "The number of high and low latents must be the same."
assert len(latents_high_test) == len(latents_low_test), "The number of high and low latents must be the same."
print(f"Loaded {len(latents_train)} train latent files in {len(latents_high_train)} batches.")
print(f"Loaded {len(latents_test)} test latent files in {len(latents_high_test)} batches.")


model_kwargs_high_train_list =  build_model_kwargs_list(prompt_high, latents_train, "train", "high", latents_low_train)
model_kwargs_high_test_list =  build_model_kwargs_list(prompt_high, latents_test, "test", "high", latents_low_test)
model_kwargs_low_train_list =  build_model_kwargs_list(prompt_low, latents_train, "train", "low", latents_high_train)
model_kwargs_low_test_list =  build_model_kwargs_list(prompt_low, latents_test, "test", "low", latents_high_test)
decode_and_export_test_conditions(model_kwargs_high_test_list, model_kwargs_low_test_list, paths, cameras, xm)
masks_train = build_masks(latents_high_train, latents_low_train)
# masks_test = build_masks(latents_high_test, latents_low_test)
# sanity_check(latents_high_train, latents_low_train, masks_train, cameras, xm, 'train')
# sanity_check(latents_high_test, latents_low_test, masks_test, cameras, xm, 'test')

sigma_max = 160
x_T_test = torch.randn((1, model.d_latent), device=device).expand(1, -1) * sigma_max
torch.save(x_T_test, paths['x_T_test'])
eps_test = torch.randn_like(x_T_test)
torch.save(eps_test, paths['eps_test'])

def train_step(scale: int, batch: torch.Tensor, timesteps: torch.Tensor, model_kwargs: dict, noise: torch.Tensor) -> int:
    try:
        network.set_lora_slider(scale)
        with network:
            losses = diffusion.training_losses(model, batch, timesteps, model_kwargs, noise)
        loss = losses['loss'] / args.grad_acc_steps
        return loss, losses['output']
    except:
        pass
        
def test_step(scale: int, epoch: int, log_data: dict, high_or_low: str, model_kwargs_test_list: list):
    for test_model_kwargs in tqdm(model_kwargs_test_list, total=len(model_kwargs_test_list), desc=f"Testing {high_or_low} epoch {epoch}"):
        test_latents = sample_latents(model=model, 
                                        diffusion=diffusion, 
                                        model_kwargs=test_model_kwargs, 
                                        guidance_scale=args.guidance_scale, 
                                        device=device, 
                                        progress=True, 
                                        sigma_max=sigma_max, 
                                        x_T=x_T_test.repeat(args.batch_size, 1),
                                        eps=eps_test,
                                        network=network,
                                        scale=scale,
                                        t_start=0,
                                        batch_size=args.batch_size)
        for i, test_latent in enumerate(test_latents):
            uid = test_model_kwargs['uid'][i]
            uid_output_dir = os.path.join(paths['samples'], uid, high_or_low)
            log_data[f"output_{uid}_{high_or_low}"] = wandb.Video(decode_and_export(test_latent, uid_output_dir, f'output_{epoch}.gif', cameras, xm))

grad_acc_step = 0
pbar = tqdm(range(args.epochs))
for i in pbar:  
    loss_for_epoch_high, loss_for_epoch_low, loss_for_epoch_masked_diff = 0, 0, 0
    for batch_high, batch_low, mask, model_kwargs_high, model_kwargs_low in tqdm(zip(latents_high_train, latents_low_train, masks_train, model_kwargs_high_train_list, model_kwargs_low_train_list)):
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
    if i % args.test_steps == 0:
        test_step(scale_high, i, log_data, "high", model_kwargs_high_test_list)
        test_step(scale_low, i, log_data, "low", model_kwargs_low_test_list)
        network.save_weights(os.path.join(paths['models'], f'model_{i}.pt'))
    wandb.log(log_data)
network.save_weights(os.path.join(paths['models'], 'model_final.pt'))
    