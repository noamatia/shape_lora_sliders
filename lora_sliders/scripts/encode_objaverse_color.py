import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import random
import objaverse
from tqdm import tqdm
from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
lvis_annotations = objaverse.load_lvis_annotations()
uids = []
for v in lvis_annotations.values():
    uids.extend(v)
random_uids = random.sample(uids, 1000)
objects = objaverse.load_objects(random_uids)
for uid, path in tqdm(objects.items(), total=len(objects)):
    for color in ['blue', 'red']:  
        batch = load_or_create_multimodal_batch(
                device,
                model_path=path,
                mv_light_mode="basic",
                mv_image_size=256,
                random_sample_count=2**17,
                color=color)
        latent = xm.encoder.encode_to_bottleneck(batch)
        torch.save(latent, os.path.join("datasets", "color", color, f'{uid}.pt'))
    