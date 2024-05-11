import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import tqdm
import torch
import numpy as np
from sklearn.decomposition import PCA
from shap_e.models.download import load_model
from jinja2 import Environment, FileSystemLoader
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

pairs = ['1eb1a8acd4185f49492d9da2668ec34c_341c9b0583d31770492d9da2668ec34c', 'c56bca4b7f353ad86debb0a33c851f8_fd9c60e969f83e6086debb0a33c851f8',
         '4c6c364af4b52751ca6910e4922d61aa_35e8b034d84f60cb4d226702c1bfe9e2', 'd107532567ee3f316663d1953862c637_9e519ddc82bb9417813635269a32e293',
         '55e6d1251fc63e1f71a782a4379556c7_ed948a3d2ece4b3b71a782a4379556c7', 'd38129a3301d31350b1fc43ca5e85e_cee98619dbdb0e34c5fc2b846c38d941',
         '64e77e9e5887ce95492d9da2668ec34c_bc61ea1b9348f456492d9da2668ec34c', 'd89e39192db6fa78492d9da2668ec34c_1eb1a8acd4185f49492d9da2668ec34c',
         '75ea5a697313a8c214c2f69de20984ee_a67a09662c39430bc8687ff9b0b4e4ac', 'e564f393acf979683c2e50348f23d3d_5c9b4af9d0e9c132b161f36d4e309050',
         '8c4ffe44076e9d4a15f62f0f1afbe530_d69aad24d253474dc984897483a49e2b', 'ea04a5915e3982aad7f7a4c4609b0913_675aaa5b883e2398d7f7a4c4609b0913',
         '969375970515e5f6492d9da2668ec34c_ef03458b97c8775b492d9da2668ec34c', 'ea3723766e96331ff91663a74ccd2338_bdd51e6d7ff84be7492d9da2668ec34c',
         '9c7b2ed3770d1a6ea6fee8e2140acec9_24bbe7f32727901aa6fee8e2140acec9', 'f4268a28d2a837a1167c009da6daa010_e4cc5cee7b011df316037b4c09d66880',
         'bdaaebf065b112da492d9da2668ec34c_8ad35dbc44f40be1492d9da2668ec34c', 'f4a2478ebfac3e56b0957d845ac33749_ca23c31de817db9b67981fccd6325b88',
         'c4ebef05a72fc4f39d62eb3fdc2d3f8a_fc3d4268406b396e71a782a4379556c7', 'fbca73a2c226a86a593a4d04856c4691_f199965dc6746de38b01ef724ff374fa']

b = 2
size = 5
n_components = 64
target_max = size
target_min = size * -1
num_selectred_entries = 8
products = [1] * (2 * size + 1)
for i in range(1, size + 1):
    products[size - i] /= b
    products[size + i] *= b
    b += 1
print('products', products)
offsets = np.arange(target_min, target_max + 1).tolist()
print('offsets', offsets)

num_chunks = 4
rows, cols = 1024, 1024
chunk_size = rows//num_chunks
base_dir = '/scratch/noam/shapetalk/latents'
output_dir = '/home/noamatia/repos/shape_lora_sliders/pca_html'
shapenet_dir = os.path.join(base_dir, 'chair', 'ShapeNet')
modelnet_dir = os.path.join(base_dir, 'chair', 'ModelNet', 'chair')
uids = [os.path.join('chair', 'ShapeNet', f) for f in os.listdir(shapenet_dir)] + [os.path.join('chair', 'ModelNet', 'chair', f) for f in os.listdir(modelnet_dir)]
selected_uids = []
for ab in pairs:
    a, b = ab.split('_')[0], ab.split('_')[1]
    selected_uids += [uid for uid in uids if a in uid or b in uid]
print('selected_uids', selected_uids)
    
chunks = None
latents = None
for uid in tqdm.tqdm(uids, total=len(uids)):
    cur_latent = torch.load(os.path.join(base_dir, uid, 'latent.pt')).cpu()
    cur_chunk = cur_latent.reshape(rows, cols)[:chunk_size, :].reshape(1, rows * chunk_size)
    if chunks is None:
        chunks = cur_chunk
        latents = cur_latent
    else:
        chunks = torch.cat((chunks, cur_chunk))
        latents = torch.cat((latents, cur_latent))
chunks = chunks.numpy() 
latents = latents.numpy()
print('chunks shape', chunks.shape)
print('latents shape', latents.shape)

pca = PCA(n_components=n_components)
chunks_transformed = pca.fit_transform(chunks)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
cameras = create_pan_cameras(256, device)

state_to_images = {}
for uid in tqdm.tqdm(selected_uids, total=len(selected_uids), desc=f'uid: {uid}'):
    uid_image_dir = os.path.join(output_dir, 'images', uid)
    os.makedirs(uid_image_dir, exist_ok=True)
    uid_latnt_dir = os.path.join(output_dir, 'latents', uid)
    os.makedirs(uid_latnt_dir, exist_ok=True)
    uid_chunk_transformed = chunks_transformed[uids.index(uid)][np.newaxis, :]
    uid_latent = latents[uids.index(uid)][np.newaxis, :]
    uid_chunk_path = os.path.join(uid_latnt_dir, 'chunk.pt')
    torch.save(uid_chunk_transformed, uid_chunk_path)
    for i in tqdm.tqdm(range(num_selectred_entries), total=num_selectred_entries):
        for j, offset in tqdm.tqdm(enumerate(offsets), total=len(offsets)):
            if j == size and i > 0:
                continue
            image_path = os.path.join(uid_image_dir, str(i) + '_' + str(j) + '.png')
            state = [0] * num_selectred_entries
            state[i] = offset
            uid_chunk_transformed_copy = uid_chunk_transformed.copy()
            uid_chunk_transformed_copy[0][i] *= products[j]
            uid_chunk = pca.inverse_transform(uid_chunk_transformed_copy)
            uid_latent_to_decode = uid_latent.reshape(rows, cols)
            uid_latent_to_decode[:chunk_size, :] = uid_chunk.reshape(chunk_size, cols)
            uid_latent_to_decode = uid_latent_to_decode.reshape(1, rows * cols)
            images = decode_latent_images(xm, torch.from_numpy(uid_latent).float().to(device), cameras, rendering_mode='nerf')
            images[7].save(image_path)
            curr = state_to_images
            for k in range(num_selectred_entries):
                curr = curr.setdefault(int(state[k]), {} if k < num_selectred_entries - 1 else [])
            curr.append(os.path.join('images', image_path))
    
template_loader = FileSystemLoader(searchpath='templates')
env = Environment(loader=template_loader)
template = env.get_template("pca.html")
output_html = template.render(
    num_entries=num_selectred_entries,
    target_min=target_min,
    target_max=target_max,
    state_to_images=state_to_images
)

with open(os.path.join(output_dir, 'pca.html'), "w") as f:
    f.write(output_html)
