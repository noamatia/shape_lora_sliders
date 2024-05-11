import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
# from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
# chamfer_raw = dist_chamfer_3D.chamfer_3DDist()

# def chamfer_loss(pc_a, pc_b, swap_axes=False, reduction='mean'):
#     """Compute the chamfer loss for batched pointclouds.
#         :param pc_a: torch.Tensor B x Na-points per point-cloud x 3
#         :param pc_b: torch.Tensor B x Nb-points per point-cloud x 3
#         :return: B floats, indicating the chamfer distances when reduction is mean, else un-reduced distances
#     """

#     n_points_a = pc_a.shape[1]
#     n_points_b = pc_b.shape[1]

#     if swap_axes:
#         pc_a = pc_a.transpose(-1, -2).contiguous()
#         pc_b = pc_b.transpose(-1, -2).contiguous()

#     dist_a, dist_b, _, _ = chamfer_raw(pc_a, pc_b)

#     if reduction == 'mean':
#         # reduce separately, sizes of points can be different
#         dist = ((n_points_a * dist_a.mean(1)) + (dist_b.mean(1) * n_points_b)) / (n_points_a + n_points_b)
#     elif reduction is None:
#         return dist_a, dist_b
#     else:
#         raise ValueError('Unknown reduction rule.')
#     return dist

obj = 'chair'
df = pd.read_csv(os.path.join('/scratch/noam', 'shapetalk', 'language', f'{obj}_train.csv'))

# sums = []
# for _, row in tqdm(df.iterrows(), total=len(df)):
#     if row['llama3_uttarance'] != 'Unkown.':
#         source_pc_path = os.path.join('/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering', row['source_uid'] + '.npz') 
#         target_pc_path = os.path.join('/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering', row['target_uid'] + '.npz')
#         source_pc = torch.Tensor(np.load(source_pc_path)['pointcloud']).unsqueeze(0).to(device)
#         target_pc = torch.Tensor(np.load(target_pc_path)['pointcloud']).unsqueeze(0).to(device)
#         cd = chamfer_loss(source_pc, target_pc).item()
#         print(cd)
#         sums.append(cd)
#     else:
#         sums.append(np.nan)
# df['chamfer_distance'] = sums
# df.to_csv(os.path.join('/scratch/noam', 'shapetalk', 'language', f'{obj}_train.csv'), index=False)
# exit()

with open(f'/home/noamatia/repos/shape_lora_sliders/lora_sliders/llama3/{obj}.json', 'r') as f:
    prompt_pairs = json.load(f)
output_dir = f'/scratch/noam/shape_lora_sliders_datasets/{obj}'
os.makedirs(output_dir, exist_ok=True)
for prompt_pair in prompt_pairs:
    pair_df = df[df['llama3_uttarance'].isin([prompt_pair[0] + '.', prompt_pair[1] + '.'])]
    pair_df = pair_df.sort_values(by='chamfer_distance')
    pair_df = pair_df.head(200)
    prompt_low, prompt_high = prompt_pair[0].replace(' ', '_'), prompt_pair[1].replace(' ', '_')
    prompt_pair_name = f"{prompt_low}_{prompt_high}"
    print(prompt_pair_name, len(pair_df))
    pair_output_dir = os.path.join(output_dir, prompt_pair_name)
    pair_df.to_csv(os.path.join(pair_output_dir, 'data.csv'), index=False)
    low_output_dir = os.path.join(pair_output_dir, prompt_low)
    low_images_dir = os.path.join(low_output_dir, 'images')
    high_output_dir = os.path.join(pair_output_dir, prompt_high)
    os.makedirs(low_images_dir, exist_ok=True)
    high_images_dir = os.path.join(high_output_dir, 'images')
    os.makedirs(high_images_dir, exist_ok=True)
    html_high = '<table>\n'
    html_high += '<tr><td>original prompt</td><td>llama 3 prompt</td><td>source</td><td>target</td><td>uid</td><td>chamfer distance</td><td>l2 distance</td></tr>\n'
    html_low = '<table>\n'
    html_low += '<tr><td>original prompt</td><td>llama 3 prompt</td><td>source</td><td>target</td><td>uid</td><td>chamfer distance</td><td>l2 distance</td></tr>\n'
    for i, row in pair_df.iterrows():
        source_uid = row['source_uid']
        target_uid = row['target_uid']
        source_image_path = os.path.join('/scratch/noam/shapetalk/images/full_size', source_uid + '.png')
        target_image_path = os.path.join('/scratch/noam/shapetalk/images/full_size', target_uid + '.png')
        if row['llama3_uttarance'] == prompt_pair[0] + '.':
            source_new_image_path = os.path.join(low_images_dir, source_uid.replace('/', '_') + '.png')
            target_new_image_path = os.path.join(low_images_dir, target_uid.replace('/', '_') + '.png')
            html_low += f"<tr><td>{row['utterance']}</td><td>{row['llama3_uttarance']}</td><td><img src='images/{source_uid.replace('/', '_')}.png' width='200'></td><td><img src='images/{target_uid.replace('/', '_')}.png' width='200'></td><td>{source_uid.replace('/', '_')}_{target_uid.replace('/', '_')}</td><td>{row['chamfer_distance']}</td><td>{row['l2_distance']}</td></tr>\n"
        else:
            source_new_image_path = os.path.join(high_images_dir, source_uid.replace('/', '_') + '.png')
            target_new_image_path = os.path.join(high_images_dir, target_uid.replace('/', '_') + '.png')
            html_high += f"<tr><td>{row['utterance']}</td><td>{row['llama3_uttarance']}</td><td><img src='images/{source_uid.replace('/', '_')}.png' width='200'></td><td><img src='images/{target_uid.replace('/', '_')}.png' width='200'></td><td>{source_uid.replace('/', '_')}_{target_uid.replace('/', '_')}</td><td>{row['chamfer_distance']}</td><td>{row['l2_distance']}</td></tr>\n"
        shutil.copy(source_image_path, source_new_image_path)
        shutil.copy(target_image_path, target_new_image_path)
    html_low += '</table>'
    html_high += '</table>'
    with open(os.path.join(low_output_dir, 'index.html'), 'w') as f:
        f.write(html_low)
    with open(os.path.join(high_output_dir, 'index.html'), 'w') as f:
        f.write(html_high)
    