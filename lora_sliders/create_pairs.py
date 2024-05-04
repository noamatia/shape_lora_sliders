import os
import json
import shutil
import pandas as pd

n = 200
obj = 'chair'
base_dir = '/scratch/noam'
output_dir = os.path.join('/scratch/noam/shape_lora_sliders_datasets', obj)
df = pd.read_csv(os.path.join(base_dir, 'shapetalk', 'language', f'{obj}_train.csv'))
with open(f'/home/noamatia/repos/shape_lora_sliders/lora_sliders/llama3/{obj}.json', 'r') as f:
    prompt_pairs = json.load(f)
    
for prompt_pair in prompt_pairs:
    prompt_low, prompt_high = prompt_pair[0], prompt_pair[1]
    prompt_pair_name = f"{prompt_low.replace(' ', '_')}_{prompt_high.replace(' ', '_')}"
    prompt_pair_output_dir = os.path.join(output_dir, prompt_pair_name)
    os.makedirs(prompt_pair_output_dir, exist_ok=True)
    prompt_pair_source_output_dir = os.path.join(prompt_pair_output_dir, 'source')
    os.makedirs(prompt_pair_source_output_dir, exist_ok=True)
    prompt_pair_target_output_dir = os.path.join(prompt_pair_output_dir, 'target')
    os.makedirs(prompt_pair_target_output_dir, exist_ok=True)
    prompt_pair_df = df[df['llama3_uttarance'].isin([prompt_low + '.', prompt_high + '.'])]
    prompt_pair_df = prompt_pair_df.sort_values('l2_distance')
    prompt_pair_df = prompt_pair_df.head(n)
    prompt_pair_df.to_csv(os.path.join(prompt_pair_output_dir, 'data.csv'), index=False)
    html = '<table>\n'
    html += '<tr><td>original prompt</td><td>llama 3 prompt</td><td>source</td><td>target</td><td>uid</td></tr>\n'
    for _, row in prompt_pair_df.iterrows():
        source_image_path = os.path.join(base_dir, 'shapetalk', 'images', 'full_size', row['source_uid'] + '.png')
        target_image_path = os.path.join(base_dir, 'shapetalk', 'images', 'full_size', row['target_uid'] + '.png')
        source_new_image_path = os.path.join(prompt_pair_source_output_dir, row['source_uid'].replace('/', '_') + '.png')
        target_new_image_path = os.path.join(prompt_pair_target_output_dir, row['target_uid'].replace('/', '_') + '.png')
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, source_new_image_path)
        if os.path.exists(target_image_path):
            shutil.copy(target_image_path, target_new_image_path)
        source_image_local_path = os.path.join('source', row['source_uid'].replace('/', '_') + '.png')
        target_image_local_path = os.path.join('target', row['target_uid'].replace('/', '_') + '.png')
        uid = row['source_uid'].replace('/', '_') + '_' + row['target_uid'].replace('/', '_')
        html += f"<tr><td>{row['utterance']}</td><td>{row['llama3_uttarance']}</td><td><img src='{source_image_local_path}' width='200'></td><td><img src='{target_image_local_path}' width='200'></td><td>{uid}</td></tr>\n"
    html += '</table>'
    with open(os.path.join(prompt_pair_output_dir, 'index.html'), 'w') as f:
        f.write(html)