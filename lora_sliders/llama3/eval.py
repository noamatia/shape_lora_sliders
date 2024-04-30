import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel

obj = 'chair'
with open(f'/home/noamatia/repos/shape_lora_sliders/lora_sliders/llama3/{obj}.json', 'r') as f:
    options = json.load(f)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
instruct = f"Use only one of the following options, if none of them is suitable return Unknown. Options: {', '.join(options)}."
input = "Can you rewrite please the following description of a {}? Description: {}."

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"/scratch/noam/shapetalk/language/llama3/{obj}/lora_model", 
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
csv_path = f'/scratch/noam/shapetalk/language/{obj}_test.csv'
df = pd.read_csv(csv_path)
df['llama3_uttarance'] = ""
for i, row in tqdm(df.iterrows(), total=len(df)):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruct,
                input.format(obj, row["utterance"]),
                "",
            )
        ], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    llama3_uttarance = tokenizer.batch_decode(outputs)[0].split("\n")[-4]
    print(row["utterance"], ' -> ', llama3_uttarance)
    df.at[i, 'llama3_uttarance'] = llama3_uttarance
df.to_csv(csv_path, index=False)
