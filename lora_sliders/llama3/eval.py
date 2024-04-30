import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
instruct = "Use only one of the following options, if none of them is suitable return Unknown. Options: a chair with a spindle backrest, a chair with a thick seat, a chair with short legs, a chair with two legs, a chair with a thin seat, a chair without a legs strecher, a chair with long legs, a chair with a rounded backrest, a chair with four legs, a chair with a curved backrest, a chair with a solid backrest, a chair without armrests, a chair with thin legs, a chair with armrests, a chair with a squared backrest, a chair with a legs strecher, a chair with thick legs, a chair with a straight backrest, a chair with a short backrest, a chair with a wide seat, a chair with a long backrest, a chair with a narrow seat."
input = "Can you rewrite please the following description of a chair? Description: {}."

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/scratch/noam/shapetalk/language/llama3/chair/lora_model", 
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
csv_path = '/scratch/noam/shapetalk/language/chair_test.csv'
df = pd.read_csv(csv_path)
df['llama3_uttarance'] = ""
for i, row in tqdm(df.iterrows(), total=len(df)):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruct,
                input.format(row["utterance"]),
                "",
            )
        ], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    llama3_uttarance = tokenizer.batch_decode(outputs)[0].split("\n")[-4]
    print(row["utterance"], ' -> ', llama3_uttarance)
    df.at[i, 'llama3_uttarance'] = llama3_uttarance
df.to_csv(csv_path, index=False)
