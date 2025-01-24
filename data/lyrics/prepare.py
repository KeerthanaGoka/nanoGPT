import os
import requests
import tiktoken
import numpy as np

############################# Lyrics Dataset - Start
import pandas as pd
df = pd.read_csv('data/lyrics/spotify_millsongdata.csv')
data = df['text'].str.cat(sep='\n')
############################# Lyrics Dataset - End

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
# Print the first 10 tokens from train_ids
print("First 10 tokens in train_ids:", train_ids[:10])

# Decode the first 10 tokens back to text using the TikToken encoder
decoded_text = enc.decode(train_ids[:10].tolist())
print("Decoded text for the first 10 tokens in train_ids:", decoded_text)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
