"""
Prepare Shakespeare dataset tokenized with DeepSeek-V3 tokenizer.
Produces train.bin, val.bin, and meta.pkl compatible with nanoGPT's get_batch.
"""
import os
import pickle
import requests
import numpy as np
from transformers import AutoTokenizer

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    print(f"downloading {data_url}")
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# Train/val split first, then tokenize
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# Load DeepSeek-V3 tokenizer
print("loading DeepSeek-V3 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-V3", trust_remote_code=True
)
print(f"tokenizer vocab size: {len(tokenizer):,}")

# Tokenize
train_ids = tokenizer.encode(train_data)
val_ids = tokenizer.encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files (int32 for DeepSeek-V3's larger vocab 129280)
train_ids = np.array(train_ids, dtype=np.int32)
val_ids = np.array(val_ids, dtype=np.int32)
out_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(out_dir, 'train.bin'))
val_ids.tofile(os.path.join(out_dir, 'val.bin'))

# Save meta information
meta = {
    'vocab_size': len(tokenizer),
    'tokenizer_name_or_path': 'deepseek-ai/DeepSeek-V3',
    'dtype': 'int32',  # train_engram.py reads the .bin files with this dtype
}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"saved to {out_dir}")
print(f"  train.bin: {len(train_ids):,} tokens")
print(f"  val.bin: {len(val_ids):,} tokens")
print(f"  meta.pkl: vocab_size={meta['vocab_size']}, tokenizer={meta['tokenizer_name_or_path']}")