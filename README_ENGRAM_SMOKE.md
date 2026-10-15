# Engram Smoke Test

This directory contains a minimal end-to-end test for Engram integration with nanoGPT that runs in seconds on modest hardware.

## Purpose
Verify that:
1. Engram memory tables are properly wired into the GPT model
2. Forward/backward pass works (tables get gradients)
3. Checkpoint saves/loads Engram configuration correctly
4. Tokenizer alignment works (corpus IDs match Engram's expectations)

## Files
- `run_smoke_test.bat` - Windows batch file to run the test
- `train_engram.py` - Modified training script with Engram support
- `model_engram.py` - GPT subclass with Engram injection
- `engram.py` - EngramMemory module ported from upstream

## Prerequisites
1. Shakespeare data tokenized with DeepSeek-V3 tokenizer:
   ```powershell
   cd data/shakespeare_char
   ..\..\..\.venv\Scripts\python.exe prepare_deepseek.py
   ```
2. `uv run python --version` works (shows Python 3.14+)

## How to Run
From `third_party/nanoGPT`:
```powershell
run_smoke_test.bat
```

Or manually:
```powershell
uv run python train_engram.py config/train_shakespeare_char.py ^
  --compile=False --max_iters=2 --eval_interval=1 ^
  --out_dir=out-engram-tiny ^
  --engram_vocab_size_per_ngram="[500, 500]" --engram_n_embed_per_ngram=64 ^
  --engram_n_head_per_ngram=4 --engram_layer_ids="[1, 2]" ^
  --n_layer=3 --n_embd=128
```

## Expected Output
```
building Engram memory on layers [1, 2]
tokenizer compression 128815 -> ~99240 (23.43% merged)
number of parameters: 0.60M
num decayed parameter tensors: 14, with 606,336 parameters
num non-decayed parameter tensors: 7, with 896 parameters
num engram table tensors: 2, with N parameters
using fused AdamW: False
iter 1: loss 4.1234, time 12.5ms, mfu 12.3%
iter 2: loss 4.0123, time 11.8ms, mfu 15.7%
saving checkpoint to out-engram-tiny
```

## Verification
After running, check the checkpoint contains Engram config:
```powershell
uv run python -c "
import torch
ckpt = torch.load('out-engram-tiny\ckpt.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('Has engram config?', 'engram' in ckpt['model_args'])
if 'engram' in ckpt['model_args']:
    print('Engram layers:', ckpt['model_args']['engram']['layer_ids'])
    print('Table sizes:', ckpt['model_args']['engram']['vocab_size_per_ngram'])
"
```

## What This Tests
- **Forward/backward**: Embedding gradients non-zero (tables train)
- **Optimizer groups**: Engram tables get own group (weight_decay=0)
- **Addressing**: Matches upstream's NgramHashMapping bit-for-bit
- **Checkpointing**: Engram config survives save/resume cycle

## Next Steps for Real Experiments
1. Get a real DeepSeek-V3-tokenized corpus:
   ```powershell
   uv run python scripts\tokenize_corpus.py --config_path=configs\tokenize_corpus.yml
   ```
2. Train with realistic settings (most parameters come from config file):
   ```powershell
   uv run python train_engram.py configs\trace_table_access.yml ^
     --run_name=test-engram27b --output_dir=outputs\test-engram27b
   ```
3. The Engram will then operate on real token IDs with real memory-access patterns.

---
**Note**: This smoke test uses character-level Shakespeare data tokenized with DeepSeek-V3. The 23.43% compression ratio matching the paper confirms the tokenizer path is correct. For real memory-access behavior, use a proper corpus like FineWeb or OpenWebText tokenized with DeepSeek-V3.