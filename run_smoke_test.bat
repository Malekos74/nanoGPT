@echo off
REM Tiny Engram smoke test - runs in seconds on modest hardware.
REM Run from: third_party/nanoGPT
REM Usage: run_smoke_test.bat
REM
REM This verifies the Engram wiring works end-to-end with DeepSeek-V3 tokenized
REM data. It uses tiny model/Engram sizes so it fits in 4GB VRAM and finishes
REM in seconds. For real experiments, use a proper corpus config instead.
REM
REM Prerequisites:
REM   - data/shakespeare_char/train.bin, val.bin, meta.pkl prepared with
REM     DeepSeek-V3 tokenizer (run prepare_deepseek.py in that directory)
REM   - uv installed and synced (uv run python --version)

setlocal

cd /d "%~dp0"

echo === Tiny Engram Smoke Test ===
echo.

uv run python train_engram.py config/train_shakespeare_char.py ^
  --compile=False ^
  --max_iters=2 ^
  --eval_interval=1 ^
  --out_dir=out-engram-tiny ^
  --engram_vocab_size_per_ngram="[500, 500]" ^
  --engram_n_embed_per_ngram=64 ^
  --engram_n_head_per_ngram=4 ^
  --engram_layer_ids="[1, 2]" ^
  --n_layer=3 ^
  --n_embd=3

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo FAILED - see error above.
    exit /b 1
)

echo.
echo === SUCCESS ===
echo Checkpoint saved to: out-engram-tiny\ckpt.pt
echo Verify it contains Engram config:
echo   uv run python -c "import torch; c=torch.load('out-engram-tiny\ckpt.pt'); print(list(c.keys())); print('engram' in c['model_args'])"
endlocal