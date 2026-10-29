# The main experiment: a Qwen2.5-3B-shaped backbone (36 layers, width 2048, 16 query
# and 2 key/value heads, SwiGLU of inner width 11008, tied embeddings) with Engram
# tables at V4.1-Flash's tables-to-backbone ratio (196B / 552B of ~3B = 1.05B) in
# fp32, a momentum-only table optimizer, full activation checkpointing, Liger's fused
# linear cross-entropy (no logits tensor), per-region torch.compile and 4096-token
# sequences, one micro-batch per step. Departs from Qwen2.5-3B where it barely
# matters for time and memory: LayerNorm instead of RMSNorm, learned positions
# instead of RoPE, no qkv bias, and the DeepSeek-V3 vocabulary that Engram's
# addressing needs (129k instead of 151k). Only ever profiled, not trained to the
# end: the schedule is a placeholder. slurm/qwen3b_profile.sbatch runs the variants
# over this base and slurm/qwen3b_max_batch.sbatch finds their batch sizes.
out_dir = 'out-qwen3b-engram'
eval_enabled = False # training steps only: the eval and its compile for eval mode are not profiled
log_interval = 1
always_save_checkpoint = False
wandb_log = False

corpus_dir = '../../artifacts/corpus/fineweb_edu_1b'
corpus_val_tokens = 1000000
gradient_accumulation_steps = 1
batch_size = 1 # every launcher sets each variant's own, from the max-batch sweep
block_size = 4096

n_layer = 36
n_head = 16
n_kv_head = 2
n_embd = 2048
mlp = 'swiglu'
mlp_hidden = 11008
dropout = 0.0
bias = False
activation_checkpointing = True
fused_cross_entropy = True

learning_rate = 6e-4
max_iters = 14 # the last recorded step: profile_warmup + profile_steps (one profiler warmup step between)
lr_decay_iters = 14
warmup_iters = 10
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
muon_momentum = 0.95
grad_clip = 1.0
# every variant's step time is flat from the second step on in the nanoGPT-scale
# profiles (job 6984, RAM variants included), so a few steps of margin suffice
profile_warmup = 4
profile_steps = 10

# Engram-27B head shape (d_mem 1280 over 8 heads, bigram and trigram), at the
# paper's injection layers; 2 layers * 2 orders * 8 heads * ~205k rows * 160 = 1.05B
engram_enabled = True
engram_layer_ids = [2, 15]
engram_max_ngram_size = 3
engram_n_embed_per_ngram = 1280
engram_n_head_per_ngram = 8
engram_vocab_size_per_ngram = [205078, 205078]
engram_table_optimizer = 'momentum' # V4.1-Flash's one state buffer per table

device = 'cuda'
dtype = 'bfloat16'
compile = True # per region, see train_engram.py
compile_mode = 'default'
