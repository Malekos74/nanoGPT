# Pretraining on FineWeb-Edu (this repo's trace corpus, see scripts/tokenize_corpus.py)
# with Engram in the paper's Engram-27B head shape on a small dense backbone.
# This is the configuration experiments/engram-profile.md profiles.
out_dir = 'out-fineweb-engram'
eval_interval = 500
eval_iters = 50
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'efficient-engram'
wandb_run_name = 'nanogpt-fineweb-engram'

# data: consecutive non-overlapping sequences of the packed corpus, in file order
corpus_dir = '../../artifacts/corpus/fineweb_edu_1b'
corpus_val_tokens = 1000000
gradient_accumulation_steps = 1
batch_size = 16
block_size = 4096

# backbone
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

# optimization: one pass over the 1B-token corpus is ~15k steps of 65k tokens
learning_rate = 6e-4
max_iters = 15000
lr_decay_iters = 15000
min_lr = 6e-5
warmup_iters = 500
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
muon_momentum = 0.95
grad_clip = 1.0

# Engram: the paper's head shape, 100k rows per head table
engram_enabled = True
engram_layer_ids = [1, 3]
engram_max_ngram_size = 3
engram_n_embed_per_ngram = 1280
engram_n_head_per_ngram = 8
engram_vocab_size_per_ngram = [100000, 100000]

device = 'cuda'
dtype = 'bfloat16'
compile = True # per region, see train_engram.py
compile_mode = 'default'
