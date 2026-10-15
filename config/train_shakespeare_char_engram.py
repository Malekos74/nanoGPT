# Miniature character-level Shakespeare model with Engram
out_dir = 'out-engram-tiny'
eval_interval = 1
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'engram-tiny'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

# GPT model
n_layer = 3
n_head = 3
n_embd = 384
dropout = 0.2

# Training
learning_rate = 1e-3
max_iters = 2
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# Engram configuration
engram_vocab_size_per_ngram = [500, 500]
engram_n_embed_per_ngram = 64
engram_n_head_per_ngram = 4
engram_layer_ids = [1, 2]

# Debugging
compile = False
device = 'cpu'

