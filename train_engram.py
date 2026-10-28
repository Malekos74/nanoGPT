"""Training script for GPT with Engram conditional memory.

Same shape as ``train.py`` (single GPU debug mode, or DDP via
``torchrun``), with two additions:

1. an Engram config section, used to build the upstream
   ``NgramHashMapping`` through ``efficient_engram.reference``
   (the pinned checkout the trace also uses, so the addressing
   here is the one the trace measured);
2. the model is ``GPTWithEngram`` from ``model_engram.py``.

To run on a single GPU, example:
$ python train_engram.py --batch_size=32 --compile=False

``compile`` compiles the model per region (the embedding, each
block, the head and each Engram sub-step) through
``model_engram.compile_regions``, so the profiler labels and the
backward markers between the regions stay in eager Python and the
per-component table survives; whole-model torch.compile would
drop them.

Note on data: Engram's addressing is defined on DeepSeek-V3
token ids, compressed through the upstream CompressedTokenizer.
The shakespeare_char dataset (GPT-2/char ids) will *run* -- the
ids are in range of the lookup table -- but the memory addresses
are meaningless. For real experiments, tokenize a corpus with
the DeepSeek-V3 tokenizer (scripts/tokenize_corpus.py builds
such corpora) and point ``dataset`` at it.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
from torch.profiler import record_function
from profiling import build_profiler
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_engram import GPTEngramConfig, GPTWithEngram
from engram import EngramMemoryConfig
from efficient_engram.reference import build_hash_mapping

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out-engram'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2-engram' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
# a corpus in this repo's trace format (artifacts/corpus/<name>: tokens.bin + meta.json),
# read as the access trace reads it: consecutive, non-overlapping sequences in file
# order. Empty keeps nanoGPT's data/<dataset> random-window loader. The corpus has no
# split, so its last corpus_val_tokens serve as val.
corpus_dir = ''
corpus_val_tokens = 1000000
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# engram memory (None disables it; see configs/trace_table_access.yml
# for the Engram-27B geometry these mirror)
engram_enabled = True
engram_tokenizer = 'deepseek-ai/DeepSeek-V3'
engram_layer_ids = [2, 15] # injection layers, paper: 2 and 15
engram_max_ngram_size = 3
engram_n_embed_per_ngram = 1280 # d_mem
engram_n_head_per_ngram = 8
engram_vocab_size_per_ngram = [1131200, 1131200] # rows per head table, per n-gram order
engram_pad_id = 2 # <|_pad|> for DeepSeek-V3, as in the reference
engram_hash_seed = 0
engram_kernel_size = 4
engram_table_device = '' # '' keeps the tables with the model; 'cpu' offloads them to host DRAM
# the pinned upstream implementation; relative to this file's directory
engram_repo = str(Path(__file__).resolve().parent.parent / 'engram')
engram_repo_sha = 'fb7f84a21f91223715394a33a1dc24bbfb7f788e'
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
muon_momentum = 0.95 # Muon (backbone matrices); AdamW takes the rest, see model_engram.configure_optimizers
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # per-region torch.compile, see module docstring
compile_mode = 'default' # torch.compile mode for the regions: 'default', 'reduce-overhead' (CUDA graphs), 'max-autotune'
# profiling: per-component time of a gradient step -> out_dir/profile.md (see profiling.py)
profile = False
profile_warmup = 5 # steps skipped before the profiler starts recording
profile_steps = 5 # steps recorded; the table is written when they are done
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=corpus_dtype, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=corpus_dtype, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return to_device(x, y)

def to_device(x, y):
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)

if corpus_dir:
    from efficient_engram.data.corpus import load_meta, open_tokens
    corpus_meta = load_meta(directory=Path(corpus_dir))
    corpus_tokens = open_tokens(directory=Path(corpus_dir))
    corpus_split_at = corpus_meta.n_tokens - corpus_val_tokens
    corpus_cursor = {'train': 0, 'val': corpus_split_at}
    corpus_end = {'train': corpus_split_at, 'val': corpus_meta.n_tokens}
    def get_batch(split):
        """Step k of a split is sequences k*B .. k*B+B-1 of the packed stream, each
        block_size tokens, exactly the trace's reading order; y is x shifted by one.
        Train raises when the corpus runs out; val wraps around."""
        per_step = batch_size * block_size
        start = corpus_cursor[split]
        if start + per_step + 1 > corpus_end[split]:
            if split == 'train':
                raise ValueError(f"train corpus exhausted at token {start}: "
                                 f"{corpus_end['train']} tokens hold {corpus_end['train'] // per_step} steps")
            start = corpus_split_at
        corpus_cursor[split] = start + per_step
        block = np.asarray(corpus_tokens[start:start + per_step + 1], dtype=np.int64)
        x = torch.from_numpy(block[:-1]).view(batch_size, block_size)
        y = torch.from_numpy(block[1:]).view(batch_size, block_size)
        return to_device(x, y)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
corpus_tokenizer_name = None
corpus_dtype = np.uint16  # GPT-2/shakespeare-char fit in uint16
if corpus_dir:
    from transformers import AutoTokenizer
    corpus_tokenizer_name = corpus_meta.tokenizer_name_or_path
    meta_vocab_size = len(AutoTokenizer.from_pretrained(corpus_tokenizer_name, trust_remote_code=True))
    print(f"corpus {corpus_dir}: {corpus_meta.n_tokens:,} tokens of {corpus_meta.dataset}, "
          f"tokenizer {corpus_tokenizer_name}, vocab_size {meta_vocab_size}, last {corpus_val_tokens:,} tokens are val")
elif os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    corpus_tokenizer_name = meta.get('tokenizer_name_or_path', None)
    # DeepSeek-V3's ids reach 128815, which overflows uint16; the corpus
    # records the dtype it was written with so get_batch reads it back
    dtype_name = meta.get('dtype', 'uint16')
    corpus_dtype = getattr(np, dtype_name)
    if meta_vocab_size > np.iinfo(corpus_dtype).max + 1:
        raise ValueError(
            f"vocab_size {meta_vocab_size} does not fit the corpus dtype {dtype_name}; "
            f"meta.pkl must record the dtype the .bin files were written with"
        )
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    print(f"corpus tokenizer: {corpus_tokenizer_name or '(none)'}")
    print(f"corpus dtype: {dtype_name}")

# engram memory: the upstream NgramHashMapping, from the pinned clone
def build_engram_config() -> EngramMemoryConfig | None:
    if not engram_enabled:
        return None
    # Use corpus tokenizer if available, otherwise fall back to configured one
    tokenizer = corpus_tokenizer_name if corpus_tokenizer_name else engram_tokenizer
    return EngramMemoryConfig(
        tokenizer_name_or_path=tokenizer,
        layer_ids=engram_layer_ids,
        max_ngram_size=engram_max_ngram_size,
        n_embed_per_ngram=engram_n_embed_per_ngram,
        n_head_per_ngram=engram_n_head_per_ngram,
        vocab_size_per_ngram=engram_vocab_size_per_ngram,
        pad_id=engram_pad_id,
        hash_seed=engram_hash_seed,
        kernel_size=engram_kernel_size,
        table_device=engram_table_device,
    )

def build_hash_mapping_for_engram(engram_config: EngramMemoryConfig):
    """Build the NgramHashMapping for this corpus's tokenizer.
    If the corpus uses DeepSeek-V3, it matches the trace; otherwise it
    builds addressing for whatever tokenizer the corpus actually uses."""
    from efficient_engram.config import EngramConfig, ReferenceConfig
    return build_hash_mapping(
        reference=ReferenceConfig(repo=Path(engram_repo), sha=engram_repo_sha),
        engram=EngramConfig(
            tokenizer_name_or_path=engram_config.tokenizer_name_or_path,
            layer_ids=engram_config.layer_ids,
            max_ngram_size=engram_config.max_ngram_size,
            n_embed_per_ngram=engram_config.n_embed_per_ngram,
            n_head_per_ngram=engram_config.n_head_per_ngram,
            vocab_size_per_ngram=engram_config.vocab_size_per_ngram,
            pad_id=engram_config.pad_id,
            hash_seed=engram_config.hash_seed,
        ),
    )

engram_config = build_engram_config()
engram_mapping = None
if engram_config is not None:
    print(f"building Engram memory on layers {engram_config.layer_ids}")
    # the mapping is built from the corpus's tokenizer (or the configured one);
    # if the corpus was tokenized with a different tokenizer, the ids won't
    # line up and the memory addresses are meaningless
    if corpus_tokenizer_name is not None and \
            corpus_tokenizer_name != engram_config.tokenizer_name_or_path:
        raise ValueError(
            f"corpus was tokenized with {corpus_tokenizer_name} but "
            f"engram.tokenizer_name_or_path is "
            f"{engram_config.tokenizer_name_or_path}; the ids must match"
        )
    engram_mapping = build_hash_mapping_for_engram(engram_config)
    original = len(engram_mapping.compressed_tokenizer.lookup_table)
    compressed = len(engram_mapping.compressed_tokenizer)
    print(f"tokenizer compression {original} -> {compressed} "
          f"({100.0 * (1.0 - compressed / original):.2f}% merged)")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  engram=asdict(engram_config) if engram_config is not None else None) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTEngramConfig(**{**model_args, 'engram': engram_config})
    model = GPTWithEngram(gptconf, mapping=engram_mapping)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'engram']:
        model_args[k] = checkpoint_model_args[k]
    # rebuild the mapping before the model: the checkpoint's engram
    # section travels with model_args, and GPTWithEngram needs it
    # the checkpoint holds the engram section as a dict (torch.load is weights-only)
    engram_config = EngramMemoryConfig(**model_args['engram']) if model_args['engram'] is not None else None
    engram_mapping = build_hash_mapping_for_engram(engram_config) if engram_config is not None else None
    # create the model
    gptconf = GPTEngramConfig(**{**model_args, 'engram': engram_config})
    model = GPTWithEngram(gptconf, mapping=engram_mapping)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    raise ValueError(
        f"init_from='{init_from}' is not supported with Engram: the OpenAI "
        "checkpoints have no memory tables to load"
    )
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
model.place_tables() # offloaded tables go back to their device

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizers = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, muon_momentum)
if init_from == 'resume':
    for name, opt in optimizers.items():
        opt.load_state_dict(checkpoint['optimizers'][name])
checkpoint = None # free up memory

# compile the model
if compile:
    print(f"compiling the model's regions (mode={compile_mode})... (takes a ~minute)")
    model.compile_regions(mode=compile_mode)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
profiler = build_profiler(out_dir, profile_warmup, profile_steps) if profile and master_process else nullcontext()
profiler.__enter__()
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for opt in optimizers.values():
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizers': {name: opt.state_dict() for name, opt in optimizers.items()},
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    step_span = record_function("step")
    step_span.__enter__()
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        with record_function("step/data"):
            X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        with record_function("backward"):
            scaler.scale(loss).backward()
    with record_function("optimizer"):
        # clip the gradient
        with record_function("optimizer/clip"):
            if grad_clip != 0.0:
                for opt in optimizers.values():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizers and the scaler if training in fp16
        for name, opt in optimizers.items():
            # muon and adamw together are the backbone's update; the tables are their own row
            with record_function("optimizer/tables" if name == "tables" else "optimizer/backbone"):
                scaler.step(opt)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        with record_function("optimizer/zero_grad"):
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
    if profile and master_process and device_type == 'cuda':
        # drain the GPU at the step boundary while profiling: otherwise the window
        # opens with up to a launch queue's worth of the previous step's kernels
        # still running (unattributable) and closes with as many of this step's
        # still queued (lost). The wait is its own row: GPU time minus host time.
        with record_function("step/sync"):
            torch.cuda.synchronize()
    step_span.__exit__(None, None, None)
    if profile and master_process:
        profiler.step()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

profiler.__exit__(None, None, None)
if ddp:
    destroy_process_group()
