"""GPT with Engram conditional memory, built on top of ``model.GPT``.

Only the injection points are new: the config gains an
``engram`` section, the forward pass injects the memory output into
the residual stream on the configured layers (upstream's placement,
before the block, per engram_demo_v1.py's TransformerBlock), and the
optimizer gets a third group for the memory tables. Everything else
is inherited from ``GPT``.

Use this the same way you use ``GPT``::

    from model_engram import GPTConfig, GPTWithEngram
    model = GPTWithEngram(config)          # or build from an NgramHashMapping
    logits, loss = model(idx, targets)

The ``NgramHashMapping`` is built by ``efficient_engram.reference``
(the same pinned upstream module the trace uses), so the addressing
this model computes is the one the trace measured.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

from engram import EngramMemoryConfig, build_engram_layers
from model import GPT, GPTConfig


# Backward-pass profiler ranges. torch.profiler does not carry forward labels
# into backward, so a pair of identity autograd Functions brackets a region:
# ``_EnterRange`` is applied to the region's *output* (its backward runs first
# and opens the range), ``_ExitRange`` to the region's *input* (its backward
# runs last and closes it). Autograd runs the most recently created nodes
# first, so a side branch such as an Engram layer is processed contiguously
# between the two markers.
_OPEN_RANGES: dict[str, record_function] = {}


class _EnterRange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        if ctx.name in _OPEN_RANGES:
            raise RuntimeError(f"backward range {ctx.name!r} opened twice")
        span = record_function(ctx.name)
        span.__enter__()
        _OPEN_RANGES[ctx.name] = span
        return grad, None


class _ExitRange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        _close_range(ctx.name)
        return grad, None


def _close_range(name: str) -> None:
    _OPEN_RANGES.pop(name).__exit__(None, None, None)


# Eager inside the compiled last segment: Liger's kernels are hand-written Triton,
# so Inductor has nothing to fuse there, and it cannot lower the fp32-accumulating
# addmm(out_dtype=...) Liger takes under AMP (torch 2.14.1, liger-kernel 0.8.4)
@torch.compiler.disable
def _fused_linear_cross_entropy(*, h: torch.Tensor, weight: torch.Tensor,
                                targets: torch.Tensor) -> torch.Tensor:
    return liger_fused_linear_cross_entropy(h, weight, targets, ignore_index=-1)


@dataclass
class GPTEngramConfig(GPTConfig):
    """``GPTConfig`` plus the Engram memory section.

    ``engram`` is ``None`` for a plain GPT, so one config class
    serves both paths.
    """

    engram: EngramMemoryConfig = None
    # recompute every block and Engram module in backward instead of keeping
    # their activations; only their inputs stay alive through the step
    activation_checkpointing: bool = False
    # what of an Engram module the checkpointing recomputes: 'full' all of it,
    # the row fetch included (offloaded rows are gathered and uploaded again);
    # 'after_fetch' keeps the fetched rows (with table_dedup, only the distinct
    # ones) and recomputes their expansion, the gate, value and short conv, so
    # the fetch runs once per step
    engram_checkpoint: str = "full"
    # the training loss through Liger's fused linear cross-entropy, which runs the
    # head and the loss over chunks of tokens and never holds the whole logits
    # tensor; the forward then returns no logits
    fused_cross_entropy: bool = False
    # the layers the backbone is cut before into segments, one compile region
    # and one bwd/backbone range each: [embedding + blocks before the first
    # cut], [blocks between two cuts]..., [blocks from the last cut + head +
    # loss]. With Engram they are its injection layers; a plain GPT takes the
    # same cuts so that both compile the same regions
    segment_cuts: list[int] = None

    def __post_init__(self):
        if not self.segment_cuts:
            raise ValueError(f"segment_cuts needs at least one layer, got {self.segment_cuts!r}")
        if self.segment_cuts != sorted(set(self.segment_cuts)) or self.segment_cuts[0] < 1 \
                or self.segment_cuts[-1] >= self.n_layer:
            raise ValueError(f"segment_cuts {self.segment_cuts} must be increasing layers in 1..{self.n_layer - 1}")
        if self.engram is not None and self.engram.layer_ids != self.segment_cuts:
            raise ValueError(f"segment_cuts {self.segment_cuts} must be the Engram layers {self.engram.layer_ids}: "
                             "a module injects between two segments")
        if self.engram_checkpoint not in ("full", "after_fetch"):
            raise ValueError(f"engram_checkpoint must be 'full' or 'after_fetch', got {self.engram_checkpoint!r}")
        if self.activation_checkpointing and self.engram is not None and \
                self.engram_checkpoint == "full" and self.engram.table_pinned:
            # the pinned paths (and everything built on them: gradient reuse,
            # prefetch) assume one fetch per backward
            raise ValueError("engram_checkpoint='full' recomputes the fetch, which the pinned "
                             "table paths do once per step; use engram_checkpoint='after_fetch'")
        if self.engram is not None:
            # Engram injects before every block it appears in, so
            # its hidden size is the model's
            if self.engram.layer_ids:
                if min(self.engram.layer_ids) < 0 or max(
                    self.engram.layer_ids
                ) >= self.n_layer:
                    raise ValueError(
                        f"engram.layer_ids {self.engram.layer_ids} are "
                        f"outside 0..{self.n_layer - 1}"
                    )


class GPTWithEngram(GPT):
    """``GPT`` with Engram memory injected into the residual stream."""

    def __init__(self, config: GPTEngramConfig, mapping=None):
        super().__init__(config)
        self.config = config
        if config.engram is None:
            self.engram_layers = None
            self._register_embedding_backward_range()
            return
        if mapping is None:
            raise ValueError(
                "GPTWithEngram needs the upstream NgramHashMapping; "
                "build it with efficient_engram.reference.build_hash_mapping"
            )
        # {layer_id: EngramMemory}, injection layers only. ModuleDict, so
        # PyTorch registers the submodules: without it the tables would be
        # invisible to named_parameters(), model.to(), state_dict() and the
        # optimizer -- the model would train without ever updating them.
        self.engram_layers = nn.ModuleDict(
            {str(layer_id): module for layer_id, module in
             build_engram_layers(
                 mapping=mapping,
                 config=config.engram,
                 backbone_hidden_size=config.n_embd,
             ).items()}
        )
        self._register_embedding_backward_range()
        # GPT.__init__ ran _init_weights before these modules existed, so
        # without this the tables and projections keep PyTorch's default
        # init (tables at std 1.0 vs 0.02 everywhere else)
        self.engram_layers.apply(self._init_weights)

    @property
    def engram_config(self) -> EngramMemoryConfig | None:
        return self.config.engram

    def _engram_at(self, layer_id: int):
        if self.engram_layers is None:
            return None
        return self.engram_layers._modules.get(str(layer_id))

    def _prefetching_memories(self) -> list:
        if self.engram_layers is None:
            return []
        return [m for m in self.engram_layers.values() if m.config.table_prefetch]

    def _register_embedding_backward_range(self) -> None:
        # the first segment starts at the token ids, which have no gradient to
        # hang an _ExitRange on, so its backward range (opened by the marker on
        # the segment's output) closes when the embedding weight's gradient has
        # been accumulated, which is the last thing autograd does with it
        self.transformer.wte.weight.register_post_accumulate_grad_hook(
            lambda _param: _close_range("bwd/backbone")
        )

    # The segments are methods so that ``compile_regions`` can compile each one
    # on its own; the labels and backward markers in ``forward`` stay eager.
    def _blocks(self, x, start: int, end: int):
        for block in self.transformer.h[start:end]:
            x = self._run(block, x=x)
        return x

    def _first_segment(self, idx, end: int):
        pos = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        return self._blocks(x, 0, end)

    def _middle_segment(self, x, start: int, end: int):
        return self._blocks(x, start, end)

    def _last_segment(self, x, start: int, targets):
        return self._head(self._blocks(x, start, self.config.n_layer), targets)

    def _head(self, x, targets):
        if targets is None:
            # inference-time mini-optimization: only the last position's logits
            return self.lm_head(self.transformer.ln_f(x[:, [-1], :])), None
        if self.config.fused_cross_entropy:
            h = self.transformer.ln_f(x)
            loss = _fused_linear_cross_entropy(
                h=h.view(-1, h.size(-1)), weight=self.lm_head.weight, targets=targets.view(-1)
            )
            return None, loss
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        return logits, loss

    def place_tables(self) -> None:
        """Offload the memory tables after ``model.to(device)`` when the Engram
        config asks for it (``table_device``); a no-op otherwise."""
        if self.engram_layers is not None:
            for module in self.engram_layers.values():
                module.place_table()

    def clip_grad_norm_(self, *, max_norm: float, table_optimizer) -> torch.Tensor:
        """``clip_grad_norm_`` over the model, returning the total norm. Layers
        with ``table_row_clip`` contribute their table through the norm of its
        row gradients, taken on the device in backward, instead of a norm over
        the dense table gradient on the host, and are scaled by the table
        optimizer rather than here: its fused kernel divides the gradient by
        ``grad_scale`` as it reads it (the attribute GradScaler sets for the same
        purpose), so the coefficient costs the host no pass over the table.
        Layers with ``table_sparse_clip`` contribute the norm of the rows their
        accumulated table gradient holds, taken on the host, and are scaled by
        the table optimizer the same way ('optimizer') or here, those rows alone
        ('rows')."""
        sparse = [m for m in (self.engram_layers or {}).values()
                  if m.config.table_row_clip or m.config.table_sparse_clip]
        if not sparse:
            return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        if len(sparse) != len(self.engram_layers):
            raise ValueError("table_row_clip or table_sparse_clip must be on for every Engram layer or none")
        scale_rows = sparse[0].config.table_sparse_clip == 'rows'
        if table_optimizer is None and not scale_rows:
            raise ValueError("table_row_clip and table_sparse_clip='optimizer' need the tables' optimizer to apply the clip")
        table_ids = {id(m.multi_head_embedding.weight) for m in sparse}
        others = [p for p in self.parameters() if id(p) not in table_ids]
        grads = [p.grad for p in others if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm([*grads, *(m.take_grad_norm() for m in sparse)])
        torch.nn.utils.clip_grads_with_norm_(others, max_norm, total_norm)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)  # clip_grads_with_norm_'s own
        table_device = sparse[0].multi_head_embedding.weight.device
        if scale_rows:
            clip_coef = clip_coef.to(table_device)
            for module in sparse:
                module.scale_table_grad(clip_coef)
            return total_norm
        table_optimizer.grad_scale = clip_coef.reciprocal().to(table_device)
        return total_norm

    def compile_regions(self, mode: str) -> None:
        """torch.compile each backbone segment and each Engram module's
        [compress + hash] and [mix] as separate graphs. Whole-model
        ``torch.compile`` would drop the record_function labels and the backward
        markers between them; per-region compilation keeps the per-component
        profile and loses only fusion across region boundaries. Liger's fused
        head and loss, when on, run eager inside the last segment."""
        self._first_segment = torch.compile(self._first_segment, mode=mode)
        self._middle_segment = torch.compile(self._middle_segment, mode=mode)
        self._last_segment = torch.compile(self._last_segment, mode=mode)
        if self.engram_layers is not None:
            for module in self.engram_layers.values():
                module.compile_regions(mode=mode)

    def _run(self, module, **kwargs):
        """``module(**kwargs)``, under activation checkpointing when the config asks for it."""
        if self.config.activation_checkpointing and self.training:
            return checkpoint(module, use_reentrant=False, **kwargs)
        return module(**kwargs)

    def _run_engram(self, *, memory, hidden_states, input_ids):
        """One Engram module, under the config's activation checkpointing."""
        if not (self.config.activation_checkpointing and self.training):
            return memory(hidden_states=hidden_states, input_ids=input_ids)
        if self.config.engram_checkpoint == "full":
            return checkpoint(memory, use_reentrant=False, hidden_states=hidden_states, input_ids=input_ids)
        rows, inverse = memory.fetch(input_ids)
        return checkpoint(memory.mix, use_reentrant=False, rows=rows, inverse=inverse,
                          hidden_states=hidden_states)

    def _inject(self, x, *, layer_id: int, idx):
        """``x`` plus the Engram module of ``layer_id``, if there is one:
        upstream injects the memory before the block, on the residual stream
        (hidden = engram(hidden, ids) + hidden)."""
        memory = self._engram_at(layer_id)
        if memory is None:
            return x
        with record_function("fwd/engram"):
            out = self._run_engram(memory=memory, hidden_states=_ExitRange.apply(x, "bwd/engram"),
                                   input_ids=idx)
            with record_function("engram/residual_add"):
                return x + _EnterRange.apply(out, "bwd/engram")

    def forward(self, idx, targets=None):
        t = idx.size(1)
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        cuts = self.config.segment_cuts

        # a prefetching memory addresses the step before the first backbone
        # kernel is queued, on its own stream, so that it never waits for one
        for memory in self._prefetching_memories():
            with record_function("fwd/engram"):
                memory.prefetch(idx)
        with record_function("fwd/backbone"):
            x = _EnterRange.apply(self._first_segment(idx, end=cuts[0]), "bwd/backbone")
        for start, end in zip(cuts[:-1], cuts[1:]):
            x = self._inject(x, layer_id=start, idx=idx)
            with record_function("fwd/backbone"):
                x = _EnterRange.apply(self._middle_segment(_ExitRange.apply(x, "bwd/backbone"), start=start, end=end),
                                      "bwd/backbone")
        x = self._inject(x, layer_id=cuts[-1], idx=idx)
        with record_function("fwd/backbone"):
            logits, loss = self._last_segment(_ExitRange.apply(x, "bwd/backbone"), start=cuts[-1], targets=targets)
            if loss is not None:
                loss = _EnterRange.apply(loss, "bwd/backbone")
        return logits, loss

    def engram_table_params(self) -> int:
        if self.engram_layers is None:
            return 0
        return sum(
            p.numel() for module in self.engram_layers.values()
            for p in module.multi_head_embedding.parameters()
        )

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """GPT's PaLM-style estimate, with the memory tables left out of the
        dense-parameter term: a row lookup is a gather, not a matmul. The
        projections, gate and short conv are ordinary parameters and stay in."""
        N = self.get_num_params() - self.engram_table_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter
        flops_promised = 312e12  # A100 bfloat16 peak, as in GPT.estimate_mfu
        return flops_per_iter * (1.0 / dt) / flops_promised

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, muon_momentum,
                             *, table_optimizer: str):
        """Three optimizers, mirroring the paper's setup: Muon for the backbone's
        2-D weight matrices (attention, MLP and the Engram projections), AdamW
        for everything Muon cannot take (token and position embeddings, norms,
        biases, the short conv), and a separate AdamW for the memory tables so
        their update is its own profiler range and gets no weight decay. The
        training loop steps all three; ``lr`` is set on every group of each."""
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        tables = {n for n in params if n.startswith("engram_layers.") and ".multi_head_embedding." in n}
        embeddings = {"transformer.wte.weight", "transformer.wpe.weight"}
        muon_params = [p for n, p in params.items() if p.dim() == 2 and n not in tables | embeddings]
        rest = {n: p for n, p in params.items() if n not in tables and not (p.dim() == 2 and n not in embeddings)}
        decay_params = [p for p in rest.values() if p.dim() >= 2]
        nodecay_params = [p for p in rest.values() if p.dim() < 2]
        table_params = [params[n] for n in sorted(tables)]
        for label, group in [("muon (2-D backbone)", muon_params), ("adamw decayed", decay_params),
                             ("adamw non-decayed", nodecay_params), ("engram tables", table_params)]:
            print(f"num {label} tensors: {len(group)}, with {sum(p.numel() for p in group):,} parameters")

        import inspect

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizers = {
            "muon": torch.optim.Muon(
                muon_params, lr=learning_rate, weight_decay=weight_decay,
                momentum=muon_momentum, nesterov=True, adjust_lr_fn="match_rms_adamw",
            ),
            "adamw": torch.optim.AdamW(
                [{"params": decay_params, "weight_decay": weight_decay},
                 {"params": nodecay_params, "weight_decay": 0.0}],
                lr=learning_rate, betas=betas, **extra_args,
            ),
        }
        if table_optimizer not in ("adamw", "momentum"):
            raise ValueError(f"table_optimizer must be 'adamw' or 'momentum', got {table_optimizer!r}")
        if table_params and table_optimizer == "adamw":  # a plain GPT (engram=None) has no tables to step
            # fused AdamW also has a CPU kernel, for offloaded tables
            optimizers["tables"] = torch.optim.AdamW(
                table_params, lr=learning_rate, betas=betas, weight_decay=0.0,
                fused=fused_available,
            )
        if table_params and table_optimizer == "momentum":
            # V4.1-Flash's state for the tables, one momentum buffer, with Nesterov
            # momentum at Muon's coefficient as there; its Sinkhorn balancing of the
            # update is left out, so this is the state's size, not V4.1's update
            optimizers["tables"] = torch.optim.SGD(
                table_params, lr=learning_rate, momentum=muon_momentum, nesterov=True,
                weight_decay=0.0, fused=True,
            )
        print(f"using fused AdamW: {use_fused}")
        return optimizers
