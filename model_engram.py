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
        _OPEN_RANGES.pop(ctx.name).__exit__(None, None, None)
        return grad, None


@dataclass
class GPTEngramConfig(GPTConfig):
    """``GPTConfig`` plus the Engram memory section.

    ``engram`` is ``None`` for a plain GPT, so one config class
    serves both paths.
    """

    engram: EngramMemoryConfig = None

    def __post_init__(self):
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

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        with record_function("fwd/backbone"):
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        for layer_id, block in enumerate(self.transformer.h):
            memory = self._engram_at(layer_id)
            if memory is not None:
                # upstream injects the memory before the block, on the
                # residual stream (hidden = engram(hidden, ids) + hidden)
                with record_function("fwd/engram"):
                    out = memory(hidden_states=_ExitRange.apply(x, "bwd/engram"), input_ids=idx)
                    x = x + _EnterRange.apply(out, "bwd/engram")
            with record_function("fwd/backbone"):
                x = _EnterRange.apply(block(_ExitRange.apply(x, "bwd/backbone")), "bwd/backbone")
        with record_function("fwd/backbone"):
            x = self.transformer.ln_f(_ExitRange.apply(x, "bwd/backbone"))
            if targets is not None:
                logits = self.lm_head(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
                loss = _EnterRange.apply(loss, "bwd/backbone")
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None

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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, muon_momentum):
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
        if table_params:  # a plain GPT (engram=None) has no tables to step
            optimizers["tables"] = torch.optim.AdamW(
                table_params, lr=learning_rate, betas=betas, weight_decay=0.0, **extra_args,
            )
        print(f"using fused AdamW: {use_fused}")
        return optimizers
