"""Engram conditional memory, ported onto the modules this repo already has.

Mirrors upstream's ``Engram`` module (third_party/engram/engram_demo_v1.py)
for nanoGPT's plain residual stream -- ``hidden_states`` is ``[B, T, n_embd]``,
not upstream's hyper-connected ``[B, T, hc_mult, D]``. Addressing (tokenizer
compression, hash multipliers, per-head table sizes) is upstream's, taken
through ``efficient_engram.reference`` so nothing is re-derived here; hashing
is the GPU ``NgramHasher``, bit-identical to upstream (tests/test_hash_matches_upstream.py).

Simplifications vs. upstream, all to keep the diff small:
- the ``hc_mult`` dimension is dropped; the gate is computed once on the whole
  hidden stream (upstream's per-slice key/query projections are an
  extra-parameter refinement of the same dot-product gate)
- the short convolution reuses nanoGPT's ``LayerNorm`` instead of upstream's
  ``RMSNorm`` (this file is meant to import only from ``model``)

Run as part of ``train_engram.py``; see that file for the config plumbing.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.profiler import record_function

from efficient_engram.hashing.ngram_hash_torch import NgramHasher
from model import LayerNorm


@dataclass
class EngramMemoryConfig:
    """Geometry of one Engram injection layer, as the paper's Table 5 has it."""

    tokenizer_name_or_path: str
    layer_ids: list[int]
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 1280  # d_mem
    n_head_per_ngram: int = 8
    vocab_size_per_ngram: list[int] = None  # one entry per n-gram order
    pad_id: int = 2  # <|_pad|> for the DeepSeek-V3 tokenizer
    hash_seed: int = 0
    kernel_size: int = 4
    # where the memory tables live: '' keeps them with the model, 'cpu' offloads
    # them to host DRAM (rows are gathered there and copied to the model's device)
    table_device: str = ''
    # with offloaded tables, stage the rows and their gradients in pinned host
    # buffers (allocated once, reused every step) so the copies run as DMA at
    # bus speed instead of through the driver's pageable-memory bounce buffer
    table_pinned: bool = False
    # keep the offloaded table itself in pinned memory (its gradient and
    # optimizer state stay pageable); nothing in the current pipeline DMAs
    # from the table, so this is a measurement, not an expected speed-up
    table_weight_pinned: bool = False

    def __post_init__(self):
        if self.table_pinned and self.table_device != 'cpu':
            raise ValueError("table_pinned needs table_device='cpu'")
        if self.table_weight_pinned and self.table_device != 'cpu':
            raise ValueError("table_weight_pinned needs table_device='cpu'")
        expected = self.max_ngram_size - 1
        if len(self.vocab_size_per_ngram) != expected:
            raise ValueError(
                f"vocab_size_per_ngram needs max_ngram_size - 1 = {expected} "
                f"entries, got {len(self.vocab_size_per_ngram)}"
            )
        if self.n_embed_per_ngram % self.n_head_per_ngram != 0:
            raise ValueError(
                f"n_embed_per_ngram ({self.n_embed_per_ngram}) must divide "
                f"n_head_per_ngram ({self.n_head_per_ngram})"
            )


class ShortConv(nn.Module):
    """Upstream's dilated depthwise short convolution, for a 3-D stream.

    Groups = channels, so each hidden feature is convolved with itself alone;
    dilation = max_ngram_size spaces the taps one n-gram apart. Input and
    output are both ``[B, T, C]``.
    """

    def __init__(self, hidden_size: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norm = LayerNorm(hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        y = self.conv(x_norm.transpose(1, 2))
        y = y[..., : x.size(1)]  # undo the look-ahead padding
        return self.act_fn(y).transpose(1, 2)


class _PinnedStaging:
    """Pinned host buffers for one Engram layer's row traffic, grown on demand
    and reused every step. ``ready`` marks the end of the last upload, which
    must finish before the row buffer is overwritten by the next gather."""

    def __init__(self):
        self.rows = torch.empty(0)
        self.grads = torch.empty(0)
        self.ready = torch.cuda.Event()

    def take(self, which: str, n_rows: int, dim: int) -> torch.Tensor:
        buf = getattr(self, which)
        if buf.shape[0] < n_rows or buf.shape[1:] != (dim,):
            buf = torch.empty((n_rows, dim), dtype=torch.float32, pin_memory=True)
            setattr(self, which, buf)
        return buf[:n_rows]


class _PinnedLookup(torch.autograd.Function):
    """Gather rows of a host table into pinned memory and upload them; in
    backward, download the row gradients into pinned memory and scatter them
    into a dense table gradient on the host."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, ids: torch.Tensor, device: torch.device,
                staging: _PinnedStaging) -> torch.Tensor:
        staging.ready.synchronize()
        rows = torch.index_select(weight, 0, ids, out=staging.take("rows", ids.numel(), weight.shape[1]))
        out = rows.to(device, non_blocking=True)
        staging.ready.record()
        ctx.save_for_backward(ids)
        ctx.staging, ctx.table_shape = staging, weight.shape
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (ids,) = ctx.saved_tensors
        host_grad = ctx.staging.take("grads", ids.numel(), grad.shape[1])
        host_grad.copy_(grad, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        with record_function("bwd/table_grad"):
            table_grad = torch.zeros(ctx.table_shape, dtype=host_grad.dtype).index_add_(0, ids, host_grad)
        return table_grad, None, None, None


class EngramMemory(nn.Module):
    """One Engram injection layer: hash the n-grams, fetch, gate, add.

    Built from an upstream ``NgramHashMapping`` (already constructed for the
    trace) so the table sizes and multipliers are exactly the ones the
    telemetry measured. The embedding is one flat ``nn.Embedding`` over all
    head tables -- upstream's ``MultiHeadEmbedding`` layout -- with per-head
    offsets, so ``hash_ids + offsets`` turns the hasher's per-table indices
    into flat row ids.
    """

    def __init__(
        self,
        layer_id: int,
        mapping,
        config: EngramMemoryConfig,
        backbone_hidden_size: int,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.config = config

        # a submodule, so its constants move with the model and the whole
        # compress -> hash -> gather chain runs on the table's device
        self.hasher = NgramHasher.from_reference(
            mapping=mapping, layer_id=layer_id, device="cpu"
        )
        self.register_buffer(
            "lookup_table",
            torch.from_numpy(mapping.compressed_tokenizer.lookup_table),
        )

        head_table_sizes = [
            int(size)
            for sizes in mapping.vocab_size_across_layers[layer_id]
            for size in sizes
        ]
        # n-gram-major order, matching hasher.moduli; the offset of each table
        # is the total size of every table before it
        offsets = [0]
        for size in head_table_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self.register_buffer("table_offsets", torch.tensor(offsets, dtype=torch.long))
        self.multi_head_embedding = nn.Embedding(
            num_embeddings=sum(head_table_sizes),
            embedding_dim=config.n_embed_per_ngram // config.n_head_per_ngram,
        )

        engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, backbone_hidden_size)
        self.key_proj = nn.Linear(engram_hidden_size, backbone_hidden_size)
        self.norm1 = nn.RMSNorm(backbone_hidden_size)
        self.norm2 = nn.RMSNorm(backbone_hidden_size)
        self.short_conv = ShortConv(
            hidden_size=backbone_hidden_size,
            kernel_size=config.kernel_size,
            dilation=config.max_ngram_size,
        )

    # The sub-steps are methods so that ``compile_regions`` can compile each
    # one on its own; the profiler labels in ``forward`` stay in eager Python
    # (torch.compile drops record_function ranges inside a compiled graph).
    def _compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lookup_table[input_ids.to(self.lookup_table.device)]

    def _fetch(self, hash_ids: torch.Tensor) -> torch.Tensor:
        table = self.multi_head_embedding.weight.device
        ids = hash_ids.to(table) + self.table_offsets
        if self.config.table_pinned:
            rows = _PinnedLookup.apply(self.multi_head_embedding.weight, ids.flatten(),
                                       hash_ids.device, self._staging)
            return rows.view(*ids.shape, -1).flatten(start_dim=-2)
        rows = self.multi_head_embedding(ids)
        # with offloaded tables this is the host-to-device copy of the rows; the
        # gradient takes the reverse path through autograd
        return rows.to(hash_ids.device).flatten(start_dim=-2)

    def place_table(self) -> None:
        """Move the table (and its offsets) to ``config.table_device`` if one is
        set; call after ``model.to(device)``, which moves everything together."""
        if self.config.table_device:
            self.multi_head_embedding.to(self.config.table_device)
            self.table_offsets = self.table_offsets.to(self.config.table_device)
        if self.config.table_pinned:
            self._staging = _PinnedStaging()
        if self.config.table_weight_pinned:
            weight = self.multi_head_embedding.weight
            weight.data = weight.data.pin_memory()

    def _gate(self, embeddings: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # one gate over the whole stream, replacing upstream's per-slice
        # (hc_mult) gates: |cos| between the memory key and the residual
        # stream, mapped to a multiplicative factor ~ sqrt(|cos|)
        key = self.norm1(self.key_proj(embeddings))
        query = self.norm2(hidden_states)
        gate = (key * query).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        return gate.sigmoid().unsqueeze(-1)

    def _value(self, embeddings: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return gate * self.value_proj(embeddings)

    def compile_regions(self, mode: str) -> None:
        """torch.compile each sub-step separately, keeping the labels between them."""
        self._compress = torch.compile(self._compress, mode=mode)
        self.hasher.forward = torch.compile(self.hasher.forward, mode=mode)
        if not self.config.table_device:  # a cross-device gather stays eager
            self._fetch = torch.compile(self._fetch, mode=mode)
        self._gate = torch.compile(self._gate, mode=mode)
        self._value = torch.compile(self._value, mode=mode)
        self.short_conv.forward = torch.compile(self.short_conv.forward, mode=mode)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        """``hidden_states`` [B, T, n_embd], ``input_ids`` [B, T] raw ids."""
        # the hasher consumes *compressed* ids
        with record_function("engram/compress"):
            compressed = self._compress(input_ids)
        with record_function("engram/hash"):
            hash_ids = self.hasher(compressed)  # [B, T, H]
        with record_function("engram/fetch"):
            embeddings = self._fetch(hash_ids)
        with record_function("engram/gate"):
            gate = self._gate(embeddings, hidden_states)
        with record_function("engram/value"):
            value = self._value(embeddings, gate)
        with record_function("engram/short_conv"):  # the conv and its sum into the value
            return value + self.short_conv(value)


def build_engram_layers(
    mapping, config: EngramMemoryConfig, backbone_hidden_size: int
):
    """Build one module per injection layer. Returns ``{layer_id: EngramMemory}``."""
    return {
        layer_id: EngramMemory(
            layer_id=layer_id,
            mapping=mapping,
            config=config,
            backbone_hidden_size=backbone_hidden_size,
        )
        for layer_id in config.layer_ids
    }
