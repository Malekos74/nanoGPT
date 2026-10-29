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
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.profiler import record_function

from efficient_engram.hashing.ngram_hash_torch import NgramHasher
from model import LayerNorm


def _forward_range(name: str):
    """``record_function(name)``, except while autograd recomputes a
    checkpointed forward in backward: that time belongs to the ``bwd/*`` range
    it runs in, not to the forward row of the same name."""
    if torch._C._current_graph_task_id() != -1:
        return nullcontext()
    return record_function(name)


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
    # look up each distinct row of the step once: the flat row ids are uniqued
    # on the model's device, only the distinct rows are gathered (and, with
    # offloaded tables, copied) and the inverse map expands them per token;
    # the row gradients are summed per distinct row on the device before they
    # reach the table. Works for any table placement; the step's counts are
    # kept in ``EngramMemory.last_dedup``
    table_dedup: bool = False
    # with pinned staging, keep one dense table gradient per layer alive across
    # steps: backward adds the row gradients into it (after clearing the rows
    # the previous step wrote) and attaches it as the weight's .grad, instead
    # of autograd allocating, zero-filling and freeing a table-sized tensor
    # every step
    table_grad_reuse: bool = False
    # with pinned staging, address the step before the backbone starts
    # (``EngramMemory.prefetch``, called at the top of the model's forward):
    # compress, hash and dedup run on a side CUDA stream and the row ids are
    # copied to the host there, so that when the forward reaches the layer the
    # gather starts at once instead of waiting behind the preceding blocks'
    # kernels; the rows go up on the same side stream and overlap whatever the
    # backbone is still computing. All within the step, so nothing is stale.
    table_prefetch: bool = False
    # with a reused gradient, run the table's backward beside autograd's CUDA
    # thread instead of on it: the lookup is split into a host half (gather,
    # scatter) and a device half (upload, download), and the download's result
    # is a CPU tensor, which makes autograd run the scatter on the thread that
    # called backward() while its CUDA thread goes on with the backbone's
    # remaining blocks. backward() returns once both are done
    table_async_backward: bool = False
    # with pinned, deduplicated tables, clip the table through its rows: the
    # norm of the row gradients, which is the dense gradient's norm since every
    # other row is zero, is taken on the device before the rows come down and
    # enters the global norm there, and the clip coefficient reaches the table
    # through its optimizer's fused kernel, which scales the gradient as it
    # reads it (``GPTWithEngram.clip_grad_norm_``). The dense table gradient is
    # never normed or scaled on the host. One backward per step: a norm is one
    # backward's, so the table gradient cannot accumulate across backwards
    table_row_clip: bool = False

    def __post_init__(self):
        if self.table_pinned and self.table_device != 'cpu':
            raise ValueError("table_pinned needs table_device='cpu'")
        if self.table_grad_reuse and not self.table_pinned:
            raise ValueError("table_grad_reuse needs table_pinned=True")
        if self.table_prefetch and not self.table_pinned:
            raise ValueError("table_prefetch needs table_pinned=True")
        if self.table_async_backward and not self.table_grad_reuse:
            raise ValueError("table_async_backward needs table_grad_reuse=True")
        if self.table_row_clip and not (self.table_pinned and self.table_dedup):
            raise ValueError("table_row_clip needs table_pinned=True and table_dedup=True")
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
    """Pinned host buffers for one Engram layer's row traffic, allocated once at
    ``capacity`` rows (the batch's lookups, which a deduplicated step never
    exceeds) and reused every step. ``ready`` marks the end of the last upload,
    which must finish before the row buffer is overwritten by the next gather;
    ``landed`` the end of the last download, which the scatter waits for.
    A prefetching layer (``table_prefetch``) also owns a side ``stream``: its
    addressing and the copy of its row ids to the host run there ahead of the
    backbone (``ids_ready`` marks the copy's end), and so does the upload."""

    def __init__(self, stream: torch.cuda.Stream | None, row_clip: bool):
        self.stream = stream
        # table_row_clip: the norm of the last backward's row gradients, a 0-dim
        # device tensor, until the clip takes it
        self.row_clip = row_clip
        self.grad_norm: torch.Tensor | None = None
        self.rows = torch.empty(0)
        self.grads = torch.empty(0)
        self.ready = torch.cuda.Event()
        self.ids_ready = torch.cuda.Event()
        self.landed = torch.cuda.Event()
        # the persistent dense table gradient (table_grad_reuse) and the rows
        # written into it since it was last cleared
        self.table_grad = torch.empty(0)
        self.touched = torch.empty(0, dtype=torch.long)

    def take(self, which: str, n_rows: int, capacity: int, dim: int) -> torch.Tensor:
        if n_rows > capacity:
            raise ValueError(f"{n_rows} rows asked of a {capacity}-row staging buffer")
        buf = getattr(self, which)
        if buf.shape[0] < capacity or buf.shape[1:] != (dim,):
            buf = torch.empty((capacity, dim), dtype=torch.float32, pin_memory=True)
            setattr(self, which, buf)
        return buf[:n_rows]

    def gather(self, weight: torch.Tensor, ids: torch.Tensor, capacity: int) -> torch.Tensor:
        """Rows ``ids`` of the host table ``weight``, in the pinned row buffer."""
        self.ready.synchronize()
        out = self.take("rows", ids.numel(), capacity, weight.shape[1])
        return torch.index_select(weight, 0, ids, out=out)

    def upload(self, rows: torch.Tensor, device: torch.device) -> torch.Tensor:
        """``rows`` (a slice of the pinned row buffer) on ``device``: one DMA,
        marked by ``ready``. On the side stream when there is one, so that it
        overlaps the backbone's kernels on the current stream, which then waits
        for it; the rows' memory is tied to the current stream, their consumer."""
        if self.stream is None:
            out = rows.to(device, non_blocking=True)
            self.ready.record()
            return out
        current = torch.cuda.current_stream(device)
        with torch.cuda.stream(self.stream):
            out = rows.to(device, non_blocking=True)
            self.ready.record()
        current.wait_event(self.ready)
        out.record_stream(current)
        return out

    def download(self, grad: torch.Tensor, capacity: int) -> torch.Tensor:
        """The row gradients ``grad`` copied from the device into the pinned
        gradient buffer, on the current stream; ``landed`` marks the copy's end."""
        host_grad = self.take("grads", grad.shape[0], capacity, grad.shape[1])
        if self.row_clip:
            self.grad_norm = torch.linalg.vector_norm(grad)
        host_grad.copy_(grad, non_blocking=True)
        self.landed.record()
        return host_grad

    def scatter(self, weight: torch.Tensor, ids: torch.Tensor, rows: torch.Tensor,
                reuse: bool) -> torch.Tensor | None:
        """The row gradients ``rows`` of ``ids`` into a dense table gradient on
        the host, once they have landed: with ``reuse`` the persistent one,
        attached to the weight (returns None), else a fresh one for autograd."""
        self.landed.synchronize()
        with record_function("bwd/table_grad"):
            if reuse:
                self.accumulate(weight, ids, rows)
                return None
            return torch.zeros(weight.shape, dtype=rows.dtype).index_add_(0, ids, rows)

    def accumulate(self, weight: torch.Tensor, ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Add the row gradients ``rows`` of ``ids`` into the persistent dense
        table gradient and attach it as ``weight.grad``. A new accumulation
        window (``weight.grad`` is None, i.e. zero_grad ran since the last
        backward) first clears the rows the previous window wrote; every other
        row is still zero from before."""
        if self.table_grad.shape != weight.shape:
            self.table_grad = torch.zeros(weight.shape, dtype=rows.dtype)
        if weight.grad is None:
            self.table_grad.index_fill_(0, self.touched, 0.0)
            self.touched = ids
        elif weight.grad is not self.table_grad:
            raise RuntimeError("table_grad_reuse: the weight's grad is not the reused buffer")
        else:
            if self.row_clip:
                raise RuntimeError("table_row_clip: the table gradient cannot accumulate across backwards")
            self.touched = torch.cat([self.touched, ids])
        self.table_grad.index_add_(0, ids, rows)
        weight.grad = self.table_grad


class _PinnedLookup(torch.autograd.Function):
    """Gather rows of a host table into pinned memory and upload them; in
    backward, download the row gradients into pinned memory and scatter them
    into a dense table gradient on the host: a fresh one returned to autograd,
    or with ``reuse`` the staging's persistent one, attached to the weight
    directly (autograd then gets no gradient for the weight). All of the
    backward runs on autograd's CUDA thread, which waits for the download."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, ids: torch.Tensor, device: torch.device,
                staging: _PinnedStaging, capacity: int, reuse: bool) -> torch.Tensor:
        out = staging.upload(staging.gather(weight, ids, capacity), device)
        ctx.save_for_backward(ids, weight)
        ctx.staging, ctx.capacity, ctx.reuse = staging, capacity, reuse
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ids, weight = ctx.saved_tensors
        host_grad = ctx.staging.download(grad, ctx.capacity)
        table_grad = ctx.staging.scatter(weight, ids, host_grad, ctx.reuse)
        return table_grad, None, None, None, None, None


class _HostGather(torch.autograd.Function):
    """The host half of a lookup with ``table_async_backward``: the gather in
    forward, the scatter in backward. Its backward's input is the CPU tensor
    that ``_Upload``'s backward returns, and autograd runs a node whose input
    gradient is on the CPU on the thread that called backward(), so the scatter
    runs there, beside the CUDA thread's remaining backbone nodes, after waiting
    for the download to land. backward() returns once both threads are done."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, ids: torch.Tensor, staging: _PinnedStaging,
                capacity: int, reuse: bool) -> torch.Tensor:
        ctx.save_for_backward(ids, weight)
        ctx.staging, ctx.reuse = staging, reuse
        return staging.gather(weight, ids, capacity)

    @staticmethod
    def backward(ctx, host_grad: torch.Tensor):
        ids, weight = ctx.saved_tensors
        table_grad = ctx.staging.scatter(weight, ids, host_grad, ctx.reuse)
        return table_grad, None, None, None, None


class _Upload(torch.autograd.Function):
    """The device half: the upload in forward, the download in backward, which
    autograd's CUDA thread issues without waiting for it (``_HostGather`` does)."""

    @staticmethod
    def forward(ctx, rows: torch.Tensor, device: torch.device, staging: _PinnedStaging,
                capacity: int) -> torch.Tensor:
        ctx.staging, ctx.capacity = staging, capacity
        return staging.upload(rows, device)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return ctx.staging.download(grad, ctx.capacity), None, None, None


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
        # between ``prefetch`` and the ``forward`` that consumes it: the step's
        # row ids (in the pinned ``ids`` buffer) and its dedup inverse map
        self._prefetched: tuple[torch.Tensor, torch.Tensor | None] | None = None

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

    # The two compile regions, [compress + hash] and [mix], are methods so that
    # ``compile_regions`` can compile each on its own; the row fetch between
    # them and the profiler labels stay in eager Python (torch.compile drops
    # record_function ranges inside a compiled graph).
    def _address(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Raw ``[B, T]`` ids -> ``[B*T*H]`` flat row ids, token-major: the
        tokenizer compression, the n-gram hash and the per-head table offsets."""
        compressed = self.lookup_table[input_ids.to(self.lookup_table.device)]
        return (self.hasher(compressed) + self.table_offsets).flatten()

    def _dedup(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The distinct row ids of the step (sorted) and, per lookup, its position
        among them. A sort on the device; the count comes back to the host."""
        unique_ids, inverse = torch.unique(ids, return_inverse=True)
        self.last_dedup = (ids.numel(), unique_ids.numel())
        return unique_ids, inverse

    def _address_step(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The step's flat row ids and, with ``table_dedup``, the distinct ids
        instead with the inverse map placing them per lookup (else ``None``)."""
        with _forward_range("engram/address"):
            ids = self._address(input_ids)
        if not self.config.table_dedup:
            return ids, None
        with _forward_range("engram/dedup"):
            return self._dedup(ids)

    def _lookup(self, ids: torch.Tensor, capacity: int) -> torch.Tensor:
        """Rows ``ids`` of the table, as ``[len(ids), dim]`` on the model's device.
        ``capacity`` is the most rows a step can ask for, the pinned buffers' size."""
        weight = self.multi_head_embedding.weight
        device = self.table_offsets.device
        if self.config.table_pinned:
            # a prefetched step hands the ids over in the pinned host buffer
            host_ids = ids if ids.device == weight.device else ids.to(weight.device)
            staging, reuse = self._staging, self.config.table_grad_reuse
            if self.config.table_async_backward:
                rows = _HostGather.apply(weight, host_ids, staging, capacity, reuse)
                return _Upload.apply(rows, device, staging, capacity)
            return _PinnedLookup.apply(weight, host_ids, device, staging, capacity, reuse)
        # with offloaded tables this is the host-to-device copy of the rows; the
        # gradient takes the reverse path through autograd
        return self.multi_head_embedding(ids.to(weight.device)).to(device)

    def _fetch(self, ids: torch.Tensor, inverse: torch.Tensor | None,
               batch_shape: torch.Size) -> torch.Tensor:
        """The embeddings ``[B, T, H*dim]`` of the flat row ids; with ``inverse``
        (deduplicated step) ``ids`` are the distinct rows and ``inverse`` places
        them per lookup, its backward summing the row gradients per distinct row."""
        capacity = ids.numel() if inverse is None else inverse.numel()
        rows = self._lookup(ids, capacity)
        if inverse is not None:
            rows = torch.index_select(rows, 0, inverse)
        return rows.view(*batch_shape, -1)

    def place_table(self) -> None:
        """Move the table to ``config.table_device`` if one is set; call after
        ``model.to(device)``, which moves everything together. The offsets stay
        with the hasher: addressing happens on the model's device, and only the
        flat row ids cross to the table."""
        if self.config.table_device:
            self.multi_head_embedding.to(self.config.table_device)
        if self.config.table_pinned:
            stream = torch.cuda.Stream() if self.config.table_prefetch else None
            self._staging = _PinnedStaging(stream=stream, row_clip=self.config.table_row_clip)
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

    def _mix(self, embeddings: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gate, value and short conv: everything after the fetch."""
        value = self._gate(embeddings, hidden_states) * self.value_proj(embeddings)
        return value + self.short_conv(value)

    def compile_regions(self, mode: str) -> None:
        """torch.compile [compress + hash] and [mix], each as one graph. The
        fetch between them stays eager wherever the table lives, so that every
        table placement compiles the same regions (an offloaded gather crosses
        devices, a deduplicated one changes size every step)."""
        self._address = torch.compile(self._address, mode=mode)
        self._mix = torch.compile(self._mix, mode=mode)

    def prefetch(self, input_ids: torch.Tensor) -> None:
        """Address the step ahead of the layer (``table_prefetch``): compress,
        hash, dedup and the copy of the row ids to pinned host memory, on the
        staging's side stream so that none of it queues behind the backbone's
        kernels. The next ``forward`` picks the result up and gathers at once.
        The ids get a tensor of their own (from the caching pinned allocator,
        so no cudaHostAlloc after the first step) rather than a reused buffer:
        backward and the reused gradient's clear keep them past the next prefetch."""
        if self._prefetched is not None:
            raise RuntimeError(f"layer {self.layer_id}: prefetch called twice without a forward")
        staging = self._staging
        current = torch.cuda.current_stream(input_ids.device)
        staging.stream.wait_stream(current)  # the batch's upload
        input_ids.record_stream(staging.stream)
        with torch.cuda.stream(staging.stream):
            ids, inverse = self._address_step(input_ids)
            if inverse is not None:
                inverse.record_stream(current)  # consumed by the expand on the current stream
            host_ids = torch.empty(ids.numel(), dtype=torch.long, pin_memory=True)
            host_ids.copy_(ids, non_blocking=True)
            staging.ids_ready.record()
        self._prefetched = (host_ids, inverse)

    def take_grad_norm(self) -> torch.Tensor:
        """The norm of the table's gradient from the last backward (``table_row_clip``),
        a 0-dim tensor on the model's device. Each backward's norm is taken once:
        a clip without a backward since the last one raises."""
        if not self.config.table_row_clip:
            raise RuntimeError(f"layer {self.layer_id}: table_row_clip is off")
        norm = self._staging.grad_norm
        if norm is None:
            raise RuntimeError(f"layer {self.layer_id}: no table gradient norm since the last clip")
        self._staging.grad_norm = None
        return norm

    def _take_prefetched(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._prefetched is None:
            raise RuntimeError(
                f"layer {self.layer_id}: table_prefetch is on but the step was not prefetched"
            )
        ids, inverse = self._prefetched
        self._prefetched = None
        self._staging.ids_ready.synchronize()
        return ids, inverse

    def fetch(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The step's embeddings ``[B, T, H*dim]`` on the model's device for raw
        ``input_ids`` [B, T]: addressing and the row fetch, the only part of the
        forward that touches the table."""
        if self.config.table_prefetch:
            ids, inverse = self._take_prefetched()
        else:
            ids, inverse = self._address_step(input_ids)  # [B*T*H] flat row ids
        with _forward_range("engram/fetch"):
            return self._fetch(ids, inverse, input_ids.shape)

    def mix(self, embeddings: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gate, value and short conv of fetched ``embeddings`` against
        ``hidden_states`` [B, T, n_embd]: everything after the fetch."""
        with _forward_range("engram/mix"):
            return self._mix(embeddings, hidden_states)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        """``hidden_states`` [B, T, n_embd], ``input_ids`` [B, T] raw ids."""
        return self.mix(embeddings=self.fetch(input_ids), hidden_states=hidden_states)


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
