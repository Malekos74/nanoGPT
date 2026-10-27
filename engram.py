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

    def __post_init__(self):
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

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        """``hidden_states`` [B, T, n_embd], ``input_ids`` [B, T] raw ids."""
        # the hasher consumes *compressed* ids
        with record_function("engram/compress"):
            compressed = self.lookup_table[input_ids.to(self.lookup_table.device)]
        with record_function("engram/hash"):
            hash_ids = self.hasher(compressed)  # [B, T, H]
        with record_function("engram/fetch"):
            embeddings = self.multi_head_embedding(
                hash_ids + self.table_offsets
            ).flatten(start_dim=-2)

        # one gate over the whole stream, replacing upstream's per-slice
        # (hc_mult) gates: |cos| between the memory key and the residual
        # stream, mapped to a multiplicative factor ~ sqrt(|cos|)
        with record_function("engram/gate"):
            key = self.norm1(self.key_proj(embeddings))
            query = self.norm2(hidden_states)
            gate = (key * query).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)

        with record_function("engram/value"):
            value = gate * self.value_proj(embeddings)
        with record_function("engram/short_conv"):
            conv = self.short_conv(value)
        return value + conv


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
