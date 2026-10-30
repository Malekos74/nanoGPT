"""What the CUDA caching allocator does during a training run.

``AllocatorLog`` takes the allocator's counters at the end of every step, so that
``allocator.json`` shows per step how often it went back to the driver: a failed
``cudaMalloc`` that freed the cache and retried, new device memory mapped or
unmapped (with ``expandable_segments`` these are cuMemMap/cuMemUnmap calls), a
synchronize-and-free of every stream's pending frees. Reading the counters costs
nothing measurable, so profiled runs keep it on.

``record_memory_history`` and ``dump_memory_history`` wrap PyTorch's allocation
recorder: every allocation, free and segment change with the stream it belongs
to and the Python stack that caused it, in a ``memory_snapshot.pickle`` that
https://pytorch.org/memory_viz draws. Collecting the stacks slows every step, so
it runs without the profiler.
"""

import json
import os

import torch

# cumulative counters, logged as their increase over the step
COUNTERS = ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_sync_all_streams")


def allocator_state() -> dict[str, int]:
    stats = torch.cuda.memory_stats()
    return {
        **{name: stats[name] for name in COUNTERS},
        "allocated_bytes": stats["allocated_bytes.all.current"],
        "reserved_bytes": stats["reserved_bytes.all.current"],
    }


class AllocatorLog:
    """Per step, the increase of each of ``COUNTERS`` and the memory allocated
    and reserved at its end."""

    def __init__(self):
        self.last = allocator_state()
        self.steps: list[dict[str, float]] = []

    def step(self) -> None:
        now = allocator_state()
        self.steps.append({
            **{name: now[name] - self.last[name] for name in COUNTERS},
            "allocated_gib": now["allocated_bytes"] / 2**30,
            "reserved_gib": now["reserved_bytes"] / 2**30,
        })
        self.last = now

    def write(self, *, out_dir: str, profile_warmup: int, profile_steps: int) -> None:
        """``allocator.json``, every step from the first; the profiled ones are the
        last ``profile_steps``. Prints the counters summed over those."""
        with open(os.path.join(out_dir, "allocator.json"), "w") as f:
            json.dump({"profile_warmup": profile_warmup, "profile_steps": profile_steps,
                       "steps": self.steps}, f, indent=1)
        recorded = self.steps[-profile_steps:]
        print("allocator over the recorded steps: " + ", ".join(
            f"{name} {sum(s[name] for s in recorded)}" for name in COUNTERS))


def record_memory_history(*, max_entries: int) -> None:
    """Start recording allocations and frees with their Python stacks; the
    newest ``max_entries`` are kept."""
    torch.cuda.memory._record_memory_history(max_entries=max_entries, stacks="python")


def dump_memory_history(*, out_dir: str) -> None:
    """Write the recorded history to ``out_dir/memory_snapshot.pickle`` and stop recording."""
    path = os.path.join(out_dir, "memory_snapshot.pickle")
    torch.cuda.memory._dump_snapshot(path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"memory history written to {path} (open it in https://pytorch.org/memory_viz)")
