"""Per-component timing of a gradient step, via torch.profiler.

``train_engram.py`` labels the step's parts with ``record_function`` (see
``model_engram.py`` and ``engram.py`` for the forward-side labels). When the
profiler's active window ends, ``write_report`` turns the aggregated ranges
into a markdown table of per-step time and share of the step, and also dumps
the raw chrome trace next to it for ``chrome://tracing`` / Perfetto.

Shares do not sum to 100%: the rows nest (``fwd/engram`` contains the
``engram/*`` rows, ``step`` contains everything). CUDA time of a range is the
time of the kernels launched inside it; CPU time is its wall time on the host,
so a host sync inside a range (e.g. a hasher that lives on the CPU) shows up
as CPU time with little CUDA time. The host runs ahead of the GPU, so the
step's CPU time is launch time, not wall time; its CUDA time is the GPU's
work per step. ``backward``'s CUDA time is derived: all kernel time minus the
forward and optimizer ranges, because autograd launches from its own thread.
"""

import os

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, schedule

# report order; hierarchy is by name prefix
COMPONENTS = [
    "step",
    "fwd/embed",
    "fwd/engram",
    "engram/compress",
    "engram/hash",
    "engram/fetch",
    "engram/gate",
    "engram/value",
    "engram/short_conv",
    "fwd/block",
    "fwd/head",
    "backward",
    "optimizer",
]


def build_profiler(out_dir: str, warmup_steps: int, active_steps: int) -> profile:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return profile(
        activities=activities,
        schedule=schedule(wait=warmup_steps, warmup=1, active=active_steps, repeat=1),
        on_trace_ready=lambda prof: write_report(prof, out_dir, active_steps),
    )


def write_report(prof: profile, out_dir: str, active_steps: int) -> None:
    # with CUDA on, every range is reported twice (host-side event and GPU-side
    # annotation); the host-side one carries both the CPU time and the time
    # of the kernels launched inside the range
    events = prof.key_averages()
    averages = {e.key: e for e in events if e.device_type == DeviceType.CPU}
    step = averages["step"]
    # backward kernels are launched from the autograd engine's own thread, so
    # they are children of no host-side range; everything the GPU ran that is
    # not under the step's forward/optimizer ranges is the backward pass
    all_kernels_us = sum(
        e.self_device_time_total for e in events
        if e.device_type != DeviceType.CPU and not e.is_user_annotation
    )
    device_us = {name: averages[name].device_time_total for name in averages}
    device_us["backward"] = all_kernels_us - device_us["step"]
    device_us["step"] = all_kernels_us
    rows = ["| component | calls/step | CPU ms/step | CPU % | CUDA ms/step | CUDA % |",
            "|---|---:|---:|---:|---:|---:|"]
    for name in COMPONENTS:
        if name not in averages:
            continue
        e = averages[name]
        cpu_ms = e.cpu_time_total / 1e3 / active_steps
        cuda_ms = device_us[name] / 1e3 / active_steps
        rows.append(
            f"| {name} | {e.count / active_steps:g} "
            f"| {cpu_ms:.2f} | {100 * e.cpu_time_total / step.cpu_time_total:.1f} "
            f"| {cuda_ms:.2f} | {100 * device_us[name] / max(device_us["step"], 1):.1f} |"
        )
    lines = [f"Profile of {active_steps} training steps (after warmup).", ""] + rows
    if torch.cuda.is_available():
        lines += ["", f"Peak CUDA memory allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"]
    report = "\n".join(lines) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "profile.md"), "w") as f:
        f.write(report)
    prof.export_chrome_trace(os.path.join(out_dir, "profile_trace.json"))
    print(report)
    print(f"profile written to {os.path.join(out_dir, 'profile.md')}")
