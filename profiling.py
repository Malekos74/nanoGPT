"""Per-component timing of a gradient step, via torch.profiler.

``train_engram.py`` labels the step's parts with ``record_function`` (see
``model_engram.py`` and ``engram.py`` for the forward-side labels; the
``bwd/*`` ranges are opened and closed from inside autograd by
``model_engram``'s marker Functions). When the profiler's active window ends,
``write_report`` exports the chrome trace and builds a markdown table from it:
every device event (kernel, copy, memset) is attributed to the host ranges
that enclose the runtime call that launched it, so a range's CUDA time is the
GPU work it issued, and copies are listed under it by direction with bytes:
h2d is host (CPU memory) to device (GPU memory), d2h the reverse, d2d within
the GPU. The step always shows all three; a range shows them once it has any.

Shares do not sum to 100%: rows nest (``fwd/engram`` contains the
``engram/*`` rows, ``backward`` the ``bwd/*`` rows, ``optimizer`` the
``optimizer/*`` rows, ``step`` everything). The host runs ahead of the GPU, so
a component's CPU time is launch time, not wall time. ``train_engram.py``
drains the GPU at the end of every profiled step (``step/sync``), so the step's
CPU time is its wall time, ``step/sync`` is how long the host waited for the
GPU, and every kernel of the step falls inside the window. ``backward``'s CUDA
time is derived: all device time minus what the step's own thread issued
(forward and optimizer), because autograd launches from its own thread; the
``bwd/*`` rows under it are measured directly.
"""

import json
import os
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, schedule

# report order; hierarchy is by name prefix
COMPONENTS = [
    "step",
    "step/sync",
    "fwd/backbone",
    "fwd/engram",
    "engram/compress",
    "engram/hash",
    "engram/fetch",
    "engram/gate",
    "engram/value",
    "engram/short_conv",
    "backward",
    "bwd/backbone",
    "bwd/engram",
    "optimizer",
    "optimizer/backbone",
    "optimizer/tables",
]
DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
COPY_DIRECTIONS = ("h2d", "d2h", "d2d")


def build_profiler(out_dir: str, warmup_steps: int, active_steps: int) -> profile:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return profile(
        activities=activities,
        schedule=schedule(wait=warmup_steps, warmup=1, active=active_steps, repeat=1),
        on_trace_ready=lambda prof: write_report(prof, out_dir, active_steps),
    )


def copy_direction(name: str) -> str | None:
    for tag, direction in (("HtoD", "h2d"), ("DtoH", "d2h"), ("DtoD", "d2d")):
        if tag in name:
            return direction
    return None


def attribute(trace_path: str) -> tuple[dict, dict, float]:
    """Sum host time, device time and copies per range name from the chrome trace.

    Returns ``(host, device, all_device_us)``: ``host[name] = {"us", "count"}``;
    ``device[name] = {"us", "h2d": {"us", "bytes", "count"}, ...}``.
    """
    events = json.load(open(trace_path))["traceEvents"]
    ranges = defaultdict(list)  # tid -> [(ts, te, name)]
    host = defaultdict(lambda: {"us": 0.0, "count": 0})
    for e in events:
        if e.get("cat") == "user_annotation":
            ranges[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
            host[e["name"]]["us"] += e["dur"]
            host[e["name"]]["count"] += 1
    # correlation id -> (tid, ts) of the launching call; cuBLAS and friends
    # launch through the driver API (cuLaunchKernel), the rest through the runtime
    launches = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = (e["tid"], e["ts"])

    def fresh():
        return {"us": 0.0, **{d: {"us": 0.0, "bytes": 0, "count": 0} for d in COPY_DIRECTIONS}}

    device = defaultdict(fresh)
    all_device_us = 0.0
    for e in events:
        if e.get("cat") not in DEVICE_CATEGORIES:
            continue
        all_device_us += e["dur"]
        launch = launches.get(e.get("args", {}).get("correlation"))
        if launch is None:
            continue  # launched before the window opened: cannot be attributed
        tid, ts = launch
        direction = copy_direction(e["name"]) if e["cat"] == "gpu_memcpy" else None
        for start, end, name in ranges[tid]:
            if start <= ts < end:
                device[name]["us"] += e["dur"]
                if direction is not None:
                    device[name][direction]["us"] += e["dur"]
                    device[name][direction]["bytes"] += int(e["args"].get("bytes", 0))
                    device[name][direction]["count"] += 1
    return host, device, all_device_us


def write_report(prof: profile, out_dir: str, active_steps: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    prof.export_chrome_trace(os.path.join(out_dir, "profile_trace.json"))
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    report_from_trace(out_dir, active_steps, peak_gib)


def report_from_trace(out_dir: str, active_steps: int, peak_gib: float | None) -> None:
    """Build ``profile.md`` from an exported ``profile_trace.json`` (also usable offline)."""
    host, device, all_device_us = attribute(os.path.join(out_dir, "profile_trace.json"))
    # autograd launches from its own thread, outside every host range of the step
    device["backward"]["us"] = all_device_us - device["step"]["us"]
    device["step"]["us"] = all_device_us
    step_host_us, step_device_us = host["step"]["us"], max(all_device_us, 1.0)

    rows = ["| component | calls/step | CPU ms/step | CPU % | CUDA ms/step | CUDA % |",
            "|---|---:|---:|---:|---:|---:|"]

    def row(label, count, cpu_us, dev_us):
        cpu = f"{cpu_us / 1e3 / active_steps:.2f} | {100 * cpu_us / step_host_us:.1f}" if cpu_us is not None else " | "
        rows.append(f"| {label} | {count / active_steps:g} | {cpu} "
                    f"| {dev_us / 1e3 / active_steps:.2f} | {100 * dev_us / step_device_us:.1f} |")

    for name in COMPONENTS:
        if name not in host:
            continue
        indent = "&nbsp;&nbsp;" if "/" in name and not name.startswith("fwd/") else ""
        row(indent + name, host[name]["count"], host[name]["us"], device[name]["us"])
        for direction in COPY_DIRECTIONS:
            copy = device[name][direction]
            if copy["count"] == 0 and name != "step":
                continue  # every range gets copy rows once it has any; step always
            mb = copy["bytes"] / 1e6 / active_steps
            row(f"{indent}&nbsp;&nbsp;· {direction} copies, {mb:.2f} MB", copy["count"], None, copy["us"])

    lines = [f"Profile of {active_steps} training steps (after warmup).", ""] + rows
    if peak_gib is not None:
        lines += ["", f"Peak CUDA memory allocated: {peak_gib:.2f} GiB"]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "profile.md"), "w") as f:
        f.write(report)
    print(report)
    print(f"profile written to {os.path.join(out_dir, 'profile.md')}")
