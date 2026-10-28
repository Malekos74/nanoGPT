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
a component's CPU time is launch time, not wall time; when the launch queue is
full the host blocks inside kernel launches, and that wait is taken out of the
ranges it fell in and reported once as ``step/launch wait`` (a launch counts as
blocked when it takes ``BLOCKED_LAUNCH_FACTOR`` times the typical launch
duration). ``train_engram.py``
drains the GPU at the end of every profiled step (``step/sync``), so the step's
CPU time is its wall time, ``step/sync`` is how long the host waited for the
GPU, and every kernel of the step falls inside the window. ``backward``'s CUDA
time is derived: all device time minus what the step's own thread issued
(forward and optimizer), because autograd launches from its own thread; the
``bwd/*`` rows under it are measured directly. Host times of ``backward``'s
children are on other threads than ``backward`` itself (the CUDA worker thread
for the marker ranges, the calling thread for CPU-device nodes such as
``bwd/table_grad``) and may overlap in time; host time is therefore projected
onto the parent's window with each instant owned by one child (see
``project_host_time``), so children and ``other`` always add up to the parent
and overlapped work is reported once, under the child listed first.
``bwd/engine start`` is the head of the backward, from the call until the
engine's first node runs (see ``_add_engine_start``).
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
    "step/launch wait",
    "step/data",
    "fwd/backbone",
    "fwd/engram",
    "engram/compress",
    "engram/hash",
    "engram/fetch",
    "engram/gate",
    "engram/value",
    "engram/short_conv",
    "engram/residual_add",
    "backward",
    "bwd/engine start",
    "bwd/backbone",
    "bwd/embedding",
    "bwd/engram",
    "bwd/table_grad",
    "optimizer",
    "optimizer/clip",
    "optimizer/backbone",
    "optimizer/tables",
    "optimizer/zero_grad",
]
# parents whose children are listed above; each gets a computed "<parent>/other"
# row (parent minus children) so that nothing in it goes unexplained
PARENTS = {
    "step": ["step/sync", "step/launch wait", "step/data", "fwd/backbone", "fwd/engram",
             "backward", "optimizer"],
    "fwd/engram": ["engram/compress", "engram/hash", "engram/fetch", "engram/gate",
                   "engram/value", "engram/short_conv", "engram/residual_add"],
    "backward": ["bwd/engine start", "bwd/backbone", "bwd/embedding", "bwd/engram",
                 "bwd/table_grad"],
    "optimizer": ["optimizer/clip", "optimizer/backbone", "optimizer/tables",
                  "optimizer/zero_grad"],
}
DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
COPY_DIRECTIONS = ("h2d", "d2h", "d2d")
LAUNCH_CALLS = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx")
# a kernel launch normally returns in a few microseconds; one that takes this
# many times the typical duration was blocked on a full launch queue, i.e. the
# host was waiting for the GPU, not working
BLOCKED_LAUNCH_FACTOR = 10
# autograd runs CPU-device nodes on the thread that called backward(), outside
# the bwd/* ranges (which live on the CUDA worker thread); these nodes are
# turned into ranges of their own by name. With offloaded tables the table's
# gradient (an eager EmbeddingBackward0 on the host) is such a node.
NODE_RANGES = {"autograd::engine::evaluate_function: EmbeddingBackward0": "bwd/table_grad"}
# the autograd engine's own node evaluations, on whichever thread they run
ENGINE_PREFIX = "autograd::engine::evaluate_function"
ENGINE_START = "bwd/engine start"


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


def attribute(trace_path: str) -> tuple[dict, dict, float, dict]:
    """Sum host time, device time and copies per range name from the chrome trace.

    Returns ``(host, device, all_device_us, launch_wait)``: ``host[name] = {"us",
    "count"}``; ``device[name] = {"us", "h2d": {"us", "bytes", "count"}, ...}``;
    ``launch_wait[name]`` is the host time inside the range spent blocked in
    kernel launches (``launch_wait["all"]`` over every thread).
    """
    events = json.load(open(trace_path))["traceEvents"]
    ranges = defaultdict(list)  # tid -> [(ts, te, name)]
    host = defaultdict(lambda: {"us": 0.0, "count": 0})
    for e in events:
        name = e["name"] if e.get("cat") == "user_annotation" else NODE_RANGES.get(e.get("name"))
        if name is not None and e.get("cat") in ("user_annotation", "cpu_op"):
            ranges[e["tid"]].append((e["ts"], e["ts"] + e["dur"], name))
            host[name]["us"] += e["dur"]
            host[name]["count"] += 1
    _add_engine_start(events, ranges, host)
    # correlation id -> (tid, ts) of the launching call; cuBLAS and friends
    # launch through the driver API (cuLaunchKernel), the rest through the runtime
    launches = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = (e["tid"], e["ts"])

    launch_calls = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")
                    and e["name"] in LAUNCH_CALLS]
    launch_wait = defaultdict(float)
    if launch_calls:
        durations = sorted(e["dur"] for e in launch_calls)
        typical = durations[len(durations) // 2]
        for e in launch_calls:
            if e["dur"] <= BLOCKED_LAUNCH_FACTOR * typical:
                continue
            excess = e["dur"] - typical
            launch_wait["all"] += excess
            for start, end, name in ranges[e["tid"]]:
                if start <= e["ts"] < end:
                    launch_wait[name] += excess
            # a wait on the autograd thread sits inside ``backward``'s wall time
            # on the calling thread, whichever bwd/* range it fell in or none
            for tid, rs in ranges.items():
                if tid == e["tid"]:
                    continue
                for start, end, name in rs:
                    if name == "backward" and start <= e["ts"] < end:
                        launch_wait["backward (other threads)"] += excess

    def fresh() -> dict:
        return {"us": 0.0, **{d: {"us": 0.0, "bytes": 0, "count": 0} for d in COPY_DIRECTIONS}}

    device = defaultdict(fresh)
    all_device_us = 0.0
    for e in events:
        if e.get("cat") not in DEVICE_CATEGORIES:
            continue
        all_device_us += e["dur"]
        direction = copy_direction(e["name"]) if e["cat"] == "gpu_memcpy" else None
        if direction is not None:
            # the step's own copy rows count every copy in the window, including
            # those autograd launches from its own thread (row gradients going home)
            device["step"][direction]["us"] += e["dur"]
            device["step"][direction]["bytes"] += int(e["args"].get("bytes", 0))
            device["step"][direction]["count"] += 1
        launch = launches.get(e.get("args", {}).get("correlation"))
        if launch is None:
            continue  # launched before the window opened: cannot be attributed
        tid, ts = launch
        for start, end, name in ranges[tid]:
            if start <= ts < end:
                device[name]["us"] += e["dur"]
                if direction is not None and name != "step":
                    device[name][direction]["us"] += e["dur"]
                    device[name][direction]["bytes"] += int(e["args"].get("bytes", 0))
                    device[name][direction]["count"] += 1
    intervals = [(ts, te, name) for tid in ranges for ts, te, name in ranges[tid]]
    return host, device, all_device_us, launch_wait, intervals


Interval = tuple[float, float]


def _add_engine_start(events: list, ranges: dict, host: dict) -> None:
    """Label the head of every ``backward`` range, from the call until the engine
    evaluates its first node, as ``bwd/engine start``: the graph's dependency
    pass and the hand-off to the autograd worker thread. No op or CUDA call runs
    in it; it is a fraction of a millisecond in most runs and a few milliseconds
    of host idling in some, independent of the launch queue, the graph size and
    the GPU's state."""
    node_starts = sorted(e["ts"] for e in events
                         if e.get("cat") == "cpu_op" and e["name"].startswith(ENGINE_PREFIX))
    for rs in ranges.values():
        for start, end, name in list(rs):
            if name != "backward":
                continue
            first = next((ts for ts in node_starts if ts >= start), end)
            rs.append((start, min(first, end), ENGINE_START))
            host[ENGINE_START]["us"] += min(first, end) - start
            host[ENGINE_START]["count"] += 1


def _merge(iv: list[Interval]) -> list[Interval]:
    out = []
    for ts, te in sorted(iv):
        if out and ts <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], te))
        else:
            out.append((ts, te))
    return out


def _clip(iv: list[Interval], windows: list[Interval]) -> list[Interval]:
    out = []
    for ts, te in iv:
        for ws, we in windows:
            lo, hi = max(ts, ws), min(te, we)
            if lo < hi:
                out.append((lo, hi))
    return out


def _subtract(iv: list[Interval], covered: list[Interval]) -> list[Interval]:
    out = []
    for ts, te in iv:
        cur = ts
        for cs, ce in covered:
            if ce <= cur or cs >= te:
                continue
            if cs > cur:
                out.append((cur, cs))
            cur = max(cur, ce)
            if cur >= te:
                break
        if cur < te:
            out.append((cur, te))
    return out


def project_host_time(intervals: list) -> dict:
    """Host microseconds per range such that, for every parent in PARENTS, the
    children plus ``<parent>/other`` tile the parent's own time window.

    Children may live on other threads than the parent (autograd's worker
    thread, or the calling thread for CPU-device nodes) and overlap in time;
    each instant of the parent window is owned by the first child in PARENTS
    order that covers it, so overlapped work is counted once.
    """
    by_name = defaultdict(list)
    for ts, te, name in intervals:
        by_name[name].append((ts, te))
    host_us = {name: sum(te - ts for ts, te in iv) for name, iv in by_name.items()}
    for parent, kids in PARENTS.items():
        if parent not in by_name:
            continue
        windows = _merge(by_name[parent])
        covered = []
        for kid in kids:
            if kid not in by_name:
                continue
            own = _merge(_clip(by_name[kid], windows))
            host_us[kid] = sum(te - ts for ts, te in _subtract(own, covered))
            covered = _merge(covered + own)
    return host_us


def write_report(prof: profile, out_dir: str, active_steps: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    prof.export_chrome_trace(os.path.join(out_dir, "profile_trace.json"))
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    report_from_trace(out_dir, active_steps, peak_gib)


def report_from_trace(out_dir: str, active_steps: int, peak_gib: float | None) -> None:
    """Build ``profile.md`` from an exported ``profile_trace.json`` (also usable offline)."""
    host, device, all_device_us, launch_wait, intervals = attribute(
        os.path.join(out_dir, "profile_trace.json")
    )
    for name, us in project_host_time(intervals).items():
        host[name]["us"] = us
    # the wait is the step's, not the range's that happened to fill the queue:
    # take it out of every range and report it once. Waits on the autograd
    # thread sit inside ``backward``'s wall time on the calling thread, so they
    # come out of ``backward`` too rather than being counted twice.
    for name in list(host):
        if name != "step":
            host[name]["us"] = max(0.0, host[name]["us"] - launch_wait[name])
    host["backward"]["us"] -= launch_wait["backward (other threads)"]
    host["step/launch wait"] = {"us": launch_wait["all"], "count": host["step"]["count"]}
    host["step/sync"] = host.get("step/sync", {"us": 0.0, "count": host["step"]["count"]})
    # autograd launches from its own thread, outside every host range of the step
    device["backward"]["us"] = all_device_us - device["step"]["us"]
    device["step"]["us"] = all_device_us
    step_host_us, step_device_us = host["step"]["us"], max(all_device_us, 1.0)

    rows = ["| component | calls/step | CPU ms/step | CPU % | CUDA ms/step | CUDA % |",
            "|---|---:|---:|---:|---:|---:|"]

    def row(label, count, cpu_us, dev_us):
        dev_us = 0.0 if abs(dev_us) < 1.0 else dev_us  # sub-microsecond rounding dust
        cpu_us = None if cpu_us is None else (0.0 if abs(cpu_us) < 1.0 else cpu_us)
        cpu = f"{cpu_us / 1e3 / active_steps:.2f} | {100 * cpu_us / step_host_us:.1f}" if cpu_us is not None else " | "
        rows.append(f"| {label} | {count / active_steps:g} | {cpu} "
                    f"| {dev_us / 1e3 / active_steps:.2f} | {100 * dev_us / step_device_us:.1f} |")

    top_level = {"step", "fwd/backbone", "fwd/engram", "backward", "optimizer"}

    def emit(name, label):
        indent = "" if name in top_level else "&nbsp;&nbsp;"
        row(indent + label, host[name]["count"], host[name]["us"], device[name]["us"])
        for direction in COPY_DIRECTIONS:
            copy = device[name][direction]
            if copy["count"] == 0 and name != "step":
                continue  # every range gets copy rows once it has any; step always
            mb = copy["bytes"] / 1e6 / active_steps
            row(f"{indent}&nbsp;&nbsp;· {direction} copies, {mb:.2f} MB", copy["count"], None, copy["us"])

    # computed remainders, printed right after the parent's last child
    other_after = {}
    for parent, kids in PARENTS.items():
        if parent not in host:
            continue
        present = [k for k in kids if k in host]
        other = parent + "/other"
        host[other] = {
            "us": host[parent]["us"] - sum(host[k]["us"] for k in present),
            "count": host[parent]["count"],
        }
        device[other]["us"] = device[parent]["us"] - sum(device[k]["us"] for k in present)
        last = present[-1] if present else parent
        # if the last child is itself a parent, wait for that child's own remainder
        other_after[last + "/other" if last in PARENTS else last] = other
    for name in COMPONENTS:
        if name not in host:
            continue
        emit(name, name)
        while name in other_after:
            name = other_after[name]
            emit(name, name)

    lines = [f"Profile of {active_steps} training steps (after warmup).", ""] + rows
    if peak_gib is not None:
        lines += ["", f"Peak CUDA memory allocated: {peak_gib:.2f} GiB"]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "profile.md"), "w") as f:
        f.write(report)
    print(report)
    print(f"profile written to {os.path.join(out_dir, 'profile.md')}")
