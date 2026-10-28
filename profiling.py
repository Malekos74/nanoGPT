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
import statistics
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
    "bwd/table_grad",
    "bwd/engram",
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
    # table_grad before engram: the table's backward is counted in full, whether it
    # runs on the calling thread beside the copy (eager) or inside bwd/engram (pinned)
    "backward": ["bwd/engine start", "bwd/backbone", "bwd/embedding", "bwd/table_grad",
                 "bwd/engram"],
    "optimizer": ["optimizer/clip", "optimizer/backbone", "optimizer/tables",
                  "optimizer/zero_grad"],
}
DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
COPY_DIRECTIONS = ("h2d", "d2h", "d2d")
LAUNCH_CALLS = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx")
# runtime calls in which the host does nothing but wait for the device (a
# pageable memcpy is one long blocking call whose host-side staging is part of
# the transfer, so it counts as communication, not host work)
WAIT_CALLS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
              "cudaMemcpy", "cudaMemcpyAsync")
# the exclusive split of a step's wall time, see ``partition_step``
PARTITION = ("gpu", "copy", "host", "launch", "data", "host_copy", "idle")
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


def attribute(trace_path: str) -> tuple[dict, dict, float, dict, list]:
    """``attribute_events`` over a whole chrome trace."""
    return attribute_events(json.load(open(trace_path))["traceEvents"])


def attribute_events(events: list) -> tuple[dict, dict, float, dict, list]:
    """Sum host time, device time and copies per range name over ``events``.

    Returns ``(host, device, all_device_us, launch_wait, intervals)``:
    ``host[name] = {"us", "count"}``; ``device[name] = {"us", "h2d": {"us",
    "bytes", "count"}, ...}``; ``launch_wait[name]`` is the host time inside the
    range spent blocked in kernel launches (``launch_wait["all"]`` over every
    thread); ``intervals`` are the host ranges as ``(ts, te, name)``.
    """
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


def _host_work_intervals(events: list, steps: list[Interval], typical_launch: float) -> list[Interval]:
    """When at least one host thread is doing work: the step's own labelled ranges
    (not ``backward``, in which the calling thread only waits for the autograd
    thread, whose work the cpu ops cover) and every cpu op, minus the calls in
    which a thread is blocked on the device."""
    work, waits = [], []
    for e in events:
        if "ts" not in e or "dur" not in e:
            continue
        iv = (e["ts"], e["ts"] + e["dur"])
        cat, name = e.get("cat"), e["name"]
        if cat == "user_annotation" and name in ("step/sync",):
            waits.append(iv)
        elif cat == "cpu_op" or (cat == "user_annotation" and name not in ("step", "backward")
                                 and not name.startswith("ProfilerStep")):
            work.append(iv)
        elif cat in ("cuda_runtime", "cuda_driver") and name in WAIT_CALLS:
            waits.append(iv)
        elif cat in ("cuda_runtime", "cuda_driver") and name in LAUNCH_CALLS \
                and e["dur"] > BLOCKED_LAUNCH_FACTOR * typical_launch:
            waits.append((iv[0] + typical_launch, iv[1]))
    return _subtract(_merge(_clip(work, steps)), _merge(waits))


def _part(active: tuple, in_data: bool) -> str:
    if not active:
        return "idle"
    if "host" not in active:
        return "gpu" if "gpu" in active else "copy"  # a copy beside a kernel counts as compute
    if len(active) == 1:
        return "host"
    if in_data:
        return "data"
    return "launch" if "gpu" in active else "host_copy"


def step_windows(events: list) -> list[Interval]:
    """The recorded steps' ``step`` ranges, in order."""
    steps = _merge([(e["ts"], e["ts"] + e["dur"]) for e in events
                    if e.get("cat") == "user_annotation" and e["name"] == "step"])
    if not steps:
        raise ValueError("no step ranges in the trace")
    return steps


def partition_steps(events: list, steps: list[Interval]) -> list[dict[str, float]]:
    """Split each step's wall time into exclusive parts, in microseconds, one
    dict per step: ``gpu`` (a kernel runs, the host waits), ``copy`` (a memcpy
    runs, nothing else), ``host`` (a host thread works, the device is idle),
    ``launch`` (the host works while the GPU computes: it is ahead, launching
    kernels and the glue around them), ``data`` (the batch is loaded while the
    device is busy), ``host_copy`` (the host works while only a transfer is in
    flight) and ``idle`` (nothing runs)."""
    launches = sorted(e["dur"] for e in events
                      if e.get("cat") in ("cuda_runtime", "cuda_driver") and e["name"] in LAUNCH_CALLS)
    typical = launches[len(launches) // 2] if launches else 0.0
    lanes = {
        "gpu": _merge(_clip([(e["ts"], e["ts"] + e["dur"]) for e in events
                             if e.get("cat") in ("kernel", "gpu_memset")], steps)),
        "copy": _merge(_clip([(e["ts"], e["ts"] + e["dur"]) for e in events
                              if e.get("cat") == "gpu_memcpy"], steps)),
        "host": _host_work_intervals(events, steps, typical),
    }
    data = _merge([(e["ts"], e["ts"] + e["dur"]) for e in events
                   if e.get("cat") == "user_annotation" and e["name"] == "step/data"])
    per_step = []
    for ws, we in steps:
        totals = dict.fromkeys(PARTITION, 0.0)
        points = sorted({ws, we} | {p for iv in [*lanes.values(), data] for s, e in iv
                                    for p in (s, e) if ws < p < we})
        for lo, hi in zip(points, points[1:], strict=False):
            active = tuple(lane for lane, iv in lanes.items()
                           if any(s <= lo and hi <= e for s, e in iv))
            in_data = any(s <= lo and hi <= e for s, e in data)
            totals[_part(active, in_data)] += hi - lo
        per_step.append(totals)
    return per_step


def partition_step(trace_path: str) -> dict[str, float]:
    """``partition_steps`` over a whole trace, averaged over its steps."""
    events = json.load(open(trace_path))["traceEvents"]
    per_step = partition_steps(events, step_windows(events))
    return {k: sum(s[k] for s in per_step) / len(per_step) for k in PARTITION}


def events_in_step(events: list, window: Interval) -> list:
    """The events of one step: host-side events that start inside the window
    and the device events launched from them (the GPU is drained at every step
    boundary, so a step's device work is launched inside its window)."""
    ws, we = window
    inside = [e for e in events if "ts" in e and e.get("cat") not in DEVICE_CATEGORIES
              and ws <= e["ts"] < we]
    correlations = {e["args"]["correlation"] for e in inside
                    if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    device = [e for e in events if e.get("cat") in DEVICE_CATEGORIES
              and e.get("args", {}).get("correlation") in correlations]
    return inside + device


def step_components(events: list) -> tuple[dict, dict]:
    """The report's rows for one step's events: ``(host, device)`` after the
    host-time projection, the launch-wait subtraction, the derived device rows
    and the computed ``<parent>/other`` rows."""
    host, device, all_device_us, launch_wait, intervals = attribute_events(events)
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
    return host, device


def _aggregate(per_step: list[tuple[dict, dict]], stat) -> tuple[dict, dict]:
    """Combine per-step ``(host, device)`` dicts with ``stat`` over the steps
    (a name absent from a step counts as zero there)."""
    def fresh() -> dict:
        return {"us": 0.0, **{d: {"us": 0.0, "bytes": 0, "count": 0} for d in COPY_DIRECTIONS}}

    names = [n for n in COMPONENTS if any(n in h for h, _ in per_step)]
    names += [n for h, _ in per_step for n in h if n not in names and n.endswith("/other")]
    host, device = {}, defaultdict(fresh)
    for name in dict.fromkeys(names):
        host[name] = {
            "us": stat([h[name]["us"] if name in h else 0.0 for h, _ in per_step]),
            "count": stat([h[name]["count"] if name in h else 0 for h, _ in per_step]),
        }
        device[name]["us"] = stat([d[name]["us"] if name in d else 0.0 for _, d in per_step])
        for direction in COPY_DIRECTIONS:
            for field in ("us", "bytes", "count"):
                device[name][direction][field] = stat(
                    [d[name][direction][field] if name in d else 0 for _, d in per_step])
    return host, device


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _table(host: dict, device: dict, spread: tuple[dict, dict] | None) -> list[str]:
    """Markdown rows of the component table from aggregated ``(host, device)``;
    with ``spread`` (another aggregate, the std), each time cell gets a ± column
    instead of a share column."""
    step_host_us, step_device_us = host["step"]["us"], max(device["step"]["us"], 1.0)
    if spread is None:
        rows = ["| component | calls/step | CPU ms/step | CPU % | CUDA ms/step | CUDA % |"]
    else:
        rows = ["| component | calls/step | CPU ms/step | ± std | CUDA ms/step | ± std |"]
    rows.append("|---|---:|---:|---:|---:|---:|")

    def cell(us: float | None, share_of: float, name: str, kind: str) -> str:
        if us is None:
            return " | "
        us = 0.0 if abs(us) < 1.0 else us  # sub-microsecond rounding dust
        if spread is None:
            return f"{us / 1e3:.2f} | {100 * us / share_of:.1f}"
        std = spread[0][name]["us"] if kind == "cpu" else spread[1][name]["us"]
        return f"{us / 1e3:.2f} | {std / 1e3:.2f}"

    def copy_cell(name: str, direction: str) -> str:
        us = device[name][direction]["us"]
        if spread is None:
            return f"{us / 1e3:.2f} | {100 * us / step_device_us:.1f}"
        return f"{us / 1e3:.2f} | {spread[1][name][direction]['us'] / 1e3:.2f}"

    top_level = {"step", "fwd/backbone", "fwd/engram", "backward", "optimizer"}

    def emit(name: str) -> None:
        indent = "" if name in top_level else "&nbsp;&nbsp;"
        rows.append(f"| {indent}{name} | {host[name]['count']:g} | "
                    f"{cell(host[name]['us'], step_host_us, name, 'cpu')} | "
                    f"{cell(device[name]['us'], step_device_us, name, 'cuda')} |")
        for direction in COPY_DIRECTIONS:
            copy = device[name][direction]
            if copy["count"] == 0 and name != "step":
                continue  # every range gets copy rows once it has any; step always
            rows.append(f"| {indent}&nbsp;&nbsp;· {direction} copies, {copy['bytes'] / 1e6:.2f} MB "
                        f"| {copy['count']:g} |  |  | {copy_cell(name, direction)} |")

    # computed remainders print right after the parent's last child
    other_after = {}
    for parent, kids in PARENTS.items():
        if parent not in host:
            continue
        present = [k for k in kids if k in host]
        last = present[-1] if present else parent
        other_after[last + "/other" if last in PARENTS else last] = parent + "/other"
    for name in COMPONENTS:
        if name not in host:
            continue
        emit(name)
        while name in other_after:
            name = other_after[name]
            emit(name)
    return rows


def write_report(prof: profile, out_dir: str, active_steps: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    prof.export_chrome_trace(os.path.join(out_dir, "profile_trace.json"))
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    report_from_trace(out_dir, active_steps, peak_gib)


def report_from_trace(out_dir: str, active_steps: int, peak_gib: float | None) -> None:
    """Build ``profile.md`` (the mean over the recorded steps, then the median
    with the std across steps) and ``steps.json`` (every step's rows and its
    wall-time partition) from an exported ``profile_trace.json``."""
    events = json.load(open(os.path.join(out_dir, "profile_trace.json")))["traceEvents"]
    steps = step_windows(events)
    if len(steps) != active_steps:
        raise ValueError(f"expected {active_steps} recorded steps, the trace has {len(steps)}")
    per_step = [step_components(events_in_step(events, w)) for w in steps]
    parts = partition_steps(events, steps)
    mean = _aggregate(per_step, statistics.fmean)
    median = _aggregate(per_step, statistics.median)
    std = _aggregate(per_step, _std)

    lines = [f"Profile of {active_steps} training steps (after warmup), mean per step.", ""]
    lines += _table(*mean, spread=None)
    lines += ["", f"Median per step over the {active_steps} steps, with the std across them.", ""]
    lines += _table(*median, spread=std)
    if peak_gib is not None:
        lines += ["", f"Peak CUDA memory allocated: {peak_gib:.2f} GiB"]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "profile.md"), "w") as f:
        f.write(report)
    print(report)
    print(f"profile written to {os.path.join(out_dir, 'profile.md')}")

    per_step_ms = [
        {name: {"cpu_ms": h[name]["us"] / 1e3, "cuda_ms": d[name]["us"] / 1e3} for name in h}
        for h, d in per_step
    ]
    with open(os.path.join(out_dir, "steps.json"), "w") as f:
        json.dump({"components": per_step_ms,
                   "partition": [{k: v / 1e3 for k, v in s.items()} for s in parts]}, f, indent=1)
    mean_parts = {k: statistics.fmean(s[k] for s in parts) / 1e3 for k in PARTITION}
    with open(os.path.join(out_dir, "partition.json"), "w") as f:
        json.dump(mean_parts, f, indent=1)  # ms per step
    print("step wall time split: " + ", ".join(f"{k} {v:.1f} ms" for k, v in mean_parts.items()))
