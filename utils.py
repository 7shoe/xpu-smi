#!/usr/bin/env python3
"""
intel_xpu/utils.py — Hardware monitoring utilities for Intel Data Center GPUs on Aurora.

Designed for integration into distributed training scripts (FSDP, DDP) where:
  - Only LOCAL_RANK == 0 per node should call xpu-smi (avoids 12 competing L0 queries)
  - Metrics are all-reduced across nodes so rank 0 can log to W&B
  - Failures must never crash training — all functions degrade gracefully

Key design decisions:
  1. xpu-smi 1.3.1 on Aurora 25.190.0 is broken (spdlog ABI conflict).
     We bypass it by running the binary with an isolated LD_LIBRARY_PATH
     pointing only at xpu-smi's own lib64 + system libs.
  2. xpu-smi dump -n 1 blocks for ~1 second (waits one sampling interval).
     To avoid stalling the training loop, we provide an async (threaded)
     reader that pre-fetches metrics in the background.
  3. CPU/RAM metrics read directly from /proc (zero overhead, no subprocess).

Usage (standalone test):
    python intel_xpu/utils.py

Usage (in training script):
    from intel_xpu.utils import XPUMonitor
    mon = XPUMonitor()                        # probes once at startup
    mon.start_async()                         # launch background thread
    ...
    metrics = mon.snapshot()                  # non-blocking, returns latest
    mon.stop_async()                          # cleanup

Usage (simple, synchronous — for infrequent health checks):
    from intel_xpu.utils import XPUMonitor
    mon = XPUMonitor()
    metrics = mon.read_xpu_smi_sync()         # blocks ~1s
    cpu_ram  = mon.read_cpu_ram()             # instant
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "XPUMonitor",
    "probe_xpu_smi",
    "read_cpu_ram",
    "read_xpu_smi_sync",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Known-good xpu-smi version on Aurora 25.190.0 (Feb '26 stack)
_PREFERRED_VERSIONS = ["1.2.42", "1.2.39", "1.2.36"]

# Search paths for xpu-smi installations
_XPU_SMI_SEARCH_GLOBS = [
    "/opt/aurora/*/support/tools/xpu-smi/*/bin/xpu-smi",
]

# Metric IDs for xpu-smi dump:
#   9  = EU Array Active (%)  — actual compute utilization (metric 0 often returns N/A)
#   1  = GPU Power (W)
#   3  = GPU Core Temperature (°C)
#   18 = GPU Memory Used (MiB)
_METRIC_IDS = "9,1,3,18"

# Column names matching _METRIC_IDS (after timestamp + device_id)
_METRIC_NAMES = ["eu_active_pct", "power_w", "temp_c", "mem_used_mib"]

# 128 GiB HBM2e per Intel Max 1550
_HBM_TOTAL_MIB = 131072

# Sentinel for "no data" in tensor slots.
# Must be negative so all_reduce MAX replaces it with any real value (>= 0),
# and so tensor_to_wandb_dict can distinguish "idle GPU at 0.0%" from
# "xpu-smi failed".
_NO_DATA = -1.0

# ── Logging helper ──────────────────────────────────────────────────────────
def _log(msg: str):
    print(f"[intel_xpu] {msg}", file=sys.stderr, flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  xpu-smi binary detection (runs once at init)
# ═══════════════════════════════════════════════════════════════════════════════

def _find_xpu_smi_candidates() -> List[Tuple[str, Path]]:
    """Discover all xpu-smi installations on the system."""
    import glob as _glob
    found: Dict[str, Path] = {}  # version -> bin path (deduplicated)

    for pattern in _XPU_SMI_SEARCH_GLOBS:
        for p in sorted(_glob.glob(pattern)):
            p = Path(p)
            if p.is_file() and os.access(p, os.X_OK):
                version = p.parent.parent.name  # .../xpu-smi/<ver>/bin/xpu-smi
                if version not in found:
                    found[version] = p

    # Order: preferred versions first, then remaining by version descending
    preferred = []
    others = []
    for ver, path in found.items():
        if ver in _PREFERRED_VERSIONS:
            preferred.append((ver, path))
        else:
            others.append((ver, path))

    preferred.sort(key=lambda x: _PREFERRED_VERSIONS.index(x[0])
                   if x[0] in _PREFERRED_VERSIONS else 999)
    others.sort(key=lambda x: x[0], reverse=True)

    return [(v, p) for v, p in preferred + others]


def _build_isolated_env(bin_path: Path) -> Dict[str, str]:
    """
    Build a CLEAN environment for xpu-smi subprocess.

    Critical: we do NOT simply inherit os.environ and override LD_LIBRARY_PATH.
    Under mpiexec, os.environ can contain LD_PRELOAD with aurora_frameworks
    libs, MPI-injected library paths, and other variables that cause spdlog
    ABI conflicts even when LD_LIBRARY_PATH is overridden (LD_PRELOAD wins).

    Instead, we construct a minimal environment with only what xpu-smi needs:
      - Its own lib64 + system libs on LD_LIBRARY_PATH
      - ZE_* / ONEAPI_* env vars for L0 driver configuration
      - Basic system vars (PATH, HOME)
    """
    lib64 = bin_path.parent.parent / "lib64"
    clean_ld = f"{lib64}:/usr/lib64:/lib64"

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LD_LIBRARY_PATH": clean_ld,
        # L0 driver needs this for sysman (metrics) API
        "ZE_ENABLE_SYSMAN": "1",
    }

    # Forward ZE_* and ONEAPI_* if present (driver config)
    for key, val in os.environ.items():
        if key.startswith("ZE_") or key.startswith("ONEAPI_"):
            env.setdefault(key, val)

    # Explicitly EXCLUDE: LD_PRELOAD, aurora_frameworks paths, CCL/MPI vars
    return env


def _build_isolated_ldpath(bin_path: Path) -> str:
    """Legacy helper — returns just the LD_LIBRARY_PATH string."""
    lib64 = bin_path.parent.parent / "lib64"
    return f"{lib64}:/usr/lib64:/lib64"


def _test_xpu_smi_discovery(
    bin_path: Path, env: Dict[str, str], timeout: float = 15.0,
) -> Optional[List[int]]:
    """
    Test if xpu-smi discovery works.  Returns list of device IDs or None.
    """
    try:
        result = subprocess.run(
            [str(bin_path), "discovery"],
            timeout=timeout,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            _log(f"discovery failed (rc={result.returncode}): "
                 f"{result.stderr[:200] if result.stderr else ''}")
            return None
        raw = result.stdout
    except subprocess.TimeoutExpired:
        _log("discovery timed out")
        return None
    except Exception as e:
        _log(f"discovery exception: {e}")
        return None

    dev_ids = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[1].isdigit():
            dev_ids.append(int(parts[1]))

    return dev_ids if dev_ids else None


def _test_xpu_smi_dump(
    bin_path: Path, env: Dict[str, str], device_ids: List[int],
    timeout: float = 15.0,
) -> bool:
    """
    Test if xpu-smi dump actually works (the command used at runtime).

    FIX: The original code only tested 'discovery' which uses different
    L0 API calls than 'dump'.  'discovery' can succeed while 'dump' fails
    under mpiexec due to LD_PRELOAD conflicts or sysman permission issues.
    """
    dev_args = [str(d) for d in device_ids]
    try:
        result = subprocess.run(
            [str(bin_path), "dump", "-d"] + dev_args +
            ["-m", _METRIC_IDS, "-n", "1"],
            timeout=timeout,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            _log(f"dump probe failed (rc={result.returncode}): "
                 f"{result.stderr[:300] if result.stderr else 'no stderr'}")
            return False

        # Verify we got parseable output (at least one data line)
        lines = [l for l in result.stdout.strip().splitlines()
                 if l.strip() and not l.startswith("Timestamp")]
        if not lines:
            _log(f"dump probe returned no data lines. "
                 f"stdout={result.stdout[:200]}")
            return False

        # Verify we can parse at least one device
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 + len(_METRIC_NAMES):
                try:
                    int(parts[1])    # device_id parseable
                    float(parts[2])  # first metric parseable
                    return True
                except ValueError:
                    continue

        _log(f"dump probe: output not parseable. first_line={lines[0][:200]}")
        return False

    except subprocess.TimeoutExpired:
        _log("dump probe timed out")
        return False
    except Exception as e:
        _log(f"dump probe exception: {e}")
        return False


class _XPUSMIBackend:
    """Resolved xpu-smi binary + its isolated environment."""

    def __init__(self, bin_path: Path, env: Dict[str, str], version: str,
                 device_ids: List[int]):
        self.bin_path = str(bin_path)
        self.env = env      # clean subprocess env (NOT os.environ)
        self.version = version
        self.device_ids = device_ids
        self.num_devices = len(device_ids)

    def __repr__(self):
        return (f"_XPUSMIBackend(v{self.version}, {self.num_devices} devices, "
                f"bin={self.bin_path})")


def probe_xpu_smi(verbose: bool = True) -> Optional[_XPUSMIBackend]:
    """
    Find a working xpu-smi binary.  Returns backend object or None.

    FIX: Now tests BOTH 'discovery' AND 'dump' commands.  The original
    only tested discovery, which could succeed while dump failed under
    mpiexec environments (different L0 calls, LD_PRELOAD interference).

    FIX: Builds a clean subprocess env instead of inheriting os.environ.
    Under mpiexec, LD_PRELOAD and aurora_frameworks paths in os.environ
    cause spdlog ABI conflicts even when LD_LIBRARY_PATH is overridden.

    This runs once at XPUMonitor init.  The probe takes 2-5 seconds
    (two subprocesses: discovery + dump).  Cache the result.
    """
    candidates = _find_xpu_smi_candidates()

    if not candidates:
        if verbose:
            _log("No xpu-smi installations found.")
        return None

    if verbose:
        _log(f"Found {len(candidates)} xpu-smi version(s), probing...")

    for version, bin_path in candidates:
        env = _build_isolated_env(bin_path)
        if verbose:
            _log(f"  Trying v{version} ...")

        # Phase 1: discovery (device enumeration)
        dev_ids = _test_xpu_smi_discovery(bin_path, env)
        if dev_ids is None:
            if verbose:
                _log(f"  x v{version}: discovery failed")
            continue

        if verbose:
            _log(f"  v{version}: discovery OK, {len(dev_ids)} device(s)")

        # Phase 2: dump (the actual runtime command) -- FIX
        if _test_xpu_smi_dump(bin_path, env, dev_ids):
            if verbose:
                _log(f"  v{version}: dump OK -- USING this version")
            return _XPUSMIBackend(bin_path, env, version, dev_ids)
        else:
            if verbose:
                _log(f"  x v{version}: discovery OK but dump FAILED")
            # Continue to next version

    if verbose:
        _log("No working xpu-smi found (all versions failed dump test). "
             "XPU hardware metrics disabled.")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  /proc readers (CPU + RAM -- zero subprocess overhead)
# ═══════════════════════════════════════════════════════════════════════════════

class _CPUSampler:
    """Stateful CPU utilization tracker via /proc/stat deltas."""

    def __init__(self):
        self._prev_idle = 0
        self._prev_total = 0
        self._read_and_update()  # prime baseline

    def _read_and_update(self) -> Optional[float]:
        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()
        except OSError:
            return None

        parts = line.split()
        if len(parts) < 5:
            return None

        values = [int(v) for v in parts[1:9]]
        idle = values[3] + values[4]  # idle + iowait
        total = sum(values)

        if self._prev_total == 0:
            self._prev_idle = idle
            self._prev_total = total
            return None

        d_idle = idle - self._prev_idle
        d_total = total - self._prev_total
        self._prev_idle = idle
        self._prev_total = total

        if d_total == 0:
            return 0.0
        return 100.0 * (1.0 - d_idle / d_total)

    def sample(self) -> Optional[float]:
        return self._read_and_update()


def read_cpu_ram() -> Dict[str, float]:
    """
    Read system RAM from /proc/meminfo.  Returns dict with:
      ram_used_gb, ram_avail_gb, ram_total_gb
    Always succeeds on Linux (returns empty dict on error).
    """
    try:
        with open("/proc/meminfo", "r") as f:
            text = f.read()
    except OSError:
        return {}

    info = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].rstrip(":")
            val_kb = int(parts[1])
            if key == "MemTotal":
                info["ram_total_gb"] = val_kb / (1024 * 1024)
            elif key == "MemAvailable":
                info["ram_avail_gb"] = val_kb / (1024 * 1024)

    if "ram_total_gb" in info and "ram_avail_gb" in info:
        info["ram_used_gb"] = info["ram_total_gb"] - info["ram_avail_gb"]

    return info


# ═══════════════════════════════════════════════════════════════════════════════
#  xpu-smi metric reading (synchronous)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_dump_output(raw: str) -> List[Dict[str, float]]:
    """
    Parse xpu-smi dump CSV output into per-device dicts.

    Input format (one line per device):
      Timestamp, DeviceId, EU Active (%), Power (W), Temp (C), Mem Used (MiB)
      01:46:02.998,    0, 0.00, 265.92, 34.00, 2958.06

    Returns list of dicts, one per device line successfully parsed.
    """
    results = []
    for line in raw.strip().splitlines():
        if line.startswith("Timestamp") or not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 + len(_METRIC_NAMES):
            continue
        try:
            dev_id = int(parts[1])
            metrics = {"device_id": dev_id}
            for i, name in enumerate(_METRIC_NAMES):
                val_str = parts[2 + i]
                if val_str == "N/A" or val_str == "":
                    metrics[name] = None
                else:
                    metrics[name] = float(val_str)
            results.append(metrics)
        except (ValueError, IndexError):
            continue
    return results


def read_xpu_smi_sync(
    backend: _XPUSMIBackend,
    timeout: float = 8.0,
) -> Dict[str, float]:
    """
    Synchronous xpu-smi dump for all devices on this node.
    Blocks for ~1 second (xpu-smi sampling interval).

    Returns node-level aggregated metrics, or empty dict on failure.
    """
    dev_args = [str(d) for d in backend.device_ids]

    try:
        result = subprocess.run(
            [backend.bin_path, "dump", "-d"] + dev_args +
            ["-m", _METRIC_IDS, "-n", "1"],
            timeout=timeout,
            capture_output=True,
            text=True,
            env=backend.env,  # clean env, NOT os.environ
        )
        if result.returncode != 0:
            return {}
        raw = result.stdout
    except Exception:
        return {}

    devices = _parse_dump_output(raw)
    if not devices:
        return {}

    return _aggregate_device_metrics(devices)


def _aggregate_device_metrics(devices: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-device metrics into node-level summary."""
    eu_vals = [d["eu_active_pct"] for d in devices if d.get("eu_active_pct") is not None]
    pwr_vals = [d["power_w"] for d in devices if d.get("power_w") is not None]
    tmp_vals = [d["temp_c"] for d in devices if d.get("temp_c") is not None]
    mem_vals = [d["mem_used_mib"] for d in devices if d.get("mem_used_mib") is not None]

    result: Dict[str, float] = {"xpu_num_devices": len(devices)}

    if eu_vals:
        result["xpu_eu_active_avg_pct"] = sum(eu_vals) / len(eu_vals)
        result["xpu_eu_active_max_pct"] = max(eu_vals)
    if pwr_vals:
        result["xpu_power_total_w"] = sum(pwr_vals)
        result["xpu_power_max_w"] = max(pwr_vals)
    if tmp_vals:
        result["xpu_temp_max_c"] = max(tmp_vals)
        result["xpu_temp_avg_c"] = sum(tmp_vals) / len(tmp_vals)
    if mem_vals:
        result["xpu_mem_used_total_mib"] = sum(mem_vals)
        result["xpu_mem_used_max_mib"] = max(mem_vals)
        result["xpu_mem_total_mib"] = len(mem_vals) * _HBM_TOTAL_MIB

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Async (threaded) reader -- avoids blocking training loop
# ═══════════════════════════════════════════════════════════════════════════════

class _AsyncXPUReader:
    """
    Background thread that periodically calls xpu-smi dump.

    FIX: Now tracks success/fail counts and logs the first failure for
    diagnosis, instead of silently swallowing all exceptions.

    Thread safety: _latest is replaced atomically (dict assignment is
    GIL-atomic in CPython), so no lock needed for reads.
    """

    def __init__(self, backend: _XPUSMIBackend, interval: float = 10.0):
        self._backend = backend
        self._interval = interval
        self._latest: Dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.success_count = 0
        self.fail_count = 0
        self._first_fail_logged = False

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="xpu-monitor",
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

    def latest(self) -> Dict[str, float]:
        """Return most recent metrics (non-blocking)."""
        return self._latest

    def _loop(self):
        while not self._stop.is_set():
            try:
                result = read_xpu_smi_sync(self._backend, timeout=8.0)
                if result:
                    self._latest = result
                    self.success_count += 1
                    if self.success_count == 1:
                        _log(f"async reader: first successful dump "
                             f"({len(result)} metrics)")
                else:
                    self.fail_count += 1
                    if not self._first_fail_logged:
                        _log(f"async reader: dump returned empty "
                             f"(after {self.success_count} successes). "
                             f"XPU hardware metrics may be missing from W&B.")
                        self._first_fail_logged = True
            except Exception as e:
                self.fail_count += 1
                if not self._first_fail_logged:
                    _log(f"async reader: dump exception: {e}")
                    self._first_fail_logged = True

            self._stop.wait(self._interval)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main interface: XPUMonitor
# ═══════════════════════════════════════════════════════════════════════════════

class XPUMonitor:
    """
    Unified hardware monitor for Intel XPU nodes on Aurora.

    Typical usage in distributed training (one instance per rank):

        # At startup (all ranks):
        monitor = XPUMonitor(async_interval=15.0)
        if LOCAL_RANK == 0:
            monitor.start_async()   # only one reader per node

        # In health logging block (all ranks participate in all-reduce):
        if global_step % health_log_every == 0:
            hw_vals = [_NO_DATA] * 10
            if LOCAL_RANK == 0:
                hw_vals = monitor.snapshot_as_tensor_values()
            hw_t = torch.tensor(hw_vals, device=device, dtype=torch.float32)
            dist.all_reduce(hw_t, op=dist.ReduceOp.MAX)
            if RANK == 0 and wandb_run:
                wandb_run.log(XPUMonitor.tensor_to_wandb_dict(hw_t.tolist()),
                              step=global_step)

        # At shutdown:
        monitor.stop_async()

    Attributes:
        available (bool):  True if xpu-smi discovery AND dump both work
        backend:           Resolved binary info (or None)
        num_devices (int): Number of XPU devices on this node (0 if unavailable)
        version (str):     xpu-smi version string (or "N/A")
    """

    def __init__(
        self,
        async_interval: float = 15.0,
        verbose: bool = True,
        probe: bool = True,
    ):
        self._verbose = verbose
        self._async_interval = async_interval
        self._cpu_sampler = _CPUSampler()
        self._async_reader: Optional[_AsyncXPUReader] = None

        self.backend: Optional[_XPUSMIBackend] = None
        if probe:
            self.backend = probe_xpu_smi(verbose=verbose)

        self.available = self.backend is not None
        self.num_devices = self.backend.num_devices if self.backend else 0
        self.version = self.backend.version if self.backend else "N/A"

    # ── Async (background thread) interface ──────────────────────────

    def start_async(self):
        """
        Start background xpu-smi polling thread.

        Call ONLY from LOCAL_RANK == 0 on each node.
        """
        if not self.available:
            if self._verbose:
                _log("start_async: no backend, skipping")
            return
        if self._async_reader is not None:
            return

        self._async_reader = _AsyncXPUReader(
            self.backend, interval=self._async_interval,
        )
        self._async_reader.start()
        if self._verbose:
            _log(f"Async reader started (interval={self._async_interval}s)")

    def stop_async(self):
        """Stop background thread (idempotent)."""
        if self._async_reader is not None:
            if self._verbose:
                _log(f"Stopping async reader -- "
                     f"successes={self._async_reader.success_count} "
                     f"failures={self._async_reader.fail_count}")
            self._async_reader.stop()
            self._async_reader = None

    # ── Snapshot: combined XPU + CPU/RAM metrics ─────────────────────

    def snapshot(self) -> Dict[str, float]:
        """
        Return latest hardware metrics.  Non-blocking if async is running.
        Empty dict entries are omitted (not set to 0).
        """
        metrics: Dict[str, float] = {}

        if self._async_reader is not None:
            metrics.update(self._async_reader.latest())
        elif self.available:
            metrics.update(read_xpu_smi_sync(self.backend))

        cpu = self._cpu_sampler.sample()
        if cpu is not None:
            metrics["cpu_util_pct"] = cpu

        metrics.update(read_cpu_ram())
        return metrics

    # ── Synchronous read (for one-off checks) ───────────────────────

    def read_xpu_sync(self) -> Dict[str, float]:
        """Synchronous xpu-smi read.  Blocks ~1s."""
        if not self.available:
            return {}
        return read_xpu_smi_sync(self.backend)

    def read_cpu_ram(self) -> Dict[str, float]:
        """Read CPU util + RAM.  Instant (no subprocess)."""
        metrics: Dict[str, float] = {}
        cpu = self._cpu_sampler.sample()
        if cpu is not None:
            metrics["cpu_util_pct"] = cpu
        metrics.update(read_cpu_ram())
        return metrics

    # ── Integration helper: build tensor for all-reduce ──────────────

    def snapshot_as_tensor_values(self) -> List[float]:
        """
        Return snapshot as a flat list of floats in a FIXED ORDER,
        suitable for packing into a torch tensor for all-reduce MAX.

        Order (10 values):
          [0] xpu_eu_active_avg_pct
          [1] xpu_eu_active_max_pct
          [2] xpu_power_total_w
          [3] xpu_power_max_w
          [4] xpu_temp_max_c
          [5] xpu_mem_used_total_mib
          [6] xpu_mem_used_max_mib
          [7] cpu_util_pct
          [8] ram_used_gb
          [9] ram_total_gb

        Missing values are _NO_DATA (-1.0), NOT 0.0.

        FIX: The v1 code used 0.0 as sentinel, which made EU Active = 0.0%
        (idle GPUs) indistinguishable from "xpu-smi failed".  After
        all_reduce MAX + tensor_to_wandb_dict filtering != 0.0, valid
        idle-GPU readings were silently dropped.  Sentinel is now -1.0:
        real values (>= 0) always win under MAX, and the dict builder
        filters on > _NO_DATA.
        """
        s = self.snapshot()
        return [
            s.get("xpu_eu_active_avg_pct", _NO_DATA),
            s.get("xpu_eu_active_max_pct", _NO_DATA),
            s.get("xpu_power_total_w", _NO_DATA),
            s.get("xpu_power_max_w", _NO_DATA),
            s.get("xpu_temp_max_c", _NO_DATA),
            s.get("xpu_mem_used_total_mib", _NO_DATA),
            s.get("xpu_mem_used_max_mib", _NO_DATA),
            s.get("cpu_util_pct", _NO_DATA),
            s.get("ram_used_gb", _NO_DATA),
            s.get("ram_total_gb", _NO_DATA),
        ]

    TENSOR_KEYS = [
        "health/xpu_eu_active_avg_pct",
        "health/xpu_eu_active_max_pct",
        "health/xpu_power_total_w",
        "health/xpu_power_max_w",
        "health/xpu_temp_max_c",
        "health/xpu_mem_used_total_mib",
        "health/xpu_mem_used_max_mib",
        "health/cpu_util_max_pct",
        "health/ram_used_max_gb",
        "health/ram_total_gb",
    ]

    @staticmethod
    def tensor_to_wandb_dict(values: List[float]) -> Dict[str, float]:
        """
        Convert the 10-element list (post all-reduce MAX) to a W&B log dict.

        FIX: Filters on > _NO_DATA (> -1.0) instead of != 0.0.
        EU Active = 0.0% now correctly logged; only true missing data
        (all nodes returned -1.0) is filtered out.
        """
        return {
            k: v for k, v in zip(XPUMonitor.TENSOR_KEYS, values)
            if v > _NO_DATA
        }

    # ── Diagnostic helper ────────────────────────────────────────────

    def diagnostic_sync_test(self) -> Dict[str, Any]:
        """
        Run a full diagnostic: probe status + timed sync dump + stderr capture.
        Call from LOCAL_RANK == 0 at startup to verify xpu-smi works under mpiexec.

        Returns dict with all diagnostic info for logging.
        """
        diag: Dict[str, Any] = {
            "available": self.available,
            "version": self.version,
            "num_devices": self.num_devices,
        }

        if not self.available:
            diag["status"] = "NO_BACKEND"
            return diag

        t0 = time.time()
        result = read_xpu_smi_sync(self.backend, timeout=10.0)
        dt = time.time() - t0

        diag["dump_latency_s"] = round(dt, 3)
        diag["dump_keys"] = len(result)
        diag["dump_result"] = {k: round(v, 2) if isinstance(v, float) else v
                               for k, v in result.items()}

        if result:
            diag["status"] = "OK"
        else:
            diag["status"] = "DUMP_FAILED"
            try:
                dev_args = [str(d) for d in self.backend.device_ids]
                r = subprocess.run(
                    [self.backend.bin_path, "dump", "-d"] + dev_args +
                    ["-m", _METRIC_IDS, "-n", "1"],
                    timeout=10, capture_output=True, text=True,
                    env=self.backend.env,
                )
                diag["dump_stderr"] = r.stderr[:500] if r.stderr else ""
                diag["dump_rc"] = r.returncode
            except Exception as e:
                diag["dump_stderr"] = str(e)

        return diag

    def print_status(self):
        """Print nvidia-smi style status to stdout."""
        print(f"\n{'='*60}")
        print(f"  Intel XPU Monitor -- v{self.version}")
        print(f"  Devices: {self.num_devices}")
        print(f"{'='*60}")

        if not self.available:
            print("  No XPU devices available (login node or xpu-smi broken)")
            print(f"{'='*60}\n")
            return

        snap = self.snapshot()
        for k, v in sorted(snap.items()):
            if isinstance(v, float):
                print(f"  {k:35s} {v:>10.2f}")
            else:
                print(f"  {k:35s} {v!s:>10s}")
        print(f"{'='*60}\n")

    def __repr__(self):
        status = "available" if self.available else "unavailable"
        return f"XPUMonitor({status}, {self.num_devices} devices, v{self.version})"


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Intel XPU Monitor -- Standalone Test")
    print("=" * 60)

    monitor = XPUMonitor(verbose=True)
    print(f"\nMonitor: {monitor}")

    # Diagnostic (most important -- reproduces the mpiexec failure mode)
    print("\n--- Diagnostic Sync Test ---")
    diag = monitor.diagnostic_sync_test()
    for k, v in sorted(diag.items()):
        print(f"  {k}: {v}")

    # CPU/RAM (always works)
    print("\n--- CPU / RAM ---")
    cr = monitor.read_cpu_ram()
    for k, v in sorted(cr.items()):
        print(f"  {k}: {v:.2f}")

    if not monitor.available:
        print("\nNo XPU backend available. Exiting.")
        sys.exit(0)

    # Synchronous XPU read (~1s)
    print("\n--- XPU Metrics (sync, ~1s) ---")
    t0 = time.time()
    xpu = monitor.read_xpu_sync()
    dt = time.time() - t0
    for k, v in sorted(xpu.items()):
        print(f"  {k}: {v:.2f}")
    print(f"  (took {dt:.2f}s)")

    # Async test
    print("\n--- Async Test (3 snapshots, 5s apart) ---")
    monitor.start_async()
    for i in range(3):
        time.sleep(5)
        snap = monitor.snapshot()
        print(f"  [{i+1}] {len(snap)} metrics: "
              f"eu={snap.get('xpu_eu_active_avg_pct', '?')}% "
              f"pwr={snap.get('xpu_power_total_w', '?')}W "
              f"temp={snap.get('xpu_temp_max_c', '?')}C "
              f"cpu={snap.get('cpu_util_pct', '?')}%")
    monitor.stop_async()

    # Tensor helper test
    print("\n--- Tensor Helper ---")
    vals = monitor.snapshot_as_tensor_values()
    print(f"  raw: {[f'{v:.2f}' for v in vals]}")
    wandb_dict = XPUMonitor.tensor_to_wandb_dict(vals)
    for k, v in wandb_dict.items():
        print(f"  {k}: {v:.2f}")

    print("\nDone.")