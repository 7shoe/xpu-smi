"""
xpu_smi.metrics — Parse xpu-smi dump output into structured dicts/tensors.

Metric IDs (from ``xpu-smi dump --help``):
    0  = GPU Utilization (%)
    1  = GPU Power (W)
    2  = GPU Frequency (MHz)
    3  = GPU Core Temperature (°C)
    5  = GPU Memory Temperature (°C)
    9  = GPU EU Array Active (%)
   18  = GPU Memory Used (MiB)
   22  = GPU Memory Bandwidth Utilization (%)
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger("xpu_smi")


# ---------------------------------------------------------------------------
# Metric ID presets
# ---------------------------------------------------------------------------

# "Standard" preset: EU activity, power, temp, memory used
METRICS_STANDARD = "9,1,3,18"

# "Extended" preset: adds frequency, mem-temp, mem-bandwidth
METRICS_EXTENDED = "9,0,1,2,3,5,18,22"


# ---------------------------------------------------------------------------
# Raw dump runner
# ---------------------------------------------------------------------------

def run_dump(
    binary: str,
    env: dict,
    num_devices: int,
    metric_ids: str = METRICS_STANDARD,
    num_samples: int = 1,
    interval_ms: int = 1000,
    timeout: int = 30,
) -> str:
    """
    Run ``xpu-smi dump`` and return raw stdout.

    Parameters
    ----------
    binary : str
        Path to the xpu-smi binary.
    env : dict
        Environment dict (use ``probe._build_env``).
    num_devices : int
        Number of XPU devices.
    metric_ids : str
        Comma-separated metric IDs (default: "9,1,3,18").
    num_samples : int
        Number of dump samples per device (-n flag).
    interval_ms : int
        Interval between samples in ms (-i flag); only used if num_samples > 1.
    timeout : int
        Subprocess timeout in seconds.
    """
    device_ids = ",".join(str(i) for i in range(num_devices))
    cmd = [binary, "dump", "-d", device_ids, "-m", metric_ids, "-n", str(num_samples)]
    if num_samples > 1:
        cmd += ["-i", str(interval_ms)]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"xpu-smi dump failed (rc={result.returncode}): {result.stderr[:300]}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _safe_float(val: str) -> Optional[float]:
    """Convert a string to float, returning None for N/A or unparseable."""
    val = val.strip()
    if not val or val.upper() in ("N/A", "NA", ""):
        return None
    try:
        return float(val)
    except ValueError:
        return None


@dataclass
class DeviceSample:
    """One sample row from xpu-smi dump for a single device."""

    timestamp: str
    device_id: int
    values: Dict[str, Optional[float]]  # metric_name → value or None


def parse_dump(stdout: str) -> List[DeviceSample]:
    """
    Parse the CSV-like output of ``xpu-smi dump``.

    Returns a list of DeviceSample, one per output row.
    """
    lines = [l.strip() for l in stdout.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return []

    # Header: "Timestamp, DeviceId, GPU EU Array Active (%), GPU Power (W), ..."
    header = [h.strip() for h in lines[0].split(",")]
    # Normalize header names to snake_case keys
    key_map = _header_to_keys(header)

    samples: List[DeviceSample] = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        ts = parts[0]
        dev_id = int(parts[1])
        values: Dict[str, Optional[float]] = {}
        for idx in range(2, min(len(parts), len(header))):
            key = key_map.get(idx)
            if key:
                values[key] = _safe_float(parts[idx])
        samples.append(DeviceSample(timestamp=ts, device_id=dev_id, values=values))

    return samples


def _header_to_keys(header: List[str]) -> Dict[int, str]:
    """Map column index → clean snake_case key."""
    mapping: Dict[int, str] = {}
    for idx, h in enumerate(header):
        if idx < 2:  # skip Timestamp, DeviceId
            continue
        key = _to_snake(h)
        mapping[idx] = key
    return mapping


def _to_snake(name: str) -> str:
    """Convert 'GPU EU Array Active (%)' → 'xpu_eu_array_active_pct'."""
    s = name.lower()
    s = s.replace("gpu ", "xpu_").replace("(%)", "pct").replace("(w)", "w")
    s = s.replace("(celsius degree)", "c").replace("(mib)", "mib")
    s = s.replace("(mhz)", "mhz")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    # Collapse double underscores
    s = re.sub(r"_+", "_", s)
    return s


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_samples(samples: List[DeviceSample]) -> Dict[str, float]:
    """
    Aggregate per-device samples into a single summary dict.

    Returns metrics like:
        xpu_num_devices, xpu_eu_active_avg_pct, xpu_eu_active_max_pct,
        xpu_power_total_w, xpu_power_max_w, xpu_temp_max_c, xpu_temp_avg_c,
        xpu_mem_used_total_mib, xpu_mem_used_max_mib, xpu_mem_total_mib
    """
    if not samples:
        return {}

    device_ids = sorted(set(s.device_id for s in samples))
    n_dev = len(device_ids)

    # Collect per-device latest values
    latest: Dict[int, Dict[str, Optional[float]]] = {}
    for s in samples:
        latest[s.device_id] = s.values  # last write wins

    result: Dict[str, float] = {"xpu_num_devices": float(n_dev)}

    # Helper to collect one metric across devices
    def _collect(key_substr: str) -> List[float]:
        vals = []
        for dev_vals in latest.values():
            for k, v in dev_vals.items():
                if key_substr in k and v is not None:
                    vals.append(v)
                    break
        return vals

    # EU active
    eu = _collect("eu")
    if eu:
        result["xpu_eu_active_avg_pct"] = sum(eu) / len(eu)
        result["xpu_eu_active_max_pct"] = max(eu)

    # Power
    pwr = _collect("power")
    if pwr:
        result["xpu_power_total_w"] = sum(pwr)
        result["xpu_power_max_w"] = max(pwr)

    # Temperature
    temp = _collect("temp")
    if temp:
        result["xpu_temp_max_c"] = max(temp)
        result["xpu_temp_avg_c"] = sum(temp) / len(temp)

    # Memory used
    mem = _collect("memory_used") or _collect("mem_used")
    if mem:
        result["xpu_mem_used_total_mib"] = sum(mem)
        result["xpu_mem_used_max_mib"] = max(mem)
        # Max 1550 has 128 GiB HBM per device = 131072 MiB
        result["xpu_mem_total_mib"] = float(n_dev * 131072)

    return result


# ---------------------------------------------------------------------------
# CPU / RAM side-channel
# ---------------------------------------------------------------------------

def cpu_ram_metrics() -> Dict[str, float]:
    """Get CPU and RAM metrics (requires psutil)."""
    if not _HAS_PSUTIL:
        logger.debug("psutil not available — skipping CPU/RAM metrics")
        return {}

    vm = psutil.virtual_memory()
    return {
        "cpu_util_pct": psutil.cpu_percent(interval=0.1),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_avail_gb": round(vm.available / (1024**3), 2),
    }


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def snapshot_sync(
    binary: str,
    env: dict,
    num_devices: int,
    metric_ids: str = METRICS_STANDARD,
    include_cpu: bool = True,
) -> Dict[str, float]:
    """
    One-shot synchronous snapshot: run dump, parse, aggregate, return dict.
    """
    t0 = time.monotonic()
    stdout = run_dump(binary, env, num_devices, metric_ids=metric_ids)
    samples = parse_dump(stdout)
    result = aggregate_samples(samples)
    if include_cpu:
        result.update(cpu_ram_metrics())
    result["_latency_s"] = round(time.monotonic() - t0, 3)
    return result


def snapshot_tensor(
    binary: str,
    env: dict,
    num_devices: int,
    keys: Optional[List[str]] = None,
) -> Tuple[List[str], List[float]]:
    """
    Return metrics as (key_names, values) lists — easy to convert to a tensor.

    Default key order matches the training health vector:
        [eu_avg, eu_max, power_total, power_max, temp_max,
         mem_used_total, mem_used_max, cpu_util, ram_used, ram_total]
    """
    DEFAULT_KEYS = [
        "xpu_eu_active_avg_pct",
        "xpu_eu_active_max_pct",
        "xpu_power_total_w",
        "xpu_power_max_w",
        "xpu_temp_max_c",
        "xpu_mem_used_total_mib",
        "xpu_mem_used_max_mib",
        "cpu_util_pct",       # requires psutil
        "ram_used_gb",        # requires psutil
        "ram_total_gb",       # requires psutil
    ]
    if keys is None:
        keys = DEFAULT_KEYS

    data = snapshot_sync(binary, env, num_devices)
    values = [data.get(k, 0.0) for k in keys]
    return keys, values
