"""
xpu_smi — Python wrapper for Intel XPU System Management Interface.

Quick start::

    from xpu_smi import XPUMonitor
    mon = XPUMonitor()           # auto-discovers best xpu-smi binary
    print(mon)                   # XPUMonitor(available, 6 devices, v1.3.1)
    print(mon.snapshot_sync())   # dict of aggregated metrics

Install::

    pip install xpu-smi
"""

from .monitor import XPUMonitor
from .probe import probe_versions, find_best_binary, XPUSMINotFoundError
from .metrics import snapshot_sync, snapshot_tensor

__version__ = "0.1.0"

__all__ = [
    "XPUMonitor",
    "probe_versions",
    "find_best_binary",
    "snapshot_sync",
    "snapshot_tensor",
    "XPUSMINotFoundError",
]
