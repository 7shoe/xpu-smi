"""
xpu_smi.monitor — High-level XPU monitor with sync and async interfaces.

Usage::

    from xpu_smi import XPUMonitor

    mon = XPUMonitor()                  # auto-probe
    print(mon)                          # XPUMonitor(available, 6 devices, v1.3.1)

    # Synchronous (blocks ~7s on first call)
    metrics = mon.snapshot()
    print(metrics["xpu_power_total_w"])

    # Tensor for ML health logging
    keys, vals = mon.as_tensor()

    # Async background reader for training loops
    mon.start_async(interval=15.0)
    latest = mon.latest()               # returns last successful snapshot (non-blocking)
    mon.stop_async()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .metrics import (
    METRICS_STANDARD,
    aggregate_samples,
    cpu_ram_metrics,
    parse_dump,
    run_dump,
    snapshot_sync,
    snapshot_tensor,
)
from .probe import XPUSMICandidate, XPUSMINotFoundError, find_best_binary, _build_env

logger = logging.getLogger("xpu_smi")


class XPUMonitor:
    """
    Central monitor object — wraps xpu-smi with Pythonic sync/async APIs.

    Parameters
    ----------
    binary_path : str, optional
        Explicit path to an xpu-smi binary.  If not given, auto-probes.
    auto_probe : bool
        If True (default), automatically find the best binary at init.
    metric_ids : str
        Comma-separated xpu-smi metric IDs for dump calls.
    """

    def __init__(
        self,
        binary_path: Optional[str] = None,
        auto_probe: bool = True,
        metric_ids: str = METRICS_STANDARD,
    ):
        self._binary: Optional[str] = None
        self._env: dict = {}
        self._num_devices: int = 0
        self._version: str = "unknown"
        self._aurora_version: str = "unknown"
        self._available: bool = False
        self._metric_ids: str = metric_ids

        # Async state
        self._async_thread: Optional[threading.Thread] = None
        self._async_stop: threading.Event = threading.Event()
        self._latest_snapshot: Dict[str, float] = {}
        self._latest_lock: threading.Lock = threading.Lock()
        self._async_successes: int = 0
        self._async_failures: int = 0

        if binary_path:
            self._init_from_path(binary_path)
        elif auto_probe:
            self._init_from_probe()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_from_probe(self) -> None:
        """Auto-discover the best xpu-smi binary."""
        try:
            best = find_best_binary()
            self._set_candidate(best)
        except XPUSMINotFoundError:
            logger.warning("No working xpu-smi found — monitor unavailable")
            self._available = False

    def _init_from_path(self, path: str) -> None:
        """Initialize from an explicit binary path."""
        from .probe import XPUSMICandidate, probe_single

        c = XPUSMICandidate(
            path=path, aurora_version="custom", smi_version="custom",
            smi_version_tuple=(999,),
        )
        probe_single(c)
        if c.discovery_ok and c.dump_ok:
            self._set_candidate(c)
        else:
            logger.warning(f"Binary {path} failed validation: {c.error}")
            self._available = False

    def _set_candidate(self, c: XPUSMICandidate) -> None:
        self._binary = c.path
        self._env = _build_env(c)
        self._num_devices = c.num_devices
        self._version = c.smi_version
        self._aurora_version = c.aurora_version
        self._available = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if a working xpu-smi binary was found."""
        return self._available

    @property
    def num_devices(self) -> int:
        return self._num_devices

    @property
    def version(self) -> str:
        return self._version

    @property
    def binary_path(self) -> Optional[str]:
        return self._binary

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def snapshot(self, include_cpu: bool = True) -> Dict[str, float]:
        """
        Blocking snapshot of all XPU + CPU/RAM metrics.

        Note: takes ~5-8s due to xpu-smi dump latency.
        """
        self._require_available()
        return snapshot_sync(
            self._binary, self._env, self._num_devices,
            metric_ids=self._metric_ids, include_cpu=include_cpu,
        )

    def as_tensor(
        self, keys: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[float]]:
        """Return (key_names, float_values) for easy conversion to a tensor."""
        self._require_available()
        return snapshot_tensor(
            self._binary, self._env, self._num_devices, keys=keys,
        )

    # ------------------------------------------------------------------
    # Asynchronous background reader
    # ------------------------------------------------------------------

    def start_async(self, interval: float = 15.0) -> None:
        """
        Start a background thread that periodically dumps metrics.

        Use ``latest()`` to read the most recent successful snapshot
        (non-blocking, zero latency from the caller's perspective).
        """
        self._require_available()
        if self._async_thread and self._async_thread.is_alive():
            logger.warning("Async reader already running")
            return

        self._async_stop.clear()
        self._async_successes = 0
        self._async_failures = 0

        def _loop():
            logger.info(f"Async reader started (interval={interval}s)")
            while not self._async_stop.is_set():
                try:
                    data = snapshot_sync(
                        self._binary, self._env, self._num_devices,
                        metric_ids=self._metric_ids, include_cpu=True,
                    )
                    with self._latest_lock:
                        self._latest_snapshot = data
                    self._async_successes += 1
                    if self._async_successes == 1:
                        logger.info(
                            f"async reader: first successful dump "
                            f"({len(data)} metrics)"
                        )
                except Exception as e:
                    self._async_failures += 1
                    logger.warning(f"async reader error: {e}")

                self._async_stop.wait(timeout=interval)

            logger.info(
                f"Stopping async reader — "
                f"successes={self._async_successes} "
                f"failures={self._async_failures}"
            )

        self._async_thread = threading.Thread(target=_loop, daemon=True)
        self._async_thread.start()

    def stop_async(self) -> None:
        """Stop the background reader."""
        self._async_stop.set()
        if self._async_thread:
            self._async_thread.join(timeout=45)
            self._async_thread = None

    def latest(self) -> Dict[str, float]:
        """
        Return the most recent async snapshot (non-blocking).

        Returns an empty dict if no successful dump has completed yet.
        Includes CPU/RAM metrics alongside XPU metrics.
        """
        with self._latest_lock:
            return dict(self._latest_snapshot)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_available(self) -> None:
        if not self._available:
            raise XPUSMINotFoundError()

    def diagnostics(self) -> Dict[str, Any]:
        """Return a diagnostic dict (useful for logging/debugging)."""
        result: Dict[str, Any] = {
            "available": self._available,
            "version": self._version,
            "aurora_version": self._aurora_version,
            "num_devices": self._num_devices,
            "binary_path": self._binary,
            "status": "OK" if self._available else "UNAVAILABLE",
        }
        if self._available:
            try:
                snap = self.snapshot(include_cpu=False)
                result["dump_keys"] = len(snap)
                result["dump_latency_s"] = snap.get("_latency_s", -1)
                result["dump_result"] = {
                    k: v for k, v in snap.items() if not k.startswith("_")
                }
            except Exception as e:
                result["dump_error"] = str(e)
        return result

    def __repr__(self) -> str:
        if self._available:
            return (
                f"XPUMonitor(available, {self._num_devices} devices, "
                f"v{self._version})"
            )
        return "XPUMonitor(unavailable)"

    def __del__(self) -> None:
        if self._async_thread and self._async_thread.is_alive():
            self.stop_async()
