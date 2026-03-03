"""
xpu_smi.__main__ — CLI entry point.

Run as::

    python -m xpu_smi              # full diagnostic
    python -m xpu_smi --probe      # just probe available versions
    python -m xpu_smi --snapshot   # single sync snapshot
    python -m xpu_smi --json       # JSON output for scripting
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .monitor import XPUMonitor
from .probe import discover_candidates, probe_versions

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[xpu_smi] %(message)s",
        level=level,
        stream=sys.stderr,
    )


def cmd_probe(args: argparse.Namespace) -> int:
    """List and probe all available xpu-smi binaries."""
    candidates = probe_versions()
    if args.json:
        out = []
        for c in candidates:
            out.append({
                "path": c.path,
                "aurora_version": c.aurora_version,
                "smi_version": c.smi_version,
                "discovery_ok": c.discovery_ok,
                "dump_ok": c.dump_ok,
                "num_devices": c.num_devices,
                "error": c.error,
            })
        print(json.dumps(out, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  xpu-smi Version Probe")
        print(f"{'='*60}")
        for c in candidates:
            status = "OK" if (c.discovery_ok and c.dump_ok) else "FAIL"
            print(
                f"  [{status}] v{c.smi_version:8s}  "
                f"aurora={c.aurora_version:12s}  "
                f"devices={c.num_devices}  "
                f"{c.error}"
            )
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Take a single synchronous snapshot."""
    mon = XPUMonitor()
    if not mon.available:
        print("ERROR: No XPU devices available", file=sys.stderr)
        return 1

    snap = mon.snapshot()
    if args.json:
        print(json.dumps(snap, indent=2))
    else:
        print(f"\n--- XPU Snapshot (v{mon.version}, {mon.num_devices} devices) ---")
        for k, v in sorted(snap.items()):
            if k.startswith("_"):
                continue
            print(f"  {k}: {v:.2f}")
        print(f"  (took {snap.get('_latency_s', '?')}s)")
    return 0


def cmd_diag(args: argparse.Namespace) -> int:
    """Full diagnostic test (mirrors the old utils.py __main__)."""
    print(f"\n{'='*60}")
    print(f"  Intel XPU Monitor — Diagnostic")
    print(f"{'='*60}")

    mon = XPUMonitor()
    print(f"\nMonitor: {mon}")

    if not mon.available:
        print("\nERROR: No XPU available. Run `python -m xpu_smi --probe` for details.")
        return 1

    # Diagnostics
    diag = mon.diagnostics()
    print(f"\n--- Diagnostics ---")
    for k, v in sorted(diag.items()):
        print(f"  {k}: {v}")

    # Sync snapshot
    print(f"\n--- Sync Snapshot ---")
    snap = mon.snapshot()
    for k, v in sorted(snap.items()):
        if k.startswith("_"):
            continue
        print(f"  {k}: {v:.2f}")
    print(f"  (took {snap.get('_latency_s', '?')}s)")

    # Tensor
    print(f"\n--- Tensor Vector ---")
    keys, vals = mon.as_tensor()
    raw = [f"{v:.2f}" for v in vals]
    print(f"  raw: {raw}")
    for k, v in zip(keys, vals):
        print(f"  health/{k}: {v:.2f}")

    # Brief async test
    print(f"\n--- Async Test (2 snapshots, 3s apart) ---")
    mon.start_async(interval=10.0)
    for i in range(2):
        time.sleep(3)
        lat = mon.latest()
        n = len(lat)
        eu = lat.get("xpu_eu_active_avg_pct", "?")
        pwr = lat.get("xpu_power_total_w", "?")
        cpu = lat.get("cpu_util_pct", "?")
        print(f"  [{i+1}] {n} metrics: eu={eu}% pwr={pwr}W cpu={cpu}%")
    mon.stop_async()

    print(f"\nDone.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="xpu-smi-py",
        description="Python wrapper for Intel xpu-smi on Aurora",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON output"
    )

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("probe", help="Probe available xpu-smi versions")
    sub.add_parser("snapshot", help="Single synchronous snapshot")
    sub.add_parser("diag", help="Full diagnostic (default)")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "probe":
        return cmd_probe(args)
    elif args.command == "snapshot":
        return cmd_snapshot(args)
    else:
        return cmd_diag(args)


if __name__ == "__main__":
    sys.exit(main())
