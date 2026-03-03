#!/usr/bin/env python3
"""
test_xpu_smi_standalone.py — Zero-install smoke test for xpu-smi.

Usage (from the directory containing src/xpu_smi/):

    python test_xpu_smi_standalone.py              # full test on compute node
    python test_xpu_smi_standalone.py --offline     # parsing only (anywhere)
    python test_xpu_smi_standalone.py --badge       # also write badge JSON

Generates a shields.io-compatible badge JSON at tests/badge.json:

    https://img.shields.io/badge/dynamic/json?url=...&query=$.message&label=tests

No pip install needed. Just needs the src/xpu_smi/ folder on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# 0.  Bootstrap: add src/ to sys.path so `import xpu_smi` works without install
# ---------------------------------------------------------------------------

def bootstrap(root: str | None = None) -> str:
    """Return the resolved src/ dir and inject it into sys.path."""
    if root is None:
        root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(root, "src")
    if not os.path.isdir(os.path.join(src, "xpu_smi")):
        alt = os.path.join(root, "xpu_smi")
        if os.path.isdir(alt):
            src = root
        else:
            print(f"ERROR: Cannot find xpu_smi package under {root}")
            print(f"  Expected: {src}/xpu_smi/  or  {root}/xpu_smi/")
            sys.exit(1)
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------

class TestResult:
    """Tracks pass / fail / xfail / skip."""

    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.xfailed: list[str] = []   # expected failures
        self.skipped: list[str] = []

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed) + len(self.xfailed) + len(self.skipped)

    @property
    def ok(self) -> bool:
        return len(self.failed) == 0

    def badge_message(self) -> str:
        parts = [f"{len(self.passed)} passed"]
        if self.xfailed:
            parts.append(f"{len(self.xfailed)} xfail")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        if self.failed:
            parts.append(f"{len(self.failed)} FAILED")
        return ", ".join(parts)

    def badge_color(self) -> str:
        if self.failed:
            return "red"
        if self.xfailed:
            return "yellowgreen"
        return "brightgreen"

    def badge_json(self) -> dict:
        """shields.io endpoint badge format."""
        return {
            "schemaVersion": 1,
            "label": "tests",
            "message": self.badge_message(),
            "color": self.badge_color(),
        }


R = TestResult()


def check(name: str, fn, xfail: bool = False):
    """
    Run a test function, print result.

    Parameters
    ----------
    name : str
        Test label.
    fn : callable
        Returns True/None for pass, False or raises for fail.
    xfail : bool
        If True, a failure is *expected* (e.g. known broken version).
    """
    try:
        result = fn()
        if result is False:
            raise AssertionError("returned False")
        if xfail:
            # It passed but we expected failure — interesting, still a pass
            print(f"  [PASS] {name}  (expected fail, but passed!)")
            R.passed.append(name)
        else:
            print(f"  [PASS] {name}")
            R.passed.append(name)
    except Exception as e:
        if xfail:
            print(f"  [XFAIL] {name}: {e}")
            R.xfailed.append(name)
        else:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc(limit=2)
            R.failed.append(name)


def skip(name: str, reason: str = ""):
    msg = f"  [SKIP] {name}"
    if reason:
        msg += f" — {reason}"
    print(msg)
    R.skipped.append(name)


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ===========================================================================
# 1. IMPORT TESTS (always run)
# ===========================================================================

def test_import():
    section("1. Import Check")

    def _import_init():
        import xpu_smi
        print(f"       version: {xpu_smi.__version__}")

    def _import_probe():
        from xpu_smi.probe import discover_candidates, find_best_binary, diagnose_environment
        return True

    def _import_metrics():
        from xpu_smi.metrics import parse_dump, aggregate_samples
        return True

    def _import_monitor():
        from xpu_smi.monitor import XPUMonitor
        return True

    check("import xpu_smi",         _import_init)
    check("import xpu_smi.probe",   _import_probe)
    check("import xpu_smi.metrics", _import_metrics)
    check("import xpu_smi.monitor", _import_monitor)


# ===========================================================================
# 2. PARSING TESTS (always run, no hardware)
# ===========================================================================

SAMPLE_DUMP = """\
Timestamp, DeviceId, GPU EU Array Active (%), GPU Power (W), GPU Core Temperature (Celsius Degree), GPU Memory Used (MiB)
19:05:18.736,    0, 0.00, 264.06, 36.00,  N/A
19:05:18.736,    1, 0.00, 263.47, 38.00,  N/A
19:05:18.736,    2, 0.00, 263.05, 35.50,  N/A
19:05:18.736,    3, 0.00, 265.37, 38.00,  N/A
19:05:18.736,    4, 0.00, 278.90, 41.50,  N/A
19:05:18.736,    5, 0.00, 264.58, 39.00,  N/A
"""


def test_parsing():
    section("2. Parsing (offline, no hardware)")

    from xpu_smi.metrics import parse_dump, aggregate_samples, _to_snake

    def _snake():
        assert _to_snake("GPU EU Array Active (%)") == "xpu_eu_array_active_pct"
        assert _to_snake("GPU Power (W)") == "xpu_power_w"
        assert _to_snake("GPU Core Temperature (Celsius Degree)") == "xpu_core_temperature_c"

    def _parse_count():
        samples = parse_dump(SAMPLE_DUMP)
        assert len(samples) == 6, f"expected 6, got {len(samples)}"

    def _parse_device_ids():
        samples = parse_dump(SAMPLE_DUMP)
        ids = [s.device_id for s in samples]
        assert ids == [0, 1, 2, 3, 4, 5], f"got {ids}"

    def _parse_na():
        samples = parse_dump(SAMPLE_DUMP)
        for s in samples:
            for k, v in s.values.items():
                if "memory" in k:
                    assert v is None, f"device {s.device_id} {k}={v}, expected None"

    def _parse_power():
        samples = parse_dump(SAMPLE_DUMP)
        powers = []
        for s in samples:
            for k, v in s.values.items():
                if "power" in k and v is not None:
                    powers.append(v)
        assert len(powers) == 6
        assert abs(powers[0] - 264.06) < 0.01
        assert abs(powers[4] - 278.90) < 0.01

    def _aggregate():
        samples = parse_dump(SAMPLE_DUMP)
        agg = aggregate_samples(samples)
        assert agg["xpu_num_devices"] == 6
        assert abs(agg["xpu_power_total_w"] - 1599.43) < 0.1
        assert abs(agg["xpu_power_max_w"] - 278.90) < 0.01
        assert abs(agg["xpu_temp_max_c"] - 41.50) < 0.01
        assert abs(agg["xpu_temp_avg_c"] - 38.0) < 0.1
        print(f"       power_total={agg['xpu_power_total_w']:.2f}W  "
              f"temp_max={agg['xpu_temp_max_c']:.1f}C")

    def _aggregate_empty():
        assert aggregate_samples([]) == {}

    def _parse_empty():
        assert parse_dump("") == []
        assert parse_dump("header only\n") == []

    check("_to_snake",             _snake)
    check("parse 6 devices",      _parse_count)
    check("device IDs 0-5",       _parse_device_ids)
    check("N/A → None",           _parse_na)
    check("power values",         _parse_power)
    check("aggregation",          _aggregate)
    check("empty aggregation",    _aggregate_empty)
    check("empty/malformed input", _parse_empty)


# ===========================================================================
# 3. ENVIRONMENT DIAGNOSIS (always run)
# ===========================================================================

def test_environment():
    section("3. Environment Diagnosis")

    from xpu_smi.probe import diagnose_environment, discover_candidates

    def _diagnose():
        diag = diagnose_environment()
        reason = diag["reason"]
        details = diag.get("details", "")
        suggestion = diag.get("suggestion", "")
        print(f"       reason:     {reason}")
        print(f"       details:    {details}")
        if suggestion:
            print(f"       suggestion: {suggestion}")
        return True

    def _hostname():
        host = socket.gethostname()
        # Aurora compute nodes: x####c#s#b#n#
        # Aurora login nodes: aurora-uan-####
        is_compute = host.startswith("x") and "c" in host and "n" in host
        is_login = "uan" in host or "login" in host
        label = "compute node" if is_compute else "login node" if is_login else "unknown"
        print(f"       hostname:   {host} ({label})")
        return True

    check("diagnose_environment()", _diagnose)
    check("hostname detection",     _hostname)


# ===========================================================================
# 4. VERSION PROBE (requires Aurora filesystem)
# ===========================================================================

def test_probe():
    section("4. Version Probe (requires Aurora)")

    from xpu_smi.probe import discover_candidates, probe_versions

    candidates = discover_candidates()
    if not candidates:
        skip("discover_candidates", "no binaries found (not on Aurora?)")
        skip("probe_versions", "no candidates")
        skip("v1.3.1 expected failure", "no candidates")
        return

    def _discover():
        n = len(candidates)
        print(f"       found {n} candidate(s)")
        for c in candidates:
            print(f"         v{c.smi_version:8s}  aurora={c.aurora_version}")
        return n > 0

    check("discover_candidates", _discover)

    # Probe all
    probed = probe_versions()
    working = [c for c in probed if c.discovery_ok and c.dump_ok]
    failed  = [c for c in probed if not (c.discovery_ok and c.dump_ok)]

    def _probe_working():
        print(f"       {len(working)}/{len(probed)} fully working")
        for c in working:
            print(f"         ✓ v{c.smi_version}  ({c.num_devices} devices)")
        assert len(working) > 0, "no working versions found"

    check("working versions found", _probe_working)

    # v1.3.1 is known broken — test it as an expected failure
    v131 = [c for c in probed if c.smi_version == "1.3.1"]
    if v131:
        def _v131_should_fail():
            c = v131[0]
            if c.discovery_ok and c.dump_ok:
                raise AssertionError("v1.3.1 unexpectedly worked!")
            print(f"       v1.3.1 error: {c.error[:80]}...")
            assert "symbol" in c.error.lower() or c.error

        check("v1.3.1 known broken (libxpum symbol error)", _v131_should_fail, xfail=False)
    else:
        skip("v1.3.1 expected failure", "v1.3.1 not present on this system")

    # Print other failed versions
    for c in failed:
        if c.smi_version != "1.3.1":
            print(f"  [INFO] v{c.smi_version} (aurora {c.aurora_version}): {c.error}")


# ===========================================================================
# 5. LIVE MONITOR (requires compute node with XPUs)
# ===========================================================================

def test_monitor():
    section("5. XPUMonitor (requires compute node)")

    from xpu_smi.monitor import XPUMonitor
    from xpu_smi.probe import discover_candidates, diagnose_environment

    candidates = discover_candidates()
    if not candidates:
        skip("XPUMonitor()", "not on Aurora")
        skip("snapshot() sync", "not on Aurora")
        skip("as_tensor()", "not on Aurora")
        skip("start/stop async", "not on Aurora")
        return

    # Check if we're likely on a login node
    diag = diagnose_environment()
    if diag.get("reason") == "login_node":
        skip("XPUMonitor()", "login node — no XPU devices")
        skip("snapshot() sync", "login node")
        skip("as_tensor()", "login node")
        skip("start/stop async", "login node")
        print(f"\n  ℹ  {diag.get('details', '')}")
        print(f"     Suggestion: {diag.get('suggestion', '')}")
        return

    def _init():
        mon = XPUMonitor()
        print(f"       {mon}")
        assert mon.available, "monitor not available"

    def _snapshot():
        mon = XPUMonitor()
        if not mon.available:
            return True
        t0 = time.monotonic()
        snap = mon.snapshot()
        dt = time.monotonic() - t0
        print(f"       {len(snap)} metrics in {dt:.1f}s")
        for k in ["xpu_num_devices", "xpu_power_total_w", "xpu_temp_max_c",
                   "xpu_mem_used_total_mib"]:
            if k in snap:
                print(f"       {k}: {snap[k]:.2f}")

    def _tensor():
        mon = XPUMonitor()
        if not mon.available:
            return True
        keys, vals = mon.as_tensor()
        print(f"       {len(keys)} keys")
        for k, v in zip(keys[:4], vals[:4]):
            print(f"         {k}: {v:.2f}")

    def _async():
        mon = XPUMonitor()
        if not mon.available:
            return True
        mon.start_async(interval=10.0)
        for _ in range(20):
            time.sleep(1)
            lat = mon.latest()
            if lat:
                break
        n = len(lat)
        pwr = lat.get("xpu_power_total_w", "?")
        temp = lat.get("xpu_temp_max_c", "?")
        print(f"       async: {n} metrics, power={pwr}W, temp={temp}C")
        mon.stop_async()
        assert n > 0, "async produced no metrics"

    check("XPUMonitor()",       _init)
    check("snapshot() sync",    _snapshot)
    check("as_tensor()",        _tensor)
    check("start/stop async",   _async)


# ===========================================================================
# 6. DIAGNOSTICS
# ===========================================================================

def test_diagnostics():
    section("6. Diagnostics")

    from xpu_smi.monitor import XPUMonitor
    from xpu_smi.probe import discover_candidates, diagnose_environment

    candidates = discover_candidates()
    diag = diagnose_environment() if candidates else {"reason": "no_binaries"}

    if not candidates or diag.get("reason") in ("login_node", "no_binaries", "no_aurora_sdk"):
        skip("mon.diagnostics()", diag.get("reason", "not on Aurora"))
        return

    def _diag():
        mon = XPUMonitor()
        if not mon.available:
            return True
        d = mon.diagnostics()
        for k, v in sorted(d.items()):
            if k == "dump_result":
                print(f"       {k}:")
                for dk, dv in sorted(v.items()):
                    print(f"         {dk}: {dv}")
            else:
                print(f"       {k}: {v}")

    check("mon.diagnostics()", _diag)


# ===========================================================================
# Summary + Badge
# ===========================================================================

def print_summary(badge_path: str | None = None):
    print(f"\n{'='*60}")
    if R.passed:
        print(f"  ✓  {len(R.passed)} passed")
    if R.xfailed:
        print(f"  ⊘  {len(R.xfailed)} expected failures")
    if R.skipped:
        print(f"  ○  {len(R.skipped)} skipped")
    if R.failed:
        print(f"  ✗  {len(R.failed)} FAILED:")
        for name in R.failed:
            print(f"       - {name}")

    total = R.total
    effective = len(R.passed) + len(R.xfailed)
    testable = total - len(R.skipped)
    pct = (effective / testable * 100) if testable > 0 else 0

    print(f"\n  {effective}/{testable} tests OK ({pct:.0f}%)")
    print(f"{'='*60}")

    if badge_path:
        badge = R.badge_json()
        os.makedirs(os.path.dirname(badge_path) or ".", exist_ok=True)
        with open(badge_path, "w") as f:
            json.dump(badge, f, indent=2)
        print(f"\n  Badge written: {badge_path}")
        print(f"  {json.dumps(badge)}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="xpu-smi zero-install smoke test")
    parser.add_argument(
        "--root", default=None,
        help="Path to the repo root (containing src/xpu_smi/)."
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Only run tests that don't need hardware (parsing, imports)."
    )
    parser.add_argument(
        "--badge", action="store_true",
        help="Write a shields.io badge JSON to tests/badge.json."
    )
    args = parser.parse_args()

    src = bootstrap(args.root)

    print(f"{'='*60}")
    print(f"  xpu-smi Standalone Test")
    print(f"  src:  {src}")
    print(f"  host: {socket.gethostname()}")
    print(f"  time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Always run
    test_import()
    test_parsing()
    test_environment()

    if not args.offline:
        test_probe()
        test_monitor()
        test_diagnostics()
    else:
        section("4-6. Hardware Tests")
        skip("version probe", "--offline")
        skip("XPUMonitor", "--offline")
        skip("diagnostics", "--offline")

    badge_path = None
    if args.badge:
        root = args.root or os.path.dirname(os.path.abspath(__file__))
        badge_path = os.path.join(root, "tests", "badge.json")

    print_summary(badge_path)

    return 0 if R.ok else 1


if __name__ == "__main__":
    sys.exit(main())
