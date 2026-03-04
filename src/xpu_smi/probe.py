"""
xpu_smi.probe — Discover and validate xpu-smi binaries on Aurora nodes.

The probing strategy:
  1. Glob /opt/aurora/*/support/tools/xpu-smi/*/bin/xpu-smi
  2. Parse semantic versions from paths, sort descending (newest first)
  3. For each candidate, run ``discovery`` to verify it works + count devices
  4. Run a quick ``dump`` to verify metrics flow
  5. Return the first fully-working binary

This avoids hard-coding paths and gracefully handles mixed Aurora SDK installs.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("xpu_smi")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class XPUSMINotFoundError(FileNotFoundError):
    """Raised when no working xpu-smi binary can be found."""

    def __init__(self, diagnosis: str = ""):
        self.diagnosis = diagnosis or _default_diagnosis()

    def __str__(self) -> str:
        return f"No working xpu-smi binary found.\n\n{self.diagnosis}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class XPUSMICandidate:
    """A single xpu-smi binary found on disk."""

    path: str
    aurora_version: str        # e.g. "25.190.0"
    smi_version: str           # e.g. "1.3.1" or "default"
    smi_version_tuple: Tuple[int, ...] = field(default_factory=tuple)
    lib64_dir: str = ""
    discovery_ok: bool = False
    dump_ok: bool = False
    num_devices: int = 0
    error: str = ""

    def __post_init__(self) -> None:
        # Derive lib64 dir from binary path
        #   .../xpu-smi/1.3.1/bin/xpu-smi  →  .../xpu-smi/1.3.1/lib64
        smi_root = str(Path(self.path).parent.parent)
        self.lib64_dir = os.path.join(smi_root, "lib64")

        # Parse version tuple for sorting
        if self.smi_version != "default":
            try:
                self.smi_version_tuple = tuple(
                    int(x) for x in self.smi_version.split(".")
                )
            except ValueError:
                self.smi_version_tuple = (0,)
        else:
            # "default" sorts below any real version
            self.smi_version_tuple = (-1,)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_GLOB_PATTERN = "/opt/aurora/*/support/tools/xpu-smi/*/bin/xpu-smi"
_PATH_RE = re.compile(
    r"/opt/aurora/(?P<aurora>[^/]+)/support/tools/xpu-smi/"
    r"(?P<smi>[^/]+)/bin/xpu-smi$"
)
_TIMEOUT_DISCOVERY = 15  # seconds
_TIMEOUT_DUMP = 30       # seconds — dump can be slow on first call


def _build_env(candidate: XPUSMICandidate) -> dict:
    """Build a minimal, clean environment for xpu-smi invocation."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "ZE_ENABLE_SYSMAN": "1",
        "LD_LIBRARY_PATH": ":".join(
            filter(None, [candidate.lib64_dir, "/usr/lib64", "/lib64"])
        ),
    }
    return env


def _run(cmd: List[str], env: dict, timeout: int) -> subprocess.CompletedProcess:
    """Run a subprocess with a clean env and timeout."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _parse_discovery_devices(stdout: str) -> int:
    """Count devices from ``xpu-smi discovery`` output."""
    # Each device block starts with "| <number>" in the Device ID column
    return len(re.findall(r"^\|\s*\d+\s*\|", stdout, re.MULTILINE))


def _validate_dump(stdout: str) -> bool:
    """Check that a dump line has at least some numeric data."""
    lines = [l.strip() for l in stdout.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    # First line is header; subsequent lines are data
    data_line = lines[1]
    # Should contain commas and at least one digit
    return "," in data_line and bool(re.search(r"\d", data_line))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_candidates(
    glob_pattern: str = _GLOB_PATTERN,
    version: Optional[str] = None,
) -> List[XPUSMICandidate]:
    """
    Find all xpu-smi binaries matching the glob pattern.

    Parameters
    ----------
    glob_pattern : str
        Glob pattern for finding binaries.
    version : str, optional
        If given, only return candidates whose smi_version matches exactly
        (e.g. ``"1.2.42"``).  This avoids probing every installed version.

    Returns candidates sorted by (smi_version descending, aurora_version descending).
    Symlinks labeled "default" are deprioritized.
    """
    paths = glob.glob(glob_pattern)

    # Also check the env override
    env_path = os.environ.get("XPU_SMI_PATH")
    if env_path and os.path.isfile(env_path):
        paths.append(env_path)

    candidates: List[XPUSMICandidate] = []
    seen: set = set()

    for p in paths:
        real = os.path.realpath(p)
        if real in seen:
            continue
        seen.add(real)

        m = _PATH_RE.search(p)
        if m:
            c = XPUSMICandidate(
                path=p,
                aurora_version=m.group("aurora"),
                smi_version=m.group("smi"),
            )
        elif p == env_path:
            c = XPUSMICandidate(
                path=p,
                aurora_version="custom",
                smi_version="custom",
                smi_version_tuple=(999,),  # env override wins
            )
        else:
            continue

        # Filter by version if requested
        if version and c.smi_version != version:
            continue

        candidates.append(c)

    # Sort: highest smi_version first; break ties by aurora_version descending
    candidates.sort(
        key=lambda c: (c.smi_version_tuple, c.aurora_version),
        reverse=True,
    )
    return candidates


def probe_single(
    candidate: XPUSMICandidate,
    skip_dump: bool = False,
) -> XPUSMICandidate:
    """
    Probe a single candidate: run discovery, then optionally a quick dump.

    Parameters
    ----------
    skip_dump : bool
        If True, skip the validation dump (~7s) after a successful discovery.
        The first real ``snapshot()`` call will serve as implicit validation.
    """
    env = _build_env(candidate)

    # --- discovery ---
    try:
        r = _run([candidate.path, "discovery"], env, _TIMEOUT_DISCOVERY)
        if r.returncode == 0:
            candidate.num_devices = _parse_discovery_devices(r.stdout)
            candidate.discovery_ok = candidate.num_devices > 0
            if not candidate.discovery_ok:
                candidate.error = "discovery returned 0 devices"
        else:
            candidate.error = f"discovery rc={r.returncode}: {r.stderr[:200]}"
    except subprocess.TimeoutExpired:
        candidate.error = "discovery timed out"
    except Exception as e:
        candidate.error = f"discovery exception: {e}"

    if not candidate.discovery_ok:
        return candidate

    if skip_dump:
        candidate.dump_ok = True
        return candidate

    # --- quick dump (1 sample, all devices, basic metrics) ---
    device_ids = ",".join(str(i) for i in range(candidate.num_devices))
    # metrics: 9=EU_active%, 1=power_W, 3=temp_C, 18=mem_used
    try:
        r = _run(
            [candidate.path, "dump", "-d", device_ids, "-m", "9,1,3,18", "-n", "1"],
            env,
            _TIMEOUT_DUMP,
        )
        if r.returncode == 0 and _validate_dump(r.stdout):
            candidate.dump_ok = True
        else:
            candidate.error = f"dump rc={r.returncode}: {r.stderr[:200]}"
    except subprocess.TimeoutExpired:
        candidate.error = "dump timed out"
    except Exception as e:
        candidate.error = f"dump exception: {e}"

    return candidate


def probe_versions(
    glob_pattern: str = _GLOB_PATTERN,
    verbose: bool = False,
    version: Optional[str] = None,
    skip_dump: bool = False,
) -> List[XPUSMICandidate]:
    """
    Discover all xpu-smi binaries and probe each one.
    Returns the full list with probe results populated.
    """
    candidates = discover_candidates(glob_pattern, version=version)
    n = len(candidates)
    logger.info(f"Found {n} xpu-smi candidate(s), probing...")

    for c in candidates:
        label = f"v{c.smi_version}" if c.smi_version != "default" else "default"
        logger.info(f"  Trying {label} ({c.aurora_version}) ...")
        probe_single(c, skip_dump=skip_dump)

        if c.discovery_ok:
            logger.info(f"  {label}: discovery OK, {c.num_devices} device(s)")
        else:
            logger.info(f"  {label}: discovery FAILED — {c.error}")
            continue

        if c.dump_ok:
            logger.info(f"  {label}: dump OK")
        else:
            logger.info(f"  {label}: dump FAILED — {c.error}")

    return candidates


def find_best_binary(
    glob_pattern: str = _GLOB_PATTERN,
    verbose: bool = False,
    version: Optional[str] = None,
    skip_dump: bool = False,
) -> XPUSMICandidate:
    """
    Return the best working xpu-smi binary.

    Parameters
    ----------
    version : str, optional
        Pin to a specific xpu-smi version (e.g. ``"1.2.42"``).
        Skips probing all other candidates.
    skip_dump : bool
        If True, skip the validation dump during probe (~7s saved).
        The first ``snapshot()`` call validates implicitly.

    Raises XPUSMINotFoundError if none work.
    """
    candidates = probe_versions(glob_pattern, verbose=verbose, version=version,
                                skip_dump=skip_dump)
    for c in candidates:
        if c.discovery_ok and c.dump_ok:
            logger.info(
                f"  Selected v{c.smi_version} "
                f"(aurora {c.aurora_version}, {c.num_devices} devices)"
            )
            return c

    raise XPUSMINotFoundError()


# ---------------------------------------------------------------------------
# Environment diagnosis
# ---------------------------------------------------------------------------

def diagnose_environment(
    glob_pattern: str = _GLOB_PATTERN,
) -> dict:
    """
    Diagnose why xpu-smi isn't working. Returns a dict with:

        reason : str
            One of: "no_aurora_sdk", "no_binaries", "login_node",
            "all_broken", "ok"
        details : str
            Human-readable explanation.
        errors : dict
            Per-version error messages (if applicable).
        suggestion : str
            Recommended fix.

    This is called automatically by XPUSMINotFoundError, but can also
    be called directly for health checks.
    """
    result: dict = {}

    # 1. Does /opt/aurora exist at all?
    if not os.path.isdir("/opt/aurora"):
        result["reason"] = "no_aurora_sdk"
        result["details"] = "/opt/aurora not found — not on an Aurora system"
        result["suggestion"] = "SSH to Aurora, then request a compute node"
        return result

    # 2. Do any xpu-smi binaries exist?
    paths = glob.glob(glob_pattern)
    if not paths:
        result["reason"] = "no_binaries"
        result["details"] = f"No binaries matching {glob_pattern}"
        result["suggestion"] = "module load xpu-smi/1.2.42"
        return result

    # 3. Probe them — are they all broken, or is it a login node?
    candidates = discover_candidates(glob_pattern)
    zero_device_count = 0
    error_map: dict = {}

    for c in candidates:
        probe_single(c)
        if c.discovery_ok and c.dump_ok:
            result["reason"] = "ok"
            result["details"] = f"v{c.smi_version} works ({c.num_devices} devices)"
            return result
        if c.discovery_ok and c.num_devices == 0:
            zero_device_count += 1
        if c.error:
            error_map[c.smi_version] = c.error

    # All candidates with 0 devices → login node
    if zero_device_count > 0:
        result["reason"] = "login_node"
        result["details"] = (
            f"{zero_device_count} binary(ies) ran but found 0 XPU devices"
        )
        result["suggestion"] = (
            "qsub -I -l select=1 -l walltime=1:00:00 -A <project>"
        )
    else:
        result["reason"] = "all_broken"
        result["details"] = f"All {len(candidates)} candidates failed"
        result["errors"] = error_map
        result["suggestion"] = "module load xpu-smi/1.2.42"

    return result


def _default_diagnosis() -> str:
    """Generate a helpful diagnosis for why xpu-smi isn't working."""
    lines = []
    try:
        diag = diagnose_environment()
    except Exception:
        return (
            "Could not diagnose environment. Ensure you are on an Aurora "
            "compute node with /opt/aurora SDK installed, or set the "
            "XPU_SMI_PATH environment variable."
        )
    reason = diag.get("reason", "unknown")
    if reason == "no_aurora_sdk":
        lines.append("  Problem:  /opt/aurora directory not found.")
        lines.append("  Likely:   You are not on an Aurora node.")
        lines.append("  Fix:      SSH to an Aurora login node, then request a compute node:")
        lines.append("              qsub -I -l select=1 -l walltime=1:00:00 -A <project>")
    elif reason == "no_binaries":
        lines.append("  Problem:  /opt/aurora exists but no xpu-smi binaries found.")
        lines.append("  Likely:   Aurora SDK not fully installed on this node.")
        lines.append("  Fix:      Try loading the module explicitly:")
        lines.append("              module load xpu-smi/1.2.42")
        lines.append("            Or set XPU_SMI_PATH to an explicit binary path.")
    elif reason == "login_node":
        lines.append("  Problem:  xpu-smi binaries found but no XPU devices detected.")
        lines.append("  Likely:   You are on a login node (no GPUs attached).")
        lines.append("  Fix:      Request a compute node:")
        lines.append("              qsub -I -l select=1 -l walltime=1:00:00 -A <project>")
        lines.append("            Or use --offline mode for parsing-only tests.")
    elif reason == "all_broken":
        broken = diag.get("errors", {})
        lines.append("  Problem:  xpu-smi binaries found but ALL failed validation.")
        lines.append(f"  Tried:    {len(broken)} version(s)")
        for ver, err in list(broken.items())[:3]:
            lines.append(f"            v{ver}: {err[:100]}")
        lines.append("  Fix:      Try a specific version:")
        lines.append("              module load xpu-smi/1.2.42")
        lines.append("            Known working versions: 1.2.36, 1.2.39, 1.2.42")
        lines.append("            Known broken: 1.3.1 (libxpum.so symbol error)")
    else:
        lines.append("  Could not determine the issue.")
        lines.append("  Set XPU_SMI_PATH to an explicit binary path, or run:")
        lines.append("    python -m xpu_smi probe")
    return "\n".join(lines)
