# xpu-smi

![tests](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2F7shoe%2Fxpu-smi%2Fmain%2Ftests%2Fbadge.json&query=%24.message&label=tests&color=brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![platform](https://img.shields.io/badge/platform-Aurora%20%7C%20Intel%20XPU-blue)
![Intel](https://img.shields.io/badge/Intel-Inside-blue?logo=intel)

Python wrapper for Intel `xpu-smi` on Aurora supercomputer nodes.

Auto-discovers the best available `xpu-smi` binary, validates it, and provides
sync/async Pythonic APIs for monitoring Intel Data Center GPU Max 1550 devices.

## Install

```bash
# minimal (no CPU/RAM metrics)
pip install xpu-smi

# w/ CPU/RAM metrics
pip install xpu-smi[cpu]

# editable
pip install -e ".[all]" --break-system-packages
```

## Quick Start

```python
from xpu_smi import XPUMonitor

# default: no subprocess calls (via v1.2.42, 6 devices)
mon = XPUMonitor()           
print(mon)

# Blocking snapshot (~7s due to xpu-smi latency)
snap = mon.snapshot()
print(f"Total power: {snap.get('xpu_power_total_w', 0):.0f} W")
print(f"Max temp:    {snap.get('xpu_temp_max_c', 0):.1f} °C")
print(f"Memory used: {snap.get('xpu_mem_used_total_mib', 0):.0f} MiB")
```

## CLI

```bash
# Full diagnostic
python -m xpu_smi

# Probe all available versions
python -m xpu_smi probe

# JSON snapshot for scripting
python -m xpu_smi snapshot --json
```

## Testing

No install required — just run from the repo root:

```bash
# Offline (parsing only — works anywhere, even a laptop)
python test_xpu_smi_standalone.py --offline

# Full test on a compute node (6 sections, ~30s)
python test_xpu_smi_standalone.py

# Generate shields.io badge JSON
python test_xpu_smi_standalone.py --badge
```

Example output on a compute node:

```
  ✓  22 passed
  ○   0 skipped
  22/22 tests OK (100%)
```

Example output on a login node:

```
  ✓  14 passed
  ○   8 skipped
  14/14 tests OK (100%)
```

## Wiring into Training

```python
from xpu_smi import XPUMonitor

# init
mon = XPUMonitor()
mon.start_async(interval=15.0)   # background thread, non-blocking reads

# inside training loop
for step, batch in enumerate(dataloader):
    loss = train_step(batch)

    if step % log_interval == 0:
        hw = mon.latest()   # zero-latency read from background cache
        logger.info(
            f"step={step} loss={loss:.4f} "
            f"pwr={hw.get('xpu_power_total_w', 0):.0f}W "
            f"temp={hw.get('xpu_temp_max_c', 0):.1f}C "
            f"eu={hw.get('xpu_eu_active_avg_pct', 0):.1f}% "
            f"mem={hw.get('xpu_mem_used_total_mib', 0):.0f}MiB"
        )

# Cleanup
mon.stop_async()
```

### Tensor Health Vector

```python
keys, vals = mon.as_tensor()
# keys = ['xpu_eu_active_avg_pct', 'xpu_power_total_w', ...]
# vals = [0.0, 1597.36, ...]

import torch
health_tensor = torch.tensor(vals, dtype=torch.float32)
```

## Troubleshooting

The library diagnoses common issues automatically. Run:

```bash
python -c "from xpu_smi.probe import diagnose_environment; print(diagnose_environment())"
```

### "No working xpu-smi binary found"

| Symptom | Cause | Fix |
|---------|-------|-----|
| `/opt/aurora` not found | Not on Aurora | SSH to Aurora, request a compute node |
| Binaries found, 0 devices | Login node (no GPUs) | `qsub -I -l select=1 -l walltime=1:00:00 -A <project>` |
| All versions fail | Broken SDK installs | `module load xpu-smi/1.2.42` or set `XPU_SMI_PATH` |

### Known xpu-smi Version Status

Based on experiments in March 2026.

| Version | Status | Notes |
|---------|--------|-------|
| 1.2.36  | ✓ Working | Oldest tested |
| 1.2.39  | ✓ Working | |
| 1.2.42  | ✓ Working | **Recommended** — default selection |
| 1.3.1   | ✗ Broken | `libxpum.so` symbol lookup error (spdlog mismatch) |

### Environment Variable Override

If auto-discovery doesn't work, point directly to a binary:

```bash
export XPU_SMI_PATH=/opt/aurora/25.190.0/support/tools/xpu-smi/1.2.42/bin/xpu-smi
python -c "from xpu_smi import XPUMonitor; print(XPUMonitor())"
```

## Version Probing Details

On Aurora nodes, `xpu-smi` is installed under multiple SDK versions:

```
/opt/aurora/24.180.3/support/tools/xpu-smi/1.2.36/bin/xpu-smi
/opt/aurora/24.347.0/support/tools/xpu-smi/1.2.39/bin/xpu-smi
/opt/aurora/25.190.0/support/tools/xpu-smi/1.2.42/bin/xpu-smi
/opt/aurora/25.190.0/support/tools/xpu-smi/1.3.1/bin/xpu-smi   ← broken
```

The library automatically:
1. Globs all candidates
2. Sorts by version (newest first, `default` symlinks deprioritized)
3. Validates each with `discovery` + `dump`
4. Selects the first fully-working binary (skips broken ones)

## Architecture

```
intel-xpu/
├── src/xpu_smi/
│   ├── __init__.py       # public API surface
│   ├── __main__.py       # CLI: python -m xpu_smi
│   ├── probe.py          # binary discovery, validation, environment diagnosis
│   ├── metrics.py        # dump parsing, aggregation, tensor helpers
│   └── monitor.py        # XPUMonitor class (sync + async)
├── tests/
│   ├── test_parsing.py   # pytest unit tests (offline)
│   └── badge.json        # shields.io badge (generated)
├── test_xpu_smi_standalone.py  # zero-install smoke test
├── pyproject.toml
└── README.md
```
