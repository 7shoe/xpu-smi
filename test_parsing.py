"""
tests/test_parsing.py — Unit tests for metrics parsing (no hardware needed).
"""

import pytest
from xpu_smi.metrics import parse_dump, aggregate_samples, _to_snake

# Real xpu-smi dump output from Aurora node
SAMPLE_DUMP = """\
Timestamp, DeviceId, GPU EU Array Active (%), GPU Power (W), GPU Core Temperature (Celsius Degree), GPU Memory Used (MiB)
19:05:18.736,    0, 0.00, 264.06, 36.00,  N/A
19:05:18.736,    1, 0.00, 263.47, 38.00,  N/A
19:05:18.736,    2, 0.00, 263.05, 35.50,  N/A
19:05:18.736,    3, 0.00, 265.37, 38.00,  N/A
19:05:18.736,    4, 0.00, 278.90, 41.50,  N/A
19:05:18.736,    5, 0.00, 264.58, 39.00,  N/A
"""


def test_to_snake():
    assert _to_snake("GPU EU Array Active (%)") == "xpu_eu_array_active_pct"
    assert _to_snake("GPU Power (W)") == "xpu_power_w"
    assert _to_snake("GPU Core Temperature (Celsius Degree)") == "xpu_core_temperature_c"
    assert _to_snake("GPU Memory Used (MiB)") == "xpu_memory_used_mib"


def test_parse_dump_count():
    samples = parse_dump(SAMPLE_DUMP)
    assert len(samples) == 6


def test_parse_dump_device_ids():
    samples = parse_dump(SAMPLE_DUMP)
    ids = [s.device_id for s in samples]
    assert ids == [0, 1, 2, 3, 4, 5]


def test_parse_dump_na_handling():
    samples = parse_dump(SAMPLE_DUMP)
    for s in samples:
        # Memory column is N/A in this dump
        mem_keys = [k for k in s.values if "memory" in k]
        for k in mem_keys:
            assert s.values[k] is None


def test_parse_dump_power_values():
    samples = parse_dump(SAMPLE_DUMP)
    powers = []
    for s in samples:
        for k, v in s.values.items():
            if "power" in k and v is not None:
                powers.append(v)
    assert len(powers) == 6
    assert abs(powers[0] - 264.06) < 0.01
    assert abs(powers[4] - 278.90) < 0.01


def test_aggregate():
    samples = parse_dump(SAMPLE_DUMP)
    agg = aggregate_samples(samples)

    assert agg["xpu_num_devices"] == 6.0
    assert agg["xpu_eu_active_avg_pct"] == 0.0
    assert abs(agg["xpu_power_total_w"] - 1599.43) < 0.01
    assert abs(agg["xpu_power_max_w"] - 278.90) < 0.01
    assert abs(agg["xpu_temp_max_c"] - 41.50) < 0.01
    assert abs(agg["xpu_temp_avg_c"] - 38.0) < 0.1


def test_aggregate_empty():
    agg = aggregate_samples([])
    assert agg == {}


def test_parse_empty():
    assert parse_dump("") == []
    assert parse_dump("header only\n") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
