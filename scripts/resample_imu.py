#!/usr/bin/env python3
"""
Resample IMU data to uniform 200Hz (5ms) spacing.

ORB-SLAM3 requires uniformly spaced IMU measurements. Its noise model assumes
dt = 1/IMU.Frequency for every step. Non-uniform timestamps cause covariance
errors and tracking failure (see docs/orbslam3_imu_uniform_spacing.md).

This script:
1. Deduplicates consecutive identical samples (from oversampling the sensor)
2. Resamples ACCL and GYRO to a uniform 5ms grid using linear interpolation
3. Writes the result as imu_data_resampled.json in the same format

Usage:
    .venv/bin/python scripts/resample_imu.py rpi_data/grabette2/imu_data.json
    .venv/bin/python scripts/resample_imu.py rpi_data/grabette2/imu_data.json -o resampled.json
    .venv/bin/python scripts/resample_imu.py rpi_data/grabette2/imu_data.json --rate 200
"""

import json
import click
import numpy as np
from pathlib import Path


def deduplicate_samples(samples):
    """Remove consecutive samples with identical values (stale sensor reads)."""
    if not samples:
        return samples
    deduped = [samples[0]]
    for s in samples[1:]:
        if s['value'] != deduped[-1]['value']:
            deduped.append(s)
    return deduped


def resample_stream(samples, target_rate_hz):
    """Resample IMU stream to uniform spacing using linear interpolation.

    Args:
        samples: list of {"cts": float_ms, "value": [x, y, z]}
        target_rate_hz: target rate (e.g. 200 for 5ms spacing)

    Returns:
        list of resampled samples with uniform cts spacing
    """
    if len(samples) < 2:
        return samples

    dt_ms = 1000.0 / target_rate_hz  # 5.0 for 200Hz

    cts = np.array([s['cts'] for s in samples])
    values = np.array([s['value'] for s in samples])
    n_axes = values.shape[1]

    # Build uniform grid starting from first timestamp
    t_start = cts[0]
    t_end = cts[-1]
    uniform_cts = np.arange(t_start, t_end, dt_ms)

    # Interpolate each axis
    resampled_values = np.zeros((len(uniform_cts), n_axes))
    for axis in range(n_axes):
        resampled_values[:, axis] = np.interp(uniform_cts, cts, values[:, axis])

    # Build output
    resampled = []
    for i in range(len(uniform_cts)):
        resampled.append({
            'cts': float(uniform_cts[i]),
            'value': resampled_values[i].tolist()
        })

    return resampled


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output', default=None, help='Output path (default: imu_data_resampled.json next to input)')
@click.option('-r', '--rate', default=200, help='Target sample rate in Hz (default: 200)')
@click.option('--no-dedup', is_flag=True, default=False, help='Skip deduplication step')
def main(input_path, output, rate, no_dedup):
    """Resample IMU data to uniform spacing for ORB-SLAM3."""
    input_path = Path(input_path)

    if output is None:
        output = input_path.parent / 'imu_data_resampled.json'
    else:
        output = Path(output)

    with open(input_path) as f:
        data = json.load(f)

    streams = data['1']['streams']

    for stream_name in ['ACCL', 'GYRO']:
        if stream_name not in streams:
            print(f"  Warning: {stream_name} not found in IMU data")
            continue

        samples = streams[stream_name]['samples']
        n_raw = len(samples)
        cts = [s['cts'] for s in samples]
        raw_duration = (cts[-1] - cts[0]) / 1000.0
        raw_rate = n_raw / raw_duration if raw_duration > 0 else 0

        # Dedup
        if not no_dedup:
            samples = deduplicate_samples(samples)
            n_deduped = len(samples)
        else:
            n_deduped = n_raw

        # Resample
        resampled = resample_stream(samples, rate)

        streams[stream_name]['samples'] = resampled

        print(f"  {stream_name}: {n_raw} raw ({raw_rate:.0f}Hz)"
              f" -> {n_deduped} deduped"
              f" -> {len(resampled)} resampled ({rate}Hz)")

    # Remove ANGL stream if present (not needed for SLAM)
    if 'ANGL' in streams:
        del streams['ANGL']

    # Write output with only the "1" key (drop frames/second etc.)
    out_data = {"1": {"streams": streams}}
    with open(output, 'w') as f:
        json.dump(out_data, f)

    print(f"\n  Written to: {output}")
    print(f"  Size: {output.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
