#!/usr/bin/env python3
"""
Analyze SLAM trajectory results.

Usage:
    uv run python scripts/analyze_slam.py rpi_data/grabette4/
    uv run python scripts/analyze_slam.py rpi_data/grabette3/ rpi_data/grabette4/
    uv run python scripts/analyze_slam.py rpi_data/*/
"""

import sys
import click
import pathlib
import numpy as np
import pandas as pd


def analyze_one(video_dir):
    """Analyze SLAM results for a single recording directory."""
    video_dir = pathlib.Path(video_dir)

    # Find trajectory CSV (mapping or localization)
    candidates = [
        'mapping_camera_trajectory.csv',
        'camera_trajectory.csv',
    ]
    csv_path = None
    for name in candidates:
        p = video_dir / name
        if p.is_file():
            csv_path = p
            break

    if csv_path is None:
        return {'dir': video_dir.name, 'error': 'no trajectory CSV found'}

    df = pd.read_csv(csv_path)
    total = len(df)
    lost = int(df['is_lost'].sum())
    tracked = total - lost

    # Find first tracked frame (end of initialization)
    first_tracked = None
    for i, row in df.iterrows():
        if row['is_lost'] == 0:
            first_tracked = i
            break

    # Trajectory geometry
    pos = df[['x', 'y', 'z']].values
    valid_mask = ~np.isnan(pos).any(axis=1) & (df['is_lost'] == 0).values
    valid = pos[valid_mask]

    traj_len = 0.0
    maxspan = 0.0
    if len(valid) > 1:
        diffs = np.diff(valid, axis=0)
        step_sizes = np.linalg.norm(diffs, axis=1)
        traj_len = float(np.sum(step_sizes))
        maxspan = float(np.max(np.ptp(valid, axis=0)))

        # Detect jumps (steps > 10x median — possible scale drift)
        median_step = np.median(step_sizes)
        n_jumps = int(np.sum(step_sizes > 10 * median_step)) if median_step > 0 else 0
    else:
        n_jumps = 0

    # Tracking continuity — longest tracked streak
    is_tracked = (df['is_lost'] == 0).values.astype(int)
    streaks = []
    current = 0
    for v in is_tracked:
        if v:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    longest_streak = max(streaks) if streaks else 0

    # After-init tracking rate (exclude initialization lost frames)
    if first_tracked is not None:
        after_init = df.iloc[first_tracked:]
        after_init_tracked = (after_init['is_lost'] == 0).sum()
        after_init_total = len(after_init)
    else:
        after_init_tracked = 0
        after_init_total = total

    return {
        'dir': video_dir.name,
        'csv': csv_path.name,
        'total': total,
        'tracked': tracked,
        'pct': 100 * tracked / total if total > 0 else 0,
        'init_frames': first_tracked if first_tracked is not None else total,
        'after_init_pct': 100 * after_init_tracked / after_init_total if after_init_total > 0 else 0,
        'longest_streak': longest_streak,
        'traj_len': traj_len,
        'maxspan': maxspan,
        'n_jumps': n_jumps,
    }


def print_result(r):
    """Print analysis for one recording."""
    if 'error' in r:
        print(f"  {r['dir']}: {r['error']}")
        return

    pct = r['pct']
    if pct >= 90:
        grade = 'EXCELLENT'
    elif pct >= 70:
        grade = 'GOOD'
    elif pct >= 50:
        grade = 'MEDIOCRE'
    else:
        grade = 'POOR'

    print(f"  {r['dir']} ({r['csv']})")
    print(f"    Tracking:   {r['tracked']}/{r['total']} ({r['pct']:.1f}%) — {grade}")
    print(f"    Init:       {r['init_frames']} frames to initialize")
    print(f"    After-init: {r['after_init_pct']:.1f}% tracked")
    print(f"    Continuity: longest streak {r['longest_streak']} frames")
    print(f"    Trajectory: {r['traj_len']:.3f}m length, {r['maxspan']:.3f}m max span")
    if r['n_jumps'] > 0:
        print(f"    WARNING:    {r['n_jumps']} large jumps detected (possible scale drift)")


@click.command()
@click.argument('dirs', nargs=-1, required=True, type=click.Path(exists=True))
def main(dirs):
    """Analyze SLAM trajectory results for one or more recording directories."""
    results = []
    for d in dirs:
        r = analyze_one(d)
        results.append(r)

    print("=" * 60)
    print("SLAM ANALYSIS")
    print("=" * 60)

    for r in results:
        print_result(r)
        print()

    # Summary if multiple
    valid = [r for r in results if 'error' not in r]
    if len(valid) > 1:
        avg_pct = np.mean([r['pct'] for r in valid])
        excellent = sum(1 for r in valid if r['pct'] >= 90)
        print("-" * 60)
        print(f"  Summary: {len(valid)} recordings, avg {avg_pct:.1f}% tracking, {excellent}/{len(valid)} excellent (>=90%)")

    print("=" * 60)


if __name__ == '__main__':
    main()
