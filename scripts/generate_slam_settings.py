#!/usr/bin/env python
"""
Generate ORB-SLAM3 settings YAML from camera intrinsics JSON.

This script reads camera intrinsics (from OpenCameraImuCalibration format)
and generates a properly scaled ORB-SLAM3 settings YAML file.

Usage:
    # Generate settings for Hero 13 at 720p
    uv run python scripts/generate_slam_settings.py \
        --intrinsics example/calibration/hero13_proper_intrinsics_2.7k.json \
        --slam_resolution 960x720 \
        --output hero13_720p_slam_settings.yaml

    # Generate settings with custom output location
    uv run python scripts/generate_slam_settings.py \
        --intrinsics gopro_intrinsics_2_7k.json \
        --slam_resolution 960x720 \
        --output /tmp/test_settings.yaml

    # Show what would be generated (dry-run)
    uv run python scripts/generate_slam_settings.py \
        --intrinsics hero13_proper_intrinsics_2.7k.json \
        --slam_resolution 960x720 \
        --dry-run
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import json

from umi.common.camera_config import generate_slam_settings, load_intrinsics, scale_intrinsics


def parse_resolution(res_str: str) -> tuple:
    """Parse resolution string like '960x720' to (width, height)."""
    parts = res_str.lower().replace('x', ',').split(',')
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution format: {res_str}. Use WxH format (e.g., 960x720)")
    return int(parts[0]), int(parts[1])


@click.command()
@click.option('-i', '--intrinsics', required=True, type=click.Path(exists=True),
              help='Path to camera intrinsics JSON file')
@click.option('-r', '--slam_resolution', required=True, type=str,
              help='Target SLAM resolution (e.g., 960x720)')
@click.option('-o', '--output', type=click.Path(),
              help='Output YAML file path (default: auto-generated name)')
@click.option('--dry-run', is_flag=True, default=False,
              help='Show generated settings without writing file')
@click.option('--show-scaling', is_flag=True, default=False,
              help='Show intrinsics scaling details')
def main(intrinsics, slam_resolution, output, dry_run, show_scaling):
    """Generate ORB-SLAM3 settings YAML from camera intrinsics."""

    intrinsics_path = pathlib.Path(intrinsics).absolute()
    slam_res = parse_resolution(slam_resolution)

    print(f"Intrinsics: {intrinsics_path.name}")
    print(f"SLAM resolution: {slam_res[0]}x{slam_res[1]}")

    # Load intrinsics
    with open(intrinsics_path) as f:
        raw_data = json.load(f)

    src_w = raw_data.get('image_width')
    src_h = raw_data.get('image_height')

    if not src_w or not src_h:
        print("Error: Intrinsics file must contain image_width and image_height")
        return 1

    source_res = (src_w, src_h)
    print(f"Source resolution: {src_w}x{src_h}")

    # Show scaling details if requested
    if show_scaling:
        intrinsics_data = load_intrinsics(intrinsics_path)
        print(f"\nOriginal intrinsics ({src_w}x{src_h}):")
        print(f"  fx: {intrinsics_data['fx']:.6f}")
        print(f"  fy: {intrinsics_data['fy']:.6f}")
        print(f"  cx: {intrinsics_data['cx']:.6f}")
        print(f"  cy: {intrinsics_data['cy']:.6f}")
        print(f"  k1: {intrinsics_data['k1']:.9f}")
        print(f"  k2: {intrinsics_data['k2']:.9f}")
        print(f"  k3: {intrinsics_data['k3']:.9f}")
        print(f"  k4: {intrinsics_data['k4']:.9f}")

        scaled = scale_intrinsics(intrinsics_data, source_res, slam_res)
        print(f"\nScaled intrinsics ({slam_res[0]}x{slam_res[1]}):")
        print(f"  fx: {scaled['fx']:.6f}")
        print(f"  fy: {scaled['fy']:.6f}")
        print(f"  cx: {scaled['cx']:.6f}")
        print(f"  cy: {scaled['cy']:.6f}")
        print(f"  k1-k4: unchanged (distortion is resolution-independent)")

        scale_x = slam_res[0] / src_w
        scale_y = slam_res[1] / src_h
        print(f"\nScale factors: x={scale_x:.6f}, y={scale_y:.6f}")

    # Generate output path if not specified
    if output is None:
        # Auto-generate name based on intrinsics file
        stem = intrinsics_path.stem.replace('intrinsics', '').replace('_', '')
        output = f"{stem}_{slam_res[0]}x{slam_res[1]}_slam_settings.yaml"
        output = pathlib.Path(output)
    else:
        output = pathlib.Path(output).absolute()

    if dry_run:
        print(f"\nWould write to: {output}")
        print("\n--- Generated YAML preview ---")
        # Generate to temp location and print
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        generate_slam_settings(intrinsics_path, slam_res, tmp_path)
        with open(tmp_path) as f:
            print(f.read())
        tmp_path.unlink()
        return 0

    # Generate settings
    result_path = generate_slam_settings(intrinsics_path, slam_res, output)
    print(f"\nGenerated: {result_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
