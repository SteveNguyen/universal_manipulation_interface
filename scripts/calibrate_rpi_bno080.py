#!/usr/bin/env python3
"""
=============================================================================
RPi Camera + BNO080 IMU Calibration Script
=============================================================================

Calibrates camera intrinsics AND camera-to-IMU transformation using OpenICC
(OpenImuCameraCalibrator).

PREREQUISITES
-------------
1. Docker with chicheng/openicc:latest image:
   docker pull chicheng/openicc:latest

2. ChArUco calibration board (same as used for GoPro/Hero13):
   - Grid: 10x8 squares
   - Square size: 19mm
   - Dictionary: DICT_ARUCO_ORIGINAL (OpenICC default)
   - Print from: OpenImuCameraCalibrator/resource/board.png

3. Three calibration video sequences with synchronized IMU:
   - cam/       : Camera-only calibration (board stationary, camera moves)
   - cam_imu/   : Camera-IMU calibration (diverse 6-DOF motion) **CRITICAL**
   - imu_bias/  : IMU bias calibration (camera stationary)

DATA STRUCTURE
--------------
calibration_dir/
├── cam/
│   ├── raw_video.mp4
│   └── imu_data.json
├── cam_imu/
│   ├── raw_video.mp4
│   └── imu_data.json
└── imu_bias/
    ├── raw_video.mp4
    └── imu_data.json

USAGE
-----
# Full calibration (recommended)
python scripts/calibrate_rpi_bno080.py \\
    --calib_dir rpi_bno080_calibration \\
    --output example/calibration/rpi_camera_intrinsics.json \\
    --generate-settings

# With custom board dimensions
python scripts/calibrate_rpi_bno080.py \\
    --calib_dir rpi_bno080_calibration \\
    --square_size 0.030 \\
    --rows 7 --cols 9 \\
    --output example/calibration/rpi_camera_intrinsics.json

# Convert IMU data format only
python scripts/calibrate_rpi_bno080.py \\
    --convert-imu \\
    --input imu_data.json \\
    --output imu_data_openicc.json

# Convert OpenICC output to UMI format only
python scripts/calibrate_rpi_bno080.py \\
    --convert-output \\
    --input cam_imu/cam_imu_calib_result_*.json \\
    --output rpi_camera_intrinsics.json

TROUBLESHOOTING
---------------
"Not enough views for calibration":
  - Reduce --voxel_grid_size (default 0.05, try 0.03)
  - Ensure board is clearly visible in videos
  - Check board dimensions match --rows/--cols

"scale too small" in SLAM:
  - cam_imu video needs MORE DIVERSE motion
  - Include rotation AND translation on all axes
  - Aim for 20-30+ different poses

"Empty IMU measurements":
  - Check IMU timestamps are in milliseconds
  - Ensure IMU recording starts before video
  - Verify consistent IMU sample rate (~200Hz)

OUTPUT FILES
------------
After successful calibration:
  - example/calibration/rpi_camera_intrinsics.json  (main calibration)
  - rpi_bno080_slam_settings.yaml                    (SLAM config)
  - calibration_dir/calibration_output.log           (full log)
  - calibration_dir/cam_imu/cam_imu_calib_result_*.json  (OpenICC output)
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import json
import click
import subprocess
import shutil
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

# Default ChArUco board (same as OpenICC/Hero13)
DEFAULT_SQUARE_SIZE = 0.019  # 19mm
DEFAULT_ROWS = 8
DEFAULT_COLS = 10

# Voxel grid size controls how many calibration views are used
# Smaller = more views (better calibration but slower)
# 0.05 works well for most cases; use 0.03 if "not enough views" error
DEFAULT_VOXEL_GRID_SIZE = 0.05

# BNO080 IMU parameters (from datasheet)
BNO080_IMU_PARAMS = {
    'NoiseGyro': 0.003,      # rad/s/sqrt(Hz)
    'NoiseAcc': 0.03,        # m/s^2/sqrt(Hz)
    'GyroWalk': 1.0e-4,      # rad/s^2/sqrt(Hz)
    'AccWalk': 0.003,        # m/s^3/sqrt(Hz)
    'Frequency': 200.0,      # Hz
}


# =============================================================================
# Helper Functions
# =============================================================================

def extract_video_info(video_path):
    """Extract video resolution and FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return width, height, fps, frame_count


def is_openicc_format(data):
    """Check if IMU data is already in OpenICC/GoPro format."""
    if "1" in data and isinstance(data["1"], dict):
        streams = data["1"].get("streams", {})
        if "ACCL" in streams or "GYRO" in streams:
            return True
    return False


def add_gopro_fields_to_openicc_data(data):
    """
    Add GoPro-specific fields that OpenICC expects.

    OpenICC's telemetry_converter expects several GoPro-specific streams:
    - CORI: Camera ORIentation (quaternion) - for image stabilization metadata
    - GRAV: Gravity vector - for orientation reference
    - IORI: Image ORIentation (quaternion) - for lens correction metadata

    For non-GoPro cameras, we add placeholder/derived values.
    """
    if "1" not in data:
        return data

    streams = data["1"].get("streams", {})

    # Get timestamps and accelerometer data
    timestamps = []
    accl_samples = []
    if "ACCL" in streams:
        timestamps = [s["cts"] for s in streams["ACCL"]["samples"]]
        accl_samples = [s["value"] for s in streams["ACCL"]["samples"]]
    elif "GYRO" in streams:
        timestamps = [s["cts"] for s in streams["GYRO"]["samples"]]

    if not timestamps:
        return data

    # CORI: Camera orientation - identity quaternion [w, x, y, z]
    if "CORI" not in streams:
        streams["CORI"] = {
            "name": "CameraOrientation",
            "units": "quaternion",
            "samples": [{"cts": t, "value": [1.0, 0.0, 0.0, 0.0]} for t in timestamps]
        }

    # GRAV: Gravity vector - derive from accelerometer (normalized)
    if "GRAV" not in streams:
        grav_samples = []
        for i, t in enumerate(timestamps):
            if i < len(accl_samples):
                acc = accl_samples[i]
                mag = (acc[0]**2 + acc[1]**2 + acc[2]**2)**0.5
                if mag > 0.1:
                    grav = [acc[0]/mag, acc[1]/mag, acc[2]/mag]
                else:
                    grav = [0.0, 0.0, -1.0]
            else:
                grav = [0.0, 0.0, -1.0]
            grav_samples.append({"cts": t, "value": grav})

        streams["GRAV"] = {
            "name": "GravityVector",
            "units": "normalized",
            "samples": grav_samples
        }

    # IORI: Image orientation - same as CORI for non-stabilized cameras
    if "IORI" not in streams:
        streams["IORI"] = {
            "name": "ImageOrientation",
            "units": "quaternion",
            "samples": [{"cts": t, "value": [1.0, 0.0, 0.0, 0.0]} for t in timestamps]
        }

    data["1"]["streams"] = streams
    return data


def convert_imu_to_openicc_format(imu_json_path, output_path, imu_frequency=200.0):
    """
    Convert BNO080 IMU data to OpenICC telemetry format.

    Supports two input formats:

    1. Simple UMI format:
       {
         "imu": [
           {"timestamp": t_seconds, "gyro": [gx, gy, gz], "accel": [ax, ay, az]},
           ...
         ]
       }

    2. OpenICC/GoPro format (timestamps in milliseconds):
       {
         "1": {
           "streams": {
             "ACCL": {"samples": [{"value": [ax, ay, az], "cts": t_ms}, ...]},
             "GYRO": {"samples": [{"value": [gx, gy, gz], "cts": t_ms}, ...]}
           }
         }
       }

    Output adds CORI, GRAV, IORI fields required by OpenICC.
    """
    with open(imu_json_path) as f:
        data = json.load(f)

    # Check if already in OpenICC format
    if is_openicc_format(data):
        print(f"    IMU data already in OpenICC format, adding CORI/GRAV/IORI")
        data = add_gopro_fields_to_openicc_data(data)
        with open(output_path, 'w') as f:
            json.dump(data, f)
        return output_path

    # Try UMI format
    imu_samples = data.get('imu', [])
    if not imu_samples:
        raise ValueError(
            f"No IMU data found in {imu_json_path}.\n"
            "Expected either:\n"
            "  - 'imu' array with {timestamp, gyro, accel} entries\n"
            "  - OpenICC format with '1.streams.ACCL/GYRO'"
        )

    # Convert to OpenICC format
    accl_samples = []
    gyro_samples = []

    for sample in imu_samples:
        ts = sample['timestamp'] * 1000  # Convert seconds to milliseconds
        accl_samples.append({
            'value': sample['accel'],
            'cts': ts
        })
        gyro_samples.append({
            'value': sample['gyro'],
            'cts': ts
        })

    # Create all required GoPro-compatible fields
    cori_samples = [{"cts": s["cts"], "value": [1.0, 0.0, 0.0, 0.0]} for s in accl_samples]
    iori_samples = [{"cts": s["cts"], "value": [1.0, 0.0, 0.0, 0.0]} for s in accl_samples]

    grav_samples = []
    for s in accl_samples:
        acc = s["value"]
        mag = (acc[0]**2 + acc[1]**2 + acc[2]**2)**0.5
        if mag > 0.1:
            grav = [acc[0]/mag, acc[1]/mag, acc[2]/mag]
        else:
            grav = [0.0, 0.0, -1.0]
        grav_samples.append({"cts": s["cts"], "value": grav})

    openicc_data = {
        "frames/second": int(imu_frequency),
        "1": {
            "streams": {
                "ACCL": {
                    "samples": accl_samples,
                    "name": "Accelerometer",
                    "units": "m/s2"
                },
                "GYRO": {
                    "samples": gyro_samples,
                    "name": "Gyroscope",
                    "units": "rad/s"
                },
                "CORI": {
                    "samples": cori_samples,
                    "name": "CameraOrientation",
                    "units": "quaternion"
                },
                "GRAV": {
                    "samples": grav_samples,
                    "name": "GravityVector",
                    "units": "normalized"
                },
                "IORI": {
                    "samples": iori_samples,
                    "name": "ImageOrientation",
                    "units": "quaternion"
                }
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(openicc_data, f)

    print(f"    Converted {len(imu_samples)} samples to OpenICC format")
    return output_path


def convert_openicc_to_umi_format(openicc_result_path, output_path, fps=49.62,
                                   image_width=1296, image_height=972):
    """Convert OpenICC calibration result to UMI format."""
    with open(openicc_result_path) as f:
        result = json.load(f)

    # Extract camera intrinsics
    intrinsics = result.get('intrinsics', {})
    T_imu_cam = result.get('T_imu_cam', {})

    # Build UMI format calibration
    umi_calib = {
        "image_width": image_width,
        "image_height": image_height,
        "intrinsic_type": "FISHEYE",
        "fps": fps,
        "intrinsics": {
            "focal_length": intrinsics.get('focal_length', 450.0),
            "aspect_ratio": intrinsics.get('aspect_ratio', 1.0),
            "principal_pt_x": intrinsics.get('principal_pt', [648.0, 486.0])[0],
            "principal_pt_y": intrinsics.get('principal_pt', [648.0, 486.0])[1],
            "radial_distortion_1": intrinsics.get('distortion_coeffs', [0, 0, 0, 0])[0],
            "radial_distortion_2": intrinsics.get('distortion_coeffs', [0, 0, 0, 0])[1],
            "radial_distortion_3": intrinsics.get('distortion_coeffs', [0, 0, 0, 0])[2],
            "radial_distortion_4": intrinsics.get('distortion_coeffs', [0, 0, 0, 0])[3],
            "skew": 0.0
        },
        "t_i_c": {
            "x": T_imu_cam.get('t_imu_cam', [0, 0, 0])[0],
            "y": T_imu_cam.get('t_imu_cam', [0, 0, 0])[1],
            "z": T_imu_cam.get('t_imu_cam', [0, 0, 0])[2]
        },
        "q_i_c": {
            "w": T_imu_cam.get('q_imu_cam', [1, 0, 0, 0])[0],
            "x": T_imu_cam.get('q_imu_cam', [1, 0, 0, 0])[1],
            "y": T_imu_cam.get('q_imu_cam', [1, 0, 0, 0])[2],
            "z": T_imu_cam.get('q_imu_cam', [1, 0, 0, 0])[3]
        },
        # Calibration metadata
        "final_reproj_error": result.get('camera_reprojection_error'),
        "cam_imu_reproj_error": result.get('cam_imu_reprojection_error'),
        "nr_calib_images": result.get('nr_calib_images', 0)
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(umi_calib, f, indent=2)

    print(f"    Saved: {output_path}")
    return umi_calib


def print_calibration_report(calib):
    """Print calibration quality report with recommendations."""
    print("\n" + "=" * 70)
    print("CALIBRATION QUALITY REPORT")
    print("=" * 70)

    # Number of views
    nr_views = calib.get('nr_calib_images', 'unknown')
    print(f"\nCalibration views: {nr_views}")
    if isinstance(nr_views, int):
        if nr_views >= 25:
            print("  [EXCELLENT] Good coverage")
        elif nr_views >= 15:
            print("  [OK] Acceptable")
        else:
            print("  [WARN] Low - may affect accuracy. Reduce voxel_grid_size.")

    # Camera intrinsics error
    reproj_err = calib.get('final_reproj_error')
    if reproj_err is not None:
        print(f"\nCamera Intrinsics RMS Error: {reproj_err:.4f} pixels")
        if reproj_err < 0.5:
            print("  [EXCELLENT]")
        elif reproj_err < 1.0:
            print("  [GOOD]")
        else:
            print("  [POOR] Consider recalibrating with better lighting/board flatness")

    # Camera-IMU error
    cam_imu_err = calib.get('cam_imu_reproj_error')
    if cam_imu_err is not None:
        print(f"\nCamera-IMU RMS Error: {cam_imu_err:.4f} pixels")
        if cam_imu_err < 2.0:
            print("  [EXCELLENT]")
        elif cam_imu_err < 4.0:
            print("  [GOOD]")
        else:
            print("  [POOR] Recalibrate cam_imu with more diverse motion")

    # T_i_c translation magnitude
    t_i_c = calib.get('t_i_c', {})
    tx, ty, tz = t_i_c.get('x', 0), t_i_c.get('y', 0), t_i_c.get('z', 0)
    t_mag = (tx**2 + ty**2 + tz**2)**0.5

    print(f"\nCamera-IMU Translation (t_i_c):")
    print(f"  x: {tx*1000:+.2f} mm")
    print(f"  y: {ty*1000:+.2f} mm")
    print(f"  z: {tz*1000:+.2f} mm")
    print(f"  magnitude: {t_mag*1000:.2f} mm")

    if t_mag < 0.002:  # < 2mm
        print("  [WARN] Very small translation - may cause 'scale too small' in SLAM")
        print("         Recalibrate cam_imu with MORE translation motion")
    elif t_mag < 0.005:  # < 5mm
        print("  [OK] Small but reasonable")
    else:
        print("  [GOOD] Reasonable magnitude")

    # Rotation
    q_i_c = calib.get('q_i_c', {})
    qw = q_i_c.get('w', 1)
    qx, qy, qz = q_i_c.get('x', 0), q_i_c.get('y', 0), q_i_c.get('z', 0)
    print(f"\nCamera-IMU Rotation (q_i_c):")
    print(f"  w: {qw:.6f}")
    print(f"  x: {qx:.6f}")
    print(f"  y: {qy:.6f}")
    print(f"  z: {qz:.6f}")

    print("\n" + "=" * 70)


def prepare_calibration_data(calib_dir, video_name='raw_video.mp4', imu_name='imu_data.json'):
    """
    Prepare calibration data directories and convert IMU format.

    Creates files with GoPro-style naming (GX*.MP4, GX*.json) for OpenICC.
    OpenICC expects specific filenames like GX010001.MP4, GX010002.MP4, etc.

    Important: The .json files are created as actual files (not symlinks)
    because they must contain CORI/GRAV/IORI data that OpenICC expects.
    """
    calib_dir = Path(calib_dir)

    # GoPro-style names expected by OpenICC
    gopro_names = {
        'cam': 'GX010001',       # Camera calibration
        'cam_imu': 'GX010002',   # Camera-IMU calibration
        'imu_bias': 'GX010003',  # IMU bias calibration
    }

    print("  Checking required directories...")
    required_dirs = ['cam', 'cam_imu', 'imu_bias']

    for subdir in required_dirs:
        dir_path = calib_dir / subdir
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Missing directory: {dir_path}")

        video_path = dir_path / video_name
        imu_path = dir_path / imu_name

        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video: {video_path}")
        if not imu_path.is_file():
            raise FileNotFoundError(f"Missing IMU data: {imu_path}")

        # Create GoPro-style files
        gopro_base = gopro_names[subdir]
        gopro_video = dir_path / f"{gopro_base}.MP4"
        gopro_imu = dir_path / f"{gopro_base}.json"
        gopro_imu_gen = dir_path / f"{gopro_base}_gen.json"

        # Video symlink
        if not gopro_video.exists():
            print(f"    Creating: {subdir}/{gopro_base}.MP4 -> {video_name}")
            gopro_video.symlink_to(video_path.name)

        # IMU files with CORI/GRAV/IORI (actual files, not symlinks)
        if not gopro_imu.exists():
            print(f"    Creating: {subdir}/{gopro_base}.json (with GoPro fields)")
            convert_imu_to_openicc_format(imu_path, gopro_imu)

        if not gopro_imu_gen.exists():
            print(f"    Creating: {subdir}/{gopro_base}_gen.json")
            convert_imu_to_openicc_format(imu_path, gopro_imu_gen)

    return True


def run_openicc_calibration(calib_dir, square_size, rows, cols, voxel_grid_size,
                            recompute_corners=False):
    """
    Run OpenICC calibration in Docker.

    Args:
        calib_dir: Path to calibration directory
        square_size: ChArUco square size in meters
        rows: Number of rows in ChArUco board
        cols: Number of columns in ChArUco board
        voxel_grid_size: Controls view selection (smaller = more views)
        recompute_corners: Force re-extraction of corners

    Returns:
        Path to calibration result file, or None if failed
    """
    calib_dir = Path(calib_dir).absolute()

    print("\n" + "=" * 70)
    print("Running OpenImuCameraCalibrator")
    print("=" * 70)
    print(f"  Board: {cols}x{rows}, {square_size*1000:.1f}mm squares")
    print(f"  Voxel grid size: {voxel_grid_size}")
    print(f"  Log: {calib_dir}/calibration_output.log")
    print()
    print("  This typically takes 15-45 minutes. Please wait...")
    print()

    recompute_flag = "1" if recompute_corners else "0"

    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{calib_dir}:/data',
        '-w', '/OpenImuCameraCalibrator/python',
        'chicheng/openicc:latest',
        'bash', '-c',
        f'''
        apt-get update -qq && apt-get install -y -qq xvfb > /dev/null 2>&1
        xvfb-run -a python run_gopro_calibration.py \
            --path_calib_dataset /data \
            --path_to_build /OpenImuCameraCalibrator/build/applications \
            --camera_model FISHEYE \
            --checker_size_m {square_size} \
            --num_squares_x {cols} \
            --num_squares_y {rows} \
            --image_downsample_factor 1.0 \
            --voxel_grid_size {voxel_grid_size} \
            --board_type charuco \
            --recompute_corners {recompute_flag} \
            --verbose 1
        '''
    ]

    log_path = calib_dir / 'calibration_output.log'
    with open(log_path, 'w') as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    # Check for result file
    cam_imu_dir = calib_dir / 'cam_imu'
    result_files = list(cam_imu_dir.glob('cam_imu_calib_result_*.json'))

    if result_files:
        print(f"\n  [OK] Calibration completed!")
        print(f"       Result: {result_files[0].name}")
        return result_files[0]

    # Check for partial result (rotation only)
    rotation_files = list(cam_imu_dir.glob('imu_to_cam_calibration_*.json'))
    if rotation_files:
        print(f"\n  [WARN] Only rotation calibration completed (no full T_i_c)")
        print(f"         This may indicate the spline optimization didn't converge.")
        print(f"         Check log: {log_path}")

    print(f"\n  [ERROR] Calibration failed!")
    print(f"          Check log: {log_path}")
    print(f"\n  Common issues:")
    print(f"    - 'Not enough views': Reduce --voxel_grid_size (try 0.03)")
    print(f"    - Board not detected: Check board is visible and matches dimensions")
    print(f"    - IMU errors: Verify IMU data format and timestamps")

    return None


def generate_slam_settings(intrinsics_path, output_path):
    """Generate SLAM settings YAML from intrinsics."""
    from umi.common.camera_config import generate_slam_settings as gen_settings

    with open(intrinsics_path) as f:
        calib = json.load(f)

    width = calib.get('image_width', 1296)
    height = calib.get('image_height', 972)

    gen_settings(
        intrinsics_path=Path(intrinsics_path),
        slam_resolution=(width, height),
        output_path=Path(output_path),
        imu_params=BNO080_IMU_PARAMS
    )

    # Fix FPS to be integer (ORB-SLAM3 requirement)
    with open(output_path, 'r') as f:
        content = f.read()

    # Replace float fps with integer
    import re
    content = re.sub(r'Camera\.fps: \d+\.\d+', 'Camera.fps: 50', content)

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"    Generated: {output_path}")


# =============================================================================
# Main CLI
# =============================================================================

@click.command()
@click.option('--calib_dir', type=click.Path(), default=None,
              help='Calibration directory with cam/, cam_imu/, imu_bias/ subdirs')
@click.option('--square_size', type=float, default=DEFAULT_SQUARE_SIZE,
              help=f'ChArUco square size in meters (default: {DEFAULT_SQUARE_SIZE} = {DEFAULT_SQUARE_SIZE*1000}mm)')
@click.option('--rows', type=int, default=DEFAULT_ROWS,
              help=f'ChArUco board rows (default: {DEFAULT_ROWS})')
@click.option('--cols', type=int, default=DEFAULT_COLS,
              help=f'ChArUco board columns (default: {DEFAULT_COLS})')
@click.option('--voxel_grid_size', type=float, default=DEFAULT_VOXEL_GRID_SIZE,
              help=f'View selection grid size - smaller = more views (default: {DEFAULT_VOXEL_GRID_SIZE})')
@click.option('--output', default='example/calibration/rpi_camera_intrinsics.json',
              help='Output calibration file')
@click.option('--convert-imu', is_flag=True, default=False,
              help='Only convert IMU data format (use with --input, --output)')
@click.option('--convert-output', is_flag=True, default=False,
              help='Only convert OpenICC output to UMI format (use with --input, --output)')
@click.option('--input', 'input_file', type=click.Path(exists=True), default=None,
              help='Input file for conversion modes')
@click.option('--generate-settings', is_flag=True, default=False,
              help='Generate SLAM settings YAML after calibration')
@click.option('--recompute-corners', is_flag=True, default=False,
              help='Force re-extraction of ChArUco corners')
def main(calib_dir, square_size, rows, cols, voxel_grid_size, output,
         convert_imu, convert_output, input_file, generate_settings, recompute_corners):
    """
    Calibrate RPi camera + BNO080 IMU using OpenImuCameraCalibrator.

    This calibrates BOTH camera intrinsics AND camera-to-IMU transformation.

    \b
    Quick start:
      1. Record calibration videos (cam/, cam_imu/, imu_bias/)
      2. Run: python scripts/calibrate_rpi_bno080.py --calib_dir my_calib --generate-settings
      3. Test: python scripts_slam_pipeline/test_slam_single_video.py video.mp4 --camera_type rpi_bno080
    """

    # Mode: Convert IMU data only
    if convert_imu:
        if not input_file:
            print("Error: --input required with --convert-imu")
            sys.exit(1)
        convert_imu_to_openicc_format(input_file, output)
        print(f"Converted: {input_file} -> {output}")
        return

    # Mode: Convert OpenICC output only
    if convert_output:
        if not input_file:
            print("Error: --input required with --convert-output")
            sys.exit(1)
        calib = convert_openicc_to_umi_format(input_file, output)
        print_calibration_report(calib)
        return

    # Mode: Full calibration
    if not calib_dir:
        print("Error: --calib_dir required for full calibration")
        print()
        print("Usage:")
        print("  python scripts/calibrate_rpi_bno080.py --calib_dir my_calibration")
        print()
        print("Or for conversion modes:")
        print("  --convert-imu --input imu.json --output imu_openicc.json")
        print("  --convert-output --input result.json --output intrinsics.json")
        sys.exit(1)

    calib_dir = Path(calib_dir)

    # Header
    print()
    print("=" * 70)
    print("RPi Camera + BNO080 IMU Calibration")
    print("=" * 70)
    print()
    print(f"  Calibration directory: {calib_dir}")
    print(f"  ChArUco board: {cols}x{rows}, {square_size*1000:.1f}mm squares")
    print(f"  Voxel grid size: {voxel_grid_size}")
    print(f"  Output: {output}")
    print()

    # Step 1: Prepare data
    print("Step 1: Preparing calibration data")
    print("-" * 40)
    try:
        prepare_calibration_data(calib_dir)
        print("  [OK] Data prepared")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)

    # Step 2: Run OpenICC
    print()
    print("Step 2: Running OpenICC calibration")
    print("-" * 40)
    result_file = run_openicc_calibration(
        calib_dir, square_size, rows, cols, voxel_grid_size,
        recompute_corners=recompute_corners
    )

    if not result_file:
        sys.exit(1)

    # Step 3: Convert to UMI format
    print()
    print("Step 3: Converting to UMI format")
    print("-" * 40)

    cam_imu_video = calib_dir / 'cam_imu' / 'raw_video.mp4'
    width, height, fps, _ = extract_video_info(cam_imu_video)

    calib = convert_openicc_to_umi_format(
        result_file, output,
        fps=fps, image_width=width, image_height=height
    )

    # Print quality report
    print_calibration_report(calib)

    # Step 4: Generate SLAM settings (optional)
    if generate_settings:
        print()
        print("Step 4: Generating SLAM settings")
        print("-" * 40)
        settings_output = 'rpi_bno080_slam_settings.yaml'
        generate_slam_settings(output, settings_output)

    # Summary
    print()
    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print()
    print("Output files:")
    print(f"  - {output}")
    if generate_settings:
        print(f"  - rpi_bno080_slam_settings.yaml")
    print(f"  - {calib_dir}/calibration_output.log")
    print()
    print("Next steps:")
    print()
    print("  1. Test SLAM with a sample video:")
    print("     uv run python scripts_slam_pipeline/test_slam_single_video.py \\")
    print("         your_video.mp4 --camera_type rpi_bno080")
    print()
    print("  2. If tracking works (>80% frames), copy to final location:")
    print(f"     cp {output} example/calibration/")
    print()
    print("  3. Run full pipeline:")
    print("     uv run python run_slam_pipeline.py /path/to/session --camera_type rpi_bno080")
    print()


if __name__ == '__main__':
    main()
