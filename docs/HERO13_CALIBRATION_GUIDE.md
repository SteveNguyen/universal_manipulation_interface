# GoPro Hero 13 Calibration Guide

Complete guide for calibrating a GoPro Hero 13 camera for use with the UMI pipeline.

## Overview

The calibration process determines:
1. **Camera intrinsics** - focal length, principal point, distortion coefficients
2. **Camera-IMU transform (T_i_c)** - physical relationship between camera and IMU
3. **Time offset** - synchronization between camera frames and IMU samples

## Prerequisites

### Hardware
- GoPro Hero 13
- ChArUco calibration board (10x8 grid, 19mm squares, 9.5mm markers)
- Flat, rigid surface for the board (no wrinkles!)
- Good lighting (avoid glare and shadows)

### Software
```bash
# Docker images
docker pull chicheng/openicc:latest
docker pull chicheng/orb_slam3:latest

# Python environment (in UMI repo)
uv sync
```

## Critical Requirements

### FPS Consistency
**ALL videos (calibration AND data collection) MUST use the SAME FPS!**

| Setting | Value |
|---------|-------|
| Resolution | 4K (4:3) = 4000x3000 |
| Lens | Ultra Wide |
| FPS | 60fps (or 50fps, but be consistent!) |
| Stabilization | **OFF** |
| Low Light | OFF |

**Why?** IMU timestamps are synchronized with video frame timing. FPS mismatch causes ~20% timing error, breaking visual-inertial fusion.

---

## Step 1: Record Calibration Videos

Record THREE separate videos at 4K resolution:

### 1.1 Camera Calibration Video (`cam/`)
**Duration:** 30-60 seconds

- Board is **STATIONARY** on a flat surface
- Move the **camera** around the board
- Cover all angles: center, corners, tilted (0°-45°)
- Various distances: board fills 30%-80% of frame
- Slow, smooth movements (avoid motion blur)

### 1.2 Camera-IMU Calibration Video (`cam_imu/`)
**Duration:** 60-90 seconds - **THIS IS THE MOST IMPORTANT VIDEO**

- **DIVERSE motion is critical!**
- Include ALL 6 degrees of freedom:
  - Rotation: pitch, yaw, roll
  - Translation: X, Y, Z movement
- Goal: 20-30+ different poses
- Keep board visible throughout
- Include figure-8 patterns

**Common mistake:** Only rotating the camera. You MUST also translate!

### 1.3 IMU Bias Video (`imu_bias/`)
**Duration:** 10-30 seconds

- Camera **COMPLETELY STATIONARY** on stable surface
- Board should be visible (but doesn't need to move)
- Used to compute IMU bias (mean offset)

---

## Step 2: Run Calibration Script

### Option A: Automated Script (Recommended)

```bash
./recalibrate_hero13_proper.sh
```

This script will:
1. Extract IMU data from 4K videos
2. Downscale videos to 2.7K (2704x2028)
3. Convert telemetry to OpenICC format
4. Run OpenICC calibration
5. Convert output to UMI format
6. Validate calibration quality

### Option B: Manual Steps

#### 2.1 Create Directory Structure
```bash
mkdir -p hero13_calibration/{cam,cam_imu,imu_bias}
# Copy your videos to respective directories
```

#### 2.2 Extract IMU Data (BEFORE downscaling!)
```bash
# IMU data is lost during video re-encoding, so extract first
docker run --rm \
    -v "$(pwd)/hero13_calibration/cam_imu":/data \
    chicheng/openicc:latest \
    node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js \
    /data/GX010001.MP4 /data/GX010001.json
```

#### 2.3 Downscale Videos to 2.7K
```bash
# Hero 13 records at 4K, but calibration works better at 2.7K
ffmpeg -i cam/GX010001.MP4 -vf "scale=2704:2028" \
    -c:v libx264 -preset slow -crf 18 cam/GX010001_2.7k.MP4
```

#### 2.4 Convert Telemetry Format
```bash
docker run --rm \
    -v "$(pwd)/hero13_calibration":/data \
    -w /OpenImuCameraCalibrator/python \
    chicheng/openicc:latest \
    python3 -c "
from telemetry_converter import TelemetryConverter
tc = TelemetryConverter()
tc.convert_gopro_telemetry_file('/data/cam_imu/GX010001.json', '/data/cam_imu/GX010001_gen.json')
tc.convert_gopro_telemetry_file('/data/imu_bias/GX010002.json', '/data/imu_bias/GX010002_gen.json')
"
```

#### 2.5 Run OpenICC Calibration
```bash
docker run --rm \
    -v "$(pwd)/hero13_calibration":/data \
    -w /OpenImuCameraCalibrator/python \
    chicheng/openicc:latest \
    bash -c "
        apt-get update -qq && apt-get install -y -qq xvfb > /dev/null 2>&1
        xvfb-run -a python run_gopro_calibration.py \
            --path_calib_dataset /data \
            --path_to_build /OpenImuCameraCalibrator/build/applications \
            --camera_model FISHEYE \
            --checker_size_m 0.019 \
            --num_squares_x 10 \
            --num_squares_y 8 \
            --image_downsample_factor 1.0 \
            --voxel_grid_size 0.10 \
            --board_type charuco \
            --verbose 1
    "
```

**Key parameter:** `voxel_grid_size`
- Controls number of calibration views selected
- Target: 20-30 views
- Too small (0.03): Too many views, slow
- Too large (0.30): Too few views, poor T_i_c

---

## Step 3: Convert to UMI Format

The OpenICC output needs to be converted to UMI JSON format:

```python
import json

# Load OpenICC calibration
with open('hero13_calibration/cam_imu/imu_to_cam_calibration_*.json') as f:
    calib = json.load(f)

# Convert to UMI format
umi_calib = {
    "final_reproj_error": calib["camera_reprojection_error"],
    "cam_imu_reproj_error": calib["cam_imu_reprojection_error"],
    "fps": 60,
    "image_height": 2028,
    "image_width": 2704,
    "intrinsic_type": "FISHEYE",
    "intrinsics": {
        "focal_length": calib["intrinsics"]["focal_length"],
        "aspect_ratio": calib["intrinsics"]["aspect_ratio"],
        "principal_pt_x": calib["intrinsics"]["principal_pt"][0],
        "principal_pt_y": calib["intrinsics"]["principal_pt"][1],
        "radial_distortion_1": calib["intrinsics"]["distortion_coeffs"][0],
        "radial_distortion_2": calib["intrinsics"]["distortion_coeffs"][1],
        "radial_distortion_3": calib["intrinsics"]["distortion_coeffs"][2],
        "radial_distortion_4": calib["intrinsics"]["distortion_coeffs"][3],
        "skew": 0.0
    },
    "t_i_c": {
        "x": calib["T_imu_cam"]["t_imu_cam"][0],
        "y": calib["T_imu_cam"]["t_imu_cam"][1],
        "z": calib["T_imu_cam"]["t_imu_cam"][2]
    },
    "q_i_c": {
        "w": calib["T_imu_cam"]["q_imu_cam"][0],
        "x": calib["T_imu_cam"]["q_imu_cam"][1],
        "y": calib["T_imu_cam"]["q_imu_cam"][2],
        "z": calib["T_imu_cam"]["q_imu_cam"][3]
    }
}

with open('hero13_intrinsics_2.7k.json', 'w') as f:
    json.dump(umi_calib, f, indent=2)
```

---

## Step 4: Generate SLAM Settings

Generate ORB-SLAM3 settings YAML from the calibration:

```bash
uv run python scripts/generate_slam_settings.py \
    --intrinsics hero13_intrinsics_2.7k.json \
    --slam_resolution 960x720 \
    --output hero13_720p_slam_settings.yaml
```

Or manually create the YAML with scaled intrinsics:

```yaml
# Scale factor from 2.7K to 720p: 960/2704 = 0.355
Camera.fx: 496.0  # focal_length * 0.355
Camera.fy: 496.0
Camera.cx: 480.0  # principal_pt_x * 0.355
Camera.cy: 360.0  # principal_pt_y * 0.355

# Distortion coefficients (unchanged)
Camera.k1: -0.0547
Camera.k2: 0.0055
Camera.k3: -0.0012
Camera.k4: 0.0001

Camera.width: 960
Camera.height: 720
Camera.fps: 60.0
Camera.type: "KannalaBrandt8"

# IMU parameters (use GoPro 10 values)
IMU.NoiseAcc: 0.017
IMU.NoiseGyro: 0.0015
IMU.AccWalk: 0.0055
IMU.GyroWalk: 0.00005

# T_imu_cam from calibration
Tbc: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [...]  # 4x4 transformation matrix
```

---

## Step 5: Validate Calibration

### Quality Metrics

| Metric | Excellent | Good | Poor |
|--------|-----------|------|------|
| Camera RMS | < 0.5 px | < 1.0 px | > 1.0 px |
| Camera-IMU RMS | < 2.0 px | < 4.0 px | > 4.0 px |
| t_i_c magnitude | > 5mm | > 2mm | < 2mm |
| Number of views | 25-35 | 15-25 | < 15 |

### Visual Validation

Test undistortion quality:
```bash
uv run python -c "
import cv2
import json
import numpy as np

with open('hero13_intrinsics_2.7k.json') as f:
    calib = json.load(f)

# Read a frame from calibration video
cap = cv2.VideoCapture('hero13_calibration/cam_imu/GX010001_2.7k.MP4')
ret, frame = cap.read()

# Build camera matrix
K = np.array([
    [calib['intrinsics']['focal_length'], 0, calib['intrinsics']['principal_pt_x']],
    [0, calib['intrinsics']['focal_length'] / calib['intrinsics']['aspect_ratio'], calib['intrinsics']['principal_pt_y']],
    [0, 0, 1]
])
D = np.array([
    calib['intrinsics']['radial_distortion_1'],
    calib['intrinsics']['radial_distortion_2'],
    calib['intrinsics']['radial_distortion_3'],
    calib['intrinsics']['radial_distortion_4']
])

# Undistort
undistorted = cv2.fisheye.undistortImage(frame, K, D, Knew=K)
cv2.imwrite('undistort_test.jpg', np.hstack([frame, undistorted]))
print('Saved undistort_test.jpg - check that straight lines are straight!')
"
```

### SLAM Test

Run SLAM on a test video:
```bash
uv run python scripts_slam_pipeline/test_slam_single_video.py \
    test_video.MP4 \
    --camera_type hero13 \
    --intrinsics hero13_intrinsics_2.7k.json \
    --slam_settings hero13_720p_slam_settings.yaml \
    --output_dir slam_validation
```

**Success criteria:**
- Map created (check `map_atlas.osa` exists)
- Tracking rate > 90%
- No "scale too small" errors

---

## Troubleshooting

### "scale too small" Error
- T_i_c magnitude too small (< 2mm)
- **Solution:** Recalibrate with more diverse cam_imu motion

### Few Calibration Views (< 15)
- `voxel_grid_size` too large
- **Solution:** Reduce to 0.08 or 0.05

### High Reprojection Error (> 1.0 px)
- Board not flat, or poor lighting
- **Solution:** Re-record with flat board, better lighting

### SLAM Fails to Initialize
- Insufficient motion in test video
- **Solution:** Start video with clear motion, visible features

### IMU Data Missing
- Forgot to extract before downscaling
- **Solution:** Re-extract from original 4K video

---

## File Locations

After successful calibration:

```
example/calibration/
├── hero13_intrinsics_2.7k.json      # Main calibration file
├── hero13_720p_slam_settings.yaml   # SLAM settings for ORB-SLAM3
└── hero13_intrinsics_720p.json      # Scaled calibration for SLAM

hero13_calibration/                   # Calibration workspace
├── cam/                              # Camera calibration video
├── cam_imu/                          # Camera-IMU calibration video + results
│   └── imu_to_cam_calibration_*.json # OpenICC output
├── imu_bias/                         # IMU bias video
└── calibration_output.log            # Full calibration log
```

---

## Quick Reference

### Calibration Command (One-liner)
```bash
./recalibrate_hero13_proper.sh
```

### Generate SLAM Settings
```bash
uv run python scripts/generate_slam_settings.py \
    --intrinsics example/calibration/hero13_intrinsics_2.7k.json \
    --slam_resolution 960x720 \
    --output example/calibration/hero13_720p_slam_settings.yaml
```

### Test SLAM
```bash
uv run python scripts_slam_pipeline/test_slam_single_video.py \
    your_video.MP4 --camera_type hero13
```

---

## References

- [OpenImuCameraCalibrator](https://github.com/urbste/OpenImuCameraCalibrator)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [UMI Project](https://umi-gripper.github.io)
