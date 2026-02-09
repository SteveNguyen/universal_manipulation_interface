# Raspberry Pi Camera + BNO080 IMU Calibration Guide

Complete guide for calibrating a Raspberry Pi camera with BNO080 IMU for use with the UMI pipeline.

## Overview

The calibration process determines:
1. **Camera intrinsics** - focal length, principal point, distortion coefficients (k1-k4 for fisheye)
2. **Camera-IMU transform (T_i_c)** - physical relationship between camera and IMU
3. **Time offset** - synchronization between camera frames and IMU samples

## Prerequisites

### Hardware
- Raspberry Pi camera module (1296×972 @ 49.62fps, wide-angle/fisheye lens)
- BNO080 IMU (Hillcrest/CEVA 9-axis IMU with sensor fusion)
- ChArUco calibration board (10×8 squares, 19mm square size - same as Hero 13)
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

### Data Recording System
The RPi camera + BNO080 data recording system should output:
- `raw_video.mp4` - Video file
- `imu_data.json` - IMU data in GoPro-compatible format

Expected IMU JSON format:
```json
{
  "imu": [
    {
      "timestamp": 0.0,
      "gyro": [gx, gy, gz],
      "accel": [ax, ay, az]
    },
    ...
  ]
}
```

---

## Critical Requirements

### FPS and Timing Consistency
**ALL videos (calibration AND data collection) MUST use the SAME FPS!**

| Setting | Value |
|---------|-------|
| Resolution | 1296×972 |
| FPS | 49.62fps |
| IMU Rate | 200 Hz |

**Why?** IMU timestamps are synchronized with video frame timing. FPS mismatch causes timing errors, breaking visual-inertial fusion.

### BNO080 Coordinate Frame
The BNO080 has a specific coordinate frame that must match what SLAM expects:
- Check datasheet for axis orientation
- May need rotation to match camera frame conventions

---

## Step 1: Print Calibration Target

### ChArUco Board Specifications

**Use the same board as Hero 13 calibration:**
- Grid: 10×8 squares (10 columns × 8 rows)
- Square size: 19mm (measure with calipers to verify!)
- Marker size: 9.5mm (half of square size)
- Dictionary: **DICT_ARUCO_ORIGINAL** (important - not DICT_4X4_50!)

If you need to generate a new board:
```bash
# Using OpenCV
uv run python -c "
import cv2
import cv2.aruco as aruco

# Create ChArUco board (DICT_ARUCO_ORIGINAL)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
board = aruco.CharucoBoard((10, 8), 0.019, 0.0095, dictionary)
img = board.generateImage((1000, 800), marginSize=20, borderBits=1)
cv2.imwrite('charuco_10x8_19mm.png', img)
print('Saved charuco_10x8_19mm.png')
"
```

### Mounting
- Print at 100% scale (no scaling!)
- Mount on rigid, flat surface (foam board, acrylic, etc.)
- **Verify dimensions with calipers** - this is critical!

---

## Step 2: Record Calibration Videos

Record THREE separate sequences:

### 2.1 Camera Calibration Video (`cam/`)
**Duration:** 30-60 seconds

- Board is **STATIONARY** on a flat surface
- Move the **camera** around the board
- Cover all angles: center, corners, tilted (0°-45°)
- Various distances: board fills 30%-80% of frame
- Slow, smooth movements (avoid motion blur)

### 2.2 Camera-IMU Calibration Video (`cam_imu/`)
**Duration:** 60-90 seconds - **THIS IS THE MOST IMPORTANT VIDEO**

- **DIVERSE motion is critical!**
- Include ALL 6 degrees of freedom:
  - Rotation: pitch, yaw, roll
  - Translation: X, Y, Z movement
- Goal: 20-30+ different poses
- Keep board visible throughout
- Include figure-8 patterns

**Common mistake:** Only rotating the camera. You MUST also translate!

### 2.3 IMU Bias Video (`imu_bias/`)
**Duration:** 10-30 seconds

- Camera **COMPLETELY STATIONARY** on stable surface
- Board should be visible (but doesn't need to move)
- Used to compute IMU bias (mean offset)

---

## Step 3: Organize Data

Create the following directory structure:
```
rpi_calibration/
├── cam/
│   ├── raw_video.mp4
│   └── imu_data.json
├── cam_imu/
│   ├── raw_video.mp4
│   └── imu_data.json
└── imu_bias/
    ├── raw_video.mp4
    └── imu_data.json
```

---

## Step 4: Run Calibration

### Option A: OpenCV Calibration (Recommended for DICT_ARUCO_ORIGINAL boards)

If your ChArUco board uses DICT_ARUCO_ORIGINAL (not DICT_4X4_50), use the OpenCV calibration script:

```bash
uv run python scripts/calibrate_camera_opencv.py \
    --video rpi_calibration/cam_imu/raw_video.mp4 \
    --output example/calibration/rpi_camera_intrinsics.json \
    --square_size 0.019 \
    --cols 10 \
    --rows 8 \
    --dictionary DICT_ARUCO_ORIGINAL \
    --fps 49.62

# Generate SLAM settings
uv run python scripts/generate_slam_settings.py \
    --intrinsics example/calibration/rpi_camera_intrinsics.json \
    --slam_resolution 1296x972 \
    --output rpi_bno080_slam_settings.yaml
```

Note: This calibrates camera intrinsics only. Camera-IMU transform (T_i_c) requires manual measurement or additional calibration.

### Option B: Automated Script (for DICT_4X4_50 boards)

```bash
./recalibrate_rpi_bno080.sh
```

### Option C: Manual OpenICC Steps

#### 4.1 Convert IMU Data to OpenICC Format

The BNO080 IMU data needs to be converted to OpenICC's expected format:

```bash
uv run python scripts/calibrate_rpi_bno080.py \
    --convert-imu \
    --input rpi_calibration/cam_imu/imu_data.json \
    --output rpi_calibration/cam_imu/imu_data_openicc.json
```

#### 4.2 Run OpenICC Calibration

```bash
docker run --rm \
    -v "$(pwd)/rpi_calibration":/data \
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

**Key parameters:**
- `checker_size_m`: Square size in meters (0.019 = 19mm, same as Hero 13 board)
- `voxel_grid_size`: Controls calibration view count (target 20-30 views)

---

## Step 5: Convert to UMI Format

```bash
uv run python scripts/calibrate_rpi_bno080.py \
    --convert-output \
    --input rpi_calibration/cam_imu/imu_to_cam_calibration_*.json \
    --output example/calibration/rpi_camera_intrinsics.json
```

This creates a UMI-format intrinsics file with:
- Camera intrinsics (fx, fy, cx, cy, k1-k4)
- IMU-camera transform (t_i_c, q_i_c)
- Resolution and FPS information

---

## Step 6: Generate SLAM Settings

```bash
uv run python scripts/generate_slam_settings.py \
    --intrinsics example/calibration/rpi_camera_intrinsics.json \
    --slam_resolution 1296x972 \
    --output rpi_bno080_slam_settings.yaml
```

Or manually update `rpi_bno080_slam_settings.yaml` with calibrated values.

---

## Step 7: Validate Calibration

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

with open('example/calibration/rpi_camera_intrinsics.json') as f:
    calib = json.load(f)

# Read a frame from calibration video
cap = cv2.VideoCapture('rpi_calibration/cam_imu/raw_video.mp4')
ret, frame = cap.read()

# Build camera matrix
intr = calib['intrinsics']
K = np.array([
    [intr['focal_length'], 0, intr['principal_pt_x']],
    [0, intr['focal_length'] * intr['aspect_ratio'], intr['principal_pt_y']],
    [0, 0, 1]
])
D = np.array([
    intr['radial_distortion_1'],
    intr['radial_distortion_2'],
    intr['radial_distortion_3'],
    intr['radial_distortion_4']
])

# Undistort
undistorted = cv2.fisheye.undistortImage(frame, K, D, Knew=K)
cv2.imwrite('undistort_test.jpg', np.hstack([frame, undistorted]))
print('Saved undistort_test.jpg - check that straight lines are straight!')
"
```

### SLAM Test

```bash
uv run python scripts_slam_pipeline/test_slam_single_video.py \
    test_video.mp4 \
    --camera_type rpi_bno080 \
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

### IMU Timing Issues
- BNO080 timestamps not synchronized with camera
- **Solution:** Verify timestamp synchronization in data recording system

### "Empty IMU measurements vector" Error
- IMU samples not matching video frame timestamps
- **Causes:**
  - Variable IMU sample rate (should be consistent ~200 Hz)
  - Timestamp offset between IMU and video start
- **Solution:** Ensure IMU recording starts before video and has consistent sample rate

### Wrong IMU Coordinate Frame
- BNO080 axes don't match expected orientation
- **Solution:** Apply rotation matrix to convert BNO080 frame to camera frame

---

## BNO080 IMU Noise Parameters

Default values from datasheet (may need tuning):

```yaml
IMU.NoiseGyro: 0.003      # rad/s/√Hz
IMU.NoiseAcc: 0.03        # m/s²/√Hz
IMU.GyroWalk: 1.0e-4      # rad/s²/√Hz
IMU.AccWalk: 0.003        # m/s³/√Hz
IMU.Frequency: 200.0      # Hz
```

### Allan Variance Analysis (Optional)
For more accurate noise parameters:
1. Record 2+ hours of static IMU data
2. Run Allan variance analysis tool
3. Extract noise density and random walk values

---

## File Locations

After successful calibration:

```
example/calibration/
├── rpi_camera_intrinsics.json          # Main calibration file
└── rpi_bno080_calibrated_slam_settings.yaml  # Calibrated SLAM settings

rpi_calibration/                         # Calibration workspace
├── cam/                                 # Camera calibration video
├── cam_imu/                             # Camera-IMU calibration video + results
│   └── imu_to_cam_calibration_*.json   # OpenICC output
├── imu_bias/                            # IMU bias video
└── calibration_output.log               # Full calibration log
```

---

## Quick Reference

### Full Calibration (One-liner)
```bash
./recalibrate_rpi_bno080.sh
```

### Generate SLAM Settings
```bash
uv run python scripts/generate_slam_settings.py \
    --intrinsics example/calibration/rpi_camera_intrinsics.json \
    --slam_resolution 1296x972 \
    --output example/calibration/rpi_bno080_calibrated_slam_settings.yaml
```

### Test SLAM (Single Video)
```bash
uv run python scripts_slam_pipeline/test_slam_single_video.py \
    your_video.mp4 --camera_type rpi_bno080
```

### Run Full Pipeline
```bash
uv run python run_slam_pipeline.py /path/to/session --camera_type rpi_bno080
```

---

## References

- [OpenImuCameraCalibrator](https://github.com/urbste/OpenImuCameraCalibrator)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [BNO080 Datasheet](https://www.ceva-ip.com/product/bno080-sensor-hub/)
- [UMI Project](https://umi-gripper.github.io)
