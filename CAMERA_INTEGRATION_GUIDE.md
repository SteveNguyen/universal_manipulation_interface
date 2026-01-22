# Camera Integration Guide for UMI SLAM Pipeline

This guide explains how to integrate a new camera (e.g., GoPro Hero 13) with the UMI SLAM pipeline.

## Overview

The SLAM pipeline requires two key calibrations:
1. **Camera intrinsics** (focal length, distortion coefficients)
2. **Camera-IMU transformation** (T_b_c: spatial relationship between camera and IMU)

## Prerequisites

- Camera with IMU (accelerometer + gyroscope)
- Ability to record at 2.7K resolution (2704×2028) or similar
  - **Note for Hero 13**: Records at 4K only. Downscale to 2.7K using metadata-preserving ffmpeg command.
- ChArUco calibration board (10×8, actual measured dimensions)
- OpenImuCameraCalibrator (OpenICC) Docker container
- ORB-SLAM3 Docker container

## Step 1: Measure ChArUco Board

**Critical**: Measure the ACTUAL printed board dimensions, not the design specs!

```bash
# Measure with calipers:
# - Square size (edge to edge of black squares)
# - Marker size (edge to edge of ArUco markers)
# Example: 19mm squares, 9.5mm markers
```

## Step 2: Record Calibration Videos

Record **three separate videos** at 2.7K resolution:

**⚠️ Important for Hero 13 users**: Hero 13 can only record at 4K (3840×2160). Record all videos in 4K, extract IMU data from the 4K videos first, then downscale to 2.7K. IMU data is lost during video downscaling.

### Video 1: Camera Calibration (`cam/`)
- **Purpose**: Calibrate camera intrinsics only
- **Duration**: 30-60 seconds
- **Motion**: Move camera smoothly around stationary ChArUco board
- **Coverage**:
  - All angles (0°, 15°, 30°, 45° tilts)
  - All positions (center, corners, edges)
  - Various distances (fill 30%-80% of frame)
- **Tips**:
  - Keep board fully visible
  - Smooth, slow movements
  - Good lighting, no motion blur

### Video 2: Camera-IMU Calibration (`cam_imu/`)
- **Purpose**: Calibrate camera-to-IMU transformation (T_b_c)
- **Duration**: 60-90 seconds
- **Motion**: **CRITICAL FOR SUCCESS**
  - Rotate camera in all axes while viewing board
  - Translate camera in all directions
  - Combine rotations and translations
  - **Goal**: Excite all 6 DOF (3 rotations + 3 translations)**
- **Coverage**:
  - 20-30+ diverse views (checked via voxel_grid_size)
  - Various orientations and positions
- **Common mistakes**:
  - ❌ Too few views (< 15) → poor T_b_c estimation
  - ❌ Only translation or only rotation → underconstrained
  - ❌ All similar angles → poor constraint on rotation

### Video 3: IMU Bias Calibration (`imu_bias/`)
- **Purpose**: Measure IMU bias (gyro/accel offsets)
- **Duration**: 10-30 seconds
- **Motion**: **Camera completely stationary, board visible**
- **Setup**: Place camera on stable surface

## Step 3: Extract and Prepare Data

```bash
# Create calibration directory structure
CALIB_DIR="camera_name_calibration"
mkdir -p "$CALIB_DIR/cam" "$CALIB_DIR/cam_imu" "$CALIB_DIR/imu_bias"

# Copy videos
cp GX010XXX.MP4 "$CALIB_DIR/cam/"
cp GX010YYY.MP4 "$CALIB_DIR/cam_imu/"
cp GX010ZZZ.MP4 "$CALIB_DIR/imu_bias/"

# For Hero 13: Extract IMU from 4K videos FIRST (before downscaling)
# IMU data is lost during video downscaling
docker run --rm \
    -v "$CALIB_DIR/cam_imu":/data \
    chicheng/openicc:latest \
    node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js \
    /data/GX010YYY.MP4 /data/GX010YYY.json

docker run --rm \
    -v "$CALIB_DIR/imu_bias":/data \
    chicheng/openicc:latest \
    node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js \
    /data/GX010ZZZ.MP4 /data/GX010ZZZ.json

# For Hero 13: Downscale 4K videos to 2.7K (after IMU extraction)
# Note: Only cam and cam_imu need downscaling - imu_bias video frames are not used
ffmpeg -i "$CALIB_DIR/cam/GX010XXX.MP4" -vf "scale=2704:2028" -c:v libx264 -preset slow -crf 18 \
    "$CALIB_DIR/cam/GX010XXX_2.7k.MP4" -y
ffmpeg -i "$CALIB_DIR/cam_imu/GX010YYY.MP4" -vf "scale=2704:2028" -c:v libx264 -preset slow -crf 18 \
    "$CALIB_DIR/cam_imu/GX010YYY_2.7k.MP4" -y

# Convert telemetry to OpenICC format (use 4K filenames for IMU JSON)
docker run --rm \
    -v "$CALIB_DIR":/data \
    -w /OpenImuCameraCalibrator/python \
    chicheng/openicc:latest \
    python3 -c "
from telemetry_converter import TelemetryConverter
tc = TelemetryConverter()
tc.convert_gopro_telemetry_file('/data/cam_imu/GX010YYY.json', '/data/cam_imu/GX010YYY_gen.json')
tc.convert_gopro_telemetry_file('/data/imu_bias/GX010ZZZ.json', '/data/imu_bias/GX010ZZZ_gen.json')
"
```

## Step 4: Run OpenICC Calibration

```bash
# IMPORTANT: Adjust these parameters based on your board!
SQUARE_SIZE=0.019      # 19mm in meters
NUM_SQUARES_X=10       # Columns
NUM_SQUARES_Y=8        # Rows
VOXEL_GRID_SIZE=0.15   # Start with 0.15, adjust if needed

# Run calibration
docker run --rm \
    -v "$CALIB_DIR":/data \
    -w /OpenImuCameraCalibrator/python \
    chicheng/openicc:latest \
    bash -c "
        apt-get update -qq && apt-get install -y -qq xvfb > /dev/null 2>&1
        xvfb-run -a python run_gopro_calibration.py \
            --path_calib_dataset /data \
            --path_to_build /OpenImuCameraCalibrator/build/applications \
            --camera_model FISHEYE \
            --checker_size_m $SQUARE_SIZE \
            --num_squares_x $NUM_SQUARES_X \
            --num_squares_y $NUM_SQUARES_Y \
            --image_downsample_factor 1.0 \
            --voxel_grid_size $VOXEL_GRID_SIZE \
            --board_type charuco \
            --verbose 1
    "
```

### Understanding voxel_grid_size

- Controls spatial downsampling of calibration views
- **Smaller values** (0.03-0.05): More views selected, slower, may get stuck
- **Larger values** (0.15-0.20): Fewer views, faster, but may miss diversity
- **Target**: 20-30 views for good camera-IMU calibration
- **Recommended**: Start with 0.15, check output view count, adjust if needed

## Step 5: Validate Calibration Quality

```bash
# Convert calibration to UMI format
uv run python3 << 'EOF'
import json
import pathlib

calib_dir = pathlib.Path("camera_name_calibration")
calib_file = calib_dir / "cam_imu" / "imu_to_cam_calibration_GX010YYY.json"

with open(calib_file) as f:
    calib = json.load(f)

# Extract and save in UMI format
output = {
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

with open('camera_name_intrinsics_2.7k.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Calibration Quality Metrics:")
print(f"  Camera RMS error: {output['final_reproj_error']:.4f} pixels")
print(f"  Camera-IMU RMS error: {output['cam_imu_reproj_error']:.4f} pixels")
print(f"  t_i_c: {output['t_i_c']}")
print()
print("Quality Assessment:")
print("  Camera RMS < 0.5 pixels: ✓ Excellent")
print("  Camera RMS 0.5-1.0: ✓ Good")
print("  Camera RMS > 1.0: ✗ Poor - recalibrate")
print()
print("  Camera-IMU RMS < 2.0: ✓ Excellent")
print("  Camera-IMU RMS 2.0-4.0: ✓ Acceptable")
print("  Camera-IMU RMS > 4.0: ✗ Poor - recalibrate cam_imu")
print()
print("  t_i_c magnitude: Check it's reasonable (2-30mm typically)")
print("  If t_i_c < 1mm: ✗ Likely underconstrained - need more diverse views")
EOF
```

**Quality Thresholds:**
- ✅ **Camera RMS < 0.5 pixels**: Excellent intrinsics
- ✅ **Camera-IMU RMS < 2.0 pixels**: Excellent T_b_c
- ⚠️ **t_i_c < 1mm**: Suspiciously small - likely underconstrained
- ❌ **Camera RMS > 1.0 pixels**: Poor calibration - redo step 2

## Step 6: Visual Validation

Test undistortion quality:

```bash
uv run python3 hero13_dev_archive_20260120/test_undistort.py \
    --video camera_name_calibration/cam_imu/GX010YYY.MP4 \
    --intrinsics camera_name_intrinsics_2.7k.json \
    --balance 1.0 \
    --show-grid \
    --output-dir undistort_test_camera_name

# Check output images:
# - Straight lines should be straight
# - Grid should be rectangular, not curved
# - No excessive warping at edges
```

## Step 7: Generate SLAM Settings

Create 720p settings (pipeline downscales 2.7K → 720p automatically):

```bash
uv run python3 << 'EOF'
import json
import numpy as np

# Load 2.7K calibration
with open('camera_name_intrinsics_2.7k.json', 'r') as f:
    calib_27k = json.load(f)

# Scale to 720p
scale_x = 960 / 2704
scale_y = 720 / 2028

intrinsics = calib_27k['intrinsics']
fx = intrinsics['focal_length'] / intrinsics['aspect_ratio'] * scale_x
fy = intrinsics['focal_length'] * scale_y
cx = intrinsics['principal_pt_x'] * scale_x
cy = intrinsics['principal_pt_y'] * scale_y

# Convert T_i_c to T_b_c
q = calib_27k['q_i_c']
t = calib_27k['t_i_c']

# Quaternion to rotation matrix
w, x, y, z = q['w'], q['x'], q['y'], q['z']
R_i_c = np.array([
    [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
    [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
])

T_i_c = np.eye(4)
T_i_c[:3, :3] = R_i_c
T_i_c[:3, 3] = [t['x'], t['y'], t['z']]
T_b_c = np.linalg.inv(T_i_c)

# Generate YAML
yaml_content = f"""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters - Generated from calibration
#--------------------------------------------------------------------------------------------
File.version: "1.0"
Camera.type: "KannalaBrandt8"

Camera1.fx: {fx:.9f}
Camera1.fy: {fy:.9f}
Camera1.cx: {cx:.1f}
Camera1.cy: {cy:.1f}

Camera1.k1: {intrinsics['radial_distortion_1']:.15f}
Camera1.k2: {intrinsics['radial_distortion_2']:.15f}
Camera1.k3: {intrinsics['radial_distortion_3']:.15f}
Camera1.k4: {intrinsics['radial_distortion_4']:.15f}

Camera.width: 960
Camera.height: 720
Camera.fps: 60
Camera.RGB: 1

# Camera-to-IMU transformation
IMU.T_b_c1: !!opencv-matrix
    rows: 4
    cols: 4
    dt: f
    data: [{T_b_c[0,0]:.8f}, {T_b_c[0,1]:.8f}, {T_b_c[0,2]:.8f}, {T_b_c[0,3]:.8f},
           {T_b_c[1,0]:.8f}, {T_b_c[1,1]:.8f}, {T_b_c[1,2]:.8f}, {T_b_c[1,3]:.8f},
           {T_b_c[2,0]:.8f}, {T_b_c[2,1]:.8f}, {T_b_c[2,2]:.8f}, {T_b_c[2,3]:.8f},
           0.0, 0.0, 0.0, 1.0]

# IMU noise parameters (standard GoPro values)
IMU.NoiseGyro: 0.0015
IMU.NoiseAcc: 0.017
IMU.GyroWalk: 5.0e-5
IMU.AccWalk: 0.0055
IMU.Frequency: 200.0

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------
ORBextractor.nFeatures: 1250
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

System.thFarPoints: 20.0

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""

with open('camera_name_720p_slam_settings.yaml', 'w') as f:
    f.write(yaml_content)

print("✓ Generated camera_name_720p_slam_settings.yaml")
EOF
```

## Step 8: Test SLAM

```bash
# Test on a sample video
mkdir -p test_slam_camera_name

# Extract IMU from test video
docker run --rm \
    -v "$(pwd)/test_video_dir":/data \
    chicheng/openicc:latest \
    node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js \
    /data/test_video.MP4 /data/test_video_imu.json

# Run SLAM
docker run --rm \
    -v "$(pwd)/test_video.MP4":/data/raw_video.mp4:ro \
    -v "$(pwd)/test_video_imu.json":/data/imu_data.json:ro \
    -v "$(pwd)/camera_name_720p_slam_settings.yaml":/data/settings.yaml:ro \
    -v "$(pwd)/test_slam_camera_name":/output \
    chicheng/orb_slam3:latest \
    /ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam \
    --vocabulary /ORB_SLAM3/Vocabulary/ORBvoc.txt \
    --setting /data/settings.yaml \
    --input_video /data/raw_video.mp4 \
    --input_imu_json /data/imu_data.json \
    --output_trajectory_csv /output/trajectory.csv

# Calculate tracking percentage
uv run python3 -c "
import pandas as pd
df = pd.read_csv('test_slam_camera_name/trajectory.csv')
tracked = (~df['is_lost']).sum()
total = len(df)
print(f'Tracking: {tracked}/{total} = {tracked/total*100:.1f}%')
print('Target: >80% for good performance')
"
```

**Expected Results:**
- ✅ **>80% tracking**: Excellent, calibration successful
- ⚠️ **50-80% tracking**: Acceptable, may need calibration refinement
- ❌ **<50% tracking**: Poor, recalibrate (likely T_b_c issue)

## Step 9: Visualize Trajectory

```bash
# Prepare visualization directory
mkdir -p test_slam_camera_name/demos/mapping
cp test_slam_camera_name/trajectory.csv test_slam_camera_name/demos/mapping/mapping_camera_trajectory.csv
cp test_video.MP4 test_slam_camera_name/demos/mapping/raw_video.mp4
cp test_video_imu.json test_slam_camera_name/demos/mapping/imu_data.json

# Visualize
uv run visualize_slam_trajectory.py \
    test_slam_camera_name/demos/mapping \
    --video-skip 5
```

Check:
- Camera view should match video content
- Trajectory should be smooth and continuous
- No jumps or discontinuities

## Common Issues and Solutions

### Issue: Camera RMS > 1.0 pixels
**Solution**:
- Check ChArUco board measurements
- Ensure good coverage in `cam` video (all angles, distances)
- Better lighting, less motion blur

### Issue: Camera-IMU RMS > 4.0 pixels or t_i_c < 1mm
**Solution**:
- Record longer `cam_imu` video (60-90 seconds)
- More diverse motion (rotate AND translate)
- Check view count: should be 20-30+
- Adjust `voxel_grid_size` (try 0.10-0.15)

### Issue: SLAM tracking < 50%
**Possible causes**:
1. **Bad T_b_c**: Recalibrate `cam_imu` with better motion
2. **Wrong intrinsics**: Verify undistortion looks correct
3. **Test video issues**: Not enough texture or motion
4. **Settings mismatch**: Verify 720p downscaling is happening

### Issue: Calibration stuck at 100% CPU
**Solution**:
- Increase `voxel_grid_size` (try 0.15-0.20)
- Check disk space for large intermediate files

## Production Deployment

Once validated:

```bash
# Copy to production location
cp camera_name_intrinsics_2.7k.json example/calibration/
cp camera_name_720p_slam_settings.yaml scripts_slam_pipeline/

# Update pipeline scripts to use new camera settings
# Commit to git
```

## Camera-Specific Settings

Different cameras may need tuning:

### ORB Features
- Higher resolution → more features
- Lower quality sensor → lower thresholds
- Default 720p: 1250 features works well

### IMU Parameters
- Check camera specs for IMU noise
- GoPro defaults (0.0015 gyro, 0.017 accel) work for most action cameras
- Professional cameras may have better IMU specs

## References

- OpenImuCameraCalibrator: https://github.com/urbste/OpenImuCameraCalibrator
- ORB-SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3
- Kannala-Brandt model: Fisheye camera calibration model used by OpenICC
