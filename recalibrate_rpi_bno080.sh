#!/bin/bash
set -e

# RPi Camera + BNO080 IMU Calibration Script
#
# This script calibrates:
# - Camera intrinsics (fx, fy, cx, cy, k1-k4 for fisheye)
# - Camera-IMU transform (T_i_c)
#
# Prerequisites:
# - docker pull chicheng/openicc:latest
# - Calibration videos recorded with synchronized camera + IMU

CALIB_DIR="rpi_bno080_calibration"
# Using same ChArUco board as Hero 13 calibration
SQUARE_SIZE=0.019  # 19mm in meters
ROWS=8
COLS=10

echo "============================================"
echo "RPi Camera + BNO080 IMU Calibration"
echo "============================================"
echo ""

# Check if calibration directory exists
if [ -d "$CALIB_DIR" ]; then
    echo "Calibration directory already exists: $CALIB_DIR"
    read -p "Continue and overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Step 1: Prepare Calibration Videos"
echo "===================================="
echo ""
echo "You need to record THREE video+IMU sequences:"
echo ""
echo "1. cam/ - Camera calibration (30-60 sec)"
echo "   - Move camera smoothly around STATIONARY board"
echo "   - Cover all angles: center, corners, tilted (0-45 deg)"
echo "   - Various distances: 30%-80% frame fill"
echo ""
echo "2. cam_imu/ - Camera-IMU calibration (60-90 sec) **CRITICAL**"
echo "   - DIVERSE motion: rotate AND translate"
echo "   - All 6 DOF: pitch, yaw, roll + X, Y, Z translation"
echo "   - Goal: 20-30+ different poses"
echo "   - Keep board visible throughout"
echo "   This is the MOST IMPORTANT video for good T_i_c!"
echo ""
echo "3. imu_bias/ - IMU bias (10-30 sec)"
echo "   - Camera COMPLETELY STATIONARY"
echo "   - Board visible"
echo ""
echo "Have you recorded all three sequences? (y/N): "
read -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please record calibration sequences first."
    echo "Refer to docs/RPI_BNO080_CALIBRATION_GUIDE.md for instructions."
    exit 1
fi

# Create directory structure
mkdir -p "$CALIB_DIR/cam" "$CALIB_DIR/cam_imu" "$CALIB_DIR/imu_bias"

echo ""
echo "Step 2: Copy Data to Calibration Directory"
echo "==========================================="
echo ""
echo "Copy your video and IMU files to:"
echo "  $CALIB_DIR/cam/       - raw_video.mp4 + imu_data.json"
echo "  $CALIB_DIR/cam_imu/   - raw_video.mp4 + imu_data.json"
echo "  $CALIB_DIR/imu_bias/  - raw_video.mp4 + imu_data.json"
echo ""
read -p "Press Enter when files are copied..."

# Verify files exist
for DIR in cam cam_imu imu_bias; do
    VIDEO="$CALIB_DIR/$DIR/raw_video.mp4"
    IMU="$CALIB_DIR/$DIR/imu_data.json"

    if [ ! -f "$VIDEO" ]; then
        echo "Error: Missing video: $VIDEO"
        exit 1
    fi
    if [ ! -f "$IMU" ]; then
        echo "Error: Missing IMU data: $IMU"
        exit 1
    fi
    echo "Found: $DIR/ (video + imu)"
done

echo ""
echo "Step 3: ChArUco Board Dimensions"
echo "================================="
echo ""
echo "Default board dimensions (same as Hero 13):"
echo "  Square size: 19mm (0.019m)"
echo "  Grid: ${COLS}x${ROWS} (columns x rows)"
echo ""
read -p "Are these dimensions correct? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Enter square size in meters (e.g., 0.030 for 30mm): " SQUARE_SIZE
    read -p "Enter columns: " COLS
    read -p "Enter rows: " ROWS
fi

echo ""
echo "Step 4: Run Calibration"
echo "========================"
echo ""
echo "This will:"
echo "  1. Convert IMU data to OpenICC format"
echo "  2. Run OpenImuCameraCalibrator"
echo "  3. Convert output to UMI format"
echo ""
echo "This process takes 10-30 minutes."
echo ""
read -p "Press Enter to start..."

# Run Python calibration script
uv run python scripts/calibrate_rpi_bno080.py \
    --calib_dir "$CALIB_DIR" \
    --square_size "$SQUARE_SIZE" \
    --rows "$ROWS" \
    --cols "$COLS" \
    --voxel_grid_size 0.10 \
    --output "example/calibration/rpi_camera_intrinsics.json" \
    --generate-settings

echo ""
echo "Step 5: Generate Final SLAM Settings"
echo "====================================="
echo ""

# Copy calibrated settings to the expected location
if [ -f "example/calibration/rpi_camera_intrinsics_slam_settings.yaml" ]; then
    cp "example/calibration/rpi_camera_intrinsics_slam_settings.yaml" \
       "example/calibration/rpi_bno080_calibrated_slam_settings.yaml"
    echo "Copied to: example/calibration/rpi_bno080_calibrated_slam_settings.yaml"
fi

# Also update the template file with calibrated values
if [ -f "example/calibration/rpi_camera_intrinsics.json" ]; then
    echo ""
    echo "Generating updated rpi_bno080_slam_settings.yaml..."
    uv run python scripts/generate_slam_settings.py \
        --intrinsics "example/calibration/rpi_camera_intrinsics.json" \
        --slam_resolution 1296x972 \
        --output rpi_bno080_slam_settings.yaml
fi

echo ""
echo "============================================"
echo "Calibration Complete!"
echo "============================================"
echo ""
echo "Output files:"
echo "  - example/calibration/rpi_camera_intrinsics.json"
echo "  - example/calibration/rpi_bno080_calibrated_slam_settings.yaml"
echo "  - rpi_bno080_slam_settings.yaml (updated with calibration)"
echo ""
echo "To test SLAM:"
echo "  uv run python scripts_slam_pipeline/test_slam_single_video.py \\"
echo "      your_test_video.mp4 --camera_type rpi_bno080"
echo ""
echo "To run full pipeline:"
echo "  uv run python run_slam_pipeline.py /path/to/session --camera_type rpi_bno080"
echo ""
