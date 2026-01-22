#!/bin/bash
set -e

# Proper Hero 13 Recalibration Script
# Based on lessons learned from initial calibration attempt
#
# Key improvements:
# - Better voxel_grid_size tuning (target 20-30 views)
# - Clear instructions for recording better cam_imu video
# - Validation checks throughout

CALIB_DIR="hero13_proper_calibration"
OPENICC_DIR="${OPENICC_DIR:-./OpenImuCameraCalibrator}"

echo "============================================"
echo "Hero 13 Proper Recalibration"
echo "============================================"
echo ""

# Check if calibration directory exists
if [ -d "$CALIB_DIR" ]; then
    echo "⚠️  Calibration directory already exists: $CALIB_DIR"
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
echo "⚠️  Hero 13 records in 4K, but we need 2.7K for calibration."
echo "We'll extract IMU from 4K videos, then downscale to 2.7K."
echo ""
echo "You need to record THREE videos at 4K resolution:"
echo ""
echo "1. cam/ - Camera calibration (30-60 sec)"
echo "   - Move camera smoothly around STATIONARY board"
echo "   - Cover all angles: center, corners, tilted (0°-45°)"
echo "   - Various distances: 30%-80% frame fill"
echo ""
echo "2. cam_imu/ - Camera-IMU calibration (60-90 sec) **CRITICAL**"
echo "   - DIVERSE motion: rotate AND translate"
echo "   - All 6 DOF: pitch, yaw, roll + X, Y, Z translation"
echo "   - Goal: 20-30+ different poses"
echo "   - Keep board visible throughout"
echo "   ⚠️  This is the MOST IMPORTANT video for good T_b_c!"
echo ""
echo "3. imu_bias/ - IMU bias (10-30 sec)"
echo "   - Camera COMPLETELY STATIONARY"
echo "   - Board visible"
echo ""
echo "Have you recorded all three videos at 4K? (y/N): "
read -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please record calibration videos at 4K first."
    echo "Refer to CAMERA_INTEGRATION_GUIDE.md for detailed instructions."
    exit 1
fi

# Create directory structure
mkdir -p "$CALIB_DIR/cam" "$CALIB_DIR/cam_imu" "$CALIB_DIR/imu_bias"

echo ""
echo "Step 2: Copy Videos to Calibration Directory"
echo "============================================="
echo ""
echo "Copy your videos to:"
echo "  $CALIB_DIR/cam/       - Camera calibration video"
echo "  $CALIB_DIR/cam_imu/   - Camera-IMU calibration video"
echo "  $CALIB_DIR/imu_bias/  - Static IMU bias video"
echo ""
read -p "Press Enter when videos are copied..."

# Detect video files
CAM_VIDEO=$(ls "$CALIB_DIR/cam/"*.MP4 2>/dev/null | head -1)
CAM_IMU_VIDEO=$(ls "$CALIB_DIR/cam_imu/"*.MP4 2>/dev/null | head -1)
IMU_BIAS_VIDEO=$(ls "$CALIB_DIR/imu_bias/"*.MP4 2>/dev/null | head -1)

if [ -z "$CAM_VIDEO" ] || [ -z "$CAM_IMU_VIDEO" ] || [ -z "$IMU_BIAS_VIDEO" ]; then
    echo "❌ Error: Missing videos!"
    echo "   cam: ${CAM_VIDEO:-NOT FOUND}"
    echo "   cam_imu: ${CAM_IMU_VIDEO:-NOT FOUND}"
    echo "   imu_bias: ${IMU_BIAS_VIDEO:-NOT FOUND}"
    exit 1
fi

echo "✓ Found videos:"
echo "  cam: $(basename "$CAM_VIDEO")"
echo "  cam_imu: $(basename "$CAM_IMU_VIDEO")"
echo "  imu_bias: $(basename "$IMU_BIAS_VIDEO")"

echo ""
echo "Step 3: Extract IMU Data from 4K Videos"
echo "========================================"
echo ""
echo "Extracting IMU data from original 4K videos..."
echo "(IMU data is lost during downscaling, so extract first!)"
echo ""

# Extract from cam_imu
echo "Extracting IMU from cam_imu video..."
docker run --rm \
    -v "$(pwd)/$CALIB_DIR/cam_imu":/data \
    chicheng/openicc:latest \
    node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js \
    /data/$(basename "$CAM_IMU_VIDEO") \
    /data/$(basename "$CAM_IMU_VIDEO" .MP4).json

# Extract from imu_bias
echo "Extracting IMU from imu_bias video..."
docker run --rm \
    -v "$(pwd)/$CALIB_DIR/imu_bias":/data \
    chicheng/openicc:latest \
    node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js \
    /data/$(basename "$IMU_BIAS_VIDEO") \
    /data/$(basename "$IMU_BIAS_VIDEO" .MP4).json

echo "✓ IMU extraction complete"

echo ""
echo "Step 4: Downscale Videos from 4K to 2.7K"
echo "========================================="
echo ""
echo "Hero 13 records at 4K (3840×2160), but we need 2.7K (2704×2028) for calibration."
echo "Downscaling videos (IMU already extracted from 4K)..."
echo ""

# Save original 4K filenames for IMU JSON files
CAM_IMU_4K=$(basename "$CAM_IMU_VIDEO" .MP4)
IMU_BIAS_4K=$(basename "$IMU_BIAS_VIDEO" .MP4)

# Downscale cam video (used for camera intrinsics)
echo "Downscaling cam video..."
ffmpeg -i "$CAM_VIDEO" -vf "scale=2704:2028" -c:v libx264 -preset slow -crf 18 \
    "$CALIB_DIR/cam/$(basename "$CAM_VIDEO" .MP4)_2.7k.MP4" -y

# Downscale cam_imu video (used for camera-IMU calibration)
echo "Downscaling cam_imu video..."
ffmpeg -i "$CAM_IMU_VIDEO" -vf "scale=2704:2028" -c:v libx264 -preset slow -crf 18 \
    "$CALIB_DIR/cam_imu/$(basename "$CAM_IMU_VIDEO" .MP4)_2.7k.MP4" -y

# Note: imu_bias video NOT downscaled - only IMU data is used, not video frames

# Update video paths to point to downscaled versions
CAM_VIDEO="$CALIB_DIR/cam/$(basename "$CAM_VIDEO" .MP4)_2.7k.MP4"
CAM_IMU_VIDEO="$CALIB_DIR/cam_imu/$(basename "$CAM_IMU_VIDEO" .MP4)_2.7k.MP4"

echo "✓ Video downscaling complete (4K originals preserved)"

echo ""
echo "Step 5: Convert Telemetry to OpenICC Format"
echo "==========================================="
echo ""

docker run --rm \
    -v "$(pwd)/$CALIB_DIR":/data \
    -w /OpenImuCameraCalibrator/python \
    chicheng/openicc:latest \
    python3 << PYTHONEOF
from telemetry_converter import TelemetryConverter
tc = TelemetryConverter()

# Use 4K filenames for IMU JSON (extracted from original 4K videos)
cam_imu_file = '$CAM_IMU_4K'
imu_bias_file = '$IMU_BIAS_4K'

tc.convert_gopro_telemetry_file(f'/data/cam_imu/{cam_imu_file}.json',
                                f'/data/cam_imu/{cam_imu_file}_gen.json')
tc.convert_gopro_telemetry_file(f'/data/imu_bias/{imu_bias_file}.json',
                                f'/data/imu_bias/{imu_bias_file}_gen.json')
print("✓ Telemetry conversion complete")
PYTHONEOF

echo ""
echo "Step 6: ChArUco Board Dimensions"
echo "================================="
echo ""
echo "Measured board dimensions (from previous calibration):"
echo "  Square size: 19mm"
echo "  Marker size: 9.5mm"
echo "  Grid: 10×8 (columns × rows)"
echo ""
read -p "Are these dimensions correct? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please measure your board with calipers and update this script."
    exit 1
fi

SQUARE_SIZE=0.019  # 19mm in meters
NUM_SQUARES_X=10
NUM_SQUARES_Y=8

echo ""
echo "Step 7: Run OpenICC Calibration"
echo "================================"
echo ""
echo "voxel_grid_size tuning:"
echo "  - Controls how many calibration views are selected"
echo "  - Target: 20-30 views for good camera-IMU calibration"
echo "  - Too small (0.03): Many views, slow, may get stuck"
echo "  - Too large (0.30): Few views, underconstrained T_b_c"
echo ""
echo "Previous attempt: voxel_grid_size=0.15 → 11 views (too few)"
echo "This attempt: voxel_grid_size=0.10 → aiming for 20-30 views"
echo ""
read -p "Press Enter to start calibration (takes 15-45 min)..."

VOXEL_GRID_SIZE=0.10

docker run --rm \
    -v "$(pwd)/$CALIB_DIR":/data \
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
    " 2>&1 | tee "$CALIB_DIR/calibration_output.log"

echo ""
echo "Step 8: Extract and Validate Calibration"
echo "========================================="
echo ""

# Convert to UMI format
uv run python3 << 'PYTHONEOF'
import json
import pathlib

calib_dir = pathlib.Path("hero13_proper_calibration")

# Find the imu_to_cam calibration file
cam_imu_files = list((calib_dir / "cam_imu").glob("imu_to_cam_calibration_*.json"))
if not cam_imu_files:
    print("❌ Error: No calibration output found!")
    exit(1)

calib_file = cam_imu_files[0]
print(f"Reading calibration from: {calib_file}")

with open(calib_file) as f:
    calib = json.load(f)

# Convert to UMI format
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
    },
    "nr_calib_images": calib.get("nr_calib_images", "unknown")
}

with open('hero13_proper_intrinsics_2.7k.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "="*60)
print("CALIBRATION QUALITY REPORT")
print("="*60)
print(f"\nNumber of calibration views: {output['nr_calib_images']}")
print(f"\nCamera Intrinsics RMS Error: {output['final_reproj_error']:.4f} pixels")
if output['final_reproj_error'] < 0.5:
    print("  ✅ Excellent!")
elif output['final_reproj_error'] < 1.0:
    print("  ✅ Good")
else:
    print("  ❌ Poor - consider recalibrating")

print(f"\nCamera-IMU RMS Error: {output['cam_imu_reproj_error']:.4f} pixels")
if output['cam_imu_reproj_error'] < 2.0:
    print("  ✅ Excellent!")
elif output['cam_imu_reproj_error'] < 4.0:
    print("  ✅ Acceptable")
else:
    print("  ❌ Poor - recalibrate cam_imu with more diverse motion")

t_mag = (output['t_i_c']['x']**2 + output['t_i_c']['y']**2 + output['t_i_c']['z']**2)**0.5
print(f"\nt_i_c (camera-IMU translation):")
print(f"  x: {output['t_i_c']['x']*1000:.2f}mm")
print(f"  y: {output['t_i_c']['y']*1000:.2f}mm")
print(f"  z: {output['t_i_c']['z']*1000:.2f}mm")
print(f"  magnitude: {t_mag*1000:.2f}mm")

if t_mag < 0.001:
    print("  ❌ Too small! Likely underconstrained.")
    print("     → Recalibrate cam_imu with MORE DIVERSE motion (20-30+ views)")
elif t_mag < 0.002:
    print("  ⚠️  Small but might be ok. Compare with previous: 7.1mm (4K), 0.4mm (2.7K bad)")
else:
    print("  ✅ Reasonable magnitude")

print("\nComparison with previous calibrations:")
print("  Hero 13 2.7K (11 views, poor):  0.4mm")
print("  Hero 13 4K (24 views, good):     7.1mm")
print("  GoPro 9 (working):              ~27mm")
print(f"  This calibration:               {t_mag*1000:.1f}mm")

print("\n" + "="*60)
print(f"✓ Saved to: hero13_proper_intrinsics_2.7k.json")
print("="*60)
PYTHONEOF

echo ""
echo "Step 9: Visual Validation"
echo "========================="
echo ""
echo "Testing undistortion quality..."

uv run python3 hero13_dev_archive_20260120/test_undistort.py \
    --video "$CAM_IMU_VIDEO" \
    --intrinsics hero13_proper_intrinsics_2.7k.json \
    --balance 1.0 \
    --show-grid \
    --output-dir undistort_test_hero13_proper

echo ""
echo "✓ Undistortion test images saved to: undistort_test_hero13_proper/"
echo ""
echo "Visual checks:"
echo "  - Straight lines should be straight"
echo "  - Grid should be rectangular, not curved"
echo "  - No excessive warping"
echo ""

echo ""
echo "============================================"
echo "Recalibration Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Review calibration quality report above"
echo "  2. Check undistortion test images in undistort_test_hero13_proper/"
echo "  3. If quality is good, generate SLAM settings and test"
echo ""
echo "To generate SLAM settings, run:"
echo "  ./generate_hero13_slam_settings_from_proper.sh"
echo ""
