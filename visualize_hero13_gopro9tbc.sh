#!/bin/bash
set -e

echo "Preparing visualization directory structure..."
echo ""

# Create demos/mapping structure with all required files
mkdir -p test_slam_hero13_gopro9tbc/demos/mapping
cp test_slam_hero13_gopro9tbc/output/trajectory.csv test_slam_hero13_gopro9tbc/demos/mapping/mapping_camera_trajectory.csv
cp test_slam_hero13_gopro9tbc/GX010073_2.7k.MP4 test_slam_hero13_gopro9tbc/demos/mapping/raw_video.mp4
cp test_slam_hero13_gopro9tbc/GX010073.json test_slam_hero13_gopro9tbc/demos/mapping/imu_data.json

echo "Visualizing Hero 13 SLAM results (with GoPro 9 T_b_c)..."
echo ""

uv run python3 visualize_slam_trajectory.py \
    test_slam_hero13_gopro9tbc/demos/mapping \
    --calibration hero13_720p_intrinsics_gopro9_tbc.json \
    --video-skip 10

echo ""
echo "Visualization complete!"
echo "Open the Rerun viewer in your browser to see the 3D trajectory."
