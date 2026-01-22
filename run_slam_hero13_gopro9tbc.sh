#!/bin/bash
set -e

echo "Running SLAM with Hero 13 intrinsics + GoPro 9 T_b_c (baseline)..."
echo ""

docker run --rm \
    -v "$(pwd)/test_slam_hero13_gopro9tbc/GX010073_2.7k.MP4":/data/raw_video.mp4:ro \
    -v "$(pwd)/test_slam_hero13_gopro9tbc/GX010073.json":/data/imu_data.json:ro \
    -v "$(pwd)/hero13_720p_slam_settings_gopro9_tbc.yaml":/data/settings.yaml:ro \
    -v "$(pwd)/test_slam_hero13_gopro9tbc/output":/output \
    chicheng/orb_slam3:latest \
    /ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam \
    --vocabulary /ORB_SLAM3/Vocabulary/ORBvoc.txt \
    --setting /data/settings.yaml \
    --input_video /data/raw_video.mp4 \
    --input_imu_json /data/imu_data.json \
    --output_trajectory_csv /output/trajectory.csv

echo ""
echo "SLAM complete! Analyzing results..."
echo ""

uv run python3 check_tracking.py test_slam_hero13_gopro9tbc/output/trajectory.csv
