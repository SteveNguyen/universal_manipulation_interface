#!/bin/bash
set -e

# Generate SLAM settings from proper Hero 13 calibration

echo "============================================"
echo "Generate Hero 13 SLAM Settings"
echo "============================================"
echo ""

if [ ! -f "hero13_proper_intrinsics_2.7k.json" ]; then
    echo "❌ Error: hero13_proper_intrinsics_2.7k.json not found"
    echo "Run ./recalibrate_hero13_proper.sh first"
    exit 1
fi

echo "Generating 720p SLAM settings from calibration..."
echo ""

uv run python3 << 'PYTHONEOF'
import json
import numpy as np

# Load 2.7K calibration
with open('hero13_proper_intrinsics_2.7k.json', 'r') as f:
    calib_27k = json.load(f)

# Scale to 720p (960×720 from 2704×2028)
scale_x = 960 / 2704
scale_y = 720 / 2028

intrinsics = calib_27k['intrinsics']
q_i_c = calib_27k['q_i_c']
t_i_c = calib_27k['t_i_c']

# Scale intrinsics
fx_720 = intrinsics['focal_length'] / intrinsics['aspect_ratio'] * scale_x
fy_720 = intrinsics['focal_length'] * scale_y
cx_720 = intrinsics['principal_pt_x'] * scale_x
cy_720 = intrinsics['principal_pt_y'] * scale_y

# Distortion coefficients stay the same
k1 = intrinsics['radial_distortion_1']
k2 = intrinsics['radial_distortion_2']
k3 = intrinsics['radial_distortion_3']
k4 = intrinsics['radial_distortion_4']

print(f"720p intrinsics:")
print(f"  fx: {fx_720:.2f}, fy: {fy_720:.2f}")
print(f"  cx: {cx_720:.1f}, cy: {cy_720:.1f}")
print(f"  k1-k4: {k1:.4f}, {k2:.4f}, {k3:.4f}, {k4:.4f}")
print()

# Convert q_i_c and t_i_c to T_b_c
w, x, y, z = q_i_c['w'], q_i_c['x'], q_i_c['y'], q_i_c['z']
R_i_c = np.array([
    [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
    [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
])

T_i_c = np.eye(4)
T_i_c[:3, :3] = R_i_c
T_i_c[:3, 3] = [t_i_c['x'], t_i_c['y'], t_i_c['z']]

# Invert to get T_b_c (body-to-camera for ORB-SLAM3)
T_b_c = np.linalg.inv(T_i_c)

print(f"T_b_c (camera-to-IMU transform):")
print(f"  Translation: ({T_b_c[0,3]*1000:.2f}, {T_b_c[1,3]*1000:.2f}, {T_b_c[2,3]*1000:.2f}) mm")
print()

# Save 720p intrinsics JSON
calib_720 = {
    "final_reproj_error": calib_27k['final_reproj_error'],
    "fps": calib_27k['fps'],
    "image_width": 960,
    "image_height": 720,
    "intrinsic_type": "FISHEYE",
    "intrinsics": {
        "focal_length": fy_720,
        "aspect_ratio": fx_720 / fy_720,
        "principal_pt_x": cx_720,
        "principal_pt_y": cy_720,
        "radial_distortion_1": k1,
        "radial_distortion_2": k2,
        "radial_distortion_3": k3,
        "radial_distortion_4": k4,
        "skew": 0.0
    },
    "cam_imu_reproj_error": calib_27k['cam_imu_reproj_error'],
    "t_i_c": t_i_c,
    "q_i_c": q_i_c
}

with open('hero13_proper_intrinsics_720p.json', 'w') as f:
    json.dump(calib_720, f, indent=2)

# Generate YAML
yaml_content = f"""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Hero 13 Proper Calibration - 720p Settings
# Generated from hero13_proper_intrinsics_2.7k.json
#--------------------------------------------------------------------------------------------
File.version: "1.0"
Camera.type: "KannalaBrandt8"

# Camera calibration scaled from 2.7K to 720p
Camera1.fx: {fx_720:.9f}
Camera1.fy: {fy_720:.9f}
Camera1.cx: {cx_720:.1f}
Camera1.cy: {cy_720:.1f}

Camera1.k1: {k1:.15f}
Camera1.k2: {k2:.15f}
Camera1.k3: {k3:.15f}
Camera1.k4: {k4:.15f}

# Camera resolution - ORB-SLAM3 will downscale 2.7K video automatically
Camera.width: 960
Camera.height: 720

# Camera frames per second
Camera.fps: 60

# Color order of the images (0: BGR, 1: RGB)
Camera.RGB: 1

# Transformation from camera to IMU (body frame)
# From proper calibration with {calib_27k.get('nr_calib_images', 'N')} views
IMU.T_b_c1: !!opencv-matrix
    rows: 4
    cols: 4
    dt: f
    data: [{T_b_c[0,0]:.8f}, {T_b_c[0,1]:.8f}, {T_b_c[0,2]:.8f}, {T_b_c[0,3]:.8f},
           {T_b_c[1,0]:.8f}, {T_b_c[1,1]:.8f}, {T_b_c[1,2]:.8f}, {T_b_c[1,3]:.8f},
           {T_b_c[2,0]:.8f}, {T_b_c[2,1]:.8f}, {T_b_c[2,2]:.8f}, {T_b_c[2,3]:.8f},
           0.0, 0.0, 0.0, 1.0]

# IMU noise parameters
IMU.NoiseGyro: 0.0015
IMU.NoiseAcc: 0.017
IMU.GyroWalk: 5.0e-5
IMU.AccWalk: 0.0055
IMU.Frequency: 200.0

#--------------------------------------------------------------------------------------------
# ORB Parameters - Standard 720p settings
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1250

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
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

with open('hero13_proper_720p_slam_settings.yaml', 'w') as f:
    f.write(yaml_content)

print("✓ Generated files:")
print("  - hero13_proper_intrinsics_720p.json")
print("  - hero13_proper_720p_slam_settings.yaml")
print()
print("Next: Test SLAM on a sample video")
PYTHONEOF

echo ""
echo "============================================"
echo "SLAM Settings Generated!"
echo "============================================"
echo ""
echo "Files created:"
echo "  ✓ hero13_proper_intrinsics_720p.json (for visualization)"
echo "  ✓ hero13_proper_720p_slam_settings.yaml (for SLAM)"
echo ""
echo "To test SLAM:"
echo "  1. Record a test video at 2.7K (2704×2028)"
echo "  2. Extract IMU with:"
echo "     docker run --rm -v \"\$(pwd)/test_dir\":/data \\"
echo "       chicheng/openicc:latest \\"
echo "       node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js \\"
echo "       /data/test_video.MP4 /data/test_video_imu.json"
echo ""
echo "  3. Run SLAM:"
echo "     docker run --rm \\"
echo "       -v \"\$(pwd)/test_video.MP4\":/data/raw_video.mp4:ro \\"
echo "       -v \"\$(pwd)/test_video_imu.json\":/data/imu_data.json:ro \\"
echo "       -v \"\$(pwd)/hero13_proper_720p_slam_settings.yaml\":/data/settings.yaml:ro \\"
echo "       -v \"\$(pwd)/test_slam_output\":/output \\"
echo "       chicheng/orb_slam3:latest \\"
echo "       /ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam \\"
echo "       --vocabulary /ORB_SLAM3/Vocabulary/ORBvoc.txt \\"
echo "       --setting /data/settings.yaml \\"
echo "       --input_video /data/raw_video.mp4 \\"
echo "       --input_imu_json /data/imu_data.json \\"
echo "       --output_trajectory_csv /output/trajectory.csv"
echo ""
echo "  4. Check tracking percentage:"
echo "     python3 -c \"import pandas as pd; df = pd.read_csv('test_slam_output/trajectory.csv'); \\"
echo "       print(f'{(~df['is_lost']).sum()}/{len(df)} = {((~df['is_lost']).sum()/len(df)*100):.1f}%')\""
echo ""
echo "Target: >80% tracking for good calibration"
echo ""
