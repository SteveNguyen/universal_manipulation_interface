#!/usr/bin/env python3
"""
Create 960x720 calibration from 4K Hero 13 OpenICC calibration.
"""
import json

print("=" * 70)
print("Creating 960x720 Calibration from 4K OpenICC Calibration")
print("=" * 70)

# Load 4K calibration (camera intrinsics with aspect_ratio)
cam_calib_path = 'hero13_openicc_dataset/cam/cam_calib_GX000001_fi_1.0.json'
cam_imu_calib_path = 'hero13_openicc_dataset/cam_imu/cam_imu_calib_result_GX000002.json'
output_path = 'hero13_720p_intrinsics.json'

with open(cam_calib_path) as f:
    calib_4k = json.load(f)

# Load camera-IMU calibration for T_i_c
with open(cam_imu_calib_path) as f:
    cam_imu_data = json.load(f)
    calib_4k['cam_imu_reproj_error'] = cam_imu_data['final_reproj_error']
    calib_4k['t_i_c'] = cam_imu_data['t_i_c']
    calib_4k['q_i_c'] = cam_imu_data['q_i_c']

print(f"\nInput camera calibration: {cam_calib_path}")
print(f"Input camera-IMU calibration: {cam_imu_calib_path}")
print(f"  Resolution: {calib_4k['image_width']}x{calib_4k['image_height']}")
print(f"  Camera RMS: {calib_4k['final_reproj_error']:.4f} pixels")
print(f"  Camera-IMU RMS: {calib_4k['cam_imu_reproj_error']:.4f} pixels")

# Target resolution
target_width = 960
target_height = 720

# Calculate scale factors
scale_x = target_width / calib_4k['image_width']
scale_y = target_height / calib_4k['image_height']

print(f"\nTarget resolution: {target_width}x{target_height}")
print(f"Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")

# Create new calibration
calib_720p = {
    "final_reproj_error": calib_4k['final_reproj_error'] * scale_x,  # Scale error proportionally
    "fps": calib_4k['fps'],
    "image_height": target_height,
    "image_width": target_width,
    "intrinsic_type": calib_4k['intrinsic_type'],
    "intrinsics": {
        "aspect_ratio": calib_4k['intrinsics'].get('aspect_ratio', 1.0),
        "focal_length": calib_4k['intrinsics']['focal_length'] * scale_x,
        "principal_pt_x": calib_4k['intrinsics']['principal_pt_x'] * scale_x,
        "principal_pt_y": calib_4k['intrinsics']['principal_pt_y'] * scale_y,
        # Distortion coefficients stay the same for fisheye
        "radial_distortion_1": calib_4k['intrinsics']['radial_distortion_1'],
        "radial_distortion_2": calib_4k['intrinsics']['radial_distortion_2'],
        "radial_distortion_3": calib_4k['intrinsics']['radial_distortion_3'],
        "radial_distortion_4": calib_4k['intrinsics']['radial_distortion_4'],
        "skew": calib_4k['intrinsics'].get('skew', 0.0)
    },
    "cam_imu_reproj_error": calib_4k['cam_imu_reproj_error'] * scale_x,
    "t_i_c": calib_4k['t_i_c'],  # Physical transformation stays the same
    "q_i_c": calib_4k['q_i_c']   # Physical transformation stays the same
}

# Save
with open(output_path, 'w') as f:
    json.dump(calib_720p, f, indent=2)

print("\n" + "=" * 70)
print("720p Calibration Parameters")
print("=" * 70)
print(f"Resolution: {calib_720p['image_width']}x{calib_720p['image_height']}")
print(f"FPS: {calib_720p['fps']}")
print(f"\nIntrinsics (scaled):")
print(f"  fx: {calib_720p['intrinsics']['focal_length']:.6f} (4K: {calib_4k['intrinsics']['focal_length']:.6f})")
print(f"  cx: {calib_720p['intrinsics']['principal_pt_x']:.6f} (4K: {calib_4k['intrinsics']['principal_pt_x']:.6f})")
print(f"  cy: {calib_720p['intrinsics']['principal_pt_y']:.6f} (4K: {calib_4k['intrinsics']['principal_pt_y']:.6f})")
print(f"\nDistortion (unchanged):")
print(f"  k1: {calib_720p['intrinsics']['radial_distortion_1']:.6f}")
print(f"  k2: {calib_720p['intrinsics']['radial_distortion_2']:.6f}")
print(f"  k3: {calib_720p['intrinsics']['radial_distortion_3']:.6f}")
print(f"  k4: {calib_720p['intrinsics']['radial_distortion_4']:.6f}")
print(f"\nExpected RMS at 720p: {calib_720p['final_reproj_error']:.4f} pixels")
print(f"Expected Camera-IMU RMS at 720p: {calib_720p['cam_imu_reproj_error']:.4f} pixels")

print(f"\n✓ Saved to: {output_path}")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
1. Downscale your test video to 960x720:
   ffmpeg -i test_slam_hero13/GX010057.MP4 -vf scale=960:720 -c:a copy test_video_720p.MP4

2. Run SLAM with 720p calibration:
   python3 test_hero13_slam.py test_video_720p.MP4 \\
     --intrinsics hero13_720p_intrinsics.json \\
     --output_dir test_hero13_slam_720p

Or use the real UMI pipeline scripts.
""")
