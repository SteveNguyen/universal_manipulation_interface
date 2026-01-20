#!/usr/bin/env python3
"""
Quick SLAM test for Hero 13 calibration.
Tests if your calibration works before running full pipeline.

Usage: python test_hero13_slam.py <test_video.mp4>
"""

import sys
import pathlib
import subprocess
import shutil
import json
import click


@click.command()
@click.argument('test_video', type=click.Path(exists=True))
@click.option('--intrinsics', default='hero13_intrinsics.json',
              help='Path to Hero 13 intrinsics file')
@click.option('--output_dir', default='test_hero13_slam',
              help='Output directory for test')
def main(test_video, intrinsics, output_dir):
    """Quick SLAM test for Hero 13."""

    test_video = pathlib.Path(test_video)
    intrinsics_path = pathlib.Path(intrinsics)
    output_dir = pathlib.Path(output_dir)

    print("=" * 60)
    print("Hero 13 SLAM Quick Test")
    print("=" * 60)
    print(f"Test video: {test_video}")
    print(f"Intrinsics: {intrinsics_path}")
    print(f"Output dir: {output_dir}")
    print()

    # Validate inputs
    if not test_video.exists():
        print(f"✗ Error: Test video not found: {test_video}")
        sys.exit(1)

    if not intrinsics_path.exists():
        print(f"✗ Error: Intrinsics file not found: {intrinsics_path}")
        print(f"  Expected location: {intrinsics_path.absolute()}")
        sys.exit(1)

    # Read intrinsics to get resolution
    with open(intrinsics_path) as f:
        calib = json.load(f)

    width = calib['image_width']
    height = calib['image_height']
    fps = calib['fps']
    print(f"Calibration resolution: {width}x{height}")
    print(f"Calibration FPS: {fps}")

    # Check if we have camera-IMU calibration data
    has_cam_imu_calib = 'q_i_c' in calib and 't_i_c' in calib
    if has_cam_imu_calib:
        print(f"Camera-IMU calibration: {calib.get('cam_imu_reproj_error', 'N/A'):.2f} pixels")

    # Detect actual video FPS
    import subprocess
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate',
         '-of', 'default=noprint_wrappers=1:nokey=1',
         str(test_video)],
        capture_output=True, text=True
    )
    video_fps_str = result.stdout.strip()
    if '/' in video_fps_str:
        num, den = map(float, video_fps_str.split('/'))
        video_fps = num / den
    else:
        video_fps = float(video_fps_str)

    print(f"Actual video FPS: {video_fps:.2f}")

    if abs(video_fps - fps) > 5:
        print(f"⚠ WARNING: Video FPS ({video_fps:.2f}) doesn't match calibration FPS ({fps:.2f})")
        print(f"  This will cause IMU synchronization issues!")
        print(f"  Using actual video FPS for SLAM...")

    # SLAM requires integer FPS
    fps = int(round(video_fps))

    print()

    # Create test directory structure
    print("Setting up test directory...")
    output_dir.mkdir(exist_ok=True)

    demos_dir = output_dir / "demos"
    demos_dir.mkdir(exist_ok=True)

    mapping_dir = demos_dir / "mapping"
    mapping_dir.mkdir(exist_ok=True)

    # Copy test video
    raw_video = mapping_dir / "raw_video.mp4"
    if raw_video.exists():
        raw_video.unlink()

    print(f"Copying test video to {raw_video}...")
    shutil.copy2(test_video, raw_video)
    print(f"✓ Video copied")
    print()

    # Generate SLAM settings from intrinsics JSON
    print(f"Generating SLAM settings from calibration (RMS={calib['final_reproj_error']:.4f})...")

    fx = calib['intrinsics']['focal_length'] / calib['intrinsics']['aspect_ratio']
    fy = calib['intrinsics']['focal_length']
    cx = calib['intrinsics']['principal_pt_x']
    cy = calib['intrinsics']['principal_pt_y']
    k1 = calib['intrinsics']['radial_distortion_1']
    k2 = calib['intrinsics']['radial_distortion_2']
    k3 = calib['intrinsics']['radial_distortion_3']
    k4 = calib['intrinsics']['radial_distortion_4']

    # Convert quaternion to rotation matrix if camera-IMU calibration is available
    if has_cam_imu_calib:
        import numpy as np
        q = calib['q_i_c']
        t = calib['t_i_c']

        # Normalize quaternion
        qw, qx, qy, qz = q['w'], q['x'], q['y'], q['z']
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx)],
            [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
        ])

        # Create 4x4 transformation matrix
        T_i_c = np.eye(4)
        T_i_c[:3, :3] = R
        T_i_c[:3, 3] = [t['x'], t['y'], t['z']]
        T_b_c_data = ', '.join(f'{x}' for x in T_i_c.flatten().tolist())

        imu_calib_comment = f"OpenICC calibration\n# Mean reprojection error: {calib.get('cam_imu_reproj_error', 'N/A'):.2f} pixels"
    else:
        # Use hardcoded approximation (old calibration)
        T_b_c_data = """-0.999961, 0.00738297, 0.00489099, 0.000871679,
          -0.00486524, 0.0035017, -0.999982, -0.0149645,
          -0.00739996, -0.999967, -0.00346565, -0.00163611,
          0.0, 0.0, 0.0, 1.0"""
        imu_calib_comment = "Approximate GoPro 10 calibration\n# WARNING: Not calibrated for this Hero 13!"

    settings_content = f"""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters for GoPro Hero 13 Ultra Wide 4K (4000x3000)
# Generated from: {intrinsics_path.name}
# RMS error: {calib['final_reproj_error']:.4f} pixels
#--------------------------------------------------------------------------------------------
File.version: "1.0"
Camera.type: "KannalaBrandt8"

# Camera calibration parameters
Camera1.fx: {fx}
Camera1.fy: {fy}
Camera1.cx: {cx}
Camera1.cy: {cy}

# Distortion parameters (k1, k2, k3, k4)
Camera1.k1: {k1}
Camera1.k2: {k2}
Camera1.k3: {k3}
Camera1.k4: {k4}

# Camera resolution
Camera.width: {width}
Camera.height: {height}

# Camera frames per second
Camera.fps: {fps}

# Color order of the images (0: BGR, 1: RGB. Ignored if images are grayscale)
Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 2000

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
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0

#--------------------------------------------------------------------------------------------
# IMU Parameters (from GoPro Hero 13)
#--------------------------------------------------------------------------------------------

# IMU noise (using GoPro 10 values as baseline)
IMU.NoiseGyro: 0.0015  # rad/s^0.5
IMU.NoiseAcc: 0.017    # m/s^1.5
IMU.GyroWalk: 5.0e-5   # rad/s^1.5
IMU.AccWalk: 0.0055    # m/s^2.5

# IMU frequency (Hz)
IMU.Frequency: 200.0

# Transformation from camera to IMU
# {imu_calib_comment}
IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [{T_b_c_data}]

# Insert KF when tracking is lost
IMU.InsertKFsWhenLost: 1
"""

    # Write generated settings
    settings_dest = mapping_dir / "hero13_4k_slam_settings.yaml"
    with open(settings_dest, 'w') as f:
        f.write(settings_content)

    print(f"✓ SLAM settings generated with FPS={fps}")
    print()

    # Step 1: Extract IMU data
    print("=" * 60)
    print("Step 1: Extracting IMU data...")
    print("=" * 60)

    cmd = [
        'python', 'scripts_slam_pipeline/01_extract_gopro_imu.py',
        str(output_dir)
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("✗ IMU extraction failed")
        sys.exit(1)

    # Check if IMU data was created
    imu_json = mapping_dir / "imu_data.json"
    if not imu_json.exists():
        print("✗ IMU data not created")
        sys.exit(1)

    print("✓ IMU data extracted")
    print()

    # Step 2: Create SLAM mask for Hero 13 resolution
    print("Creating SLAM mask for Hero 13 (4000x3000)...")
    try:
        import cv2
        import numpy as np
        from umi.common.cv_util import draw_predefined_mask

        # Create empty mask (all black = all pixels active) for testing
        slam_mask = np.zeros((height, width), dtype=np.uint8)
        # slam_mask = draw_predefined_mask(
        #     slam_mask, color=255, mirror=True, gripper=False, finger=True)
        mask_path = mapping_dir / "slam_mask.png"
        cv2.imwrite(str(mask_path), slam_mask)
        print(f"✓ SLAM mask created (disabled for testing): {mask_path}")
    except ImportError:
        # Create minimal black PNG without cv2
        print("⚠ cv2 not available, creating minimal black mask...")
        import struct

        mask_path = mapping_dir / "slam_mask.png"
        # Create minimal 1x1 black PNG (ORB-SLAM3 will resize)
        png_data = (
            b'\x89PNG\r\n\x1a\n'  # PNG signature
            b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'  # 1x1 image
            b'\x08\x00\x00\x00\x00:~\x9bU'  # grayscale, no compression
            b'\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01'  # black pixel
            b'H\xaf\xa4q'
            b'\x00\x00\x00\x00IEND\xaeB`\x82'  # PNG end
        )
        with open(mask_path, 'wb') as f:
            f.write(png_data)
        print(f"✓ Minimal black mask created: {mask_path}")
    print()

    # Step 3: Run SLAM with custom Hero 13 settings
    print("=" * 60)
    print("Step 3: Running SLAM with Hero 13 settings...")
    print("=" * 60)
    print("This may take 5-10 minutes for a short video...")
    print()

    # Pull docker image
    print("Ensuring ORB_SLAM3 docker image is available...")
    docker_image = "chicheng/orb_slam3:latest"
    cmd = ['docker', 'pull', docker_image]
    subprocess.run(cmd)
    print()

    # Prepare paths for docker (need absolute paths)
    map_file = mapping_dir / "map_atlas.osa"
    mapping_dir_abs = mapping_dir.absolute()
    mapping_parent_abs = mapping_dir_abs.parent

    cmd = [
        'docker', 'run', '--rm',
        '--volume', f'{mapping_dir_abs}:/data',
        '--volume', f'{mapping_parent_abs}:/map',
        docker_image,
        '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
        '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        '--setting', '/data/hero13_4k_slam_settings.yaml',  # Use Hero 13 settings
        '--input_video', '/data/raw_video.mp4',
        '--input_imu_json', '/data/imu_data.json',
        '--output_trajectory_csv', '/data/mapping_camera_trajectory.csv',
        '--save_map', '/map/mapping/map_atlas.osa',
        '--mask_img', '/data/slam_mask.png'
    ]

    stdout_path = mapping_dir / 'slam_stdout.txt'
    stderr_path = mapping_dir / 'slam_stderr.txt'

    with open(stdout_path, 'w') as stdout_f, open(stderr_path, 'w') as stderr_f:
        result = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f)

    if result.returncode != 0:
        print("✗ SLAM failed")
        print()
        print("Check the logs:")
        print(f"  stdout: {stdout_path}")
        print(f"  stderr: {stderr_path}")
        sys.exit(1)

    print("✓ SLAM completed")
    print()

    # Step 4: Check outputs
    print("=" * 60)
    print("Step 4: Checking SLAM outputs...")
    print("=" * 60)

    trajectory_csv = mapping_dir / "mapping_camera_trajectory.csv"
    map_file = mapping_dir / "map_atlas.osa"

    success = True

    if trajectory_csv.exists():
        # Count successfully tracked poses (not lost)
        import pandas as pd
        df = pd.read_csv(trajectory_csv)
        num_poses = len(df[df['is_lost'] == False])

        print(f"✓ Camera trajectory: {num_poses} successfully tracked poses")

        if num_poses < 10:
            print("  ✗ VERY FEW poses tracked - tracking mostly failed!")
            print("    Calibration quality is insufficient")
            success = False
        elif num_poses < 50:
            print("  ⚠ Warning: Significant tracking loss")
            print("    Calibration needs improvement")
            success = False
        else:
            print("  ✓ Good tracking!")
    else:
        print("✗ No camera trajectory generated")
        success = False

    if map_file.exists():
        print(f"✓ Map file created: {map_file}")
    else:
        print("✗ No map file created")
        success = False

    print()

    # Final verdict
    print("=" * 60)
    if success:
        print("✓ SLAM TEST PASSED!")
        print("=" * 60)
        print()
        print("Your Hero 13 calibration works for SLAM!")
        print()
        print("Next steps:")
        print("1. Copy calibration to project:")
        print(f"   cp {intrinsics_path} example/calibration/")
        print()
        print("2. Update pipeline scripts to use Hero 13 intrinsics:")
        print("   - scripts_slam_pipeline/04_detect_aruco.py")
        print("   - Update default --intrinsics_json path")
        print()
        print(f"3. Update SLAM mask resolution to {width}x{height}:")
        print("   - scripts_slam_pipeline/02_create_map.py line 62")
        print()
        print("4. Start collecting data with your Hero 13!")
    else:
        print("✗ SLAM TEST FAILED")
        print("=" * 60)
        print()
        print("The calibration doesn't work well enough for SLAM.")
        print()
        print("Recommended actions:")
        print("1. Re-record calibration video with:")
        print("   - Perfectly flat ChArUco board (no wrinkles)")
        print("   - Better coverage (more angles/distances)")
        print("   - Very slow movement (avoid any blur)")
        print()
        print("2. Run calibration again:")
        print("   python calibrate_hero13_safe.py ...")
        print()
        print("3. Target: RMS error < 1.0 pixels")

    print()


if __name__ == '__main__':
    main()
