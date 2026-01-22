#!/usr/bin/env python3
"""
Visualize SLAM trajectory in 3D using Rerun.

Usage:
    python visualize_slam_trajectory.py <slam_output_dir>
    python visualize_slam_trajectory.py test_hero13_slam_720p/demos/mapping

Requirements:
    pip install rerun-sdk opencv-python pandas numpy scipy
"""

import sys
import pathlib
import click
import numpy as np
import pandas as pd
import cv2
import rerun as rr
from scipy.spatial.transform import Rotation


@click.command()
@click.argument('slam_dir', type=click.Path(exists=True))
@click.option('--show-video/--no-video', default=True, help='Show video frames')
@click.option('--video-skip', default=10, help='Show every Nth video frame')
@click.option('--calibration', type=click.Path(exists=True), default=None, help='Path to calibration JSON for accurate projection')
@click.option('--show-imu-frame', is_flag=True, help='Show IMU frame axes for debugging')
@click.option('--app-id', default='slam_viz', help='Rerun application ID')
def main(slam_dir, show_video, video_skip, calibration, show_imu_frame, app_id):
    """Visualize SLAM trajectory from ORB-SLAM3 output."""

    slam_dir = pathlib.Path(slam_dir)

    # Check for required files
    traj_csv = slam_dir / "mapping_camera_trajectory.csv"
    video_path = slam_dir / "raw_video.mp4"
    imu_json = slam_dir / "imu_data.json"

    if not traj_csv.exists():
        print(f"Error: Trajectory file not found: {traj_csv}")
        print("Expected: <slam_dir>/mapping_camera_trajectory.csv")
        sys.exit(1)

    # Load calibration if provided
    calib_intrinsics = None
    target_width = None
    target_height = None
    T_cam_imu = None
    if calibration:
        import json
        with open(calibration) as f:
            calib_data = json.load(f)
            if 'intrinsics' in calib_data:
                calib_intrinsics = calib_data
                print(f"Loaded calibration from {calibration}")
                # Get expected resolution from calibration
                if 'image_width' in calib_data and 'image_height' in calib_data:
                    target_width = calib_data['image_width']
                    target_height = calib_data['image_height']
                    print(f"Calibration resolution: {target_width}x{target_height}")

                # Get IMU-camera transform (T_imu_cam)
                if 'q_i_c' in calib_data and 't_i_c' in calib_data:
                    # q_i_c: quaternion from IMU to camera (R_imu_cam)
                    # t_i_c: translation from IMU to camera
                    # Convention: v_cam = R_i_c * v_imu + t_i_c
                    q_i_c = calib_data['q_i_c']
                    t_i_c = calib_data['t_i_c']
                    T_cam_imu = {
                        'q': np.array([q_i_c['x'], q_i_c['y'], q_i_c['z'], q_i_c['w']]),  # xyzw format
                        't': np.array([t_i_c['x'], t_i_c['y'], t_i_c['z']])
                    }
                    print(f"Loaded T_imu_cam transform from calibration")

    # Load IMU data if available
    imu_data = None
    if imu_json.exists():
        import json
        print(f"Loading IMU data from {imu_json}...")
        with open(imu_json) as f:
            raw_imu_data = json.load(f)

        # Parse GoPro metadata format
        # Structure: {frame_num: {streams: {ACCL: {samples: [...]}, GYRO: {samples: [...]}}}}
        # Note: JSON may also contain metadata like "frames/second" at top level
        imu_samples = {'accel': [], 'gyro': []}
        for frame_num, frame_data in raw_imu_data.items():
            # Skip non-dictionary entries (like "frames/second": 59.94)
            if not isinstance(frame_data, dict):
                continue
            if 'streams' not in frame_data:
                continue

            streams = frame_data['streams']

            # Extract accelerometer data
            if 'ACCL' in streams and 'samples' in streams['ACCL']:
                for sample in streams['ACCL']['samples']:
                    imu_samples['accel'].append({
                        'timestamp': sample['cts'] / 1000.0,  # Convert ms to seconds
                        'value': sample['value']  # [x, y, z]
                    })

            # Extract gyroscope data
            if 'GYRO' in streams and 'samples' in streams['GYRO']:
                for sample in streams['GYRO']['samples']:
                    imu_samples['gyro'].append({
                        'timestamp': sample['cts'] / 1000.0,  # Convert ms to seconds
                        'value': sample['value']  # [x, y, z]
                    })

        if imu_samples['accel'] or imu_samples['gyro']:
            imu_data = imu_samples
            print(f"  Loaded {len(imu_samples['accel'])} accel samples, {len(imu_samples['gyro'])} gyro samples")

    # Initialize rerun
    rr.init(app_id, spawn=True)

    # Log IMU data if available
    if imu_data:
        print("\nLogging IMU data...")

        # Log accelerometer data
        for sample in imu_data['accel']:
            timestamp = sample['timestamp']
            value = sample['value']
            rr.set_time_seconds(timeline="timestamp", seconds=timestamp)
            rr.log("imu/accelerometer/x", rr.Scalars(value[0]))
            rr.log("imu/accelerometer/y", rr.Scalars(value[1]))
            rr.log("imu/accelerometer/z", rr.Scalars(value[2]))

        # Log gyroscope data
        for sample in imu_data['gyro']:
            timestamp = sample['timestamp']
            value = sample['value']
            rr.set_time_seconds(timeline="timestamp", seconds=timestamp)
            rr.log("imu/gyroscope/x", rr.Scalars(value[0]))
            rr.log("imu/gyroscope/y", rr.Scalars(value[1]))
            rr.log("imu/gyroscope/z", rr.Scalars(value[2]))

    # Read trajectory
    print(f"\nLoading trajectory from {traj_csv}...")
    df_all = pd.read_csv(traj_csv)

    print(f"Loaded {len(df_all)} total frames")
    print(f"Columns: {df_all.columns.tolist()}")

    # Separate valid and lost frames
    if 'is_lost' in df_all.columns:
        df_valid = df_all[df_all['is_lost'] == False].copy()
        print(f"Valid tracked poses: {len(df_valid)}")
        print(f"Lost/untracked frames: {len(df_all) - len(df_valid)}")

        if len(df_valid) > 0:
            first_valid = df_valid.iloc[0]['frame_idx']
            last_valid = df_valid.iloc[-1]['frame_idx']
            print(f"Tracking range: frames {int(first_valid)} to {int(last_valid)}")
    else:
        df_valid = df_all
        print(f"All frames appear to be tracked")

    if len(df_valid) == 0:
        print("Error: No valid poses found!")
        sys.exit(1)

    df = df_valid

    # Extract positions
    positions = df[['x', 'y', 'z']].values

    # Extract rotations (quaternions)
    # CSV format: q_x, q_y, q_z, q_w (xyzw format)
    # Rerun expects: w, x, y, z (wxyz format)
    quaternions_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].values

    # Extract timestamps/frames
    if 'timestamp' in df.columns:
        times = df['timestamp'].values
    else:
        times = df['frame_idx'].values

    print(f"\nTrajectory statistics:")
    print(f"  Valid poses: {len(positions)}")
    print(f"  Position range:")
    print(f"    X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m")
    print(f"    Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m")
    print(f"    Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m")

    # Calculate total distance
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_distance = np.sum(distances)
    print(f"  Total distance: {total_distance:.3f} m")

    # Set up world coordinate frame
    # SLAM world frame = IMU body frame at initialization (arbitrary orientation)
    # Don't force any specific axis interpretation
    # Using default right-handed coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Add coordinate axes at origin for reference (world/IMU frame at initialization)
    axis_length = 0.5
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ
        ),
        static=True
    )

    # Log the full trajectory as a line strip (green)
    rr.log("world/trajectory_full", rr.LineStrips3D(positions, colors=[0, 255, 0]))

    # Log start and end points
    rr.log("world/start", rr.Points3D(positions[0], colors=[0, 255, 0], radii=0.01))
    rr.log("world/end", rr.Points3D(positions[-1], colors=[255, 0, 0], radii=0.01))

    # Open video if requested
    video_cap = None
    if show_video and video_path.exists():
        print(f"\nLoading video from {video_path}...")
        video_cap = cv2.VideoCapture(str(video_path))
        if not video_cap.isOpened():
            print("Warning: Could not open video file")
            video_cap = None
        else:
            fps = video_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video: {total_frames} frames at {fps:.2f} fps")

    # Create a mapping from frame_idx to pose for valid frames
    pose_map = {}
    for i, row in df.iterrows():
        frame_idx = int(row['frame_idx'])
        pose_map[frame_idx] = {
            'pos': np.array([row['x'], row['y'], row['z']]),
            'quat': np.array([row['q_x'], row['q_y'], row['q_z'], row['q_w']]),
            'timestamp': row['timestamp']
        }

    # Animate through ALL frames to show full video
    print(f"\nVisualizing full video with trajectory overlay...")
    print(f"Total frames: {len(df_all)}")
    print(f"Tracked frames: {len(df)}")

    # Calculate actual intrinsics once
    if calib_intrinsics:
        aspect_ratio = calib_intrinsics['intrinsics'].get('aspect_ratio', 1.0)
        fy = calib_intrinsics['intrinsics']['focal_length']
        fx = fy / aspect_ratio
        cx = calib_intrinsics['intrinsics']['principal_pt_x']
        cy = calib_intrinsics['intrinsics']['principal_pt_y']

        # Use calibration resolution if available, otherwise default to 720p
        if target_width is None:
            target_width = 960
            target_height = 720
            print(f"No resolution in calibration, defaulting to {target_width}x{target_height}")
    else:
        fx, fy, cx, cy = None, None, None, None
        target_width = 960
        target_height = 720

    trajectory_so_far = []

    # Camera axes length (for visualization)
    cam_axis_length = 0.1

    for frame_i in range(len(df_all)):
        row = df_all.iloc[frame_i]
        frame_idx = int(row['frame_idx'])

        # Skip frames for performance
        if frame_i % video_skip != 0:
            continue

        # Set timeline based on actual frame index
        rr.set_time_sequence(timeline="frame", sequence=frame_idx)
        rr.set_time_seconds(timeline="timestamp", seconds=row['timestamp'])

        # Check if this frame has a valid pose
        if frame_idx in pose_map:
            pose_data = pose_map[frame_idx]
            pos_imu = pose_data['pos']  # IMU position in world
            quat_imu_xyzw = pose_data['quat']  # IMU orientation in world (world-to-IMU rotation)

            # SLAM outputs IMU poses in world frame
            # We need to compute camera pose from IMU pose using T_imu_cam
            if T_cam_imu is not None:
                # Convert quaternions to scipy Rotation objects
                R_world_imu = Rotation.from_quat(quat_imu_xyzw)  # IMU orientation in world
                R_imu_cam = Rotation.from_quat(T_cam_imu['q'])  # Rotation from IMU to camera (R_i_c)

                # Compute camera orientation in world: R_world_cam = R_world_imu * R_imu_cam
                R_world_cam = R_world_imu * R_imu_cam

                # Compute camera position in world: p_world_cam = p_world_imu + R_world_imu * t_imu_cam
                # t_i_c is translation from IMU to camera expressed in IMU frame
                pos_cam = pos_imu + R_world_imu.apply(T_cam_imu['t'])

                quat_cam_xyzw = R_world_cam.as_quat()  # xyzw format
            else:
                # No transform available, assume poses are already camera poses
                pos_cam = pos_imu
                quat_cam_xyzw = quat_imu_xyzw

            # Log camera pose
            # Transform3D at "world/camera" positions camera relative to world
            # translation: camera position in world
            # rotation: world-to-camera (parent-to-child transform)
            # SLAM outputs world-to-camera quaternions (Twc), so use directly
            quat_world_to_cam = quat_cam_xyzw

            rr.log(
                "world/camera",
                rr.Transform3D(
                    translation=pos_cam,
                    rotation=rr.Quaternion(xyzw=quat_world_to_cam),
                )
            )

            # Log current position (use IMU position for trajectory, camera position for visualization)
            rr.log("world/current_position", rr.Points3D(pos_cam, colors=[255, 0, 0], radii=0.005))

            # Build trajectory history (use IMU positions for consistency)
            trajectory_so_far.append(pos_imu)
            if len(trajectory_so_far) > 1:
                rr.log("world/trajectory_history", rr.LineStrips3D(np.array(trajectory_so_far), colors=[0, 128, 255]))

            # Add camera-oriented axes to show camera frame
            # Standard camera convention (OpenCV/SLAM):
            # - X: right (in image plane)
            # - Y: down (in image plane)
            # - Z: forward (optical axis, pointing INTO the scene)
            # quat_cam_xyzw is world-to-camera, invert to get camera-to-world for drawing axes
            rot_cam = Rotation.from_quat(quat_cam_xyzw).inv()
            cam_x = rot_cam.apply([cam_axis_length, 0, 0])  # X = right
            cam_y = rot_cam.apply([0, cam_axis_length, 0])  # Y = down
            cam_z = rot_cam.apply([0, 0, cam_axis_length])  # Z = forward (optical axis)

            rr.log(
                "world/camera_axes",
                rr.Arrows3D(
                    origins=[pos_cam, pos_cam, pos_cam],
                    vectors=[cam_x, cam_y, cam_z],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ
                )
            )

            # Optionally show IMU frame for debugging
            if show_imu_frame:
                rot_imu = Rotation.from_quat(quat_imu_xyzw)
                imu_x = rot_imu.apply([cam_axis_length, 0, 0])
                imu_y = rot_imu.apply([0, cam_axis_length, 0])
                imu_z = rot_imu.apply([0, 0, cam_axis_length])

                rr.log(
                    "world/imu_axes",
                    rr.Arrows3D(
                        origins=[pos_imu, pos_imu, pos_imu],
                        vectors=[imu_x, imu_y, imu_z],
                        colors=[[200, 100, 100], [100, 200, 100], [100, 100, 200]]  # Faded RGB for XYZ
                    )
                )

        # Log video frame if available
        if video_cap is not None:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_cap.read()

            if ret:
                # Get original frame size
                h_orig, w_orig = frame.shape[:2]

                # Downscale to target resolution for performance and correct projection
                if target_width and target_height:
                    frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame

                h, w = frame_resized.shape[:2]

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # Set intrinsics if not set (use simple approximation)
                if fx is None:
                    fx = w / 2
                    fy = h / 2
                    cx = w / 2
                    cy = h / 2

                # Log pinhole camera - defines the camera projection
                # IMPORTANT: Resolution must match the actual image size
                rr.log(
                    "world/camera",
                    rr.Pinhole(
                        resolution=[w, h],
                        focal_length=[fx, fy],
                        principal_point=[cx, cy],
                    )
                )

                # Log the image under the camera
                rr.log("world/camera", rr.Image(frame_rgb))

        # Print progress (only when actually processing)
        if frame_i % (video_skip * 5) == 0:
            print(f"  Processing frame {frame_i}/{len(df_all)} (frame_idx: {frame_idx}, skip={video_skip})", end='\r')

    print(f"\n✓ Visualization complete!")
    print(f"\nRerun viewer should be open.")
    print(f"\nVisualization elements:")
    print(f"  Coordinate system: SLAM world frame (IMU body frame at initialization)")
    print(f"    - Origin: First IMU position")
    print(f"    - Axes: Arbitrary orientation (depends on how camera was held)")
    print(f"    - World axes at origin: Red=X, Green=Y, Blue=Z")
    print(f"  Trajectory:")
    print(f"    - Green line: full IMU trajectory")
    print(f"    - Blue line: trajectory up to current frame")
    print(f"    - Red point: current camera position")
    print(f"  Camera axes (bright RGB):")
    print(f"    - Red arrow: camera X (right in image)")
    print(f"    - Green arrow: camera Y (down in image)")
    print(f"    - Blue arrow: camera Z (forward/viewing direction)")
    if show_imu_frame:
        print(f"  IMU axes (faded RGB): Shows IMU body frame for debugging")
    print(f"  Video:")
    print(f"    - Downscaled to {target_width}x{target_height} for performance")
    if imu_data:
        print(f"  IMU data:")
        print(f"    - imu/accelerometer/{'{x,y,z}'}: acceleration plots")
        print(f"    - imu/gyroscope/{'{x,y,z}'}: angular velocity plots")
    print(f"\nControls:")
    print(f"  - Use timeline slider to scrub through trajectory")
    print(f"  - Mouse: rotate view, scroll: zoom")
    print(f"  - Select entities in left panel to show/hide")

    if video_cap:
        video_cap.release()

    # Keep the script running so the viewer stays open
    print("\nPress Ctrl+C to exit...")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
