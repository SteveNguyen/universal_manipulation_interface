#!/usr/bin/env python3
"""
Visualize UMI dataset episodes with 3D trajectory, video, IMU, and gripper data.

Usage:
    python visualize_dataset_episode.py data/dataset_redcube2
    python visualize_dataset_episode.py data/dataset_redcube2 --episode 5
    python visualize_dataset_episode.py data/dataset_redcube2 --episode 0 --video-skip 5

Requirements:
    pip install rerun-sdk opencv-python pandas numpy scipy
"""

import sys
import os

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import pickle
import json
import click
import numpy as np
import pandas as pd
import cv2
import rerun as rr
from scipy.spatial.transform import Rotation

from umi.common.pose_util import pose_to_mat


def load_imu_data(imu_json_path):
    """Load and parse IMU data from GoPro GPMF format."""
    if not imu_json_path.exists():
        return None

    with open(imu_json_path) as f:
        raw_imu_data = json.load(f)

    imu_samples = {'accel': [], 'gyro': []}
    for frame_num, frame_data in raw_imu_data.items():
        if not isinstance(frame_data, dict):
            continue
        if 'streams' not in frame_data:
            continue

        streams = frame_data['streams']

        if 'ACCL' in streams and 'samples' in streams['ACCL']:
            for sample in streams['ACCL']['samples']:
                imu_samples['accel'].append({
                    'timestamp': sample['cts'] / 1000.0,
                    'value': sample['value']
                })

        if 'GYRO' in streams and 'samples' in streams['GYRO']:
            for sample in streams['GYRO']['samples']:
                imu_samples['gyro'].append({
                    'timestamp': sample['cts'] / 1000.0,
                    'value': sample['value']
                })

    if imu_samples['accel'] or imu_samples['gyro']:
        return imu_samples
    return None


def load_camera_trajectory(csv_path):
    """Load camera trajectory from SLAM output."""
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    return df


def load_tx_slam_tag(session_dir):
    """Load the SLAM-to-tag transform from mapping calibration."""
    tx_path = session_dir / 'demos' / 'mapping' / 'tx_slam_tag.json'
    if not tx_path.exists():
        return None

    with open(tx_path) as f:
        data = json.load(f)

    tx_slam_tag = np.array(data['tx_slam_tag'])
    return tx_slam_tag


def transform_pose_slam_to_world(slam_pos, slam_quat, tx_tag_slam):
    """Transform camera pose from SLAM frame to world/tag frame."""
    # Build 4x4 pose matrix in SLAM frame
    slam_rot = Rotation.from_quat(slam_quat)
    pose_slam = np.eye(4)
    pose_slam[:3, :3] = slam_rot.as_matrix()
    pose_slam[:3, 3] = slam_pos

    # Transform to world frame: pose_world = tx_tag_slam @ pose_slam
    pose_world = tx_tag_slam @ pose_slam

    world_pos = pose_world[:3, 3]
    world_rot = Rotation.from_matrix(pose_world[:3, :3])
    world_quat = world_rot.as_quat()

    return world_pos, world_quat


@click.command()
@click.argument('session_dir', type=click.Path(exists=True))
@click.option('--episode', '-e', type=int, default=0, help='Episode index to visualize')
@click.option('--video-skip', default=5, help='Show every Nth video frame')
@click.option('--calibration', '-c', type=click.Path(exists=True), default=None,
              help='Path to camera intrinsics JSON')
@click.option('--app-id', default='umi_dataset_viz', help='Rerun application ID')
@click.option('--list-episodes', is_flag=True, help='List all episodes and exit')
def main(session_dir, episode, video_skip, calibration, app_id, list_episodes):
    """Visualize UMI dataset episode with trajectory, video, IMU, and gripper data."""

    session_dir = pathlib.Path(session_dir)
    demos_dir = session_dir / 'demos'

    # Load dataset plan
    plan_path = session_dir / 'dataset_plan.pkl'
    if not plan_path.exists():
        print(f"Error: Dataset plan not found: {plan_path}")
        sys.exit(1)

    with open(plan_path, 'rb') as f:
        plan = pickle.load(f)

    print(f"Loaded dataset plan with {len(plan)} episodes")

    if list_episodes:
        print("\nAvailable episodes:")
        for i, ep in enumerate(plan):
            n_steps = len(ep['episode_timestamps'])
            duration = ep['episode_timestamps'][-1] - ep['episode_timestamps'][0]
            video_path = ep['cameras'][0]['video_path']
            print(f"  {i}: {video_path} ({n_steps} steps, {duration:.2f}s)")
        return

    if episode >= len(plan):
        print(f"Error: Episode {episode} not found. Valid range: 0-{len(plan)-1}")
        sys.exit(1)

    ep = plan[episode]

    # Extract episode data
    timestamps = ep['episode_timestamps']
    gripper_data = ep['grippers'][0]
    camera_data = ep['cameras'][0]

    tcp_pose = gripper_data['tcp_pose']  # (N, 6) - position + rotvec
    gripper_width = gripper_data['gripper_width']  # (N,)

    video_path = demos_dir / camera_data['video_path']
    video_start, video_end = camera_data['video_start_end']

    video_dir = video_path.parent
    n_steps = len(timestamps)
    fps = 60.0  # GoPro fps

    print(f"\n=== Episode {episode} ===")
    print(f"Video: {video_path}")
    print(f"Frames: {video_start} - {video_end}")
    print(f"Steps: {n_steps}")
    print(f"Duration: {timestamps[-1] - timestamps[0]:.2f}s")
    print(f"Gripper opening: {gripper_width.min()*1000:.1f} - {gripper_width.max()*1000:.1f} mm")

    # Load SLAM-to-tag transform
    tx_slam_tag = load_tx_slam_tag(session_dir)
    if tx_slam_tag is not None:
        tx_tag_slam = np.linalg.inv(tx_slam_tag)
        print(f"Loaded SLAM-to-World transform")
    else:
        tx_tag_slam = None
        print(f"Warning: No tx_slam_tag.json found, SLAM and World frames will be the same")

    # Load auxiliary data
    imu_data = load_imu_data(video_dir / 'imu_data.json')
    if imu_data:
        print(f"IMU: {len(imu_data['accel'])} accel, {len(imu_data['gyro'])} gyro samples")

    camera_traj = load_camera_trajectory(video_dir / 'camera_trajectory.csv')
    if camera_traj is not None:
        print(f"Camera trajectory: {len(camera_traj)} poses")

    # Load calibration if provided
    fx, fy, cx, cy = None, None, None, None
    if calibration:
        with open(calibration) as f:
            calib_data = json.load(f)
            if 'intrinsics' in calib_data:
                aspect_ratio = calib_data['intrinsics'].get('aspect_ratio', 1.0)
                fy = calib_data['intrinsics']['focal_length']
                fx = fy / aspect_ratio
                cx = calib_data['intrinsics']['principal_pt_x']
                cy = calib_data['intrinsics']['principal_pt_y']
                print(f"Loaded calibration: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Initialize rerun
    rr.init(app_id, spawn=True)

    # ==================== WORLD FRAME (Tag/Real coordinates) ====================
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    axis_length = 0.2
    rr.log(
        "world/origin_axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ),
        static=True
    )

    # Gripper trajectory in world frame (GREEN)
    gripper_positions = tcp_pose[:, :3]
    gripper_rotvecs = tcp_pose[:, 3:]

    rr.log("world/gripper_trajectory", rr.LineStrips3D(gripper_positions, colors=[0, 255, 0]), static=True)
    rr.log("world/gripper_start", rr.Points3D(gripper_positions[0], colors=[0, 255, 0], radii=0.01), static=True)
    rr.log("world/gripper_end", rr.Points3D(gripper_positions[-1], colors=[255, 0, 0], radii=0.01), static=True)

    # ==================== SLAM FRAME (Raw SLAM coordinates) ====================
    rr.log("slam", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    rr.log(
        "slam/origin_axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ),
        static=True
    )

    # Camera trajectory in SLAM frame (BLUE)
    if camera_traj is not None:
        valid_cam = camera_traj[~camera_traj['is_lost']]
        if len(valid_cam) > 0:
            slam_cam_positions = valid_cam[['x', 'y', 'z']].values
            rr.log("slam/camera_trajectory", rr.LineStrips3D(slam_cam_positions, colors=[0, 128, 255]), static=True)
            rr.log("slam/camera_start", rr.Points3D(slam_cam_positions[0], colors=[0, 255, 0], radii=0.01), static=True)
            rr.log("slam/camera_end", rr.Points3D(slam_cam_positions[-1], colors=[255, 0, 0], radii=0.01), static=True)

            # Also show camera trajectory transformed to world frame
            if tx_tag_slam is not None:
                world_cam_positions = []
                for pos in slam_cam_positions:
                    pos_h = np.array([*pos, 1.0])
                    world_pos = (tx_tag_slam @ pos_h)[:3]
                    world_cam_positions.append(world_pos)
                world_cam_positions = np.array(world_cam_positions)
                rr.log("world/camera_trajectory", rr.LineStrips3D(world_cam_positions, colors=[0, 128, 255]), static=True)

    # ==================== TIME SERIES SETUP ====================
    # Configure plot appearances
    rr.log("gripper/opening", rr.SeriesLines(colors=[[255, 128, 0]], names=["opening (mm)"]), static=True)

    if imu_data:
        rr.log("accel/x", rr.SeriesLines(colors=[[255, 0, 0]], names=["X"]), static=True)
        rr.log("accel/y", rr.SeriesLines(colors=[[0, 255, 0]], names=["Y"]), static=True)
        rr.log("accel/z", rr.SeriesLines(colors=[[0, 0, 255]], names=["Z"]), static=True)
        rr.log("gyro/x", rr.SeriesLines(colors=[[255, 0, 0]], names=["X"]), static=True)
        rr.log("gyro/y", rr.SeriesLines(colors=[[0, 255, 0]], names=["Y"]), static=True)
        rr.log("gyro/z", rr.SeriesLines(colors=[[0, 0, 255]], names=["Z"]), static=True)

    # ==================== LOG TIME SERIES DATA ====================
    # Use video time (seconds from video start) as common timeline
    # Episode starts at frame video_start, so t=0 corresponds to frame video_start
    episode_start_time = video_start / fps

    # Log gripper data
    print("\nLogging gripper time series...")
    for step_i in range(n_steps):
        # Video time for this step
        frame_idx = video_start + step_i
        video_time = frame_idx / fps

        rr.set_time_seconds(timeline="video_time", seconds=video_time)
        rr.set_time_sequence(timeline="step", sequence=step_i)

        gw_mm = gripper_width[step_i] * 1000
        rr.log("gripper/opening", rr.Scalars(gw_mm))

    # Log IMU data (already in video time)
    if imu_data:
        print("Logging IMU data...")
        for sample in imu_data['accel']:
            t = sample['timestamp']
            # Only log IMU data within episode time range
            if episode_start_time <= t <= (video_end / fps):
                v = sample['value']
                rr.set_time_seconds(timeline="video_time", seconds=t)
                rr.log("accel/x", rr.Scalars(v[0]))
                rr.log("accel/y", rr.Scalars(v[1]))
                rr.log("accel/z", rr.Scalars(v[2]))

        for sample in imu_data['gyro']:
            t = sample['timestamp']
            if episode_start_time <= t <= (video_end / fps):
                v = sample['value']
                rr.set_time_seconds(timeline="video_time", seconds=t)
                rr.log("gyro/x", rr.Scalars(v[0]))
                rr.log("gyro/y", rr.Scalars(v[1]))
                rr.log("gyro/z", rr.Scalars(v[2]))

    # ==================== OPEN VIDEO ====================
    video_cap = None
    if video_path.exists():
        video_cap = cv2.VideoCapture(str(video_path))
        if video_cap.isOpened():
            actual_fps = video_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video: {total_frames} frames at {actual_fps:.2f} fps, {frame_w}x{frame_h}")
            fps = actual_fps  # Use actual fps
        else:
            print(f"Warning: Could not open video: {video_path}")
            video_cap = None

    # Build camera pose map from trajectory (in SLAM coordinates)
    cam_pose_map = {}
    if camera_traj is not None:
        for _, row in camera_traj.iterrows():
            if row['is_lost']:
                continue
            frame_idx = int(row['frame_idx'])
            cam_pose_map[frame_idx] = {
                'pos': np.array([row['x'], row['y'], row['z']]),
                'quat': np.array([row['q_x'], row['q_y'], row['q_z'], row['q_w']]),
                'timestamp': row['timestamp']
            }

    # ==================== ANIMATE THROUGH EPISODE ====================
    print(f"\nLogging 3D visualization...")

    gripper_axis_length = 0.05
    cam_axis_length = 0.08

    for step_i in range(n_steps):
        frame_idx = video_start + step_i

        # Skip frames for performance
        if step_i % video_skip != 0:
            continue

        video_time = frame_idx / fps

        rr.set_time_seconds(timeline="video_time", seconds=video_time)
        rr.set_time_sequence(timeline="step", sequence=step_i)
        rr.set_time_sequence(timeline="frame", sequence=frame_idx)

        # ===== WORLD FRAME: Gripper pose =====
        pos = gripper_positions[step_i]
        rotvec = gripper_rotvecs[step_i]
        rot = Rotation.from_rotvec(rotvec)

        rr.log("world/gripper_current", rr.Points3D(pos, colors=[255, 128, 0], radii=0.008))

        gx = rot.apply([gripper_axis_length, 0, 0])
        gy = rot.apply([0, gripper_axis_length, 0])
        gz = rot.apply([0, 0, gripper_axis_length])

        rr.log(
            "world/gripper_axes",
            rr.Arrows3D(
                origins=[pos, pos, pos],
                vectors=[gx, gy, gz],
                colors=[[255, 100, 100], [100, 255, 100], [100, 100, 255]]
            )
        )

        # ===== CAMERA POSES =====
        if frame_idx in cam_pose_map:
            cam_data = cam_pose_map[frame_idx]
            slam_cam_pos = cam_data['pos']
            slam_cam_quat = cam_data['quat']

            # SLAM frame: camera position and axes
            rr.log("slam/camera_current", rr.Points3D(slam_cam_pos, colors=[255, 128, 0], radii=0.008))

            slam_cam_rot = Rotation.from_quat(slam_cam_quat).inv()
            scx = slam_cam_rot.apply([cam_axis_length, 0, 0])
            scy = slam_cam_rot.apply([0, cam_axis_length, 0])
            scz = slam_cam_rot.apply([0, 0, cam_axis_length])

            rr.log(
                "slam/camera_axes",
                rr.Arrows3D(
                    origins=[slam_cam_pos, slam_cam_pos, slam_cam_pos],
                    vectors=[scx, scy, scz],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
                )
            )

            # SLAM frame: camera transform for video projection
            rr.log(
                "slam/camera",
                rr.Transform3D(
                    translation=slam_cam_pos,
                    rotation=rr.Quaternion(xyzw=slam_cam_quat),
                )
            )

            # WORLD frame: transform camera pose from SLAM to world
            if tx_tag_slam is not None:
                world_cam_pos, world_cam_quat = transform_pose_slam_to_world(
                    slam_cam_pos, slam_cam_quat, tx_tag_slam)
            else:
                world_cam_pos = slam_cam_pos
                world_cam_quat = slam_cam_quat

            rr.log("world/camera_current", rr.Points3D(world_cam_pos, colors=[0, 128, 255], radii=0.008))

            world_cam_rot = Rotation.from_quat(world_cam_quat).inv()
            wcx = world_cam_rot.apply([cam_axis_length, 0, 0])
            wcy = world_cam_rot.apply([0, cam_axis_length, 0])
            wcz = world_cam_rot.apply([0, 0, cam_axis_length])

            rr.log(
                "world/camera_axes",
                rr.Arrows3D(
                    origins=[world_cam_pos, world_cam_pos, world_cam_pos],
                    vectors=[wcx, wcy, wcz],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
                )
            )

            # WORLD frame: camera transform for video projection
            rr.log(
                "world/camera",
                rr.Transform3D(
                    translation=world_cam_pos,
                    rotation=rr.Quaternion(xyzw=world_cam_quat),
                )
            )

            has_camera_pose = True
        else:
            has_camera_pose = False

        # ===== VIDEO FRAME (only once) =====
        if video_cap is not None:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_cap.read()

            if ret:
                h, w = frame.shape[:2]

                # Downscale for performance
                target_w, target_h = 960, 720
                if w != target_w or h != target_h:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    h, w = target_h, target_w

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Set intrinsics
                if fx is None:
                    fx_use, fy_use = w * 0.8, w * 0.8
                    cx_use, cy_use = w / 2, h / 2
                else:
                    scale_x = target_w / 2704 if w == target_w else 1.0
                    scale_y = target_h / 2028 if h == target_h else 1.0
                    fx_use = fx * scale_x
                    fy_use = fy * scale_y
                    cx_use = cx * scale_x
                    cy_use = cy * scale_y

                # Log video frame only once under video/
                rr.log("video/frame", rr.Image(frame_rgb))

                # Log pinhole + image for 3D reprojection (only if camera pose available)
                if has_camera_pose:
                    pinhole = rr.Pinhole(
                        resolution=[w, h],
                        focal_length=[fx_use, fy_use],
                        principal_point=[cx_use, cy_use],
                    )
                    rr.log("slam/camera", pinhole)
                    rr.log("slam/camera", rr.Image(frame_rgb))
                    rr.log("world/camera", pinhole)
                    rr.log("world/camera", rr.Image(frame_rgb))

        # Progress
        if step_i % (video_skip * 10) == 0:
            print(f"  Step {step_i}/{n_steps}", end='\r')

    print(f"\n\nVisualization complete!")
    print(f"\n=== Layout ===")
    print(f"")
    print(f"3D VIEWS:")
    print(f"  world/ - Real coordinates (ArUco tag = origin)")
    print(f"    - gripper_trajectory (GREEN): Gripper path")
    print(f"    - camera_trajectory (BLUE): Camera path transformed to world")
    print(f"    - camera: Video reprojection")
    print(f"")
    print(f"  slam/ - Raw SLAM coordinates")
    print(f"    - camera_trajectory (BLUE): Camera path")
    print(f"    - camera: Video reprojection")
    print(f"")
    print(f"TIME SERIES (drag each folder to a Time Series view):")
    print(f"  gripper/ - Gripper opening (mm)")
    if imu_data:
        print(f"  accel/ - Accelerometer X,Y,Z (m/s^2)")
        print(f"  gyro/ - Gyroscope X,Y,Z (rad/s)")
    print(f"")
    print(f"VIDEO:")
    print(f"  video/frame - Raw video")
    print(f"")
    print(f"=== Timeline ===")
    print(f"Use 'video_time' timeline to sync all data")

    if video_cap:
        video_cap.release()

    # Keep running
    print("\nPress Ctrl+C to exit...")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
