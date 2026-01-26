# Mapping Video with ArUco Tag - Purpose and Analysis

## Overview

The mapping video serves as the **foundation for establishing a global coordinate system** for the entire dataset. It creates a bridge between two coordinate systems:

1. **SLAM coordinate system** - arbitrary, determined by where ORB-SLAM3 initializes
2. **World/Tag coordinate system** - fixed, defined by the ArUco marker's position

## What Happens During Mapping

### Step 1: SLAM Map Creation (`02_create_map_hero13.py`)

- Records a video walking around the workspace while keeping the ArUco tag (ID 13) visible
- ORB-SLAM3 builds a visual feature map (`map_atlas.osa`) using:
  - Monocular camera images (720p downscaled for speed)
  - IMU data (accelerometer + gyroscope at 200Hz)
- Outputs: `map_atlas.osa` (20MB map), `mapping_camera_trajectory.csv`

### Step 2: ArUco Detection (`04_detect_aruco.py`)

- Detects the ArUco tag in every frame of the mapping video
- Computes the tag's 6-DoF pose relative to the camera using camera intrinsics
- Outputs: `tag_detection.pkl`

### Step 3: Calibrate SLAM-to-Tag Transform (`05_run_calibrations.py` → `calibrate_slam_tag.py`)

- For each frame where both SLAM tracking and ArUco detection succeeded:
  ```
  tx_slam_tag = tx_slam_cam @ tx_cam_tag
  ```
  Where:
  - `tx_slam_cam` = camera pose in SLAM coordinates (from trajectory)
  - `tx_cam_tag` = tag pose relative to camera (from ArUco detection)
  - `tx_slam_tag` = tag pose in SLAM coordinates

- Uses geometric median + filtering to find a robust estimate
- Outputs: `tx_slam_tag.json` (4x4 transformation matrix)

## Why This Matters

The `tx_slam_tag` transform (and its inverse `tx_tag_slam`) enables:

1. **Unified World Frame**: All demo trajectories can be converted from SLAM coordinates to a fixed world frame anchored at the ArUco tag (z-up convention)

2. **Relocalization**: Subsequent demo videos use `map_atlas.osa` to relocalize - finding their position in the same SLAM map. This means all demos share the same coordinate system.

3. **Left/Right Gripper Disambiguation**: The tag frame's z-up orientation allows determining which gripper is "left" vs "right" by projecting camera positions.

4. **Robot Deployment**: During inference, detecting the same ArUco tag allows the robot to know where it is in the learned coordinate frame.

## The Transform Chain

```
tx_tag_slam = inverse(tx_slam_tag)

For any demo frame:
  tx_tag_cam = tx_tag_slam @ tx_slam_cam

This gives the camera pose in world coordinates (tag frame).
```

## Key Files Produced

| File | Purpose |
|------|---------|
| `map_atlas.osa` | Visual feature map for relocalization |
| `mapping_camera_trajectory.csv` | Camera trajectory during mapping |
| `tag_detection.pkl` | ArUco poses for every frame |
| `tx_slam_tag.json` | The critical SLAM↔World transform |

## Summary

The ArUco tag essentially acts as a **ground control point** that anchors the entire dataset to a physical reference in the real world. Without this mapping step, each demo video would have its own arbitrary coordinate system, making it impossible to learn consistent spatial relationships across demonstrations.
