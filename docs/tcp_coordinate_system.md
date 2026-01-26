# TCP (Tool Center Point) Coordinate System

## Overview

The TCP (Tool Center Point) in the UMI pipeline represents the position and orientation of the **gripper fingertips**. This document explains how the TCP is computed from camera poses and the relevant offsets.

## TCP Definition

The TCP is located at the **tip of the gripper fingers** when closed, centered between the two fingers.

### Offset from Camera to TCP

The transform from camera optical center to TCP is defined in `scripts_slam_pipeline/06_generate_dataset_plan.py` (lines 103-116):

```python
# tcp to camera transform
# all unit in meters
# y axis in camera frame
cam_to_center_height = 0.086  # constant for UMI

# optical center to mounting screw, positive is when optical center is in front of the mount
CAM_TO_MOUNT_OFFSETS = {
    'gopro9': 0.01465,  # GoPro Hero 9/10/11
    'hero13': 0.01465,  # Hero 13 (same mount design)
}
cam_to_mount_offset = CAM_TO_MOUNT_OFFSETS.get(camera_type, 0.01465)
cam_to_tip_offset = cam_to_mount_offset + tcp_offset  # tcp_offset default = 0.205

pose_cam_tcp = np.array([0, cam_to_center_height, cam_to_tip_offset, 0, 0, 0])
tx_cam_tcp = pose_to_mat(pose_cam_tcp)
```

### Offset Values

| Parameter | Value | Description |
|-----------|-------|-------------|
| `cam_to_center_height` | 8.6 cm | Y offset - perpendicular distance from optical axis |
| `cam_to_mount_offset` | 1.47 cm | Z offset - optical center to mounting screw |
| `tcp_offset` | 20.5 cm | Z offset - mounting screw to gripper fingertip |
| **Total Z offset** | **21.97 cm** | Total distance along optical axis |

### TCP Position in Camera Frame

```
pose_cam_tcp = [X, Y, Z, rx, ry, rz]
             = [0, 0.086, 0.2197, 0, 0, 0]
```

- **X axis:** 0 (centered horizontally)
- **Y axis:** 8.6 cm (below optical center, in camera's "down" direction)
- **Z axis:** 22 cm (in front of camera, along optical axis)
- **Rotation:** Identity (TCP frame aligned with camera frame)

## Coordinate Frames

### Camera Frame (OpenCV/SLAM convention)
- **X:** Right (in image plane)
- **Y:** Down (in image plane)
- **Z:** Forward (optical axis, into the scene)

### World Frame (Tag frame)
- **Origin:** ArUco tag position
- **Z:** Up
- **X, Y:** Horizontal plane

## Transform Chain

To compute TCP pose in world frame:

```
tx_world_tcp = tx_world_slam @ tx_slam_cam @ tx_cam_tcp
```

Where:
- `tx_world_slam` = `tx_tag_slam` = inverse of `tx_slam_tag` (from mapping calibration)
- `tx_slam_cam` = Camera pose from SLAM trajectory
- `tx_cam_tcp` = Fixed offset from camera to TCP (defined above)

## Visualization

When visualizing the dataset:
- **Green trajectory:** TCP/gripper fingertip position
- **Blue trajectory:** Camera position

The expected offset between these trajectories is approximately **23 cm**, which corresponds to the physical distance between the GoPro camera and the gripper fingertips.

### Measured Offset Example (Episode 0)

```
Step 0:
  Gripper (TCP):  [0.13, -0.47, 0.26]
  Camera:         [0.12, -0.58, 0.46]
  Offset:         [0.01,  0.12, -0.20]
  Distance:       23.6 cm
```

The offset is consistent across all timesteps, confirming the rigid mounting between camera and gripper.

## Related Files

- `scripts_slam_pipeline/06_generate_dataset_plan.py` - TCP offset definition (lines 103-116)
- `umi/common/pose_util.py` - Pose transformation utilities
- `scripts/calibrate_slam_tag.py` - SLAM-to-tag calibration
