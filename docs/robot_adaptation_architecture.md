# Robot Adaptation Architecture

This document explains how UMI adapts trajectory data for different robot arms (UR5, Franka, ARX X5).

## Key Finding: Robot-Agnostic Training

The UMI approach is **robot-agnostic during training**. The policy learns in **Cartesian/task space** (TCP pose), not joint space. Inverse kinematics is only used at **execution time** by the robot controller.

## Kinematic-Based Data Filtering (Paper vs Implementation)

The UMI paper describes kinematic-based filtering:

> "When the robot's base location and kinematics are known, the absolute end-effector pose recovered by SLAM allows kinematics and dynamics feasibility filtering on the demonstration data."

**Current implementation status**: This filtering is marked as TODO in the codebase.

**File**: `scripts_slam_pipeline/06_generate_dataset_plan.py` (line 698)
```python
# TODO: handle optinal robot cal based filtering
is_step_valid = is_tracked.copy()
```

### What IS Implemented (SLAM-based filtering)
- Drops frames where SLAM tracking failed
- Drops episodes with too few valid frames (<60)
- Drops episodes with manual `check_result.txt=false` markers

### What is NOT Implemented (kinematic filtering)
To implement the paper's kinematic filtering, you would need:
1. Robot URDF/kinematics model
2. Robot base position relative to ArUco tag origin
3. IK solver to check pose feasibility
4. Joint limit and velocity limit checks

### Runtime Safety (Implemented)
Instead of filtering training data, the execution scripts have **reactive collision avoidance**:

```yaml
# example/eval_robots_config.yaml
"height_threshold": -0.024,  # Table collision avoidance
"sphere_radius": 0.1,        # Inter-gripper collision
"sphere_center": [0, -0.06, -0.185]
```

```python
# scripts_real/replay_real_bimanual_umi.py
def solve_table_collision(ee_pose, gripper_width, height_threshold):
    # Lifts gripper if it would go below table

def solve_sphere_collision(ee_poses, robots_config):
    # Pushes grippers apart if collision spheres overlap
```

## Architecture Overview

```
Recording (any robot)     Training (robot-agnostic)     Execution (robot-specific)
        │                          │                            │
   TCP Pose ──────────────> Policy learns ──────────────> Robot Controller
 [x,y,z,rx,ry,rz]         in task space              handles IK internally
```

This design allows demonstrations recorded with one robot to potentially be executed on a different robot, as long as the workspace and task are compatible.

## Robot-Specific Adaptation Points

### 1. Execution Controllers (robot-specific)

| Robot | Controller | Key Differences |
|-------|------------|-----------------|
| UR5 | `RTDEInterpolationController` | 500Hz, RTDE protocol, lookahead smoothing |
| Franka | `FrankaInterpolationController` | 200Hz, impedance control (Kx, Kxd gains) |

**File**: `umi/real_world/umi_env.py` (lines 231-260)

```python
if robot_type.startswith('ur5'):
    robot = RTDEInterpolationController(
        frequency=500,          # UR5 CB3 RTDE frequency
        lookahead_time=0.1,     # Trajectory smoothing
        gain=300,               # Proportional gain
        tcp_offset_pose=[0, 0, tcp_offset, 0, 0, 0],
        ...
    )
elif robot_type.startswith('franka'):
    robot = FrankaInterpolationController(
        frequency=200,          # Lower frequency for Franka
        Kx_scale=1.0,           # Impedance stiffness
        Kxd_scale=np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0]),  # Damping
        ...
    )
```

### 2. TCP Offset (hardware-specific)

The `tcp_offset` parameter (default 0.21m) accounts for the gripper mounting. This is the distance from the robot flange to the tool center point (gripper fingertips).

```python
tcp_offset_pose = [0, 0, tcp_offset, 0, 0, 0]  # Tool center point offset
```

See `docs/tcp_coordinate_system.md` for detailed offset calculations.

### 3. Pose Representation (training abstraction)

**File**: `diffusion_policy/common/pose_repr_util.py`

The policy uses **relative pose representation** to be robot-agnostic:

| Representation | Description | Use Case |
|----------------|-------------|----------|
| `abs` | Absolute world frame | Robot-dependent, not recommended |
| `relative` | Relative to current pose | **Default**, robot-independent |
| `delta` | Differential poses | Alternative representation |

```python
def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep='relative', backward=False):
    """
    Convert pose representation between different formats.

    backward=False: Training transform (world → policy space)
    backward=True:  Execution transform (policy space → world)
    """
    if pose_rep == 'relative':
        if not backward:
            out = np.linalg.inv(base_pose_mat) @ pose_mat  # To relative
        else:
            out = base_pose_mat @ pose_mat  # Back to absolute
```

### 4. Trajectory Filtering (in sampler)

**File**: `diffusion_policy/common/sampler.py`

The sampler applies temporal filtering and interpolation:

- **Position**: Linear interpolation
- **Rotation**: SLERP (Spherical Linear Interpolation) for smooth rotations
- **Temporal downsampling**: Configurable step skipping via `key_down_sample_steps`
- **Latency compensation**: Adjustable via `key_latency_steps`

```python
# For rotation data: SLERP interpolation
if 'rot' in key:
    slerp = st.Slerp(times=..., rotations=st.Rotation.from_rotvec(input_arr))
    output = slerp(idx_with_latency)
else:
    # For position: linear interpolation
    interp = si.interp1d(x=times, y=input_arr, axis=0)
    output = interp(idx_with_latency)
```

## Data Flow

### Training Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ TRAINING DATA PIPELINE                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Recording          Dataset Transform           Policy Input     │
│  ─────────────          ─────────────────           ────────────     │
│  TCP Pose (6D)    →     Pose Repr (relative)   →    10D tensor       │
│  [x,y,z,rx,ry,rz]       + temporal filtering        [pos3 + rot6d    │
│  + gripper_width        + SLERP interpolation        + gripper]      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Execution Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ EXECUTION PIPELINE                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Policy Output      Inverse Transform       Robot Controller         │
│  ─────────────      ─────────────────       ──────────────────       │
│  10D action    →    Absolute TCP pose  →    IK + Motion Planning     │
│  (relative)         (backward=True)         (UR5 RTDE / Franka)      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Pose Representations

### 6D Pose (Recording/Execution)

Standard pose representation used in recording and robot commands:
- Position: `[x, y, z]` in meters
- Orientation: `[rx, ry, rz]` axis-angle representation

### 10D Pose (Policy)

Internal policy representation for better learning:
- Position: `[x, y, z]` (3D)
- Orientation: 6D rotation representation (first two columns of rotation matrix)
- Gripper: `[width]` (1D)

**File**: `umi/common/pose_util.py`

```python
def mat_to_pose10d(mat):
    """Convert 4x4 matrix to 10D pose [x, y, z, rot6d]"""
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    d6 = mat_to_rot6d(rotmat)  # First 2 rows of rotation matrix
    return np.concatenate([pos, d6], axis=-1)

def rot6d_to_mat(d6):
    """Convert 6D rotation back to 3x3 matrix using Gram-Schmidt"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)
```

## Multi-Robot (Bimanual) Support

For bimanual setups, additional relative poses are computed:

**File**: `diffusion_policy/dataset/umi_dataset.py`

```python
# Generate relative poses between robots
for robot_id in range(self.num_robot):
    for other_robot_id in range(self.num_robot):
        if robot_id == other_robot_id:
            continue
        # Compute robot0 pose relative to robot1, and vice versa
        rel_pose = convert_pose_mat_rep(
            pose_mat,
            base_pose_mat=other_pose_mat[-1],
            pose_rep='relative'
        )
        obs_dict[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_pose[:, :3]
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `umi/real_world/umi_env.py` | Robot type selection, TCP offset configuration |
| `umi/real_world/rtde_interpolation_controller.py` | UR5 controller (RTDE protocol) |
| `umi/real_world/franka_interpolation_controller.py` | Franka controller (impedance control) |
| `diffusion_policy/common/pose_repr_util.py` | Pose representation conversion |
| `diffusion_policy/common/sampler.py` | Temporal filtering, SLERP interpolation |
| `diffusion_policy/dataset/umi_dataset.py` | Multi-robot dataset handling |
| `umi/common/pose_util.py` | Low-level pose/rotation math (10D, 6D) |
| `umi/real_world/real_inference_util.py` | Real-time obs↔action transforms |

## Task Configurations

### Single-arm (`diffusion_policy/config/task/umi.yaml`)
- Action shape: `[10]` (3D pos + 6D rot + 1D gripper)
- Observations: `camera0_rgb`, `robot0_eef_pos`, `robot0_eef_rot_axis_angle`, `robot0_gripper_width`

### Bimanual (`diffusion_policy/config/task/umi_bimanual.yaml`)
- Action shape: `[20]` (two sets of 10D actions)
- Observations: Both robot poses plus relative poses (`robot0_eef_pos_wrt1`, etc.)

## ARX X5 Support

ARX X5 support is provided in a separate repository: [umi-arx](https://github.com/real-stanford/umi-arx)

The adaptation follows the same pattern:
1. New controller class handling ARX-specific protocol
2. Same Cartesian TCP command interface
3. Robot-specific parameters (frequency, gains, offsets)

## Adding a New Robot

To add support for a new robot arm:

1. **Create a controller** in `umi/real_world/`:
   - Inherit from or follow the pattern of `RTDEInterpolationController`
   - Implement `schedule_waypoint(pose, target_time)` method
   - Handle robot-specific communication protocol

2. **Register in `umi_env.py`**:
   ```python
   elif robot_type.startswith('new_robot'):
       robot = NewRobotController(...)
   ```

3. **Configure TCP offset** for your gripper mounting

4. **Test** with existing trained policies (they should work if workspace is similar)
