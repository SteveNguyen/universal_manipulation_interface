# GoPro Hero 13 Support Guide

This guide explains how to use the UMI SLAM pipeline with GoPro Hero 13 cameras.

## Overview

The Hero 13 requires separate calibration files and SLAM settings due to differences in sensor characteristics compared to the GoPro 9/10/11 series. The pipeline has been adapted to support both camera types.

## Camera Calibration

### Using Pre-calibrated Settings (Recommended)

The repository includes pre-calibrated files for Hero 13:

- **SLAM settings**: `hero13_720p_slam_settings_gopro9_tbc.yaml` - ORB-SLAM3 configuration for 720p processing
- **Camera intrinsics (2.7K)**: `hero13_proper_intrinsics_2.7k.json` - For ArUco detection at full resolution

**Note**: The current SLAM settings file uses GoPro 9's IMU-to-camera transform (`T_b_c`) as a workaround. This produces good SLAM tracking results in practice. A proper Hero 13 calibration file (`hero13_proper_720p_slam_settings.yaml`) exists but may require further validation.

### Performing Your Own Calibration

If you need to recalibrate for your specific Hero 13 unit:

1. **Camera Intrinsics Calibration**
   - Use [OpenImuCameraCalibrator](https://github.com/urbste/OpenImuCameraCalibrator/)
   - Record calibration videos at both 720p and 2.7K resolutions
   - Extract intrinsics for each resolution

2. **IMU-Camera Calibration**
   - Record synchronized IMU and camera data
   - Use OpenImuCameraCalibrator to compute `T_b_c` (body-to-camera transform)
   - Update the SLAM settings YAML file with new parameters

## Running the SLAM Pipeline

### Basic Usage

For Hero 13 data, use the dedicated pipeline script:

```bash
# Using uv run (recommended - no need to activate venv)
uv run python run_slam_pipeline_hero13.py /path/to/your/session_directory

# Or with activated virtual environment
source .venv/bin/activate
python run_slam_pipeline_hero13.py /path/to/your/session_directory
```

### Pipeline Steps

The pipeline executes these steps automatically:

1. **Step 00**: Extract videos and metadata from GoPro MP4 files
2. **Step 01**: Downsample videos to 720p for SLAM processing
3. **Step 02**: Create SLAM map from the mapping video (with ArUco tag)
4. **Step 03**: Run batch SLAM on all demo videos using the created map
5. **Step 04**: Detect ArUco markers for coordinate frame calibration
6. **Step 05**: Run hand-eye calibration (SLAM-to-tag transform)
7. **Step 06**: Generate dataset plan with trajectory and gripper data

### Expected Output

A successful run will show:
- SLAM tracking success rate (aim for >80%)
- Number of usable demos vs dropped demos
- Final dataset plan JSON file

Example output:
```
99% of raw data are used.
n_dropped_demos 0
```

### Troubleshooting SLAM Issues

If SLAM tracking fails frequently:

1. **Check video quality**: Ensure good lighting and avoid motion blur
2. **Verify mapping video**: The ArUco tag should be clearly visible throughout
3. **Review IMU data**: Ensure GoPro's gyro/accelerometer are working correctly

## Data Visualization

After running the pipeline, visualize the processed data:

```bash
# Using uv run (recommended)
uv run python visualize_dataset_episode.py /path/to/session_directory --episode 0

# Or with activated venv
python visualize_dataset_episode.py /path/to/session_directory --episode 0
```

### Command-line Options

```bash
uv run python visualize_dataset_episode.py --help
```

| Option | Description |
|--------|-------------|
| `--episode, -e` | Episode index to visualize (default: 0) |
| `--video-skip` | Show every Nth video frame for performance (default: 5) |
| `--calibration, -c` | Path to camera intrinsics JSON for accurate reprojection |
| `--app-id` | Rerun application ID (default: 'umi_dataset_viz') |
| `--list-episodes` | List all available episodes and exit |

### Examples

```bash
# List all episodes in a dataset
uv run python visualize_dataset_episode.py data/my_session --list-episodes

# Visualize episode 5 with every 10th frame (faster)
uv run python visualize_dataset_episode.py data/my_session -e 5 --video-skip 10

# Visualize with camera intrinsics for accurate 3D reprojection
uv run python visualize_dataset_episode.py data/my_session -e 0 \
    --calibration hero13_proper_intrinsics_2.7k.json
```

### Visualization Features

The script displays using [Rerun](https://rerun.io/):

- **3D Views**:
  - `world/gripper`: Green trajectory - gripper TCP (Tool Center Point) in world coordinates
  - `slam/camera`: Blue trajectory - camera position in SLAM coordinates
  - Camera reprojection showing video frames at camera locations

- **Time Series**:
  - `gripper/opening`: Gripper finger separation over time
  - `accel/x,y,z`: Accelerometer data
  - `gyro/x,y,z`: Gyroscope data

- **Video**:
  - `video/frame`: Raw video frames synchronized with trajectory data

## Coordinate Systems

### World Frame (Tag Frame)
- Origin: ArUco marker position
- Z-axis: Up
- Used for: Training data, robot deployment

### SLAM Frame
- Origin: Where ORB-SLAM3 initializes
- Arbitrary orientation
- Converted to world frame via `tx_slam_tag` transform

### TCP (Tool Center Point)
- Located at gripper fingertips (when closed)
- Approximately 22cm in front of camera along optical axis
- See `docs/tcp_coordinate_system.md` for detailed offset values

## File Structure

After running the pipeline, your session directory will contain:

```
session_directory/
├── demos/
│   ├── mapping/
│   │   ├── raw_video.mp4
│   │   ├── map_atlas.osa          # SLAM map
│   │   ├── tag_detection.pkl      # ArUco detections
│   │   └── tx_slam_tag.json       # SLAM-to-world transform
│   ├── demo_*/
│   │   ├── raw_video.mp4
│   │   ├── camera_trajectory.csv
│   │   └── ...
│   └── gripper_*/
│       └── gripper_range.json
├── dataset_plan.json              # Final dataset plan
└── ...
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `run_slam_pipeline_hero13.py` | Main pipeline script for Hero 13 |
| `hero13_720p_slam_settings_gopro9_tbc.yaml` | SLAM configuration |
| `hero13_proper_intrinsics_2.7k.json` | Camera intrinsics for ArUco |
| `visualize_dataset_episode.py` | Data visualization tool |
| `docs/tcp_coordinate_system.md` | TCP offset documentation |
| `docs/mapping_video_analysis.md` | Mapping video purpose |

## Differences from GoPro 9/10/11

| Aspect | GoPro 9/10/11 | Hero 13 |
|--------|---------------|---------|
| Pipeline script | `run_slam_pipeline.py` | `run_slam_pipeline_hero13.py` |
| SLAM settings | Built into Docker | `hero13_720p_slam_settings_gopro9_tbc.yaml` |
| Intrinsics | `gopro_intrinsics_2_7k.json` | `hero13_proper_intrinsics_2.7k.json` |
| Resolution | Same (2.7K for ArUco, 720p for SLAM) | Same |
| Mask dimensions | 2028x2704 | 2028x2704 (same) |

## Known Limitations

1. **IMU Calibration**: Currently using GoPro 9's `T_b_c` transform as a workaround
2. **Metadata**: Hero 13 may store camera serial number differently; the pipeline handles this with fallback logic
