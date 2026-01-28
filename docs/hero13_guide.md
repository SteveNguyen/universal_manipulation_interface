# GoPro Hero 13 Support Guide

This guide explains how to use the UMI SLAM pipeline with GoPro Hero 13 cameras.

## Overview

The Hero 13 requires separate calibration files and SLAM settings due to differences in sensor characteristics compared to the GoPro 9/10/11 series. The pipeline has been adapted to support both camera types.

## Camera Calibration

### Using Pre-calibrated Settings (Recommended)

The repository includes pre-calibrated files for Hero 13 at multiple resolutions:

**Camera Intrinsics:**
- `example/calibration/hero13_proper_intrinsics_720p.json` - For 720p pipeline (960×720)
- `example/calibration/hero13_proper_intrinsics_2.7k.json` - For 2.7K pipeline (2704×2028)
- `example/calibration/hero13_proper_intrinsics_4k.json` - For native 4K (4000×3000)

**SLAM Settings:**
- Auto-generated per pipeline run based on video resolution
- Stored in `demos/mapping/slam_settings_auto.yaml`

All intrinsics files include:
- Focal length and principal point (scaled appropriately)
- Fisheye distortion coefficients (k1-k4, resolution-independent)
- IMU-to-camera transform (T_i_c)
- FPS and resolution metadata

**Quality metrics** (from 720p calibration):
- Camera RMS: 0.30 pixels (excellent)
- Camera-IMU RMS: 1.10 pixels (excellent)

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

### Resolution Options

The pipeline supports multiple resolutions for Hero 13. All use ffmpeg pre-downscaling for optimal quality.

#### 720p Pipeline (Default - Recommended)

**Best results: ~90% success rate.**

```bash
# Using uv run (recommended)
uv run python run_slam_pipeline.py /path/to/session --camera_type hero13

# Or with activated virtual environment
source .venv/bin/activate
python run_slam_pipeline.py /path/to/session --camera_type hero13
```

**What happens:**
- 4K videos (4000×3000) → downscaled to 720p (960×720) using ffmpeg lanczos
- SLAM processes at 720p with no internal resize
- ArUco detection uses `hero13_proper_intrinsics_720p.json`
- SLAM settings auto-generated for 720p
- Masks automatically scaled to resolution

**Why this works best:**
- Pre-downscaling with ffmpeg preserves quality better than ORB-SLAM internal resize
- Avoids feature descriptor inconsistency from internal downscaling
- Optimal ORB feature extraction at 720p resolution
- No relocalization failures

#### 2.7K Pipeline (Alternative)

For comparison with original GoPro 9/10 pipeline:

```bash
uv run python run_slam_pipeline.py /path/to/session \
    --camera_type hero13 \
    --slam_resolution 2704x2028
```

**What happens:**
- 4K videos → downscaled to 2704×2028 using ffmpeg
- SLAM processes at 2.7K resolution
- ArUco detection uses `hero13_proper_intrinsics_2.7k.json`
- Uses pre-generated `hero13_proper_2.7k_slam_settings.yaml`

**Expected results:** May achieve 70-80% success rate

#### Custom Resolution

For experimentation:

```bash
uv run python run_slam_pipeline.py /path/to/session \
    --camera_type hero13 \
    --slam_resolution 1920x1080
```

**Requirements:**
1. Generate calibration JSON at target resolution
2. Place in `example/calibration/` directory
3. Pipeline will auto-generate SLAM settings

### Pipeline Steps

The pipeline executes these steps automatically:

1. **00_process_videos** - Extract and organize videos from GoPro MP4 files
2. **01_extract_gopro_imu** - Extract IMU data from GPMF metadata
3. **02_create_map** - Create ORB-SLAM3 map from mapping video
   - Applies quality downscaling (ffmpeg lanczos)
   - Auto-generates SLAM settings for target resolution
   - Two-pass mapping for better coverage
4. **03_batch_slam** - Localize all demo videos against map
   - Parallel processing
   - Per-video quality downscaling
5. **04_detect_aruco** - Detect ArUco markers for coordinate frame calibration
   - Uses resolution-matched intrinsics
6. **05_run_calibrations** - Compute SLAM-to-tag transform (hand-eye calibration)
7. **06_generate_dataset_plan** - Generate dataset plan with trajectory and gripper data

### Expected Output

The pipeline includes comprehensive progress reporting:

**Mapping Results:**
```
==========================================================
MAPPING RESULTS
==========================================================
Pass 2 (final): 1847/1875 frames tracked (98.5%)
Pass 1: 1702/1875 frames tracked (90.8%)
Map size: 45.2 MB
==========================================================
```

**Batch SLAM Results:**
```
==========================================================
BATCH SLAM RESULTS
==========================================================
Demos processed: 48/53 (90.6%)

Tracking quality:
  Excellent (>=90%): 45/48 (93.8%)
  Good (80-90%):     2/48 (4.2%)
  Medium (50-80%):   1/48 (2.1%)
  Poor (<50%):       0/48 (0.0%)

Average tracking quality: 95.2%
==========================================================
```

**Final Summary:**
```
==========================================================
PIPELINE SUMMARY
==========================================================
Dataset: your_session
Total demos: 53
Demos with SLAM tracking: 48/53 (90.6%)
Demos with ArUco detection: 48/53 (90.6%)

✓ Dataset plan generated: dataset_plan.pkl
  Ready for training!
==========================================================
```

### Visualizing SLAM Results

Inspect SLAM trajectory and tracking quality:

```bash
# Visualize mapping
python visualize_slam_trajectory.py /path/to/session/demos/mapping \
    --calibration example/calibration/hero13_proper_intrinsics_720p.json

# Visualize individual demo with video
python visualize_slam_trajectory.py /path/to/session/demos/demo_XXX \
    --calibration example/calibration/hero13_proper_intrinsics_720p.json \
    --show-video
```

**Note:** Use the calibration file matching your pipeline resolution (720p or 2.7k)

**Visualization features:**
- 3D camera trajectory with pose axes
- Synchronized video playback
- IMU data plots (accelerometer/gyroscope)
- Lost frame identification
- Frame-by-frame navigation

### Troubleshooting SLAM Issues

If SLAM tracking fails frequently:

1. **Check video quality**: Ensure good lighting and avoid motion blur
2. **Verify mapping video**: The ArUco tag should be clearly visible throughout
3. **Review IMU data**: Ensure GoPro's gyro/accelerometer are working correctly
4. **Try 720p**: If using custom resolution, try default 720p pipeline

## Generating Training Dataset

After the SLAM pipeline completes, generate the training dataset (replay buffer):

```bash
uv run python scripts_slam_pipeline/07_generate_replay_buffer.py \
    -o /path/to/session_directory/dataset.zarr.zip \
    --camera_type hero13 \
    /path/to/session_directory
```

**Important**: The `--camera_type hero13` flag is required. It enables two Hero 13-specific behaviors:

1. **No cropping**: Unlike GoPro 9 (which has circular fisheye with black borders), Hero 13 produces full rectangular images. The entire image is resized to 224x224, preserving the mirrors.

2. **Correct mask geometry**: Uses a smaller gripper-body-only mask that doesn't cover the mirrors.

### What the mask does

During dataset generation, a mask blacks out the gripper mechanism to prevent the model from overfitting to hardware appearance. For Hero 13:
- **Masked**: Gripper body/mechanism at the bottom center only
- **NOT masked**: Mirrors (they provide useful workspace views)
- **NOT masked**: Gripper fingers (they show gripper state)

### Tuning the mask (optional)

If you need to adjust the Hero 13 mask for your specific setup:

```bash
uv run python scripts/hero13_mask_tuner.py \
    --video /path/to/session_directory/demos/demo_*/raw_video.mp4
```

Edit `GRIPPER_BODY_PTS` in the script and re-run until the mask looks correct.

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
| `run_slam_pipeline.py` | Main pipeline script (use `--camera_type hero13` and optional `--slam_resolution`) |
| `example/calibration/hero13_proper_intrinsics_720p.json` | Camera intrinsics for 720p pipeline |
| `example/calibration/hero13_proper_intrinsics_2.7k.json` | Camera intrinsics for 2.7K pipeline |
| `example/calibration/hero13_proper_intrinsics_4k.json` | Camera intrinsics for native 4K |
| `hero13_proper_2.7k_slam_settings.yaml` | Pre-generated SLAM settings for 2.7K |
| `scripts/generate_slam_settings.py` | Tool to generate SLAM settings from intrinsics |
| `umi/common/camera_config.py` | Camera configuration profiles |
| `visualize_dataset_episode.py` | Dataset visualization tool |
| `visualize_slam_trajectory.py` | SLAM trajectory visualization with IMU |
| `docs/tcp_coordinate_system.md` | TCP offset documentation |
| `docs/mapping_video_analysis.md` | Mapping video purpose |
| `docs/HERO13_CALIBRATION_GUIDE.md` | Complete calibration guide |
| `HERO13_SETUP.md` | Technical setup and calibration guide |

## Differences from GoPro 9/10/11

| Aspect | GoPro 9/10/11 | Hero 13 |
|--------|---------------|---------|
| Pipeline command | `run_slam_pipeline.py` | `run_slam_pipeline.py --camera_type hero13` |
| Native resolution | 2704×2028 (2.7K) | 4000×3000 (4K) |
| SLAM processing | 960×720 (internal resize) | 960×720 (ffmpeg pre-downscale, default) |
| Resolution options | Fixed | 720p (default), 2.7K, or custom via `--slam_resolution` |
| SLAM settings | Built into Docker | Auto-generated per resolution |
| Intrinsics | `gopro_intrinsics_2_7k.json` | `hero13_proper_intrinsics_720p.json` (or 2.7k/4k) |
| Image type | Circular fisheye (black borders) | Full rectangular (no black borders) |
| Dataset generation | Center crop + resize | Resize only (preserves mirrors) |
| Training mask | Large gripper mask (covers corners) | Small gripper body mask (preserves mirrors/fingers) |

## Additional Resources

For detailed technical information:

- **Calibration Process**: See `HERO13_SETUP.md` for complete calibration procedure
- **Resolution Scaling**: See `umi/common/camera_config.py` for intrinsics scaling implementation
- **SLAM Settings Generation**: Use `scripts/generate_slam_settings.py` to create settings for custom resolutions
- **Coordinate Systems**: See `docs/tcp_coordinate_system.md` for gripper TCP definition
