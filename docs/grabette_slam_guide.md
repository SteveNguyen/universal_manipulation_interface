# Grabette SLAM Guide

Step-by-step procedure to run and analyze SLAM on Grabette recordings.

## Prerequisites

- Docker with `chicheng/orb_slam3:latest` image pulled
- Python environment: `uv` with the project `.venv`
- A recording directory containing `raw_video.mp4` and `imu_data.json`

## 1. Run SLAM

```bash
uv run python scripts_slam_pipeline/02_create_map.py \
  --input_dir rpi_data/grabette3 \
  --camera_type grabette \
  --no_docker_pull \
  --retries 3
```

**What this does:**
1. Resamples IMU to uniform 200Hz (`imu_data_resampled.json`) — required for ORB-SLAM3
2. Auto-selects `rpi_bmi088_slam_settings.yaml` (BMI088 noise parameters)
3. Creates a SLAM mask to exclude the device from feature detection
4. Runs SLAM up to 4 times (1 + 3 retries), keeping the best result
5. If best result >= 90%, runs a second pass with `--load_map` to recover initialization frames
6. Keeps whichever pass gives better tracking

**Options:**
- `--no_docker_pull`: skip pulling Docker image (faster if already cached)
- `--retries N`: retry up to N extra times (default: 0 = single run)
- `--no_mask`: skip masking (for bare camera, not mounted on gripper)
- `-s <path>`: use custom SLAM settings YAML instead of auto-selected one

**Output files:**
- `mapping_camera_trajectory.csv` — trajectory (pose per frame)
- `map_atlas.osa` — ORB-SLAM3 map atlas
- `slam_mask.png` — mask used for SLAM
- `slam_stdout.txt` / `slam_stderr.txt` — SLAM logs

## 2. Analyze Results

```bash
uv run python scripts/analyze_slam.py rpi_data/grabette3/
```

Analyze multiple recordings at once:
```bash
uv run python scripts/analyze_slam.py rpi_data/grabette*/
```

**Output example:**
```
grabette3 (mapping_camera_trajectory.csv)
  Tracking:   715/715 (100.0%) — EXCELLENT
  Init:       0 frames to initialize
  After-init: 100.0% tracked
  Continuity: longest streak 715 frames
  Trajectory: 4.147m length, 0.586m max span
```

**Key metrics:**
- **Tracking %**: overall frames tracked (includes init loss)
- **After-init %**: tracking quality after initialization — this is what matters
- **Init frames**: how many frames lost to VIBA initialization
- **Trajectory length / max span**: sanity check for scale (should match real motion)
- **Large jumps**: possible scale drift or tracking glitches

## 3. Visualize Trajectory

```bash
uv run python visualize_slam_trajectory.py rpi_data/grabette3/
```

Opens a Rerun viewer with 3D trajectory, video frames, and IMU data.

**Options:**
- `--no-video`: skip video frames (faster)
- `--video-skip N`: show every Nth frame (default: 10)
- `--show-imu-frame`: display IMU coordinate axes

## Troubleshooting

### Low tracking (< 50%)

- **Check after-init rate**: if after-init is ~100%, the problem is just slow initialization. Use `--retries 3`.
- **Check SLAM logs**: `cat rpi_data/grabette3/slam_stdout.txt | tail -20`
- **Segfault (exit 139)**: usually means raw IMU was used instead of resampled. Check that `imu_data_resampled.json` exists.

### SLAM crashes with "Empty IMU measurements vector"

Raw IMU has non-uniform timestamps. The pipeline should auto-resample, but if running manually, resample first:
```bash
uv run python scripts/resample_imu.py rpi_data/grabette3/imu_data.json
```

### Inconsistent results across runs

VIBA initialization is non-deterministic. Use `--retries 3` and the pipeline will keep the best result. See `docs/lab_notebook_grabette_slam.md` for details.

### Scale looks wrong (trajectory too large/small)

- Check that `rpi_bmi088_slam_settings.yaml` has correct noise: NoiseGyro=0.009, NoiseAcc=0.09
- Max span should match your actual motion range (e.g., 0.3-0.8m for tabletop manipulation)
