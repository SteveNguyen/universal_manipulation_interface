# GoPro Hero 13 Setup Guide for UMI

This guide documents how to use GoPro Hero 13 (4K Ultra Wide mode) with the Universal Manipulation Interface pipeline.

## Summary

The Hero 13 requires different calibration than GoPro 10 MaxLens due to:
- Different resolution: 4000x3000 vs 2704x2028
- Different lens distortion profile
- SLAM must run at 720p (960x720) not at full 4K resolution

## ⚠️ CRITICAL: FPS Requirement

**ALL videos (calibration and test) MUST use the SAME FPS!**

| ✅ Correct | ❌ Wrong |
|-----------|----------|
| Calibration at 60 FPS<br>Test at 60 FPS | Calibration at 60 FPS<br>Test at 50 FPS |
| Calibration at 50 FPS<br>Test at 50 FPS | Calibration at 50 FPS<br>Test at 60 FPS |

**Why:** IMU timestamps are synchronized with video frame timing. FPS mismatch causes 20% timing error, breaking visual-inertial fusion.

**Common mistake:** Recording calibration at 60 FPS (NTSC), then switching to 50 FPS (PAL) to avoid flickering.

**Solution:** Choose 60 or 50 FPS at calibration and **never change it**.

📖 **See:** [`HERO13_FPS_REQUIREMENTS.md`](HERO13_FPS_REQUIREMENTS.md) for detailed explanation.

## Calibration Files

**Final working calibration:** `hero13_720p_intrinsics.json`
- Resolution: 960x720 (downscaled from 4K for SLAM)
- Camera RMS: 0.19 pixels (excellent)
- Camera-IMU RMS: 0.65 pixels (excellent)
- IMU noise: GoPro 10 values (0.017 m/s^1.5, 0.0015 rad/s^0.5)

## Calibration Procedure

### 1. Record Calibration Videos

Follow OpenICC standard procedure:

**a) Camera calibration video (`cam/`):**
- ChArUco board perfectly flat (no wrinkles!)
- 30+ frames with varied angles and distances
- Record in 4K Ultra Wide mode
- Slow, smooth movement (avoid blur)

**b) Static IMU bias video (`imu_bias/`):**
- Camera completely still on stable surface
- 30+ seconds
- Used to compute IMU bias (mean offset)
- NOTE: This is NOT used for noise estimation

**c) Camera-IMU calibration video (`cam_imu/`):**
- ChArUco board with dynamic motion
- Include rotation, translation, figure-8 patterns
- 60+ seconds recommended
- This data determines T_cam_imu transformation

### 2. Run OpenICC Calibration

```bash
# Using the OpenImuCameraCalibrator Docker image
python calibrate_hero13_openicc.py
```

**Key parameters:**
- `--camera_model FISHEYE` (Kannala-Brandt with 4 distortion coefficients)
- `--image_downsample_factor 1.0` (use full 4K resolution)
- ChArUco board: 10x8 squares, 21mm square size

**Expected output:**
- `hero13_openicc_dataset/cam/cam_calib_*.json` - Camera intrinsics at 4K
- `hero13_openicc_dataset/cam_imu/cam_imu_calib_result_*.json` - T_cam_imu
- Target: Camera RMS < 1.0 pixels, Camera-IMU RMS < 3.0 pixels

### 3. Generate 720p Calibration

ORB-SLAM3 requires 720p resolution (matching GoPro 10 pipeline):

```bash
python create_720p_calibration.py
```

This script:
- Loads 4K OpenICC calibration results
- Scales intrinsics by 0.24 (4000→960, 3000→720)
- Preserves distortion coefficients (resolution-independent)
- Preserves T_cam_imu (physical transformation)
- Outputs: `hero13_720p_intrinsics.json`

### 4. Validate with SLAM Test

```bash
python test_hero13_slam.py <test_video.MP4> \
  --intrinsics hero13_720p_intrinsics.json \
  --output_dir test_slam_validation
```

**Success criteria:**
- Map created (not "0 maps in atlas")
- 50+ tracked poses
- No "scale too small" errors

## Technical Details

### Why 720p for SLAM?

The UMI pipeline was designed for GoPro 10 MaxLens (2704x2028) which gets processed at 960x720:
1. **Performance**: Monocular-inertial SLAM is computationally intensive
2. **ORB features**: 1250 features at 720p is optimal for tracking
3. **Calibration accuracy**: Lower resolution = lower absolute pixel errors
4. **Internal downscaling**: ORB-SLAM3 automatically resizes based on Camera.width/height

Running at 4K (4000x3000):
- ❌ SLAM fails to initialize or loses tracking
- ❌ "scale too small" errors
- ❌ Insufficient feature matches

### IMU Noise Parameters

**Current approach:** Use GoPro 10 values
- `IMU.NoiseAcc`: 0.017 m/s^1.5
- `IMU.NoiseGyro`: 0.0015 rad/s^0.5
- `IMU.GyroWalk`: 5.0e-5 rad/s^1.5
- `IMU.AccWalk`: 0.0055 m/s^2.5

**Why not measure from Hero 13?**
- Static video method is unreliable (contaminated by motion)
- OpenICC residual method includes spline approximation errors
- Proper method is Allan variance analysis (requires 2+ hours static video)
- GoPro 10/13 have similar IMU specifications

### Camera-IMU Transformation

OpenICC calibrates:
- **T_i_c**: 4x4 transformation matrix from IMU to camera frame
- **Time offset**: IMU to camera synchronization (~-12ms for Hero 13)
- **Reprojection error**: How well IMU+visual align (target: <3 pixels)

The T_i_c is critical for monocular-inertial SLAM to estimate scale.

### Resolution Scaling Math

From 4K to 720p (scale factor = 0.24):

**Scale linearly:**
- fx: 1397.48 → 335.39
- fy: 1397.48 → 335.39
- cx: 1992.55 → 478.21
- cy: 1505.93 → 361.42

**Keep unchanged:**
- k1, k2, k3, k4 (distortion coefficients)
- T_i_c (physical transformation)
- aspect_ratio (intrinsic property)

## Files Reference

```
hero13_720p_intrinsics.json          # Final calibration for SLAM
create_720p_calibration.py            # Script to generate 720p from OpenICC
test_hero13_slam.py                   # SLAM validation tool

hero13_openicc_dataset/
├── cam/
│   └── cam_calib_*.json             # 4K camera intrinsics
├── cam_imu/
│   └── cam_imu_calib_result_*.json  # T_cam_imu calibration
└── imu_bias/
    └── imu_bias_*.json              # IMU bias estimates
```

## Running the Pipeline

### Resolution Options

The pipeline now supports multiple resolutions for Hero 13. All use ffmpeg pre-downscaling for optimal quality.

#### 720p Pipeline (Default - Recommended)

Best results: 90.6% success rate on test dataset.

```bash
uv run python run_slam_pipeline.py data/your_dataset --camera_type hero13
```

**What happens:**
- 4K videos → downscaled to 960x720 using ffmpeg lanczos (high quality)
- SLAM runs at 720p with no internal resize
- ArUco detection uses `hero13_proper_intrinsics_720p.json`
- SLAM settings auto-generated for 720p resolution
- Masks automatically scaled to match resolution

**Why 720p works best:**
- Avoids ORB-SLAM internal resize (which causes feature descriptor inconsistency)
- Pre-downscaling with ffmpeg lanczos preserves quality better
- ORB feature extraction optimized at this resolution
- No relocalization failures from descriptor mismatches

#### 2.7K Pipeline (Alternative)

For testing or comparison with original GoPro 9/10 pipeline:

```bash
uv run python run_slam_pipeline.py data/your_dataset \
    --camera_type hero13 \
    --slam_resolution 2704x2028
```

**What happens:**
- 4K videos → downscaled to 2704x2028 using ffmpeg
- SLAM runs at 2.7K resolution
- ArUco detection uses `hero13_proper_intrinsics_2.7k.json`
- Uses pre-generated `hero13_proper_2.7k_slam_settings.yaml`

**Expected results:**
- May achieve 70-80% success rate
- Higher resolution but more demanding on ORB-SLAM
- Useful for comparison or specific use cases

#### Custom Resolution

For experimentation:

```bash
uv run python run_slam_pipeline.py data/your_dataset \
    --camera_type hero13 \
    --slam_resolution 1920x1080
```

**Requirements for custom resolutions:**
1. Generate intrinsics file at target resolution using `scripts/generate_slam_settings.py`
2. Create calibration JSON using intrinsics scaling
3. Place in `example/calibration/` directory

### Pipeline Steps

The pipeline automatically runs all steps in sequence:

1. **00_process_videos** - Organize raw videos into demo structure
2. **01_extract_gopro_imu** - Extract IMU data from GoPro GPMF metadata
3. **02_create_map** - Create ORB-SLAM3 map from mapping video
   - Applies quality downscaling if resolution specified
   - Auto-generates SLAM settings for target resolution
   - Two-pass mapping for better coverage
4. **03_batch_slam** - Localize all demo videos against map
   - Parallel processing of demos
   - Quality downscaling per video
5. **04_detect_aruco** - Detect ArUco markers for gripper tracking
   - Uses resolution-matched intrinsics
6. **05_run_calibrations** - Compute gripper-camera calibration
7. **06_generate_dataset_plan** - Generate training dataset plan

### Progress Reporting

The pipeline now includes comprehensive progress reporting:

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
Dataset: redcube4k
Total demos: 53
Demos with SLAM tracking: 48/53 (90.6%)
Demos with ArUco detection: 48/53 (90.6%)

✓ Dataset plan generated: dataset_plan.pkl
  Ready for training!
==========================================================
```

### Visualizing Results

Inspect SLAM results with Rerun visualization:

```bash
# Visualize mapping
python visualize_slam_trajectory.py data/your_dataset/demos/mapping \
    --calibration example/calibration/hero13_proper_intrinsics_720p.json

# Visualize individual demo
python visualize_slam_trajectory.py data/your_dataset/demos/demo_XXX \
    --calibration example/calibration/hero13_proper_intrinsics_720p.json \
    --show-video
```

**Visualization features:**
- 3D trajectory with camera poses
- Synchronized video playback
- IMU data plots (accelerometer/gyroscope)
- Frame-by-frame navigation
- Lost frame identification

**Note:** Use the calibration file matching your pipeline resolution (720p or 2.7k).

## Troubleshooting

### SLAM fails to initialize
- Check video has sufficient motion (not too static)
- Verify IMU data extracted: `imu_data.json` should exist
- Check mask isn't blocking too many pixels

### "scale too small" error
- Usually means calibration quality issue
- Re-record calibration videos with better coverage
- Ensure using 720p calibration, not 4K

### High reprojection error (>3 pixels)
- Check ChArUco board flatness
- Ensure good lighting (no glare/shadows)
- Record more varied poses
- Try reducing `--image_downsample_factor`

## Recording Settings for Hero 13

**Recommended GoPro settings:**
- Resolution: 4K (4:3)
- Lens: Ultra Wide
- FPS: 60fps
- Stabilization: OFF (important!)
- Low light: OFF
- Bitrate: High

**Why these settings:**
- 4K 4:3 gives 4000x3000 matching calibration
- Ultra Wide has widest FOV for manipulation tasks
- 60fps matches IMU rate (200Hz with 3-4 IMU samples per frame)
- Stabilization OFF preserves raw IMU data

## Future Improvements

1. **Record proper Allan variance data** (2+ hours static)
   - More accurate IMU noise parameters
   - Better scale estimation in SLAM

2. **Create Docker image with Hero 13 settings**
   - Pre-configured SLAM parameters
   - No need to mount custom settings

3. **Automated calibration validation**
   - Script to check reprojection errors
   - Automated SLAM test on sample video

## Credits

- OpenImuCameraCalibrator: https://github.com/urbste/OpenImuCameraCalibrator
- ORB-SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3
- UMI: https://umi-gripper.github.io
