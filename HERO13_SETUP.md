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

## Integration with UMI Pipeline

### 1. Copy Calibration

```bash
cp hero13_720p_intrinsics.json example/calibration/
```

### 2. Update Pipeline Scripts

**scripts_slam_pipeline/02_create_map.py:**
```python
# Change line 80 from:
'--setting', '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml',

# To (requires updating Docker image or mounting custom settings):
'--setting', '/data/hero13_720p_slam_settings.yaml',
```

**scripts_slam_pipeline/04_detect_aruco.py:**
```python
# Update default intrinsics path to Hero 13
@click.option('-ci', '--intrinsics_json',
              default='example/calibration/hero13_720p_intrinsics.json')
```

### 3. SLAM Mask Resolution

Update mask creation in `02_create_map.py` line 62:
```python
# Change from:
slam_mask = np.zeros((2028, 2704), dtype=np.uint8)

# To:
slam_mask = np.zeros((720, 960), dtype=np.uint8)
```

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
