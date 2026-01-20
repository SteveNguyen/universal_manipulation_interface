# Hero 13 Technical Investigation Notes

This document captures technical insights gained during Hero 13 integration with UMI.

## Key Findings

### 1. Resolution is Critical for SLAM

**Discovery:** ORB-SLAM3 monocular-inertial works at 720p but fails at 4K.

**Evidence:**
- 4K (4000x3000): SLAM initializes but gets "scale too small" errors
- 720p (960x720): SLAM works reliably with 170+ tracked poses
- Same calibration quality, same video, only difference is resolution in settings

**Why:**
- Computational complexity increases dramatically with resolution
- Feature matching becomes harder at high resolution
- The pipeline was designed/tuned for GoPro 10 at 720p
- ORB-SLAM3 downscales internally based on Camera.width/height parameters

**Lesson:** Always match SLAM settings resolution to what the system was designed for.

### 2. IMU Noise Parameters Cannot Be Extracted from OpenICC Residuals

**Initial hypothesis:** OpenICC residuals (IMU - visual spline) represent IMU sensor noise.

**Result:** WRONG - residuals are 100-500x higher than actual sensor noise.

**Evidence:**
- Hero 13 residuals: NoiseAcc=5.46, NoiseGyro=2.71 (321x higher than GoPro 10)
- GoPro 9 residuals: NoiseAcc=8.26, NoiseGyro=1.19 (485x higher than GoPro 10)
- But GoPro 9 works fine with values 0.017, 0.0015 in SLAM!

**Explanation:**
OpenICC residuals include:
- Spline smoothness constraints (spline can't match every measurement exactly)
- Time synchronization errors
- Rolling shutter effects
- Camera model approximations
- AND sensor noise (small component)

The spline is optimized using BOTH visual and IMU data, so residuals represent total optimization error, not pure sensor noise.

**Lesson:** IMU noise must come from Allan variance analysis or manufacturer specs, not calibration residuals.

### 3. Static IMU Video Purpose is Only for Bias

**Discovery:** OpenICC's `get_imu_biases.py` only computes mean (bias), not std (noise).

**Code evidence:**
```python
bias_gyro  = np.mean(gyro_np, 0)  # Only mean!
bias_accel = np.mean(accl_np, 0)  # No np.std() anywhere
```

**Purpose of static video:**
- Compute IMU bias (offset to subtract from measurements)
- Used during camera-IMU spline optimization
- NOT used for noise parameter estimation

**Proper noise estimation:**
- Allan variance analysis (requires 2+ hours of static data)
- See `OpenImuCameraCalibrator/docs/imu_noise_parameters.md`
- Or use manufacturer specifications

**Lesson:** Don't try to extract noise from short static videos - the method doesn't work.

### 4. Camera-IMU Reprojection Error Scale

**Initial concern:** 2.71 pixels seemed "high" for camera-IMU error.

**Reality:** It's actually normal and good!

**Context:**
- At 4K resolution: 2.71 / 4000 = 0.068% relative error
- Visual-inertial systems typically achieve 1-3 pixels absolute error
- Our 2.71 pixels at 4K → 0.65 pixels at 720p
- GoPro 9 example: 1.70 pixels at 1920x1080

**Lesson:** Judge calibration error relative to image resolution, not absolute pixel values.

### 5. SLAM Settings File Format Matters

**Discovery:** Test script generated settings differently than real pipeline.

**Key differences found:**
```yaml
# Real pipeline (works):
ORBextractor.nFeatures: 1250
Camera.width: 960
Camera.height: 720

# Test script initially (failed):
ORBextractor.nFeatures: 2000
Camera.width: 4000
Camera.height: 3000
```

**Lesson:** When validating, use exact same settings as production system.

### 6. Calibration File Format Variations

**OpenICC camera calibration output:**
```json
{
  "intrinsics": {
    "aspect_ratio": 0.9975,
    "focal_length": 1397.48,  // This is fy
    "skew": 0.0,
    // ...
  }
}
```

**Where fx = fy / aspect_ratio**

**Our convert script output (initially wrong):**
```json
{
  "intrinsics": {
    "focal_length": 1397.48,  // Missing aspect_ratio!
    // ...
  }
}
```

**Lesson:** Carefully match expected calibration file format for tools consuming it.

## Dead Ends Investigated

### 1. Trying Multiple Calibration Approaches
- **Pinhole model:** Wrong model for fisheye lens
- **Different downsampling:** Didn't fix core issues
- **Manual tuning of distortion:** Calibration quality was already good

### 2. Extensive IMU Noise Exploration
- Extracting from static video (motion contaminated)
- Extracting from camera-IMU residuals (includes optimization errors)
- Computing from trimmed static video (still questionable)
- **Final answer:** Just use GoPro 10 values (they work!)

### 3. Investigating T_cam_imu Quality
- Plotting trajectories
- Checking rotation matrix validity
- Analyzing Euler angles
- **Result:** Calibration was always fine, resolution was the issue

## What Actually Worked

1. **Use OpenICC FISHEYE calibration at full 4K**
   - Camera RMS: 0.81 pixels
   - Camera-IMU RMS: 2.71 pixels

2. **Scale intrinsics to 720p** (960x720)
   - Multiply fx, fy, cx, cy by 0.24
   - Keep k1, k2, k3, k4 unchanged
   - Keep T_cam_imu unchanged

3. **Use GoPro 10 IMU noise parameters**
   - NoiseAcc: 0.017 m/s^1.5
   - NoiseGyro: 0.0015 rad/s^0.5

4. **Let ORB-SLAM3 downscale video internally**
   - Feed 4K video
   - Settings specify 720p
   - SLAM handles conversion

## Recommended Workflow

```
1. Record calibration videos in 4K Ultra Wide
2. Run OpenICC calibration at full resolution
3. Scale intrinsics to 720p
4. Use GoPro 10 IMU noise values
5. Validate with test_hero13_slam.py
6. Deploy in production pipeline
```

## Open Questions

1. **Would Hero 13-specific Allan variance improve results?**
   - Probably marginally
   - Effort: High (2+ hours recording)
   - Benefit: Uncertain (GoPro 10 values already work)

2. **Could we run SLAM at higher resolution?**
   - Maybe 1280x960 (1.5x scale)?
   - Would need parameter tuning
   - Worth investigating for better accuracy

3. **Why does higher IMU noise help initialization?**
   - With very low noise (GoPro 10 values at 4K): Failed
   - With higher noise (11x at 4K): Initialized but scale too small
   - At 720p with GoPro 10 values: Works perfectly
   - Suggests interaction between resolution and noise parameters

## Useful Commands

```bash
# Extract IMU data from video
docker run --rm -v $(pwd):/data chicheng/openicc:latest \
  node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js \
  /data/video.mp4 /data/imu_data.json

# Check video resolution
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height -of csv=p=0 video.mp4

# Test SLAM with calibration
python test_hero13_slam.py video.mp4 \
  --intrinsics calibration.json \
  --output_dir test_output

# Create 720p calibration from OpenICC 4K
python create_720p_calibration.py
```

## References

- [OpenICC Documentation](https://github.com/urbste/OpenImuCameraCalibrator)
- [ORB-SLAM3 Paper](https://arxiv.org/abs/2007.11898)
- [Kannala-Brandt Fisheye Model](https://april.eecs.umich.edu/wiki/Camera_suite)
- [Allan Variance for IMUs](https://github.com/gaowenliang/imu_utils)
